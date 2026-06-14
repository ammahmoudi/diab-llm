import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow importing fairness losses from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fairness.loss_functions.fairness_losses import EqualizedOddsLoss, HypoglycemiaTPREqualityLoss


class DistillationTrainer:
    def __init__(
        self,
        teacher,
        student,
        dataloader,
        optimizer,
        device,
        accelerator=None,
        scheduler=None,
        early_stopping=None,
        alpha=0.5,
        beta=0.5,
        train_epochs=10,
        logger=None,
        fairness_weight: float = 0.0,
        target_threshold: float = 70.0,
        pred_threshold: float = 70.0,
        teacher_calibration_enabled: bool = False,
        teacher_calibration_feature: str = None,
        teacher_calibration_group0_offset: float = 0.0,
        teacher_calibration_group1_offset: float = 0.0,
    ):
        self.teacher = teacher
        self.student = student
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.alpha = alpha
        self.beta = beta
        self.loss_fn = nn.MSELoss()
        self.train_epochs = train_epochs
        self.logger = logger or logging.getLogger(__name__)
        self.pred_len = self.teacher.prediction_length
        self.context_len = self.teacher.sequence_length

        # K1: calibrated soft labels (group-conditional teacher output shifts)
        self.teacher_calibration_enabled = bool(teacher_calibration_enabled)
        self.teacher_calibration_feature = teacher_calibration_feature
        self.teacher_calibration_group0_offset = float(teacher_calibration_group0_offset)
        self.teacher_calibration_group1_offset = float(teacher_calibration_group1_offset)

        # Fairness-aware loss (disabled when fairness_weight == 0)
        self.fairness_weight = fairness_weight
        if fairness_weight > 0:
            # HypoglycemiaTPREqualityLoss fixes the core issue with EqualizedOddsLoss:
            # rare hypoglycemia windows (~1-5% of batches) caused near-zero gradients.
            # The new loss uses soft-TPR equalization + focal regression upweighting
            # on true hypo timesteps, giving a stable gradient even with rare events.
            self.eo_loss_fn = HypoglycemiaTPREqualityLoss(
                base_loss=None,
                fairness_weight=1.0,
                hypo_threshold=target_threshold,   # 70.0 mg/dL
                focal_gamma=3.0,
                soft_slope=0.1,
            )
        else:
            self.eo_loss_fn = None

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _apply_teacher_group_calibration(self, y_teacher, batch_groups):
        """Apply group-conditional additive offsets to teacher outputs.

        Group labels follow project convention: group 0 / group 1 (e.g. Female/Male
        for gender). Offsets are scalar glucose shifts in mg/dL.
        """
        if (not self.teacher_calibration_enabled) or (batch_groups is None):
            return y_teacher

        offsets = torch.where(
            batch_groups == 0,
            torch.tensor(self.teacher_calibration_group0_offset, device=y_teacher.device, dtype=y_teacher.dtype),
            torch.tensor(self.teacher_calibration_group1_offset, device=y_teacher.device, dtype=y_teacher.dtype),
        )
        # Broadcast offsets over [pred_len, channels]
        return y_teacher + offsets.view(-1, 1, 1)

    def train(self):
        train_loss_l = []
        for epoch in range(self.train_epochs):
            self.student.train()
            total_loss = 0.0
            total_loss_gt = 0.0
            total_loss_teacher = 0.0
            total_loss_fairness = 0.0

            mse_loss_fn = nn.MSELoss()

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                # Support 4-element batches (standard) and 5-element batches (with group labels)
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_groups = batch
                    batch_groups = batch_groups.to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_groups = None

                batch_x, batch_y, batch_x_mark, batch_y_mark = [
                    b.float().to(self.device) for b in (batch_x, batch_y, batch_x_mark, batch_y_mark)
                ]
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.context_len, :], dec_inp], dim=1)

                with torch.no_grad():
                    y_teacher = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    y_teacher = self._apply_teacher_group_calibration(y_teacher, batch_groups)

                y_student = self.student(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                y_true = batch_y[:, -self.pred_len :, :]

                # 1. Ground-truth loss
                loss_gt = mse_loss_fn(y_student, y_true)
                # 2. Teacher distillation loss (match teacher output)
                loss_teacher = mse_loss_fn(y_student, y_teacher)

                # Combine losses
                loss = self.alpha * loss_gt + self.beta * loss_teacher

                # 3. Optional fairness regularisation (EO Gap on hypoglycemia detection)
                loss_fairness = torch.tensor(0.0, device=self.device)
                if self.eo_loss_fn is not None and batch_groups is not None:
                    unique_groups = torch.unique(batch_groups)
                    if len(unique_groups) == 2:
                        # EqualizedOddsLoss returns base_loss + fairness_penalty;
                        # here we only want the fairness penalty, so we pass a
                        # zero base_loss by using predictions == targets as base.
                        y_flat = y_student.reshape(y_student.shape[0], -1).mean(dim=1)
                        t_flat = y_true.reshape(y_true.shape[0], -1).mean(dim=1)
                        eo_combined = self.eo_loss_fn(y_flat, t_flat, batch_groups)
                        # eo_combined = base_loss(0) + fairness_penalty; subtract base
                        base_part = mse_loss_fn(y_flat, t_flat)
                        loss_fairness = eo_combined - base_part
                        loss = loss + self.fairness_weight * loss_fairness

                self.optimizer.zero_grad()
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                total_loss += loss.item()
                total_loss_gt += loss_gt.item()
                total_loss_teacher += loss_teacher.item()
                total_loss_fairness += loss_fairness.item()

            avg_loss = total_loss / len(self.dataloader)
            avg_loss_gt = total_loss_gt / len(self.dataloader)
            avg_loss_teacher = total_loss_teacher / len(self.dataloader)
            avg_loss_fairness = total_loss_fairness / len(self.dataloader)

            train_loss_l.append(avg_loss)

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1} | Total Loss: {avg_loss:.7f} | GT Loss: {avg_loss_gt:.7f} "
                    f"| Teacher Loss: {avg_loss_teacher:.7f} | Fairness Loss: {avg_loss_fairness:.7f}"
                )

            if self.early_stopping:
                # Provide a path to save the best model
                save_path = os.path.join("logs", "best_student.pth")
                self.early_stopping(avg_loss, self.student, save_path)
                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info("Early stopping triggered.")
                    break

        return train_loss_l
