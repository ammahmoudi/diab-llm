import logging
import os
import torch
import torch.nn as nn
from tqdm import tqdm

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
        kl_weight=0.1,
        temperature=3.0,
        train_epochs=10,
        logger=None,
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
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.loss_fn = nn.MSELoss()
        self.train_epochs = train_epochs
        self.logger = logger or logging.getLogger(__name__)
        self.pred_len = self.teacher.prediction_length
        self.context_len = self.teacher.sequence_length

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self):
        train_loss_l = []
        for epoch in range(self.train_epochs):
            self.student.train()
            total_loss = 0.0
            total_loss_gt = 0.0
            total_loss_teacher = 0.0
            total_loss_kl = 0.0

            mse_loss_fn = nn.MSELoss()
            kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                batch_x, batch_y, batch_x_mark, batch_y_mark = [
                    b.float().to(self.device) for b in batch
                ]
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.context_len, :], dec_inp], dim=1)

                with torch.no_grad():
                    y_teacher = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                y_student = self.student(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                y_true = batch_y[:, -self.pred_len :, :]

                # 1. Ground-truth loss
                loss_gt = mse_loss_fn(y_student, y_true)
                # 2. Distillation loss (match teacher output)
                loss_teacher = mse_loss_fn(y_student, y_teacher)
                # 3. KL divergence (soft targets)
                student_log_probs = nn.functional.log_softmax(y_student / self.temperature, dim=-1)
                teacher_probs = nn.functional.softmax(y_teacher / self.temperature, dim=-1)
                # Ensure they are probability distributions
                assert torch.all(teacher_probs >= 0) and torch.all(student_log_probs >= 0), "Probabilities must be non-negative."
                assert torch.allclose(teacher_probs.sum(dim=-1), torch.ones_like(teacher_probs.sum(dim=-1))), "Teacher probs must sum to 1."
                loss_kl = kl_loss_fn(student_log_probs, teacher_probs)

                # Combine losses
                loss = (
                    self.alpha * loss_gt +
                    self.beta * loss_teacher +
                    self.kl_weight * loss_kl
                )

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
                total_loss_kl += loss_kl.item()

            avg_loss = total_loss / len(self.dataloader)
            avg_loss_gt = total_loss_gt / len(self.dataloader)
            avg_loss_teacher = total_loss_teacher / len(self.dataloader)
            avg_loss_kl = total_loss_kl / len(self.dataloader)

            train_loss_l.append(avg_loss)

            if self.logger:
                self.logger.info(f"Epoch {epoch+1} | Total Loss: {avg_loss:.7f} | GT Loss: {avg_loss_gt:.7f} | Teacher Loss: {avg_loss_teacher:.7f} | KL Loss: {avg_loss_kl:.7f}")

            if self.early_stopping:
                # Provide a path to save the best model
                save_path = os.path.join("logs", "best_student.pth")
                self.early_stopping(avg_loss, self.student, save_path)
                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info("Early stopping triggered.")
                    break

        return train_loss_l
