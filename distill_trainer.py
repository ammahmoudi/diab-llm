import torch
import torch.nn as nn
from tqdm import tqdm

class DistillationTrainer:
    def __init__(self, teacher, student, dataloader, optimizer, device, alpha=0.5, beta=0.5):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.loss_fn = nn.MSELoss()
        self.pred_len = self.teacher.prediction_length
        self.context_len = self.teacher.sequence_length


        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.prompt_template =  "Forecast the next 9 values given 6 input steps."



    def train_epoch(self):
        self.student.train()
        total_loss = 0.0
        temperature = 4.0  # For KL-div scaling
        mse_loss_fn = nn.MSELoss()
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        for batch in tqdm(self.dataloader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = [b.float().to(self.device) for b in batch]

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.context_len, :], dec_inp], dim=1)

            # Teacher output (frozen)
            with torch.no_grad():
                y_teacher = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Student output
            y_student = self.student(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Ground truth: last prediction_length slice
            y_true = batch_y[:, -self.pred_len:, :]
            # print("Student output shape:", y_student.shape)
            # print("Teacher output shape:", y_teacher.shape)
            # teacher_loss = self.loss_fn(y_teacher, y_true).item()
            # print("🔎 Teacher loss vs true:", teacher_loss)
            # student_loss = self.loss_fn(y_student, y_true).item()
            # print("🔎 Student loss vs true:", student_loss)
            
            # print("y_student:", y_student[0].flatten()[:5])
            # print("y_teacher:", y_teacher[0].flatten()[:5])
            # print("y_true   :", y_true[0].flatten()[:5])



            # Compute losses
            loss_gt = mse_loss_fn(y_student, y_true)
            loss_teacher = mse_loss_fn(y_student, y_teacher)

            # KL Divergence (use softmax for probability dist)
            student_log_probs = nn.functional.log_softmax(y_student / temperature, dim=-1)
            teacher_probs = nn.functional.softmax(y_teacher / temperature, dim=-1)
            loss_kl = kl_loss_fn(student_log_probs, teacher_probs)

            # Combine losses with weights
            loss = (
                self.alpha * loss_gt +
                self.beta * loss_teacher +
                0.1 * loss_kl  # gamma = 0.1 as KL weight
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)

