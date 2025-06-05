import os
import gin
import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from absl import app, flags

from distill_trainer import DistillationTrainer
from models.time_llm import Model as TeacherModel
from models.layers.student_model import Model as StudentModel
from data_processing.data_loader import TimeLLMDataHandler

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "configs/config_distil_584_train.gin", "Path to gin config file.")

@gin.configurable
def run(log_dir="./logs", llm_settings=None, data_settings=None):
    # Set random seed
    if "seed" in llm_settings:
        seed = llm_settings["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    os.makedirs(log_dir, exist_ok=True)

    # Set dtype globally
    if llm_settings["torch_dtype"] == "bfloat16":
        torch.set_default_dtype(torch.bfloat16)
    elif llm_settings["torch_dtype"] == "float32":
        torch.set_default_dtype(torch.float32)
    elif llm_settings["torch_dtype"] == "float16":
        torch.set_default_dtype(torch.float16)

    # Initialize data loader
    data_loader = TimeLLMDataHandler(settings={
        "input_features": data_settings["input_features"],
        "labels": data_settings["labels"],
        "preprocessing_method": data_settings["preprocessing_method"],
        "preprocess_input_features": data_settings["preprocess_input_features"],
        "preprocess_label": data_settings["preprocess_label"],
        "frequency": data_settings["frequency"],
        "num_workers": llm_settings["num_workers"],
        "sequence_length": llm_settings["sequence_length"],
        "context_length": llm_settings["context_length"],
        "prediction_length": llm_settings["prediction_length"],
        "percent": data_settings["percent"],
        "val_split": data_settings["val_split"],
    })

    _, train_loader = data_loader.load_from_csv(
        data_settings["path_to_train_data"],
        batch_size=llm_settings["train_batch_size"],
        split="train"
    )

    # -------------------
    # Teacher (frozen)
    # -------------------
    # teacher = TeacherModel(llm_settings)
    # for p in teacher.parameters():
    #     p.requires_grad = False
    # teacher.eval()
    
    
    teacher = TeacherModel(llm_settings)

    # 🔁 Load a fine-tuned teacher checkpoint
    teacher_ckpt_path = llm_settings.get("teacher_checkpoint_path", None)
    if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
        teacher.load_state_dict(torch.load(teacher_ckpt_path, map_location="cpu"))
        print(f"✅ Loaded fine-tuned teacher model from: {teacher_ckpt_path}")
    else:
        print("⚠️ No teacher checkpoint provided or file not found. Using fresh teacher model.")

    # Freeze teacher parameters
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # -------------------
    # Student
    # -------------------
    student_cfg = llm_settings.copy()
    student_cfg["llm_model"] = "DistilBERT"
    student_cfg["llm_layers"] = 6
    student_cfg["llm_dim"] = 768
    student_cfg["d_ff"] = 32
    # student = StudentModel(student_cfg)
    student  = TeacherModel(student_cfg)
    for param in student.llm_model.parameters():
            param.requires_grad = True


    optimizer = optim.Adam(student.parameters(), lr=llm_settings["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        alpha=0.5,
        beta=0.5
    )

    for epoch in range(llm_settings["train_epochs"]):
        loss = trainer.train_epoch()
        print(f"[Epoch {epoch+1}] Distillation Loss: {loss:.4f}")

    # Save model
    student_path = os.path.join(log_dir, "student_distilled_2.pth")
    torch.save(student.state_dict(), student_path)
    print(f"✅ Saved distilled model to: {student_path}")

def main(argv):
    gin.parse_config_file(FLAGS.config_path)
    run()

if __name__ == "__main__":
    app.run(main)
