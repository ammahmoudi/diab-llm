import logging
import os
import gin
import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from absl import app, flags
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from utils.time_llm_utils import EarlyStopping, adjust_learning_rate

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
    # for param in student.llm_model.parameters():
    #         param.requires_grad = True


    # Accelerator setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./configs/ds_config_zero2.json")
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    teacher.to(accelerator.device)  # <-- Add this line

    # Prepare optimizer, scheduler, early stopping
    optimizer = optim.Adam(student.parameters(), lr=llm_settings["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        epochs=llm_settings["train_epochs"],
        max_lr=llm_settings["learning_rate"],
    )
    early_stopping = EarlyStopping(accelerator=accelerator, patience=llm_settings["patience"], verbose=True, save_mode=False)

    # Prepare with accelerator
    train_loader, student, optimizer, scheduler = accelerator.prepare(
        train_loader, student, optimizer, scheduler
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        optimizer=optimizer,
        device=accelerator.device,
        accelerator=accelerator,
        scheduler=scheduler,
        early_stopping=early_stopping,
        alpha=0.5,
        beta=0.5,
        train_epochs=llm_settings["train_epochs"],
        logger=logging.getLogger(__name__),
    )

    train_loss_l = trainer.train()

    # Save model
    student_path = os.path.join(log_dir, "student_distilled_3.pth")
    torch.save(student.state_dict(), student_path)
    print(f"✅ Saved distilled model to: {student_path}")

def main(argv):
    gin.parse_config_file(FLAGS.config_path)
    run()

if __name__ == "__main__":
    app.run(main)
