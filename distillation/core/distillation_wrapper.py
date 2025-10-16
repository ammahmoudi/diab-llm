import logging
import os
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from distillation.core.distillation_trainer import DistillationTrainer
from models import time_llm as TimeLLMModel
import numpy as np
import pickle

class DistillationWrapper:
    """Wrapper for DistillationTrainer to integrate with the pipeline"""
    
    def __init__(self, settings, data_settings, log_dir, teacher_checkpoint_path):
        self.settings = settings
        self.data_settings = data_settings
        self.log_dir = log_dir
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Distillation parameters
        self.alpha = settings.get('distillation_alpha', 0.5)
        self.beta = settings.get('distillation_beta', 0.5) 
        self.kl_weight = settings.get('distillation_kl_weight', 0.1)
        self.temperature = settings.get('distillation_temperature', 3.0)
        
        logging.info(f"🎓 Initializing Distillation Wrapper")
        logging.info(f"  📍 Log Directory: {log_dir}")
        logging.info(f"  👨‍🏫 Teacher Checkpoint: {teacher_checkpoint_path}")
        logging.info(f"  🔥 Distillation Parameters: α={self.alpha}, β={self.beta}, KL={self.kl_weight}, T={self.temperature}")
        
    def _create_model_config(self, is_student=False):
        """Create model configuration based on settings"""
        if is_student:
            # Student model - use the main llm_model parameters
            llm_model = self.settings.get('llm_model', 'TinyBERT')
            llm_dim = self.settings.get('llm_dim', 312)
            llm_layers = self.settings.get('llm_layers', 4)
            d_ff = self.settings.get('d_ff', 32)
        else:
            # Teacher model - use teacher-specific parameters or infer from teacher_model
            teacher_model = self.settings.get('teacher_model', 'BERT')
            if teacher_model == 'BERT':
                llm_model = 'BERT'
                llm_dim = 768
                llm_layers = 12
                # Use the same d_ff as the training config to match projection layers
                d_ff = self.settings.get('d_ff', 32)  # Use the actual training d_ff
            elif teacher_model == 'DistilBERT':
                llm_model = 'DistilBERT'
                llm_dim = 768
                llm_layers = 6
                d_ff = self.settings.get('d_ff', 32)  # Use the actual training d_ff
            else:
                # Fallback to provided values
                llm_model = teacher_model
                llm_dim = self.settings.get('teacher_dim', 768)
                llm_layers = self.settings.get('teacher_layers', 12)
                d_ff = self.settings.get('d_ff', 32)  # Use the actual training d_ff
        
        config = {
            'task_name': self.settings.get('task_name', 'long_term_forecast'),
            'sequence_length': self.settings.get('sequence_length', 6),
            'context_length': self.settings.get('context_length', 6), 
            'prediction_length': self.settings.get('prediction_length', 9),
            'enc_in': self.settings.get('enc_in', 1),
            'dec_in': self.settings.get('dec_in', 1),
            'c_out': self.settings.get('c_out', 1),
            'd_model': self.settings.get('d_model', 32),
            'd_ff': d_ff,
            'n_heads': self.settings.get('n_heads', 8),
            'e_layers': self.settings.get('e_layers', 2),
            'd_layers': self.settings.get('d_layers', 1),
            'dropout': self.settings.get('dropout', 0.1),
            'moving_avg': self.settings.get('moving_avg', 25),
            'factor': self.settings.get('factor', 1),
            'activation': self.settings.get('activation', 'gelu'),
            'embed': self.settings.get('embed', 'timeF'),
            'prompt_domain': self.settings.get('prompt_domain', 0),
            'llm_model': llm_model,
            'llm_dim': llm_dim,
            'llm_layers': llm_layers,
            'patch_len': self.settings.get('patch_len', 6),
            'stride': self.settings.get('stride', 8)
        }
        return config
    
    def _load_teacher_model(self):
        """Load the teacher model from checkpoint"""
        logging.info("👨‍🏫 Loading teacher model...")
        
        teacher_config = self._create_model_config(is_student=False)
        teacher_model = TimeLLMModel.Model(teacher_config)
        
        # Load teacher checkpoint
        checkpoint = torch.load(self.teacher_checkpoint_path, map_location=self.device)
        teacher_model.load_state_dict(checkpoint)
        teacher_model.to(self.device)
        teacher_model.eval()  # Freeze teacher model
        
        # Freeze all teacher parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        logging.info("✅ Teacher model loaded and frozen")
        return teacher_model
    
    def _create_student_model(self):
        """Create the student model"""
        logging.info("👨‍🎓 Creating student model...")
        
        student_config = self._create_model_config(is_student=True)
        student_model = TimeLLMModel.Model(student_config)
        student_model.to(self.device)
        
        logging.info(f"✅ Student model created: {student_config['llm_model']}-{student_config['llm_layers']}L-{student_config['llm_dim']}D")
        return student_model
    
    def distill_knowledge(self, train_loader, val_loader=None, epochs=10):
        """Perform knowledge distillation training"""
        logging.info(f"🎓 Starting Knowledge Distillation for {epochs} epochs...")
        
        # Load teacher and create student
        teacher_model = self._load_teacher_model()
        student_model = self._create_student_model()
        
        # Setup optimizer
        optimizer = Adam(student_model.parameters(), lr=self.settings.get('learning_rate', 0.001))
        
        # Create trainer
        trainer = DistillationTrainer(
            teacher=teacher_model,
            student=student_model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            kl_weight=self.kl_weight,
            temperature=self.temperature,
            train_epochs=epochs,
            logger=logging.getLogger()
        )
        
        # Add context_len and pred_len to trainer (needed for training loop)
        trainer.context_len = self.settings['context_length']
        trainer.pred_len = self.settings['prediction_length']
        
        # Train the student model
        train_losses = trainer.train()
        
        # Save the trained student model
        checkpoint_path = os.path.join(self.log_dir, "student_distilled.pth")
        torch.save(student_model.state_dict(), checkpoint_path)
        
        logging.info(f"✅ Knowledge Distillation completed!")
        logging.info(f"📁 Student checkpoint saved to: {checkpoint_path}")
        
        return checkpoint_path, train_losses, None
    
    def predict(self, test_loader, output_dir=None):
        """Run inference with the distilled student model"""
        logging.info("🔮 Running inference with distilled student model...")
        
        # Load the distilled student model
        checkpoint_path = os.path.join(self.log_dir, "student_distilled.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Distilled model checkpoint not found: {checkpoint_path}")
        
        student_model = self._create_student_model()
        student_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        student_model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = [
                    b.float().to(self.device) for b in batch
                ]
                
                dec_inp = torch.zeros_like(batch_y[:, -self.settings['prediction_length']:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.settings['context_length'], :], dec_inp], dim=1)
                
                outputs = student_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                predictions.append(outputs.detach().cpu().numpy())
                targets.append(batch_y[:, -self.settings['prediction_length']:, :].detach().cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Save predictions
        if output_dir:
            pred_path = os.path.join(output_dir, "predictions.pkl")
            target_path = os.path.join(output_dir, "targets.pkl")
            
            with open(pred_path, 'wb') as f:
                pickle.dump(predictions, f)
            with open(target_path, 'wb') as f:
                pickle.dump(targets, f)
                
            logging.info(f"📁 Predictions saved to: {pred_path}")
            logging.info(f"📁 Targets saved to: {target_path}")
        
        return predictions, targets, None
    
    def evaluate(self, predictions, targets, metrics):
        """Evaluate model performance"""
        from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
        
        # Reshape if needed
        if len(predictions.shape) == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
        if len(targets.shape) == 3:
            targets = targets.reshape(-1, targets.shape[-1])
        
        results = {}
        for metric in metrics:
            if metric.lower() == 'rmse':
                results['rmse'] = calculate_rmse(predictions, targets)
            elif metric.lower() == 'mae':
                results['mae'] = calculate_mae(predictions, targets)
            elif metric.lower() == 'mape':
                results['mape'] = calculate_mape(predictions, targets)
        
        return results