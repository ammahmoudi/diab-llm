import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from distillation.core.distillation_trainer import DistillationTrainer
from models import time_llm as TimeLLMModel
import numpy as np
import pickle

# Single source of truth: import patient demographics from fairness module
try:
    from fairness.utils.analyzer_utils import get_ohiot1dm_default_data
    _OHIOT1DM_PATIENT_DATA = get_ohiot1dm_default_data()
except ImportError:
    # Fallback if fairness module unavailable — matches Table 1, OhioT1DM paper
    _OHIOT1DM_PATIENT_DATA = {
        '540': {'gender': 'Male',   'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '544': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
        '552': {'gender': 'Male',   'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '559': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '563': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '567': {'gender': 'Female', 'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '570': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '575': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '584': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
        '588': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '591': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '596': {'gender': 'Male',   'age': '60-80', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
    }

# Binary label mappings per feature (two groups: 0 and 1)
_GROUP_LABEL_MAPS = {
    'gender':  lambda v: 1 if v == 'Male' else 0,
    'age':     lambda v: 1 if v == '60-80' else 0,   # Old (60-80) vs Young/Middle
    'pump':    lambda v: 1 if v == '630G'  else 0,   # Closed-loop vs older pump
    'sensor':  lambda v: 1 if v == 'Basis' else 0,   # Basis vs Empatica
    'cohort':  lambda v: 1 if v == '2020'  else 0,   # 2020 cohort vs 2018
}


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

        # Fairness regularisation parameters (disabled by default)
        self.fairness_weight   = float(settings.get('fairness_weight', 0.0))
        self.fairness_feature  = settings.get('fairness_feature', None)   # e.g. "gender"
        self.target_threshold  = float(settings.get('target_threshold', 70.0))
        self.pred_threshold    = float(settings.get('pred_threshold',   70.0))
        self.hypo_oversample   = bool(settings.get('hypo_oversample',   False))

        # K1: calibrated soft labels (teacher output shifts by group)
        self.teacher_calibration_enabled = bool(settings.get('teacher_calibration_enabled', False))
        self.teacher_calibration_feature = settings.get('teacher_calibration_feature', None)
        self.teacher_calibration_group0_offset = float(settings.get('teacher_calibration_group0_offset', 0.0))
        self.teacher_calibration_group1_offset = float(settings.get('teacher_calibration_group1_offset', 0.0))

        logging.info(f"🎓 Initializing Distillation Wrapper")
        logging.info(f"  📍 Log Directory: {log_dir}")
        logging.info(f"  👨‍🏫 Teacher Checkpoint: {teacher_checkpoint_path}")
        logging.info(f"  🔥 Distillation Parameters: α={self.alpha}, β={self.beta}")
        if self.fairness_weight > 0:
            logging.info(
                f"  ⚖️  Fairness Regularisation: weight={self.fairness_weight}, "
                f"feature={self.fairness_feature}, threshold={self.target_threshold} mg/dL"
            )
        if self.teacher_calibration_enabled:
            logging.info(
                f"  🧪 K1 Calibrated Soft Labels: feature={self.teacher_calibration_feature}, "
                f"group0_offset={self.teacher_calibration_group0_offset:+.3f}, "
                f"group1_offset={self.teacher_calibration_group1_offset:+.3f}"
            )
        
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

    # ------------------------------------------------------------------
    # Fairness helpers
    # ------------------------------------------------------------------

    def _build_group_labels(self, dataset, feature: str) -> np.ndarray:
        """Build a 1-D integer label array (length = len(dataset)) for a given
        demographic feature.

        Each window index maps to the patient whose data starts at that row.
        For windows that span a patient boundary the majority patient is used.

        Args:
            dataset:  A Dataset_T1DM instance (must have .data_path and size attributes).
            feature:  One of "gender", "age", "pump", "sensor", "cohort".

        Returns:
            np.ndarray of shape (len(dataset),) with integer group labels (0 or 1).
            Returns None if the feature is unknown or the CSV has no item_id column.
        """
        import pandas as pd
        from scipy.stats import mode as scipy_mode

        if feature not in _GROUP_LABEL_MAPS:
            logging.warning(f"Unknown fairness_feature '{feature}'. Fairness loss disabled.")
            return None

        label_fn = _GROUP_LABEL_MAPS[feature]

        try:
            df = pd.read_csv(dataset.data_path)
            if 'item_id' not in df.columns:
                logging.warning("Training CSV has no 'item_id' column. Fairness loss disabled.")
                return None

            # Replicate the same preprocessing as Dataset_T1DM.__read_data__
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d-%m-%Y %H:%M:%S")
            except Exception:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
                except Exception:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Apply percent truncation
            percent = getattr(dataset, 'percent', 100)
            df = df.iloc[:int(len(df) * percent / 100)]

            # Apply train/val split
            num_samples = len(df)
            val_split = getattr(dataset, 'val_split', 0)
            num_train = int(num_samples * (100 - val_split) / 100)
            # We only care about the training slice
            df_train = df.iloc[0:num_train].reset_index(drop=True)

            # Resolve patient IDs to binary group labels
            patient_ids = df_train['item_id'].astype(str).values

            def pid_to_label(pid):
                info = _OHIOT1DM_PATIENT_DATA.get(pid, {})
                val = info.get(feature, None)
                return label_fn(val) if val is not None else -1

            row_labels = np.array([pid_to_label(pid) for pid in patient_ids])

            # Assign per-window label using the majority patient in the window
            seq_len  = dataset.sequence_length
            pred_len = dataset.prediction_length
            n_windows = len(df_train) - seq_len - pred_len + 1

            if n_windows <= 0:
                logging.warning("Not enough training rows to build group labels.")
                return None

            window_labels = np.empty(n_windows, dtype=np.int64)
            for i in range(n_windows):
                window_rows = row_labels[i: i + seq_len]
                valid = window_rows[window_rows >= 0]
                if len(valid) == 0:
                    window_labels[i] = 0
                else:
                    counts = np.bincount(valid)
                    window_labels[i] = int(np.argmax(counts))

            logging.info(
                f"⚖️  Group labels built: {n_windows} windows, feature='{feature}', "
                f"label_0={np.sum(window_labels==0)}, label_1={np.sum(window_labels==1)}"
            )
            return window_labels

        except Exception as e:
            logging.warning(f"Could not build group labels: {e}. Fairness loss disabled.")
            return None

    def distill_knowledge(self, train_loader, val_loader=None, epochs=10):
        """Perform knowledge distillation training"""
        logging.info(f"🎓 Starting Knowledge Distillation for {epochs} epochs...")

        # Load teacher and create student
        teacher_model = self._load_teacher_model()
        student_model = self._create_student_model()

        # Optionally wrap the training DataLoader with per-window group labels
        # and/or hypo-prevalence oversampling to reduce group imbalance
        active_train_loader = train_loader
        _need_group_labels = (
            (self.fairness_weight > 0 and self.fairness_feature) or
            (self.hypo_oversample and self.fairness_feature) or
            (self.teacher_calibration_enabled and self.teacher_calibration_feature)
        )
        if _need_group_labels:
            label_feature = self.fairness_feature or self.teacher_calibration_feature
            group_labels = self._build_group_labels(train_loader.dataset, label_feature)
            if group_labels is not None:
                from data_processing.data_sets import GroupLabeledDataset
                from torch.utils.data import WeightedRandomSampler
                labeled_dataset = GroupLabeledDataset(train_loader.dataset, group_labels)

                # Fix A: Hypo-prevalence oversampling
                # Weight each window inversely proportional to its group's hypo
                # prevalence so the model sees balanced hypo events per gender.
                sampler = None
                if self.hypo_oversample:
                    import numpy as np
                    targets_np = np.array([float(labeled_dataset[i][1].mean())
                                          for i in range(len(labeled_dataset))])
                    grp_np = np.array(group_labels)
                    HYPO_THRESH = self.target_threshold  # 70 mg/dL
                    sample_weights = np.ones(len(labeled_dataset))
                    unique_groups = np.unique(grp_np)
                    # Compute per-group hypo prevalence
                    group_hypo_rates = {}
                    for g in unique_groups:
                        g_mask = grp_np == g
                        g_targets = targets_np[g_mask]
                        group_hypo_rates[g] = (g_targets < HYPO_THRESH).mean() + 1e-8
                    max_rate = max(group_hypo_rates.values())
                    # Oversample windows from underrepresented-hypo groups
                    for g in unique_groups:
                        g_mask = grp_np == g
                        g_targets = targets_np[g_mask]
                        is_hypo = g_targets < HYPO_THRESH
                        oversample_factor = max_rate / group_hypo_rates[g]
                        # Upweight hypo windows of underrepresented group
                        indices = np.where(g_mask)[0]
                        for idx, hypo in zip(indices, is_hypo):
                            if hypo:
                                sample_weights[idx] = oversample_factor
                    sampler = WeightedRandomSampler(
                        weights=sample_weights.tolist(),
                        num_samples=len(labeled_dataset),
                        replacement=True,
                    )
                    n_groups = {g: (grp_np == g).sum() for g in unique_groups}
                    rates_str = {g: f"{r:.3f}" for g, r in group_hypo_rates.items()}
                    max_factor = max_rate / min(group_hypo_rates.values())
                    logging.info(f"⚖️  Hypo oversampling: groups={n_groups}, "
                                 f"rates={rates_str}, "
                                 f"max_oversample={max_factor:.1f}x")

                active_train_loader = DataLoader(
                    labeled_dataset,
                    batch_size=train_loader.batch_size,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    num_workers=train_loader.num_workers,
                    drop_last=train_loader.drop_last,
                )
                logging.info("⚖️  Fairness-labeled DataLoader created successfully.")

        # Setup optimizer
        optimizer = Adam(student_model.parameters(), lr=self.settings.get('learning_rate', 0.001))

        # Create trainer
        trainer = DistillationTrainer(
            teacher=teacher_model,
            student=student_model,
            dataloader=active_train_loader,
            optimizer=optimizer,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            train_epochs=epochs,
            logger=logging.getLogger(),
            fairness_weight=self.fairness_weight,
            target_threshold=self.target_threshold,
            pred_threshold=self.pred_threshold,
            teacher_calibration_enabled=self.teacher_calibration_enabled,
            teacher_calibration_feature=self.teacher_calibration_feature,
            teacher_calibration_group0_offset=self.teacher_calibration_group0_offset,
            teacher_calibration_group1_offset=self.teacher_calibration_group1_offset,
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