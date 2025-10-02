import os
import subprocess
import pandas as pd
import argparse
import datetime
import gin
import json
from shutil import copyfile

import ast
import re


@gin.configurable
def run(log_dir="./logs", llm_settings=None, data_settings=None):
    pass

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8').strip())
    process.stdout.close()
    return process.wait()

def extract_metrics_from_log(log_path):
    """Extract evaluation metrics from the log file"""
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Metric results:" in line or "Student Model Evaluation Metrics:" in line:
                    # Extract the part after ':'
                    match = re.search(r':\s*(\{.*\})', line)
                    if match:
                        metrics_str = match.group(1)
                        # Replace np.float32(x) with x
                        metrics_str = re.sub(r'np\.float32\(([^\)]+)\)', r'\1', metrics_str)
                        # Safely evaluate dict
                        metrics_dict = ast.literal_eval(metrics_str)
                        return metrics_dict
    except Exception as e:
        print(f"Error extracting metrics from log: {e}")
    return metrics


def create_test_config(template_path, output_path, model_params, is_student=False):
    """
    Create a test config file from template with proper model parameters
    
    Args:
        template_path: Path to the template config file
        output_path: Where to save the new config
        model_params: Dictionary with model parameters
        is_student: Whether this is for a student model
    """
    with open(template_path, 'r') as f:
        config_content = f.read()

    # --- Update run.log_dir ---
    if is_student:
        model_name = model_params['model_type'].lower()
        new_log_dir = f"./experiment_configs_distil/student_{model_name}_dim_{model_params['dim']}_layers_{model_params['layers']}_test/logs"

        config_content = re.sub(
            r'run\.log_dir\s*=\s*"([^"]+)"',
            f'run.log_dir = "{new_log_dir}"',
            config_content
        )
    else:
        # For teacher, you can skip or set a new_log_dir if needed
        pass

    # --- Update run.llm_settings ---
    # Extract llm_settings dict block using regex
    llm_settings_pattern = r'run\.llm_settings\s*=\s*({.*?})'  # no newline required

    match = re.search(llm_settings_pattern, config_content, re.DOTALL)

    if not match:
        print("Could not find run.llm_settings block in template config.")
        return

    llm_settings_str = match.group(1)

    # Parse the dict safely
    llm_settings_dict = ast.literal_eval(llm_settings_str)

    # Update the relevant keys
    llm_settings_dict['restore_checkpoint_path'] = model_params['checkpoint_path']
    llm_settings_dict['llm_model'] = model_params['model_type']
    llm_settings_dict['llm_layers'] = model_params['layers']
    llm_settings_dict['llm_dim'] = model_params['dim']
    
    if 'd_ff' in model_params:
        llm_settings_dict['d_ff'] = model_params['d_ff']

    # Convert dict back to nicely formatted string
    updated_llm_settings_str = "{\n"
    for key, value in llm_settings_dict.items():
        if isinstance(value, str):
            updated_llm_settings_str += f"    '{key}': '{value}',\n"
        else:
            updated_llm_settings_str += f"    '{key}': {value},\n"
    updated_llm_settings_str += "}\n\n"

    # Replace old block with updated one
    config_content = re.sub(
        llm_settings_pattern,
        f'run.llm_settings = {updated_llm_settings_str}',
        config_content,
        flags=re.DOTALL
    )

    # --- Write updated config ---
    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"Created test config at {output_path}")



def update_train_config_with_student_params(config_path, output_path, student_params,distill_log_dir):
    """Update the training config with student model parameters"""
    with open(config_path, 'r') as f:
        config_lines = f.readlines()

    llm_settings_start = None
    llm_settings_end = None

    # Step 1: Find where run.llm_settings starts
    for i, line in enumerate(config_lines):
        if 'run.llm_settings' in line and '=' in line and '{' in line:
            llm_settings_start = i
            break

    if llm_settings_start is None:
        print("Could not find run.llm_settings in config")
        return config_path

    # Step 2: Find where run.llm_settings block ends (matching '}')
    brace_count = 0
    for i in range(llm_settings_start, len(config_lines)):
        line = config_lines[i]
        brace_count += line.count('{')
        brace_count -= line.count('}')
        if brace_count == 0:
            llm_settings_end = i
            break

    if llm_settings_end is None:
        print("Could not find end of run.llm_settings block")
        return config_path

    # Step 3: Insert student params BEFORE the closing '}'
    # So insert before llm_settings_end line

    # Check if line before '}' ends with comma, if not add it
    prev_line_index = llm_settings_end - 1
    if not config_lines[prev_line_index].strip().endswith(','):
        config_lines[prev_line_index] = config_lines[prev_line_index].rstrip() + ',\n'

    # Now insert student params
    # Update run.log_dir = "logs/distill_xxx"
    for i, line in enumerate(config_lines):
        if line.strip().startswith("run.log_dir"):
            config_lines[i] = f'run.log_dir = "{distill_log_dir}"\n'
            print(f"✅ Forced run.log_dir = {distill_log_dir}")
            break

    student_params_str = [
        f"    'student_model': '{student_params['model']}',\n",
        f"    'student_layers': {student_params['layers']},\n",
        f"    'student_dim': {student_params['dim']},\n",
        f"    'student_d_ff': {student_params['d_ff']},\n"
    ]

    # Insert the lines
    config_lines = config_lines[:llm_settings_end] + student_params_str + config_lines[llm_settings_end:]


    # Save updated config
    with open(output_path, 'w') as f:
        f.writelines(config_lines)

    print(f"Updated training config at {output_path} with student parameters")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Automated Model Distillation and Evaluation')
    parser.add_argument('--teacher_config', type=str, default='configs/dist/config_distil_584_train.gin',
                        help='Path to teacher training config')
                        
    parser.add_argument('--test_config_template', type=str, default='configs/dist/config_distil_584_test.gin',
                        help='Path to test config template')
    parser.add_argument('--results_csv', type=str, default='experiment_results_distillation.csv',
                        help='Path to save comparison results')
    # Add student model configuration arguments
    parser.add_argument('--student_model', type=str, default='DistilBERT',
                        help='Type of student model to use')
    parser.add_argument('--student_layers', type=int, default=6,
                        help='Number of layers in the student model')
    parser.add_argument('--student_dim', type=int, default=768,
                        help='Hidden dimension size for the student model')
    parser.add_argument('--student_d_ff', type=int, default=32,
                        help='Feed-forward dimension for the student model')
    args = parser.parse_args()

    gin.parse_config_file(args.teacher_config)

    teacher_config = gin.query_parameter('run.llm_settings')
    teacher_checkpoint = teacher_config.get('teacher_checkpoint_path', '')

    print(f"Using teacher checkpoint: {teacher_checkpoint}")

    if not teacher_checkpoint:
        raise ValueError("Teacher checkpoint path is not set in the config. Please check your teacher config file.")

    # Get timestamps and create folders
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    distill_log_dir = f"logs/distill_{timestamp}"
    os.makedirs(distill_log_dir, exist_ok=True)
    
    # Copy the teacher config for reference
    copyfile(args.teacher_config, f"{distill_log_dir}/teacher_config.gin")
    
    #Create a modified training config with student parameters
    student_params = {
        'model': args.student_model,
        'layers': args.student_layers,
        'dim': args.student_dim,
        'd_ff': args.student_d_ff
    }
    
    modified_train_config = f"{distill_log_dir}/modified_train_config.gin"
    updated_config_path = update_train_config_with_student_params(
        args.teacher_config, 
        modified_train_config, 
        student_params,
        distill_log_dir
    )
    
    # 1. Run the distillation process with the modified config
    print("\n=== STEP 1: TRAINING STUDENT MODEL ===")
    distill_command = f"python run_distill.py --config_path {updated_config_path} " 
    run_command(distill_command)

    
    # Get the path where the student model was saved
    student_model_path = os.path.join(
        distill_log_dir,
        "student_distilled.pth"
    )

    
    # 2. Create test configs for both teacher and student
    print("\n=== STEP 2: PREPARING TEST CONFIGS ===")
    teacher_test_config = f"{distill_log_dir}/teacher_test_config.gin"
    student_test_config = f"{distill_log_dir}/student_test_config.gin"
    
    # Teacher config with parameters from teacher_config
    teacher_params = {
        'checkpoint_path': teacher_checkpoint,
        'model_type': teacher_config.get('llm_model', 'BERT'),
        'layers': teacher_config.get('llm_layers', 12),
        'dim': teacher_config.get('llm_dim', 768),
        'd_ff': teacher_config.get('d_ff', 32)
    }

    create_test_config(
        args.test_config_template, 
        teacher_test_config, 
        teacher_params,
        is_student=False
    )
    
    # Student config with parameters from command line arguments
    student_params = {
        'checkpoint_path': student_model_path,
        'model_type': args.student_model,
        'layers': args.student_layers,
        'dim': args.student_dim,
        'd_ff': args.student_d_ff
    }

    create_test_config(
        args.test_config_template, 
        student_test_config, 
        student_params,
        is_student=True
    )
    
    # 3. Test the teacher model
    print("\n=== STEP 3: EVALUATING TEACHER MODEL ===")
    teacher_log = f"{distill_log_dir}/teacher_test.log"
    teacher_test_command = f"./run_main.sh --config_path {teacher_test_config} --log_level INFO --remove_checkpoints False > {teacher_log} 2>&1"
    run_command(teacher_test_command)
    
    # 4. Test the student model
    print("\n=== STEP 4: EVALUATING STUDENT MODEL ===")
    student_log = f"{distill_log_dir}/student_test.log"
    student_test_command = f"./run_main.sh --config_path {student_test_config} --log_level INFO --remove_checkpoints False > {student_log} 2>&1"
    run_command(student_test_command)

        
    # 5. Extract metrics and create comparison
    print("\n=== STEP 5: COMPARING RESULTS ===")
    teacher_metrics = extract_metrics_from_log(teacher_log)
    student_metrics = extract_metrics_from_log(student_log)
    
    # === STEP 6: TRAINING TimeLLM WITH STUDENT BACKBONE ===
    print("\n=== STEP 6: TRAINING TimeLLM WITH STUDENT BACKBONE ===")

    # Prepare new config
    student_time_llm_config = f"{distill_log_dir}/student_time_llm_train_config.gin"

    # Copy the modified_train_config first
    copyfile(modified_train_config, student_time_llm_config)

    # Now patch it:
    with open(student_time_llm_config, 'r') as f:
        config_lines = f.readlines()

    # Overwrite the relevant keys
    for i, line in enumerate(config_lines):
        if "'method'" in line:
            config_lines[i] = "    'method': 'time_llm',\n"
        elif "'restore_from_checkpoint'" in line:
            config_lines[i] = "    'restore_from_checkpoint': False,\n"
        elif "'teacher_checkpoint_path'" in line:
            config_lines[i] = "    'teacher_checkpoint_path': '',\n"
        elif "'llm_model'" in line:
            config_lines[i] = f"    'llm_model': '{student_params['model_type']}',\n"
        elif "'llm_layers'" in line:
            config_lines[i] = f"    'llm_layers': {student_params['layers']},\n"
        elif "'llm_dim'" in line:
            config_lines[i] = f"    'llm_dim': {student_params['dim']},\n"
        elif "'d_ff'" in line:
            config_lines[i] = f"    'd_ff': {student_params['d_ff']},\n"

    # Remove any 'student_model', 'student_layers', etc
    filtered_lines = []
    for line in config_lines:
        if "'student_model'" in line or "'student_layers'" in line or "'student_dim'" in line or "'student_d_ff'" in line:
            continue  # Skip this line
        filtered_lines.append(line)

    # Save the modified config
    with open(student_time_llm_config, 'w') as f:
        f.writelines(filtered_lines)

    print(f"Created student TimeLLM config at {student_time_llm_config}")

    # Run training
    student_time_llm_log = f"{distill_log_dir}/student_time_llm_train.log"
    time_llm_command = f"./run_main.sh --config_path {student_time_llm_config} --log_level INFO --remove_checkpoints False > {student_time_llm_log} 2>&1"

    run_command(time_llm_command)

    # Optional: extract metrics again if desired (reuse extract_metrics_from_log)
    time_llm_metrics = extract_metrics_from_log(student_time_llm_log)

    print("\n=== TIME LLM (STUDENT BACKBONE) METRICS ===")
    print(time_llm_metrics)



    # Create a DataFrame with comparison results
    results = {
        'patient_id':'584',
        'seed': teacher_config.get('seed', 0),
        'timestamp': timestamp,
        'teacher_model': teacher_config.get('llm_model', 'BERT'),
        'teacher_layers': teacher_config.get('llm_layers', 12),
        'teacher_dim': teacher_config.get('llm_dim', 768),
        'student_model': args.student_model,  
        'student_layers': args.student_layers,  
        'student_dim': args.student_dim,  
        'seq_len': teacher_config.get('sequence_length', 6),
        'context_len': teacher_config.get('context_length', 6),
        'pred_len': teacher_config.get('prediction_length', 9),
        'teacher_rmse': teacher_metrics.get('rmse', 0),
        'teacher_mae': teacher_metrics.get('mae', 0),
        'teacher_mape': teacher_metrics.get('mape', 0),
        'student_rmse': student_metrics.get('rmse', 0),
        'student_mae': student_metrics.get('mae', 0),
        'student_mape': student_metrics.get('mape', 0),
        'rmse_diff': teacher_metrics.get('rmse', 0) - student_metrics.get('rmse', 0),
        'mae_diff': teacher_metrics.get('mae', 0) - student_metrics.get('mae', 0),
        'mape_diff': teacher_metrics.get('mape', 0) - student_metrics.get('mape', 0),
        'compression_ratio': teacher_config.get('llm_layers', 12) / 6,
        'time_llm_rmse': time_llm_metrics.get('rmse', 0),
'time_llm_mae': time_llm_metrics.get('mae', 0),
'time_llm_mape': time_llm_metrics.get('mape', 0),

    }
    
    # Add to results CSV
    try:
        if os.path.exists(args.results_csv):
            results_df = pd.read_csv(args.results_csv)
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        else:
            results_df = pd.DataFrame([results])
        
        results_df.to_csv(args.results_csv, index=False)
        print(f"Results saved to {args.results_csv}")
        
        # Print comparison summary
        print("\n=== DISTILLATION RESULTS SUMMARY ===")
        print(f"Teacher model: {results['teacher_model']} (Layers: {results['teacher_layers']}, Dim: {results['teacher_dim']})")
        print(f"Student model: {results['student_model']} (Layers: {results['student_layers']}, Dim: {results['student_dim']})")
        print(f"Compression ratio: {results['compression_ratio']:.2f}x")

        print("\nPerformance metrics:")
        print(f"  RMSE - Teacher: {results['teacher_rmse']:.4f}, Student: {results['student_rmse']:.4f}, Diff: {results['rmse_diff']:.4f}")
        print(f"  MAE  - Teacher: {results['teacher_mae']:.4f}, Student: {results['student_mae']:.4f}, Diff: {results['mae_diff']:.4f}")
        print(f"  MAPE - Teacher: {results['teacher_mape']:.4f}, Student: {results['student_mape']:.4f}, Diff: {results['mape_diff']:.4f}")

        print("\n=== TIME LLM (STUDENT BACKBONE) METRICS ===")
        print(f"  RMSE: {results['time_llm_rmse']:.4f}")
        print(f"  MAE : {results['time_llm_mae']:.4f}")
        print(f"  MAPE: {results['time_llm_mape']:.4f}")

    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()