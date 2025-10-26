#!/usr/bin/env python3
"""
Age Group Fairness Analyzer for OhioT1DM Dataset

Analyzes fairness across age groups (20-40, 40-60, 60-80) using actual experiment results.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
from pathlib import Path

class AgeFairnessAnalyzer:
    def __init__(self, data_path="/workspace/LLM-TIME/data/ohiot1dm/data.csv"):
        """Initialize analyzer with OhioT1DM patient data"""
        self.data_path = data_path
        self.patient_data = self._load_patient_data()
        
        # Create results directory
        self.results_dir = Path("/workspace/LLM-TIME/fairness/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_patient_data(self):
        """Load patient demographics from data.csv"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"📊 Loaded data for {len(df)} patients")
            
            # Create age group mapping
            age_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                age_group = row['Age']  # Already in format "20-40", "40-60", "60-80"
                age_mapping[patient_id] = age_group
                
            return age_mapping
        except Exception as e:
            print(f"❌ Error loading patient data: {e}")
            return {}
    
    def analyze_latest(self):
        """Analyze age group fairness for the latest experiment"""
        
        distillation_dir = "/workspace/LLM-TIME/distillation_experiments"
        
        if not os.path.exists(distillation_dir):
            raise FileNotFoundError(f"💥 Distillation directory not found: {distillation_dir}")
        
        experiment_dirs = [d for d in os.listdir(distillation_dir) 
                          if os.path.isdir(os.path.join(distillation_dir, d)) and d.startswith('pipeline_')]
        
        if not experiment_dirs:
            raise FileNotFoundError("💥 No pipeline experiment directories found")
        
        latest_experiment = sorted(experiment_dirs)[-1]
        experiment_path = os.path.join(distillation_dir, latest_experiment)
        
        print(f"🔍 Analyzing experiment: {latest_experiment}")
        
        # Load patient results from experiment directory
        patient_results = self._load_patient_results(experiment_path)
        
        print(f"\n🔍 Age Group Fairness Analysis")
        print("=" * 50)
        
        # Group patients by age
        age_groups = defaultdict(list)
        
        # Collect results by age group
        for patient_id, patient_data in patient_results.items():
            if patient_id in self.patient_data:
                age_group = self.patient_data[patient_id]
                age_groups[age_group].append({
                    'patient_id': patient_id,
                    'mse': patient_data.get('mse', 0),
                    'mae': patient_data.get('mae', 0),
                    'rmse': patient_data.get('rmse', 0)
                })
        
        # Print group information
        for age_group, patients in age_groups.items():
            print(f"🎂 {age_group}: {len(patients)} patients")
            print(f"   Patient IDs: {[p['patient_id'] for p in patients]}")
        
        return age_groups
    
    def _load_patient_results(self, experiment_path):
        """Load all patient results from experiment directory"""
        patient_results = {}
        
        # Find patient directories (handle nested structure)
        patient_dirs = []
        for root, dirs, files in os.walk(experiment_path):
            for d in dirs:
                if d.startswith('patient_'):
                    patient_dirs.append(os.path.join(root, d))
        
        print(f"📊 Found {len(patient_dirs)} patient directories")
        
        for patient_dir in patient_dirs:
            try:
                patient_id = os.path.basename(patient_dir).split('_')[1]
                
                # Try to find distillation summary first, then student summary
                distill_file = os.path.join(patient_dir, "phase_3_distillation", "distillation_summary.json")
                student_file = os.path.join(patient_dir, "phase_2_student", "student_baseline_summary.json")
                teacher_file = os.path.join(patient_dir, "phase_1_teacher", "teacher_training_summary.json")
                
                if os.path.exists(distill_file):
                    with open(distill_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,  # Convert RMSE to MSE
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                elif os.path.exists(student_file):
                    with open(student_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                elif os.path.exists(teacher_file):
                    with open(teacher_file, 'r') as f:
                        data = json.load(f)
                        patient_results[patient_id] = {
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
                            'mae': data.get('performance_metrics', {}).get('mae', 0),
                            'rmse': data.get('performance_metrics', {}).get('rmse', 0)
                        }
                        
            except Exception as e:
                patient_name = os.path.basename(patient_dir)
                print(f"⚠️  Could not load results for {patient_name}: {e}")
                continue
        
        return patient_results
    
    def analyze_latest_experiment(self):
        """Analyze age group fairness for the latest experiment"""
        
        # Find latest experiment directory
        distillation_dir = "/workspace/LLM-TIME/distillation_experiments"
        
        if not os.path.exists(distillation_dir):
            raise FileNotFoundError(f"❌ Distillation directory not found: {distillation_dir}")
        
        # Get latest experiment folder
        experiment_dirs = [d for d in os.listdir(distillation_dir) 
                          if os.path.isdir(os.path.join(distillation_dir, d))]
        
        if not experiment_dirs:
            raise FileNotFoundError("❌ No experiment directories found")
        
        latest_experiment = sorted(experiment_dirs)[-1]
        experiment_path = os.path.join(distillation_dir, latest_experiment)
        
        print(f"🔍 Analyzing experiment: {latest_experiment}")
        return self.analyze_experiment(experiment_path)
    
    def analyze_experiment(self, experiment_path):
        """Analyze age group fairness for a specific experiment"""
        
        # Load patient results from experiment directory
        patient_results = self._load_patient_results(experiment_path)
        
        print(f"\n🔍 Age Group Fairness Analysis")
        print("=" * 50)
        
        # Group patients by age
        age_groups = {
            '20–40': [],
            '40–60': [], 
            '60–80': []
        }
        
        # Collect results by age group
        for patient_id, patient_data in patient_results.items():
            if patient_id in self.patient_data:
                age_group = self.patient_data[patient_id]
                if age_group in age_groups:
                    age_groups[age_group].append({
                        'patient_id': patient_id,
                        'mse': patient_data.get('mse', 0),
                        'mae': patient_data.get('mae', 0)
                    })
        
        # Calculate performance for each age group
        age_performance = {}
        for age_group, patients in age_groups.items():
            if patients:
                group_mse = np.mean([p['mse'] for p in patients])
                group_mae = np.mean([p['mae'] for p in patients])
                
                age_performance[age_group] = {
                    'count': len(patients),
                    'mse': group_mse,
                    'mae': group_mae,
                    'patients': patients
                }
                
                print(f"👥 Age {age_group}: {len(patients)} patients, MSE = {group_mse:.6f}")
        
        # Calculate fairness metrics
        if len(age_performance) < 2:
            print("⚠️  Not enough age groups for fairness analysis")
            return None
        
        mse_values = [perf['mse'] for perf in age_performance.values()]
        mae_values = [perf['mae'] for perf in age_performance.values()]
        
        # Fairness ratio (worst performing group / best performing group)
        mse_ratio = max(mse_values) / min(mse_values) if min(mse_values) > 0 else float('inf')
        mae_ratio = max(mae_values) / min(mae_values) if min(mae_values) > 0 else float('inf')
        
        # Coefficient of variation (standard deviation / mean)
        mse_cv = np.std(mse_values) / np.mean(mse_values) if np.mean(mse_values) > 0 else float('inf')
        mae_cv = np.std(mae_values) / np.mean(mae_values) if np.mean(mae_values) > 0 else float('inf')
        
        # Determine fairness level
        fairness_level = self._classify_fairness_level(mse_ratio)
        
        print(f"\n📊 Age Group Fairness Metrics:")
        print(f"⚖️  MSE Fairness Ratio: {mse_ratio:.3f}")
        print(f"⚖️  MAE Fairness Ratio: {mae_ratio:.3f}")
        print(f"📈 MSE Coefficient of Variation: {mse_cv:.3f}")
        print(f"📈 MAE Coefficient of Variation: {mae_cv:.3f}")
        print(f"🎯 Age Group Fairness Level: {fairness_level}")
        
        # Individual patient analysis
        print(f"\n👤 Individual Patient Analysis:")
        for age_group, performance in age_performance.items():
            patients = performance['patients']
            best_patient = min(patients, key=lambda x: x['mse'])
            worst_patient = max(patients, key=lambda x: x['mse'])
            
            print(f"Age {age_group}:")
            print(f"  Best: {best_patient['patient_id']} (MSE: {best_patient['mse']:.6f})")
            print(f"  Worst: {worst_patient['patient_id']} (MSE: {worst_patient['mse']:.6f})")
        
        return {
            'age_performance': age_performance,
            'mse_ratio': mse_ratio,
            'mae_ratio': mae_ratio,
            'mse_cv': mse_cv,
            'mae_cv': mae_cv,
            'fairness_level': fairness_level,
            'patient_data': self.patient_data
        }
    
    def _classify_fairness_level(self, ratio):
        """Classify fairness level based on ratio"""
        if ratio <= 1.10:
            return "Excellent"
        elif ratio <= 1.25:
            return "Good"
        elif ratio <= 1.50:
            return "Acceptable"
        else:
            return "Poor"
    
    def create_comprehensive_visualizations(self, age_groups, age_performance):
        """Create comprehensive visualizations matching gender analyzer format exactly"""
        
        if not age_groups or not age_performance:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        # Create 2x2 subplot layout exactly like gender analyzer
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RMSE Comparison by Age Group (matching gender format)
        age_group_names = list(age_groups.keys())
        age_rmse = [age_performance[group]['avg_rmse'] for group in age_group_names]
        
        x = np.arange(len(age_group_names))
        width = 0.6
        
        colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(age_group_names)]
        bars1 = ax1.bar(x, age_rmse, width, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('RMSE (Lower = Better)')
        ax1.set_title('Performance by Age Group')
        ax1.set_xticks(x)
        ax1.set_xticklabels(age_group_names)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Fairness Score Progression (single point for ratio)
        fairness_ratio = age_performance.get('fairness_ratio', 1.0)
        ax2.plot([0], [fairness_ratio], 'o-', linewidth=3, markersize=10, color='purple')
        ax2.set_xlabel('Analysis')
        ax2.set_ylabel('Fairness Ratio (Lower = Better)')
        ax2.set_title('Age Group Fairness Ratio')
        ax2.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Fairness')
        ax2.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Good Fairness Threshold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Current'])
        
        # Annotate point
        ax2.annotate(f'{fairness_ratio:.2f}', (0, fairness_ratio), 
                    textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, fontweight='bold')
        
        # 3. Performance Ratios (matching gender format)
        ratios = [fairness_ratio]
        model_names = ['Age Groups']
        
        colors = ['lightgreen' if r < 1.2 else 'gold' if r < 1.5 else 'salmon' for r in ratios]
        bars3 = ax3.bar(model_names, ratios, color=colors, alpha=0.7)
        ax3.set_xlabel('Comparison Type')
        ax3.set_ylabel('Performance Ratio (Worse/Better)')
        ax3.set_title('Age Group Performance Ratios')
        ax3.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Fairness')
        ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
        ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Poor Fairness')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        for bar, ratio in zip(bars3, ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # 4. Experiment Summary (matching gender format)
        ax4.text(0.1, 0.9, 'EXPERIMENT SUMMARY', fontsize=14, fontweight='bold',
                transform=ax4.transAxes)
        
        summary_text = []
        for group in age_group_names:
            if group in age_performance:
                count = age_performance[group]['count']
                rmse = age_performance[group]['avg_rmse']
                summary_text.append(f"{group} patients analyzed: {count}")
        
        summary_text.append("")
        summary_text.append(f"Fairness Ratio: {fairness_ratio:.2f}x")
        fairness_level = age_performance.get('fairness_level', 'Unknown')
        summary_text.append(f"Fairness Level: {fairness_level}")
        summary_text.append("")
        summary_text.append("CONCLUSION:")
        if fairness_ratio < 1.2:
            summary_text.append("✅ Good fairness across age groups")
        elif fairness_ratio < 1.5:
            summary_text.append("⚠️ Moderate age-based disparity")
        else:
            summary_text.append("🚨 Poor fairness - significant age bias")
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=11, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"age_group_fairness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_file}")
        return plot_file

    def generate_fairness_report(self, age_groups, age_performance):
        """Generate comprehensive report like gender analyzer"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("AGE GROUP FAIRNESS ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if age_performance:
            report_lines.append("PERFORMANCE RESULTS:")
            for age, perf in age_performance.items():
                report_lines.append(f"  {age}:")
                report_lines.append(f"    Patients: {perf['count']}")
                report_lines.append(f"    Avg RMSE: {perf['avg_rmse']:.3f}")
                report_lines.append(f"    Avg MAE: {perf['avg_mae']:.3f}")
                report_lines.append("")
            
            # Calculate fairness metrics
            rmse_vals = [perf['avg_rmse'] for perf in age_performance.values()]
            rmse_ratio = max(rmse_vals) / min(rmse_vals) if min(rmse_vals) > 0 else float('inf')
            
            if rmse_ratio <= 1.10:
                fairness_level = "EXCELLENT"
            elif rmse_ratio <= 1.25:
                fairness_level = "GOOD"
            elif rmse_ratio <= 1.50:
                fairness_level = "ACCEPTABLE"
            else:
                fairness_level = "POOR"
            
            report_lines.append("FAIRNESS ASSESSMENT:")
            report_lines.append(f"  RMSE Ratio: {rmse_ratio:.2f}x")
            report_lines.append(f"  Fairness Level: {fairness_level}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_file = self.results_dir / f"age_group_fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        for line in report_lines:
            print(line)
        
        print(f"\n📁 Full report saved to: {report_file}")
        return report_file

def main():
    """Main function to run age fairness analysis"""
    
    print("� Starting Age Group Fairness Analysis")
    print("=" * 50)
    
    try:
        analyzer = AgeFairnessAnalyzer()
        
        # Show age group distribution
        age_counts = {}
        for patient_id, age_group in analyzer.patient_data.items():
            age_counts[age_group] = age_counts.get(age_group, 0) + 1
        
        print(f"\n👥 Age Group Distribution:")
        for age_group, count in age_counts.items():
            percentage = (count / len(analyzer.patient_data)) * 100
            print(f"   {age_group}: {count} patients ({percentage:.1f}%)")
        
        # Analyze latest experiment
        age_groups = analyzer.analyze_latest()
        
        if age_groups:
            # Calculate performance metrics
            age_performance = {}
            for age_group, patients in age_groups.items():
                if patients:
                    rmse_values = [p['rmse'] for p in patients]
                    mae_values = [p['mae'] for p in patients]
                    
                    age_performance[age_group] = {
                        'count': len(patients),
                        'avg_rmse': np.mean(rmse_values),
                        'avg_mae': np.mean(mae_values),
                        'std_rmse': np.std(rmse_values),
                        'std_mae': np.std(mae_values),
                        'patients': patients
                    }
            
            if age_performance:
                # Calculate fairness metrics
                rmse_vals = [perf['avg_rmse'] for perf in age_performance.values()]
                rmse_ratio = max(rmse_vals) / min(rmse_vals) if min(rmse_vals) > 0 else float('inf')
                
                if rmse_ratio <= 1.10:
                    fairness_level = "EXCELLENT"
                elif rmse_ratio <= 1.25:
                    fairness_level = "GOOD"
                elif rmse_ratio <= 1.50:
                    fairness_level = "ACCEPTABLE"
                else:
                    fairness_level = "POOR"
                
                print(f"\n⚖️ AGE GROUP FAIRNESS ASSESSMENT: {fairness_level}")
                print(f"📊 RMSE Fairness Ratio: {rmse_ratio:.2f}x")
                
                # Create visualizations
                analyzer.create_comprehensive_visualizations(age_groups, age_performance)
                
                # Generate report
                analyzer.generate_fairness_report(age_groups, age_performance)
                
                print(f"\n� All results saved in: {analyzer.results_dir}")
        
        print("\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(age_groups)} age groups analyzed")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
