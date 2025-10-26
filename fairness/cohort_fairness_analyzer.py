#!/usr/bin/env python3
"""
Cohort Fairness Analyzer for OhioT1DM Dataset
"""

import os
import json
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

class CohortFairnessAnalyzer:
    def __init__(self, data_path="/workspace/LLM-TIME/data/ohiot1dm/data.csv"):
        self.data_path = data_path
        self.patient_data = self._load_patient_data()
        
        # Create results directory
        self.results_dir = Path("/workspace/LLM-TIME/fairness/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Loaded cohort data for {len(self.patient_data)} patients")
        
    def _load_patient_data(self):
        """Load patient cohort info from data.csv"""
        try:
            df = pd.read_csv(self.data_path)
            cohort_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                cohort = row['Cohort']
                cohort_mapping[patient_id] = cohort
            return cohort_mapping
        except Exception as e:
            print(f"❌ Error loading patient data: {e}")
            return {}

    def _load_patient_results(self, experiment_path):
        """Load patient results from experiment directory structure"""
        patient_results = {}
        
        # Find patient directories (handle nested structure)
        patient_dirs = []
        for root, dirs, files in os.walk(experiment_path):
            for d in dirs:
                if d.startswith('patient_'):
                    patient_dirs.append(os.path.join(root, d))
        
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
                            'mse': data.get('performance_metrics', {}).get('rmse', 0) ** 2,
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
    
    def analyze_latest(self):
        """Analyze cohort fairness for the latest experiment"""
        
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
        return self.analyze_experiment(experiment_path)
    
    def analyze_experiment(self, experiment_path):
        """Analyze cohort fairness for a specific experiment"""
        
        patient_results = self._load_patient_results(experiment_path)
        
        print(f"\n🔍 Cohort Fairness Analysis")
        print("=" * 50)
        
        # Group patients by cohort
        cohort_groups = defaultdict(list)
        
        for patient_id, patient_data in patient_results.items():
            if patient_id in self.patient_data:
                cohort = self.patient_data[patient_id]
                cohort_groups[cohort].append({
                    'patient_id': patient_id,
                    'mse': patient_data.get('mse', 0),
                    'mae': patient_data.get('mae', 0), 
                    'rmse': patient_data.get('rmse', 0)
                })
        
        # Print group information
        for cohort, patients in cohort_groups.items():
            print(f"👥 Cohort {cohort}: {len(patients)} patients")
            print(f"   Patient IDs: {[p['patient_id'] for p in patients]}")
        
        return cohort_groups

    def create_comprehensive_visualizations(self, cohort_groups, cohort_performance):
        """Create comprehensive visualizations matching gender analyzer format exactly"""
        
        if not cohort_groups or not cohort_performance:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        # Create 2x2 subplot layout exactly like gender analyzer
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RMSE Comparison by Cohort (matching gender format)
        cohort_names = list(cohort_groups.keys())
        cohort_rmse = [cohort_performance[cohort]['avg_rmse'] for cohort in cohort_names]
        
        x = np.arange(len(cohort_names))
        width = 0.6
        
        colors = ['skyblue', 'lightcoral'] if len(cohort_names) == 2 else ['skyblue', 'lightcoral', 'lightgreen'][:len(cohort_names)]
        bars1 = ax1.bar(x, cohort_rmse, width, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Cohort')
        ax1.set_ylabel('RMSE (Lower = Better)')
        ax1.set_title('Performance by Cohort')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cohort_names)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Fairness Score Progression (single point for ratio)
        fairness_ratio = cohort_performance.get('fairness_ratio', 1.0)
        ax2.plot([0], [fairness_ratio], 'o-', linewidth=3, markersize=10, color='purple')
        ax2.set_xlabel('Analysis')
        ax2.set_ylabel('Fairness Ratio (Lower = Better)')
        ax2.set_title('Cohort Fairness Ratio')
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
        model_names = ['Cohorts']
        
        colors = ['lightgreen' if r < 1.2 else 'gold' if r < 1.5 else 'salmon' for r in ratios]
        bars3 = ax3.bar(model_names, ratios, color=colors, alpha=0.7)
        ax3.set_xlabel('Comparison Type')
        ax3.set_ylabel('Performance Ratio (Worse/Better)')
        ax3.set_title('Cohort Performance Ratios')
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
        for cohort in cohort_names:
            if cohort in cohort_performance:
                count = cohort_performance[cohort]['count']
                rmse = cohort_performance[cohort]['avg_rmse']
                summary_text.append(f"{cohort} patients analyzed: {count}")
        
        summary_text.append("")
        summary_text.append(f"Fairness Ratio: {fairness_ratio:.2f}x")
        fairness_level = cohort_performance.get('fairness_level', 'Unknown')
        summary_text.append(f"Fairness Level: {fairness_level}")
        summary_text.append("")
        summary_text.append("CONCLUSION:")
        if fairness_ratio < 1.2:
            summary_text.append("✅ Good fairness across cohorts")
        elif fairness_ratio < 1.5:
            summary_text.append("⚠️ Moderate cohort-based disparity")
        else:
            summary_text.append("🚨 Poor fairness - significant cohort bias")
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=11, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"cohort_fairness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_file}")
        return plot_file

    def generate_fairness_report(self, cohort_groups, cohort_performance):
        """Generate comprehensive report like gender analyzer"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COHORT FAIRNESS ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        if cohort_performance:
            report_lines.append("PERFORMANCE RESULTS:")
            for cohort in cohort_groups.keys():
                if cohort in cohort_performance:
                    perf = cohort_performance[cohort]
                    report_lines.append(f"  {cohort}:")
                    report_lines.append(f"    Patients: {perf['count']}")
                    report_lines.append(f"    Avg RMSE: {perf['avg_rmse']:.3f}")
                    report_lines.append(f"    Avg MAE: {perf['avg_mae']:.3f}")
                    report_lines.append("")
            
            report_lines.append("FAIRNESS ASSESSMENT:")
            report_lines.append(f"  RMSE Ratio: {cohort_performance['fairness_ratio']:.2f}x")
            report_lines.append(f"  Fairness Level: {cohort_performance['fairness_level']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_file = self.results_dir / f"cohort_fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        for line in report_lines:
            print(line)
        
        print(f"\n📁 Full report saved to: {report_file}")
        return report_file


def main():
    """Main function to run cohort fairness analysis"""
    
    print("🚀 Starting Cohort Fairness Analysis")
    print("=" * 50)
    
    
    try:
        analyzer = CohortFairnessAnalyzer()
        cohort_groups = analyzer.analyze_latest()
        
        if cohort_groups:
            # Calculate performance metrics
            cohort_performance = {}
            for cohort, patients in cohort_groups.items():
                if patients:
                    rmse_values = [p['rmse'] for p in patients]
                    mae_values = [p['mae'] for p in patients]
                    
                    cohort_performance[cohort] = {
                        'count': len(patients),
                        'avg_rmse': np.mean(rmse_values),
                        'avg_mae': np.mean(mae_values),
                        'patients': patients
                    }
            
            if cohort_performance:
                # Calculate fairness metrics
                rmse_vals = [perf['avg_rmse'] for perf in cohort_performance.values()]
                rmse_ratio = max(rmse_vals) / min(rmse_vals) if min(rmse_vals) > 0 else float('inf')
                
                if rmse_ratio <= 1.10:
                    fairness_level = "EXCELLENT"
                elif rmse_ratio <= 1.25:
                    fairness_level = "GOOD"
                elif rmse_ratio <= 1.50:
                    fairness_level = "ACCEPTABLE"
                else:
                    fairness_level = "POOR"
                
                print(f"\n⚖️ COHORT FAIRNESS ASSESSMENT: {fairness_level}")
                print(f"📊 RMSE Fairness Ratio: {rmse_ratio:.2f}x")
                
                # Create visualizations
                cohort_performance['fairness_ratio'] = rmse_ratio
                cohort_performance['fairness_level'] = fairness_level
                analyzer.create_comprehensive_visualizations(cohort_groups, cohort_performance)
                
                # Generate report
                analyzer.generate_fairness_report(cohort_groups, cohort_performance)
                
                print(f"\n📁 All results saved in: {analyzer.results_dir}")
        
        print("\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(cohort_groups)} cohorts analyzed")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
