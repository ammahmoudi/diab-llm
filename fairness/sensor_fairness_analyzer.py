#!/usr/bin/env python3
"""
Sensor Fairness Analyzer for OhioT1DM Dataset
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

class SensorFairnessAnalyzer:
    def __init__(self, data_path="/workspace/LLM-TIME/data/ohiot1dm/data.csv"):
        self.data_path = data_path
        self.patient_data = self._load_patient_data()
        
        # Create results directory
        self.results_dir = Path("/workspace/LLM-TIME/fairness/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Loaded sensor band data for {len(self.patient_data)} patients")
        
    def _load_patient_data(self):
        """Load patient sensor band info from data.csv"""
        try:
            df = pd.read_csv(self.data_path)
            sensor_mapping = {}
            for _, row in df.iterrows():
                patient_id = str(row['ID'])
                sensor_band = row['Sensor Band']
                sensor_mapping[patient_id] = sensor_band
            return sensor_mapping
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
        """Analyze sensor fairness for the latest experiment"""
        
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
        """Analyze sensor fairness for a specific experiment"""
        
        patient_results = self._load_patient_results(experiment_path)
        
        print(f"\n🔍 Sensor Band Fairness Analysis")
        print("=" * 50)
        
        # Group patients by sensor band
        sensor_groups = defaultdict(list)
        
        for patient_id, patient_data in patient_results.items():
            if patient_id in self.patient_data:
                sensor_band = self.patient_data[patient_id]
                sensor_groups[sensor_band].append({
                    'patient_id': patient_id,
                    'mse': patient_data.get('mse', 0),
                    'mae': patient_data.get('mae', 0), 
                    'rmse': patient_data.get('rmse', 0)
                })
        
        # Print group information
        for sensor_band, patients in sensor_groups.items():
            print(f"📡 {sensor_band}: {len(patients)} patients")
            print(f"   Patient IDs: {[p['patient_id'] for p in patients]}")
        
        return sensor_groups
    
    def create_comprehensive_visualizations(self, sensor_groups):
        """Create comprehensive visualizations matching gender analyzer format exactly"""
        
        if not sensor_groups:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        # Calculate performance metrics
        sensor_performance = {}
        for band, patients in sensor_groups.items():
            rmse_values = [p.get('rmse', 0) for p in patients]
            mae_values = [p.get('mae', 0) for p in patients]
            
            sensor_performance[band] = {
                'count': len(patients),
                'avg_rmse': np.mean(rmse_values) if rmse_values else 0,
                'avg_mae': np.mean(mae_values) if mae_values else 0
            }
        
        # Calculate fairness ratio
        bands = list(sensor_performance.keys())
        if len(bands) >= 2:
            rmse_values = [sensor_performance[band]['avg_rmse'] for band in bands]
            fairness_ratio = max(rmse_values) / min(rmse_values) if min(rmse_values) > 0 else 1.0
        else:
            fairness_ratio = 1.0
        
        # Create 2x2 subplot layout exactly like gender analyzer
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RMSE Comparison by Sensor Band (matching gender format)
        band_names = list(sensor_groups.keys())
        band_rmse = [sensor_performance[band]['avg_rmse'] for band in band_names]
        
        x = np.arange(len(band_names))
        width = 0.6
        
        colors = ['skyblue', 'lightcoral'] if len(band_names) == 2 else ['skyblue', 'lightcoral', 'lightgreen'][:len(band_names)]
        bars1 = ax1.bar(x, band_rmse, width, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Sensor Band')
        ax1.set_ylabel('RMSE (Lower = Better)')
        ax1.set_title('Performance by Sensor Band')
        ax1.set_xticks(x)
        ax1.set_xticklabels(band_names)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Fairness Score Progression (single point for ratio)
        ax2.plot([0], [fairness_ratio], 'o-', linewidth=3, markersize=10, color='purple')
        ax2.set_xlabel('Analysis')
        ax2.set_ylabel('Fairness Ratio (Lower = Better)')
        ax2.set_title('Sensor Band Fairness Ratio')
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
        model_names = ['Sensor Bands']
        
        colors = ['lightgreen' if r < 1.2 else 'gold' if r < 1.5 else 'salmon' for r in ratios]
        bars3 = ax3.bar(model_names, ratios, color=colors, alpha=0.7)
        ax3.set_xlabel('Comparison Type')
        ax3.set_ylabel('Performance Ratio (Worse/Better)')
        ax3.set_title('Sensor Band Performance Ratios')
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
        for band in band_names:
            if band in sensor_performance:
                count = sensor_performance[band]['count']
                rmse = sensor_performance[band]['avg_rmse']
                summary_text.append(f"{band} patients analyzed: {count}")
        
        summary_text.append("")
        summary_text.append(f"Fairness Ratio: {fairness_ratio:.2f}x")
        
        if fairness_ratio < 1.2:
            fairness_level = "GOOD"
            conclusion = "✅ Good fairness between sensor bands"
        elif fairness_ratio < 1.5:
            fairness_level = "MODERATE"
            conclusion = "⚠️ Moderate sensor-based disparity"
        else:
            fairness_level = "POOR"
            conclusion = "🚨 Poor fairness - significant sensor bias"
        
        summary_text.append(f"Fairness Level: {fairness_level}")
        summary_text.append("")
        summary_text.append("CONCLUSION:")
        summary_text.append(conclusion)
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=11, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"sensor_band_fairness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_file}")
        
        # Return performance results for consistency
        sensor_performance['fairness_ratio'] = fairness_ratio
        sensor_performance['fairness_level'] = fairness_level
        return sensor_performance


def main():
    """Main function to run sensor fairness analysis"""
    
    print("🚀 Starting Sensor Band Fairness Analysis")
    print("=" * 50)
    
    try:
        analyzer = SensorFairnessAnalyzer()
        sensor_groups = analyzer.analyze_latest()
        
        if sensor_groups:
            # Create comprehensive visualizations
            sensor_performance = analyzer.create_comprehensive_visualizations(sensor_groups)
            
            if sensor_performance:
                # Extract fairness metrics
                rmse_vals = [perf['avg_rmse'] for band, perf in sensor_performance.items() 
                            if band not in ['fairness_ratio', 'fairness_level']]
                rmse_ratio = max(rmse_vals) / min(rmse_vals) if min(rmse_vals) > 0 else float('inf')
                
                if rmse_ratio <= 1.10:
                    fairness_level = "EXCELLENT"
                elif rmse_ratio <= 1.25:
                    fairness_level = "GOOD"
                elif rmse_ratio <= 1.50:
                    fairness_level = "ACCEPTABLE"
                else:
                    fairness_level = "POOR"
                
                print(f"\n⚖️ SENSOR BAND FAIRNESS ASSESSMENT: {fairness_level}")
                print(f"📊 RMSE Fairness Ratio: {rmse_ratio:.2f}x")
                
                print(f"\n📁 All results saved in: {analyzer.results_dir}")
        
        print("\n🎉 Analysis Complete!")
        print(f"📊 Results: {len(sensor_groups)} sensor bands analyzed")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
