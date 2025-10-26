"""
Gender Fairness Analysis using Distillation Results
==================================================

This script analyzes your distillation experiment results to determine
if distillation makes gender fairness worse using actual performance metrics.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob
import sys

sys.path.append('/workspace/LLM-TIME')
from fairness.analysis.patient_analyzer import PatientAnalyzer
from fairness.metrics.fairness_metrics import FairnessMetrics
from fairness.visualization.fairness_plots import FairnessVisualizer


class GenderFairnessAnalyzer:
    """Analyze gender fairness using your actual distillation experiment results."""
    
    def __init__(self, experiments_dir="/workspace/LLM-TIME/distillation_experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.patient_analyzer = PatientAnalyzer()
        self.fairness_metrics = FairnessMetrics()
        self.visualizer = FairnessVisualizer()
        
        # Create results directory
        self.results_dir = Path("/workspace/LLM-TIME/fairness/analysis_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def find_experiment_directories(self):
        """Find all available experiment directories."""
        experiment_dirs = list(self.experiments_dir.glob("pipeline_*"))
        experiment_dirs.sort(reverse=True)  # Most recent first
        
        print(f"📁 Found {len(experiment_dirs)} experiment directories:")
        for i, exp_dir in enumerate(experiment_dirs):
            print(f"  {i}: {exp_dir.name}")
        
        return experiment_dirs
    
    def load_patient_results(self, experiment_dir):
        """Load all patient results from an experiment directory."""
        patient_results = {}
        
        # Find all patient directories
        patient_dirs = list(experiment_dir.glob("**/patient_*"))
        
        print(f"📊 Loading results from {len(patient_dirs)} patients...")
        
        for patient_dir in patient_dirs:
            patient_id = int(patient_dir.name.split('_')[1])
            
            try:
                # Load teacher results
                teacher_file = patient_dir / "phase_1_teacher" / "teacher_training_summary.json"
                student_file = patient_dir / "phase_2_student" / "student_baseline_summary.json" 
                distilled_file = patient_dir / "phase_3_distillation" / "distillation_summary.json"
                
                patient_data = {'patient_id': patient_id}
                
                # Load teacher results
                if teacher_file.exists():
                    with open(teacher_file) as f:
                        teacher_data = json.load(f)
                        patient_data['teacher'] = {
                            'rmse': teacher_data['performance_metrics']['rmse'],
                            'mae': teacher_data['performance_metrics']['mae'],
                            'mape': teacher_data['performance_metrics']['mape']
                        }
                
                # Load student baseline results
                if student_file.exists():
                    with open(student_file) as f:
                        student_data = json.load(f)
                        patient_data['student_baseline'] = {
                            'rmse': student_data['performance_metrics']['rmse'],
                            'mae': student_data['performance_metrics']['mae'],
                            'mape': student_data['performance_metrics']['mape']
                        }
                
                # Load distilled results
                if distilled_file.exists():
                    with open(distilled_file) as f:
                        distilled_data = json.load(f)
                        patient_data['distilled'] = {
                            'rmse': distilled_data['performance_metrics']['rmse'],
                            'mae': distilled_data['performance_metrics']['mae'],
                            'mape': distilled_data['performance_metrics']['mape']
                        }
                
                patient_results[patient_id] = patient_data
                
            except Exception as e:
                print(f"⚠️  Could not load results for patient {patient_id}: {e}")
        
        return patient_results
    
    def analyze_gender_fairness(self, patient_results):
        """Analyze gender fairness using real performance metrics."""
        
        print("🔍 ANALYZING GENDER FAIRNESS")
        print("="*50)
        
        # Get gender groups
        gender_groups = self.patient_analyzer.create_fairness_groups('Gender')
        
        # Group patients by gender
        male_patients = gender_groups['male']
        female_patients = gender_groups['female']
        
        print(f"👨 Male patients: {male_patients}")
        print(f"👩 Female patients: {female_patients}")
        
        # Calculate performance by gender group
        gender_performance = {}
        
        for gender, patient_ids in [('male', male_patients), ('female', female_patients)]:
            # Filter to patients we have results for
            available_patients = [pid for pid in patient_ids if pid in patient_results]
            
            if not available_patients:
                print(f"⚠️  No results available for {gender} patients")
                continue
            
            print(f"\n📊 {gender.upper()} GROUP ANALYSIS:")
            print(f"   Available patients: {available_patients}")
            
            gender_perf = {}
            
            # Calculate averages for each model type
            for model_type in ['teacher', 'student_baseline', 'distilled']:
                metrics = []
                
                for patient_id in available_patients:
                    if model_type in patient_results[patient_id]:
                        metrics.append(patient_results[patient_id][model_type])
                
                if metrics:
                    # Calculate average performance across patients in this gender group
                    avg_rmse = np.mean([m['rmse'] for m in metrics])
                    avg_mae = np.mean([m['mae'] for m in metrics])
                    avg_mape = np.mean([m['mape'] for m in metrics])
                    
                    gender_perf[model_type] = {
                        'rmse': avg_rmse,
                        'mae': avg_mae,
                        'mape': avg_mape,
                        'sample_count': len(metrics)
                    }
                    
                    print(f"   {model_type.upper()}:")
                    print(f"     RMSE: {avg_rmse:.3f}")
                    print(f"     MAE: {avg_mae:.3f}")
                    print(f"     MAPE: {avg_mape:.3f}")
                    print(f"     Samples: {len(metrics)}")
            
            gender_performance[gender] = gender_perf
        
        return gender_performance
    
    def calculate_fairness_metrics(self, gender_performance):
        """Calculate fairness metrics using real performance data."""
        
        if 'male' not in gender_performance or 'female' not in gender_performance:
            print("❌ Cannot calculate fairness - missing gender group data")
            return None
        
        print("⚖️ FAIRNESS ANALYSIS")
        print("="*50)
        
        fairness_analysis = {}
        
        # Analyze each model type
        for model_type in ['teacher', 'student_baseline', 'distilled']:
            if (model_type in gender_performance['male'] and 
                model_type in gender_performance['female']):
                
                male_rmse = gender_performance['male'][model_type]['rmse']
                female_rmse = gender_performance['female'][model_type]['rmse']
                male_mae = gender_performance['male'][model_type]['mae']
                female_mae = gender_performance['female'][model_type]['mae']
                
                # Calculate fairness metrics using RMSE (lower is better)
                rmse_difference = abs(male_rmse - female_rmse)
                rmse_ratio = max(male_rmse, female_rmse) / min(male_rmse, female_rmse)
                
                # Fairness through awareness (coefficient of variation)
                mean_rmse = (male_rmse + female_rmse) / 2
                std_rmse = np.std([male_rmse, female_rmse])
                fairness_score = std_rmse / mean_rmse if mean_rmse > 0 else 0
                
                # Determine which gender performs worse
                worse_gender = 'male' if male_rmse > female_rmse else 'female'
                better_gender = 'female' if male_rmse > female_rmse else 'male'
                
                fairness_analysis[model_type] = {
                    'male_rmse': male_rmse,
                    'female_rmse': female_rmse,
                    'male_mae': male_mae,
                    'female_mae': female_mae,
                    'rmse_difference': rmse_difference,
                    'rmse_ratio': rmse_ratio,
                    'fairness_score': fairness_score,
                    'worse_gender': worse_gender,
                    'better_gender': better_gender,
                    'fairness_assessment': self.assess_fairness(fairness_score)
                }
                
                print(f"\n{model_type.upper()} MODEL:")
                print(f"  Male RMSE:   {male_rmse:.3f}")
                print(f"  Female RMSE: {female_rmse:.3f}")
                print(f"  Difference:  {rmse_difference:.3f}")
                print(f"  Ratio:       {rmse_ratio:.2f}x")
                print(f"  Fairness:    {fairness_score:.4f} ({fairness_analysis[model_type]['fairness_assessment']})")
                print(f"  Worse for:   {worse_gender}")
        
        return fairness_analysis
    
    def assess_fairness(self, fairness_score):
        """Assess fairness level based on score."""
        if fairness_score < 0.05:
            return "EXCELLENT"
        elif fairness_score < 0.1:
            return "GOOD"
        elif fairness_score < 0.2:
            return "MODERATE"
        else:
            return "POOR"
    
    def analyze_distillation_impact(self, fairness_analysis):
        """Analyze if distillation makes gender fairness worse."""
        
        if 'teacher' not in fairness_analysis or 'distilled' not in fairness_analysis:
            print("❌ Cannot analyze distillation impact - missing teacher or distilled results")
            return None
        
        print("\n🎯 DISTILLATION IMPACT ON GENDER FAIRNESS")
        print("="*60)
        
        teacher_fairness = fairness_analysis['teacher']['fairness_score']
        distilled_fairness = fairness_analysis['distilled']['fairness_score']
        
        fairness_change = distilled_fairness - teacher_fairness
        percent_change = (fairness_change / teacher_fairness) * 100 if teacher_fairness > 0 else 0
        
        # Compare ratios
        teacher_ratio = fairness_analysis['teacher']['rmse_ratio']
        distilled_ratio = fairness_analysis['distilled']['rmse_ratio']
        ratio_change = distilled_ratio - teacher_ratio
        
        print(f"📊 FAIRNESS SCORES:")
        print(f"   Teacher:    {teacher_fairness:.4f} ({fairness_analysis['teacher']['fairness_assessment']})")
        print(f"   Distilled:  {distilled_fairness:.4f} ({fairness_analysis['distilled']['fairness_assessment']})")
        print(f"   Change:     {fairness_change:+.4f} ({percent_change:+.1f}%)")
        
        print(f"\n📊 PERFORMANCE RATIOS:")
        print(f"   Teacher:    {teacher_ratio:.2f}x")
        print(f"   Distilled:  {distilled_ratio:.2f}x") 
        print(f"   Change:     {ratio_change:+.2f}x")
        
        # Determine conclusion
        makes_worse = distilled_fairness > teacher_fairness
        
        print(f"\n🎯 CONCLUSION:")
        if makes_worse:
            if abs(fairness_change) > 0.05:
                conclusion = "🚨 DISTILLATION SIGNIFICANTLY WORSENS GENDER FAIRNESS"
                severity = "CRITICAL"
            elif abs(fairness_change) > 0.02:
                conclusion = "⚠️ DISTILLATION MODERATELY WORSENS GENDER FAIRNESS" 
                severity = "MODERATE"
            else:
                conclusion = "➖ DISTILLATION SLIGHTLY WORSENS GENDER FAIRNESS"
                severity = "MINOR"
        else:
            conclusion = "✅ DISTILLATION MAINTAINS OR IMPROVES GENDER FAIRNESS"
            severity = "GOOD"
        
        print(f"   {conclusion}")
        print(f"   Severity: {severity}")
        
        return {
            'makes_fairness_worse': makes_worse,
            'fairness_change': fairness_change,
            'percent_change': percent_change,
            'ratio_change': ratio_change,
            'conclusion': conclusion,
            'severity': severity,
            'teacher_fairness': teacher_fairness,
            'distilled_fairness': distilled_fairness
        }
    
    def create_fairness_visualizations(self, gender_performance, fairness_analysis):
        """Create visualizations using real data."""
        
        if not gender_performance or not fairness_analysis:
            print("⚠️ Cannot create visualizations - insufficient data")
            return
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. RMSE Comparison by Gender
        models = ['Teacher', 'Student', 'Distilled']
        male_rmse = []
        female_rmse = []
        
        for model_type in ['teacher', 'student_baseline', 'distilled']:
            if model_type in fairness_analysis:
                male_rmse.append(fairness_analysis[model_type]['male_rmse'])
                female_rmse.append(fairness_analysis[model_type]['female_rmse'])
        
        if male_rmse and female_rmse:
            x = np.arange(len(models[:len(male_rmse)]))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, male_rmse, width, label='Male', color='skyblue', alpha=0.8)
            bars2 = ax1.bar(x + width/2, female_rmse, width, label='Female', color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Model Type')
            ax1.set_ylabel('RMSE (Lower = Better)')
            ax1.set_title('Performance by Gender Group')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models[:len(male_rmse)])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Fairness Score Progression
        if fairness_analysis:
            fairness_scores = [fairness_analysis[model]['fairness_score'] 
                             for model in ['teacher', 'student_baseline', 'distilled'] 
                             if model in fairness_analysis]
            model_names = [model.replace('_', ' ').title() for model in ['teacher', 'student_baseline', 'distilled'] 
                          if model in fairness_analysis]
            
            ax2.plot(model_names, fairness_scores, 'o-', linewidth=3, markersize=10, color='purple')
            ax2.set_xlabel('Model Type')
            ax2.set_ylabel('Fairness Score (Lower = Better)')
            ax2.set_title('Fairness Score Progression')
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Good Fairness Threshold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Annotate points
            for i, (model, score) in enumerate(zip(model_names, fairness_scores)):
                ax2.annotate(f'{score:.3f}', (i, score), textcoords="offset points", 
                            xytext=(0,15), ha='center', fontsize=10, fontweight='bold')
        
        # 3. Performance Ratios
        if fairness_analysis:
            ratios = [fairness_analysis[model]['rmse_ratio'] 
                     for model in ['teacher', 'student_baseline', 'distilled'] 
                     if model in fairness_analysis]
            
            colors = ['lightgreen' if r < 1.2 else 'gold' if r < 1.5 else 'salmon' for r in ratios]
            bars3 = ax3.bar(model_names, ratios, color=colors, alpha=0.7)
            ax3.set_xlabel('Model Type')
            ax3.set_ylabel('Performance Ratio (Worse/Better)')
            ax3.set_title('Gender Performance Ratios')
            ax3.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect Fairness')
            ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Acceptable Threshold')
            ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Poor Fairness')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            for bar, ratio in zip(bars3, ratios):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{ratio:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # 4. Individual Patient Results (if available)
        ax4.text(0.1, 0.9, 'EXPERIMENT SUMMARY', fontsize=14, fontweight='bold', 
                transform=ax4.transAxes)
        
        summary_text = []
        if 'male' in gender_performance and 'female' in gender_performance:
            male_count = gender_performance['male'].get('teacher', {}).get('sample_count', 0)
            female_count = gender_performance['female'].get('teacher', {}).get('sample_count', 0)
            summary_text.append(f"Male patients analyzed: {male_count}")
            summary_text.append(f"Female patients analyzed: {female_count}")
        
        if fairness_analysis and 'teacher' in fairness_analysis and 'distilled' in fairness_analysis:
            impact = fairness_analysis['distilled']['fairness_score'] - fairness_analysis['teacher']['fairness_score']
            summary_text.append(f"")
            summary_text.append(f"Fairness Impact:")
            summary_text.append(f"Teacher: {fairness_analysis['teacher']['fairness_score']:.3f}")
            summary_text.append(f"Distilled: {fairness_analysis['distilled']['fairness_score']:.3f}")
            summary_text.append(f"Change: {impact:+.3f}")
            summary_text.append(f"")
            summary_text.append("CONCLUSION:")
            if impact > 0.02:
                summary_text.append("🚨 Distillation worsens fairness")
            elif impact > 0:
                summary_text.append("⚠️ Distillation slightly worsens fairness")
            else:
                summary_text.append("✅ Distillation maintains fairness")
        
        for i, text in enumerate(summary_text):
            ax4.text(0.1, 0.8 - i*0.08, text, fontsize=11, transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"gender_fairness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_file}")
    
    def generate_fairness_report(self, gender_performance, fairness_analysis, distillation_impact):
        """Generate comprehensive report using real data."""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("GENDER FAIRNESS ANALYSIS REPORT")
        report_lines.append("Using Distillation Experiment Results")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Experiment details
        report_lines.append("EXPERIMENT DETAILS:")
        if gender_performance:
            for gender in ['male', 'female']:
                if gender in gender_performance:
                    sample_count = gender_performance[gender].get('teacher', {}).get('sample_count', 0)
                    report_lines.append(f"  {gender.capitalize()} patients analyzed: {sample_count}")
        report_lines.append("")
        
        # Performance results
        if fairness_analysis:
            report_lines.append("PERFORMANCE RESULTS:")
            for model_type, results in fairness_analysis.items():
                report_lines.append(f"  {model_type.upper().replace('_', ' ')} MODEL:")
                report_lines.append(f"    Male RMSE:     {results['male_rmse']:.3f}")
                report_lines.append(f"    Female RMSE:   {results['female_rmse']:.3f}")
                report_lines.append(f"    Difference:    {results['rmse_difference']:.3f}")
                report_lines.append(f"    Ratio:         {results['rmse_ratio']:.2f}x")
                report_lines.append(f"    Fairness:      {results['fairness_score']:.4f} ({results['fairness_assessment']})")
                report_lines.append(f"    Worse for:     {results['worse_gender']}")
                report_lines.append("")
        
        # Distillation impact
        if distillation_impact:
            report_lines.append("DISTILLATION IMPACT:")
            report_lines.append(f"  {distillation_impact['conclusion']}")
            report_lines.append(f"  Fairness change: {distillation_impact['fairness_change']:+.4f}")
            report_lines.append(f"  Percent change:  {distillation_impact['percent_change']:+.1f}%")
            report_lines.append(f"  Severity:        {distillation_impact['severity']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if distillation_impact and distillation_impact['makes_fairness_worse']:
            report_lines.append("  🎯 IMMEDIATE ACTION NEEDED:")
            report_lines.append("    - Implement fairness-aware distillation")
            report_lines.append("    - Use demographic parity loss (fairness_weight=0.5)")
            report_lines.append("    - Balance training data across gender groups")
            report_lines.append("    - Monitor fairness metrics during training")
        else:
            report_lines.append("  ✅ Current approach maintains acceptable fairness")
            report_lines.append("    - Continue monitoring fairness in future experiments")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Save report
        report_file = self.results_dir / f"gender_fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Print to console
        for line in report_lines:
            print(line)
        
        print(f"\n📁 Full report saved to: {report_file}")
        return report_file


def main():
    """Main function to run gender fairness analysis."""
    
    print("🎯 GENDER FAIRNESS ANALYSIS")
    print("Using Your Distillation Experiment Results")
    print("="*60)
    
    analyzer = GenderFairnessAnalyzer()
    
    # Find available experiments
    experiment_dirs = analyzer.find_experiment_directories()
    if not experiment_dirs:
        print("❌ No experiment directories found!")
        return
    
    # Use the most recent experiment
    latest_experiment = experiment_dirs[0]
    print(f"\n✅ Using latest experiment: {latest_experiment.name}")
    
    # Load patient results
    patient_results = analyzer.load_patient_results(latest_experiment)
    if not patient_results:
        print("❌ No patient results found!")
        return
    
    print(f"✅ Loaded results for {len(patient_results)} patients")
    
    # Analyze gender fairness
    gender_performance = analyzer.analyze_gender_fairness(patient_results)
    if not gender_performance:
        print("❌ Could not analyze gender performance!")
        return
    
    # Calculate fairness metrics
    fairness_analysis = analyzer.calculate_fairness_metrics(gender_performance)
    if not fairness_analysis:
        print("❌ Could not calculate fairness metrics!")
        return
    
    # Analyze distillation impact
    distillation_impact = analyzer.analyze_distillation_impact(fairness_analysis)
    
    # Create visualizations
    analyzer.create_fairness_visualizations(gender_performance, fairness_analysis)
    
    # Generate report
    analyzer.generate_fairness_report(gender_performance, fairness_analysis, distillation_impact)
    
    print("\n🎉 GENDER FAIRNESS ANALYSIS COMPLETE!")
    print(f"📁 All results saved in: {analyzer.results_dir}")


if __name__ == "__main__":
    main()
