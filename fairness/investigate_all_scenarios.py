#!/usr/bin/env python3
"""
Comprehensive Fairness Investigation Across All Scenarios
Analyzes fairness for:
1. Inference experiments (inference_only, trained_standard, trained_noisy, trained_denoised)
2. Distillation (teacher -> student -> distilled)
3. All-patients trained then inference mode
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

class ComprehensiveFairnessInvestigator:
    def __init__(self):
        self.results_dir = Path(__file__).parent / "analysis_results"
        self.inference_dir = self.results_dir / "inference_scenarios"
        self.distillation_all_patients_dir = self.results_dir / "distillation_all_patients"
        self.distillation_per_patient_dir = self.results_dir / "distillation_per_patient"
        
        # Match constants from investigate_fairness_issues.py
        self.POOR_RATIO = 1.5  # Fairness ratio above this is POOR
        self.SIGNIFICANT_DEGRADATION = 0.1  # 10% degradation is significant
    
    def find_latest_report(self, directory: Path, pattern: str) -> Path:
        """Find the most recent report file."""
        reports = sorted(directory.glob(pattern))
        if not reports:
            raise FileNotFoundError(f"No reports found matching {pattern} in {directory}")
        return reports[-1]
    
    def load_inference_data(self) -> Dict:
        """Load inference scenario data."""
        report_file = self.find_latest_report(
            self.inference_dir,
            "legendary_inference_scenarios_report_*.json"
        )
        print(f"📂 Loading inference: {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def load_distillation_all_patients_data(self) -> Dict:
        """Load distillation all-patients data."""
        report_file = self.find_latest_report(
            self.distillation_all_patients_dir,
            "legendary_distillation_report_*.json"
        )
        print(f"📂 Loading distillation (all-patients): {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def load_distillation_per_patient_data(self) -> Dict:
        """Load distillation per-patient data."""
        report_file = self.find_latest_report(
            self.distillation_per_patient_dir,
            "legendary_distillation_report_*.json"
        )
        print(f"📂 Loading distillation (per-patient): {report_file.name}")
        
        with open(report_file, 'r') as f:
            return json.load(f)
    
    def analyze_all_scenarios(self) -> Dict:
        """Analyze fairness across all scenarios."""
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE FAIRNESS ANALYSIS")
        print("=" * 80)
        
        results = {
            'inference': {},
            'distillation_all_patients': {},
            'distillation_per_patient': {},
            'all_scenarios': [],
            'poor_fairness': [],
            'significant_degradation': [],
            'top_issues': []
        }
        
        # 1. Analyze inference scenarios
        print("\n📊 Analyzing inference scenarios...")
        inference_data = self.load_inference_data()
        results['inference'] = self._analyze_inference(inference_data)
        
        # 2. Analyze distillation (all-patients)
        print("📊 Analyzing distillation (all-patients)...")
        distillation_all_data = self.load_distillation_all_patients_data()
        results['distillation_all_patients'] = self._analyze_distillation(
            distillation_all_data, 'all_patients'
        )
        
        # 3. Analyze distillation (per-patient)
        print("📊 Analyzing distillation (per-patient)...")
        distillation_per_data = self.load_distillation_per_patient_data()
        results['distillation_per_patient'] = self._analyze_distillation(
            distillation_per_data, 'per_patient'
        )
        
        # 4. Compile all scenarios for comparison
        results['all_scenarios'] = self._compile_all_scenarios(results)
        
        # 4. Find poor fairness cases
        results['poor_fairness'] = [s for s in results['all_scenarios'] 
                                    if s['fairness_ratio'] >= self.POOR_RATIO]
        
        # 5. Find significant degradation cases
        results['significant_degradation'] = [s for s in results['all_scenarios'] 
                                              if s.get('degradation', 0) >= self.SIGNIFICANT_DEGRADATION]
        
        # 6. Get top issues
        results['top_issues'] = sorted(results['all_scenarios'], 
                                       key=lambda x: x['fairness_ratio'], 
                                       reverse=True)[:20]
        
        return results
    
    def _analyze_inference(self, data: Dict) -> Dict:
        """Analyze inference scenarios."""
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']
        results = []
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['results']:
                continue
            
            feature_data = data['results'][display_name]
            
            for scenario in scenarios:
                if scenario not in feature_data:
                    continue
                
                scenario_data = feature_data[scenario]
                
                # Get baseline for comparison
                baseline_ratio = feature_data['inference_only']['fairness_ratio']
                degradation = scenario_data['fairness_ratio'] - baseline_ratio if scenario != 'inference_only' else 0
                
                # Get worst and best groups
                if 'groups' in scenario_data:
                    groups = scenario_data['groups']
                    rmse_values = [(group_name, group_data['rmse_mean']) 
                                 for group_name, group_data in groups.items() 
                                 if 'rmse_mean' in group_data]
                    
                    if rmse_values:
                        worst_group = max(rmse_values, key=lambda x: x[1])
                        best_group = min(rmse_values, key=lambda x: x[1])
                        
                        results.append({
                            'context': 'inference',
                            'scenario': scenario,
                            'feature': feature_key,
                            'fairness_ratio': scenario_data['fairness_ratio'],
                            'baseline_ratio': baseline_ratio,
                            'degradation': degradation,
                            'degradation_pct': (degradation / baseline_ratio * 100) if baseline_ratio > 0 else 0,
                            'worst_group': worst_group[0],
                            'worst_rmse': worst_group[1],
                            'best_group': best_group[0],
                            'best_rmse': best_group[1],
                            'gap': worst_group[1] - best_group[1]
                        })
        
        return {'scenarios': results}
    
    def _analyze_distillation(self, data: Dict, mode: str) -> Dict:
        """Analyze distillation scenarios.
        
        Args:
            data: The distillation data
            mode: Either 'all_patients' or 'per_patient'
        """
        feature_map = {
            'Gender': 'gender',
            'Age Group': 'age_group',
            'Pump Model': 'pump_model',
            'Sensor Band': 'sensor_band',
            'Cohort': 'cohort'
        }
        
        # Three phases: teacher, student_baseline, distilled
        phases = ['teacher', 'student_baseline', 'distilled']
        results = []
        
        if 'feature_analysis' not in data:
            return {'scenarios': results}
        
        for display_name, feature_key in feature_map.items():
            if display_name not in data['feature_analysis']:
                continue
            
            feature_data = data['feature_analysis'][display_name]
            
            # For each phase, we need to calculate fairness ratios
            # Teacher and distilled are in feature_analysis
            # Student_baseline needs to be calculated from individual reports
            
            for phase in ['teacher', 'distilled']:
                phase_ratio_key = f'{phase}_fairness_ratio'
                if phase_ratio_key not in feature_data:
                    continue
                
                fairness_ratio = feature_data[phase_ratio_key]
                
                # Get baseline (teacher)
                baseline_ratio = feature_data['teacher_fairness_ratio']
                degradation = fairness_ratio - baseline_ratio if phase != 'teacher' else 0
                
                # Get per-group impacts
                worst_group = None
                best_group = None
                worst_rmse = None
                best_rmse = None
                
                if 'per_group_impacts' in data and display_name in data['per_group_impacts']:
                    group_impacts = data['per_group_impacts'][display_name]
                    rmse_key = f'{phase}_rmse'
                    
                    rmse_values = [(group_name, group_data.get(rmse_key, 0)) 
                                 for group_name, group_data in group_impacts.items() 
                                 if rmse_key in group_data]
                    
                    if rmse_values:
                        worst = max(rmse_values, key=lambda x: x[1])
                        best = min(rmse_values, key=lambda x: x[1])
                        worst_group, worst_rmse = worst
                        best_group, best_rmse = best
                
                results.append({
                    'context': f'distillation_{mode}',
                    'scenario': phase,
                    'feature': feature_key,
                    'fairness_ratio': fairness_ratio,
                    'baseline_ratio': baseline_ratio,
                    'degradation': degradation,
                    'degradation_pct': (degradation / baseline_ratio * 100) if baseline_ratio > 0 else 0,
                    'worst_group': worst_group,
                    'worst_rmse': worst_rmse,
                    'best_group': best_group,
                    'best_rmse': best_rmse,
                    'gap': (worst_rmse - best_rmse) if (worst_rmse and best_rmse) else 0
                })
            
            # Calculate student_baseline fairness ratio from groups data
            # This requires loading individual feature report
            student_ratio = self._calculate_student_fairness_ratio(display_name, mode)
            if student_ratio:
                results.append({
                    'context': f'distillation_{mode}',
                    'scenario': 'student_baseline',
                    'feature': feature_key,
                    'fairness_ratio': student_ratio,
                    'baseline_ratio': feature_data['teacher_fairness_ratio'],
                    'degradation': student_ratio - feature_data['teacher_fairness_ratio'],
                    'degradation_pct': ((student_ratio - feature_data['teacher_fairness_ratio']) / feature_data['teacher_fairness_ratio'] * 100),
                    'worst_group': None,
                    'worst_rmse': None,
                    'best_group': None,
                    'best_rmse': None,
                    'gap': 0
                })
        
        return {'scenarios': results}
    
    def _calculate_student_fairness_ratio(self, feature_name: str, mode: str) -> Optional[float]:
        """Calculate student_baseline fairness ratio from groups data."""
        try:
            # Map feature display names to file names
            file_map = {
                'Gender': 'gender',
                'Age Group': 'age_group',
                'Pump Model': 'pump_model',
                'Sensor Band': 'sensor_band',
                'Cohort': 'cohort'
            }
            
            file_name = file_map.get(feature_name)
            if not file_name:
                return None
            
            # Load the individual feature report
            if mode == 'all_patients':
                dir_path = self.distillation_all_patients_dir
            else:
                dir_path = self.distillation_per_patient_dir
            
            pattern = f"{file_name}_fairness_report_{mode}_*.json"
            reports = sorted(dir_path.glob(pattern))
            if not reports:
                return None
            
            with open(reports[-1], 'r') as f:
                data = json.load(f)
            
            # Extract student_baseline RMSE for each group
            if 'groups' not in data:
                return None
            
            student_rmses = []
            for group_name, group_data in data['groups'].items():
                if 'phases' in group_data and 'student_baseline' in group_data['phases']:
                    student_rmse = group_data['phases']['student_baseline'].get('rmse_mean', 0)
                    if student_rmse > 0:
                        student_rmses.append(student_rmse)
            
            if len(student_rmses) >= 2:
                # Fairness ratio = worst / best
                return max(student_rmses) / min(student_rmses)
            
            return None
            
        except Exception as e:
            print(f"  Warning: Could not calculate student fairness for {feature_name} ({mode}): {e}")
            return None
    
    def _compile_all_scenarios(self, results: Dict) -> List[Dict]:
        """Compile all scenarios into one list from inference and both distillation types."""
        all_scenarios = []
        
        # Add inference scenarios
        for s in results['inference']['scenarios']:
            all_scenarios.append(s)
        
        # Add distillation all_patients scenarios
        for s in results['distillation_all_patients']['scenarios']:
            all_scenarios.append(s)
        
        # Add distillation per_patient scenarios
        for s in results['distillation_per_patient']['scenarios']:
            all_scenarios.append(s)
        
        return all_scenarios
    
    def print_summary(self, results: Dict):
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE SUMMARY")
        print("=" * 80)
        
        total_scenarios = len(results['all_scenarios'])
        inference_count = len(results['inference']['scenarios'])
        distillation_all_count = len(results['distillation_all_patients']['scenarios'])
        distillation_per_count = len(results['distillation_per_patient']['scenarios'])
        
        print(f"\n📈 Total Scenarios Analyzed: {total_scenarios}")
        print(f"   - Inference: {inference_count}")
        print(f"   - Distillation (All Patients): {distillation_all_count}")
        print(f"   - Distillation (Per Patient): {distillation_per_count}")
        
        # Poor fairness cases
        print(f"\n🔴 Poor Fairness Cases (≥{self.POOR_RATIO}): {len(results['poor_fairness'])}")
        if results['poor_fairness']:
            for case in sorted(results['poor_fairness'], key=lambda x: x['fairness_ratio'], reverse=True):
                print(f"   • {case['context'].upper()} - {case['feature'].replace('_', ' ').title()}")
                print(f"     Scenario: {case['scenario'].replace('_', ' ').title()}")
                print(f"     Fairness Ratio: {case['fairness_ratio']:.3f}x")
                if case['worst_group'] and case['best_group']:
                    print(f"     {case['worst_group']} vs {case['best_group']}: {case['gap']:.2f} RMSE gap")
        
        # Significant degradation
        print(f"\n⚠️  Significant Degradation (≥{self.SIGNIFICANT_DEGRADATION}): {len(results['significant_degradation'])}")
        if results['significant_degradation']:
            for case in sorted(results['significant_degradation'], key=lambda x: x['degradation'], reverse=True)[:10]:
                print(f"   • {case['context'].upper()} - {case['feature'].replace('_', ' ').title()}")
                print(f"     Scenario: {case['scenario'].replace('_', ' ').title()}")
                print(f"     {case['baseline_ratio']:.3f}x → {case['fairness_ratio']:.3f}x")
                print(f"     Degradation: {case['degradation']:+.3f}x ({case['degradation_pct']:+.1f}%)")
        
        # Top 10 worst fairness ratios
        print(f"\n🎯 TOP 10 WORST FAIRNESS RATIOS (All Contexts):")
        for i, case in enumerate(results['top_issues'][:10], 1):
            if case['context'].startswith('distillation'):
                context_label = "🔬 DISTILL"
            else:
                context_label = "💉 INFER"
            print(f"\n{i}. {context_label} | {case['feature'].replace('_', ' ').title()} - {case['scenario'].replace('_', ' ').title()}")
            print(f"   Fairness Ratio: {case['fairness_ratio']:.3f}x")
            if case['worst_group'] and case['best_group']:
                print(f"   Worst: {case['worst_group']} (RMSE={case['worst_rmse']:.2f})")
                print(f"   Best: {case['best_group']} (RMSE={case['best_rmse']:.2f})")
                print(f"   Gap: {case['gap']:.2f} RMSE points")
        
        # Context comparison
        print("\n" + "=" * 80)
        print("📊 CONTEXT COMPARISON")
        print("=" * 80)
        
        # Average by context
        inference_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios']]
        distillation_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios']]
        distillation_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios']]
        all_distillation_ratios = distillation_all_ratios + distillation_per_ratios
        
        print(f"\n📈 Average Fairness Ratios:")
        print(f"   Inference: {np.mean(inference_ratios):.3f}x (σ={np.std(inference_ratios):.3f})")
        print(f"   Distillation (All Patients): {np.mean(distillation_all_ratios):.3f}x (σ={np.std(distillation_all_ratios):.3f})")
        print(f"   Distillation (Per Patient): {np.mean(distillation_per_ratios):.3f}x (σ={np.std(distillation_per_ratios):.3f})")
        print(f"   All Distillation: {np.mean(all_distillation_ratios):.3f}x (σ={np.std(all_distillation_ratios):.3f})")
        
        # Worst in each context
        worst_inference = max(results['inference']['scenarios'], key=lambda x: x['fairness_ratio'])
        worst_distillation_all = max(results['distillation_all_patients']['scenarios'], key=lambda x: x['fairness_ratio'])
        worst_distillation_per = max(results['distillation_per_patient']['scenarios'], key=lambda x: x['fairness_ratio'])
        
        print(f"\n🔴 Worst Cases by Context:")
        print(f"   Inference: {worst_inference['feature'].replace('_', ' ').title()} - {worst_inference['scenario'].replace('_', ' ').title()}")
        print(f"              Ratio: {worst_inference['fairness_ratio']:.3f}x")
        print(f"   Distillation (All): {worst_distillation_all['feature'].replace('_', ' ').title()} - {worst_distillation_all['scenario'].replace('_', ' ').title()}")
        print(f"                       Ratio: {worst_distillation_all['fairness_ratio']:.3f}x")
        print(f"   Distillation (Per): {worst_distillation_per['feature'].replace('_', ' ').title()} - {worst_distillation_per['scenario'].replace('_', ' ').title()}")
        print(f"                       Ratio: {worst_distillation_per['fairness_ratio']:.3f}x")
    
    def generate_visual_report(self, results: Dict):
        """Generate comprehensive visual report."""
        print("\n" + "=" * 80)
        print("📊 GENERATING COMPREHENSIVE VISUAL REPORT")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"comprehensive_fairness_report_{timestamp}.png"
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.35)
        
        fig.suptitle('COMPREHENSIVE FAIRNESS ANALYSIS\nAll Scenarios: Inference + Distillation', 
                    fontsize=22, fontweight='bold', y=0.985)
        
        # 1. Overall fairness ratio distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_overall_distribution(ax1, results)
        
        # 2. Context comparison (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_context_comparison(ax2, results)
        
        # 3. Status summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_status_summary(ax3, results)
        
        # 4. Comprehensive heatmap with all scenarios (second row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_comprehensive_heatmap(ax4, results)
        
        # 5. Distillation comparison (third row, left 2/3)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_distillation_comparison(ax5, results)
        
        # 6. Top 15 worst cases (third row, right 1/3)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_top_worst_cases(ax6, results)
        
        # 7. Degradation analysis (fourth row, left)
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_degradation_analysis(ax7, results)
        
        # 8. Feature comparison across contexts (fourth row, middle)
        ax8 = fig.add_subplot(gs[3, 1])
        self._plot_feature_comparison(ax8, results)
        
        # 9. Recommendations (fourth row, right)
        ax9 = fig.add_subplot(gs[3, 2])
        self._plot_recommendations(ax9, results)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Visual report saved: {output_file}")
        return output_file
    
    def _plot_overall_distribution(self, ax, results: Dict):
        """Plot overall fairness ratio distribution."""
        ratios = [s['fairness_ratio'] for s in results['all_scenarios']]
        
        ax.hist(ratios, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(x=self.POOR_RATIO, color='red', linestyle='--', linewidth=2, label=f'Poor Threshold ({self.POOR_RATIO})')
        ax.axvline(x=np.median(ratios), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(ratios):.2f})')
        
        ax.set_xlabel('Fairness Ratio', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Fairness Ratio Distribution\n(All Scenarios)', fontweight='bold', fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_context_comparison(self, ax, results: Dict):
        """Plot comparison between contexts."""
        contexts = ['Inference', 'Distill\n(All)', 'Distill\n(Per)']
        
        inference_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios']]
        distillation_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios']]
        distillation_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios']]
        
        avgs = [np.mean(inference_ratios), np.mean(distillation_all_ratios), np.mean(distillation_per_ratios)]
        stds = [np.std(inference_ratios), np.std(distillation_all_ratios), np.std(distillation_per_ratios)]
        
        bars = ax.bar(contexts, avgs, yerr=stds, capsize=10, 
                     color=['#3498db', '#e74c3c', '#9b59b6'], alpha=0.7)
        
        # Position text above error bars to avoid overlapping
        for bar, avg, std in zip(bars, avgs, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{avg:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.axhline(y=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Average Fairness Ratio', fontweight='bold')
        ax.set_title('Context Comparison', fontweight='bold', fontsize=11)
        ax.set_ylim(0, max(avgs) + max(stds) + 0.15)  # Add extra space at top
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_status_summary(self, ax, results: Dict):
        """Plot status summary."""
        ax.axis('off')
        
        total = len(results['all_scenarios'])
        poor = len(results['poor_fairness'])
        degraded = len(results['significant_degradation'])
        
        inference_scenarios = len(results['inference']['scenarios'])
        distillation_all_scenarios = len(results['distillation_all_patients']['scenarios'])
        distillation_per_scenarios = len(results['distillation_per_patient']['scenarios'])
        
        summary_text = f"""
SUMMARY STATISTICS

Total Scenarios: {total}
  • Inference: {inference_scenarios}
  • Distillation (All): {distillation_all_scenarios}
  • Distillation (Per): {distillation_per_scenarios}

Poor Fairness: {poor}
Significant Degradation: {degraded}

Status: {'GOOD' if poor == 0 else 'NEEDS ATTENTION'}
"""
        
        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def _plot_comprehensive_heatmap(self, ax, results: Dict):
        """Plot comprehensive heatmap with all scenarios (inference + both distillation types)."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        # All scenarios: inference (4) + distillation all_patients (3) + distillation per_patient (3)
        scenarios = ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised',
                    'teacher_all', 'student_all', 'distilled_all',
                    'teacher_per', 'student_per', 'distilled_per']
        scenario_labels = ['Inference\nOnly', 'Trained\nStandard', 'Trained\nNoisy', 'Trained\nDenoised',
                          'Teacher\n(All)', 'Student\n(All)', 'Distilled\n(All)',
                          'Teacher\n(Per)', 'Student\n(Per)', 'Distilled\n(Per)']
        
        matrix = np.zeros((len(features), len(scenarios)))
        
        # Fill in inference scenarios
        for s in results['inference']['scenarios']:
            if s['feature'] in features and s['scenario'] in ['inference_only', 'trained_standard', 'trained_noisy', 'trained_denoised']:
                i = features.index(s['feature'])
                j = scenarios.index(s['scenario'])
                matrix[i, j] = s['fairness_ratio']
        
        # Fill in distillation all_patients scenarios
        for s in results['distillation_all_patients']['scenarios']:
            if s['feature'] in features:
                i = features.index(s['feature'])
                # Map phase to scenario index
                if s['scenario'] == 'teacher':
                    j = scenarios.index('teacher_all')
                elif s['scenario'] == 'student_baseline':
                    j = scenarios.index('student_all')
                elif s['scenario'] == 'distilled':
                    j = scenarios.index('distilled_all')
                else:
                    continue
                matrix[i, j] = s['fairness_ratio']
        
        # Fill in distillation per_patient scenarios
        for s in results['distillation_per_patient']['scenarios']:
            if s['feature'] in features:
                i = features.index(s['feature'])
                # Map phase to scenario index
                if s['scenario'] == 'teacher':
                    j = scenarios.index('teacher_per')
                elif s['scenario'] == 'student_baseline':
                    j = scenarios.index('student_per')
                elif s['scenario'] == 'distilled':
                    j = scenarios.index('distilled_per')
                else:
                    continue
                matrix[i, j] = s['fairness_ratio']
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.0)
        
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(scenario_labels, fontsize=7.5)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features], fontsize=10)
        
        # Add values to cells
        for i in range(len(features)):
            for j in range(len(scenarios)):
                if matrix[i, j] > 0:  # Only show non-zero values
                    text_color = 'white' if matrix[i, j] > 1.4 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha="center", va="center", color=text_color, fontsize=7, fontweight='bold')
        
        # Add vertical separators
        ax.axvline(x=3.5, color='white', linestyle='-', linewidth=3)
        ax.axvline(x=6.5, color='white', linestyle='-', linewidth=3)
        
        # Add context labels - keep at reasonable position
        ax.text(1.5, -0.9, 'INFERENCE', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        ax.text(5.0, -0.9, 'DISTILL (All Patients)', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
        ax.text(8.5, -0.9, 'DISTILL (Per Patient)', ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#9b59b6', alpha=0.3))
        
        # Move title further up with more padding
        ax.set_title('Comprehensive Fairness Ratios: All Scenarios', fontweight='bold', fontsize=13, pad=35)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fairness Ratio', rotation=270, labelpad=20, fontweight='bold')
        
        # Add threshold legend
        ax.text(4.5, 5.5, '< 1.1: Excellent | 1.1-1.25: Good | 1.25-1.5: Acceptable | > 1.5: Poor', 
                fontsize=8, ha='center', va='top')
    
    def _plot_distillation_comparison(self, ax, results: Dict):
        """Plot distillation comparison across both experiment types."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        
        feature_labels = [f.replace('_', ' ').title() for f in features]
        x = np.arange(len(features))
        width = 0.13  # Narrower bars to fit 6 groups
        
        # All patients: teacher, student, distilled
        phases_all = ['teacher', 'student_baseline', 'distilled']
        colors_all = ['#3498db', '#5dade2', '#85c1e9']
        
        for i, phase in enumerate(phases_all):
            ratios = []
            for feature in features:
                scenarios = [s for s in results['distillation_all_patients']['scenarios'] 
                           if s['feature'] == feature and s['scenario'] == phase]
                ratio = scenarios[0]['fairness_ratio'] if scenarios else 0
                ratios.append(ratio)
            
            offset = (i - 2.5) * width
            bars = ax.bar(x + offset, ratios, width, 
                         label=f"{phase.replace('_', ' ').title()} (All)",
                         alpha=0.8, color=colors_all[i])
            
            # Only show text if bar is tall enough to avoid crowding
            for bar, ratio in zip(bars, ratios):
                if ratio > 0 and bar.get_height() > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                           f'{ratio:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=0)
        
        # Per patient: teacher, student, distilled
        phases_per = ['teacher', 'student_baseline', 'distilled']
        colors_per = ['#e74c3c', '#ec7063', '#f1948a']
        
        for i, phase in enumerate(phases_per):
            ratios = []
            for feature in features:
                scenarios = [s for s in results['distillation_per_patient']['scenarios'] 
                           if s['feature'] == feature and s['scenario'] == phase]
                ratio = scenarios[0]['fairness_ratio'] if scenarios else 0
                ratios.append(ratio)
            
            offset = (i + 0.5) * width
            bars = ax.bar(x + offset, ratios, width, 
                         label=f"{phase.replace('_', ' ').title()} (Per)",
                         alpha=0.8, color=colors_per[i])
            
            # Only show text if bar is tall enough to avoid crowding
            for bar, ratio in zip(bars, ratios):
                if ratio > 0 and bar.get_height() > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                           f'{ratio:.2f}', ha='center', va='bottom', fontsize=5.5, rotation=0)
        
        ax.axhline(y=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor Threshold')
        ax.set_xlabel('Feature', fontweight='bold')
        ax.set_ylabel('Fairness Ratio', fontweight='bold')
        ax.set_title('Distillation: All Patients vs Per Patient', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=0, fontsize=9)
        ax.legend(loc='upper left', fontsize=6.5, ncol=2)
        ax.set_ylim(0, max([s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios'] + 
                            results['distillation_per_patient']['scenarios']]) * 1.15)  # Add 15% space at top
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_worst_cases(self, ax, results: Dict):
        """Plot top 15 worst cases."""
        top15 = results['top_issues'][:15]
        
        # Create single-line compact labels
        labels = []
        for s in top15:
            feature = s['feature'].replace('_', ' ').title()[:8]
            scenario = s['scenario'].replace('_', ' ')[:10]
            context = 'Inf' if s['context'] == 'inference' else 'Dst'
            # Single line format: "Feature: Scenario (Context)"
            labels.append(f"{feature}: {scenario} ({context})")
        
        ratios = [s['fairness_ratio'] for s in top15]
        colors = ['#e74c3c' if r >= self.POOR_RATIO else '#f39c12' if r >= 1.25 else '#3498db' for r in ratios]
        
        bars = ax.barh(labels, ratios, color=colors, alpha=0.8)
        
        # Position text inside bars to avoid overlapping
        for bar, ratio in zip(bars, ratios):
            # Place text at the end of the bar, inside it
            ax.text(ratio - 0.05, bar.get_y() + bar.get_height()/2,
                   f'{ratio:.2f}', va='center', ha='right', fontsize=7, fontweight='bold', color='white')
        
        ax.axvline(x=self.POOR_RATIO, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Fairness Ratio', fontweight='bold', fontsize=9)
        ax.set_title('Top 15 Worst Cases', fontweight='bold', fontsize=11)
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=7)  # Slightly larger for readability
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(ratios) + 0.1)  # Add a bit of space on right
    
    def _plot_degradation_analysis(self, ax, results: Dict):
        """Plot degradation analysis."""
        degraded = [s for s in results['all_scenarios'] if s['degradation'] > 0]
        
        if not degraded:
            ax.text(0.5, 0.5, 'No Degradation\nDetected', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='green')
            ax.axis('off')
            return
        
        # Group by context
        inference_deg = [s['degradation_pct'] for s in degraded if s['context'] == 'inference']
        distill_deg = [s['degradation_pct'] for s in degraded if s['context'] == 'distillation']
        
        data = [inference_deg, distill_deg]
        tick_labels = ['Inference', 'Distillation']
        
        bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylabel('Degradation (%)', fontweight='bold')
        ax.set_title('Degradation Distribution', fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_feature_comparison(self, ax, results: Dict):
        """Plot feature comparison across contexts."""
        features = ['gender', 'age_group', 'pump_model', 'sensor_band', 'cohort']
        
        inference_avgs = []
        distill_all_avgs = []
        distill_per_avgs = []
        
        for feature in features:
            inf_ratios = [s['fairness_ratio'] for s in results['inference']['scenarios'] 
                         if s['feature'] == feature]
            dist_all_ratios = [s['fairness_ratio'] for s in results['distillation_all_patients']['scenarios'] 
                              if s['feature'] == feature]
            dist_per_ratios = [s['fairness_ratio'] for s in results['distillation_per_patient']['scenarios'] 
                              if s['feature'] == feature]
            
            inference_avgs.append(np.mean(inf_ratios) if inf_ratios else 0)
            distill_all_avgs.append(np.mean(dist_all_ratios) if dist_all_ratios else 0)
            distill_per_avgs.append(np.mean(dist_per_ratios) if dist_per_ratios else 0)
        
        x = np.arange(len(features))
        width = 0.25
        
        bars1 = ax.bar(x - width, inference_avgs, width, label='Inference', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, distill_all_avgs, width, label='Distill (All)', color='#e74c3c', alpha=0.8)
        bars3 = ax.bar(x + width, distill_per_avgs, width, label='Distill (Per)', color='#9b59b6', alpha=0.8)
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Avg Fairness Ratio', fontweight='bold')
        ax.set_title('Feature Comparison', fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace('_', '\n').title() for f in features], fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_recommendations(self, ax, results: Dict):
        """Plot recommendations."""
        ax.axis('off')
        
        top_issues = results['top_issues'][:5]
        
        rec_text = "TOP PRIORITIES\n" + "=" * 30 + "\n\n"
        
        for i, issue in enumerate(top_issues, 1):
            context = issue['context'].upper()
            feature = issue['feature'].replace('_', ' ').title()
            scenario = issue['scenario'].replace('_', ' ').title()
            ratio = issue['fairness_ratio']
            
            rec_text += f"{i}. [{context}] {feature}\n"
            rec_text += f"   {scenario}\n"
            rec_text += f"   Ratio: {ratio:.2f}x\n\n"
        
        ax.text(0.05, 0.95, rec_text, ha='left', va='top', fontsize=9,
               family='monospace', transform=ax.transAxes)
    
    def run(self):
        """Run comprehensive analysis."""
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE FAIRNESS INVESTIGATOR")
        print("Analyzing: Inference + Distillation + All Patients Trained")
        print("=" * 80)
        
        try:
            # Analyze all scenarios
            results = self.analyze_all_scenarios()
            
            # Print summary
            self.print_summary(results)
            
            # Generate visual report
            visual_report = self.generate_visual_report(results)
            
            print("\n" + "=" * 80)
            print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
            print(f"📊 Visual report: {visual_report}")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    investigator = ComprehensiveFairnessInvestigator()
    investigator.run()
