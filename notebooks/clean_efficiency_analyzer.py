"""
Clean, Simple LLM Efficiency Analysis Module
===========================================

This module replaces the messy notebook with clean, reusable functions.
No estimations - only real experimental data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Optional, Union, Any


class LLMEfficiencyAnalyzer:
    """Clean, organized LLM efficiency analysis with real data only"""
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize analyzer with base project path"""
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.outputs_path = self.base_path / "notebooks" / "outputs"
        self.latex_dir = self.outputs_path / "latex_tables"
        
        # Create output directories
        self.outputs_path.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)
        
        # Store loaded data
        self.efficiency_df = None
        self.inference_summary = None
        
        print(f"✅ LLMEfficiencyAnalyzer initialized")
        print(f"   Base path: {self.base_path}")
        print(f"   Output dir: {self.outputs_path}")
    
    def load_experimental_data(self) -> Dict[str, pd.DataFrame]:
        """Load all experimental data from pickle and CSV files"""
        print("🔄 Loading experimental data...")
        
        data = {}
        
        # Load efficiency data
        try:
            eff_files = list(self.data_path.rglob("*efficiency*.pkl"))
            if eff_files:
                with open(eff_files[0], 'rb') as f:
                    efficiency_data = pickle.load(f)
                    self.efficiency_df = pd.DataFrame(efficiency_data)
                    data['efficiency'] = self.efficiency_df
                    print(f"   ✅ Efficiency data: {len(self.efficiency_df)} records")
        except Exception as e:
            print(f"   ⚠️  Could not load efficiency data: {e}")
        
        # Load inference summary
        try:
            inference_files = list(self.outputs_path.rglob("*inference_summary*.csv"))
            if inference_files:
                self.inference_summary = pd.read_csv(inference_files[0])
                data['inference'] = self.inference_summary
                print(f"   ✅ Inference summary: {len(self.inference_summary)} models")
        except Exception as e:
            print(f"   ⚠️  Could not load inference summary: {e}")
        
        return data
    
    def generate_inference_latex_table(self, include_gpu: bool = False) -> str:
        """Generate clean LaTeX table for inference metrics using real data only"""
        print(f"🔄 Generating LaTeX table...")
        
        # Use real inference summary data if available
        if self.inference_summary is None:
            print("   ⚠️  No inference data available - loading data first...")
            self.load_experimental_data()
        
        if self.inference_summary is None:
            print("   ❌ No inference data found")
            return "No data available"
        
        # Create LaTeX content based on real data
        latex_content = self._create_latex_table(self.inference_summary)
        filename = "real_inference_performance.tex"
        
        # Save file
        latex_file = self.latex_dir / filename
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"   ✅ LaTeX saved: {latex_file.name}")
        return str(latex_file)
    
    def _create_latex_table(self, df: pd.DataFrame) -> str:
        """Create LaTeX table with real experimental data"""
        latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{array}
\usepackage{adjustbox}

\begin{document}

\begin{table*}[htbp]
    \centering
    \caption{LLM Inference Performance: Real Experimental Results}
    \label{tab:llm_real_performance}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Model} & \textbf{Size (MB)} & \textbf{Latency (ms)} & \textbf{Peak RAM (MB)} & \textbf{Power (W)} & \textbf{Throughput (samples/s)} \\
        \midrule
"""
        
        for _, row in df.iterrows():
            model = row.get('model_name', 'Unknown')
            size = f"{row.get('model_size_mb', 0):.0f}" if pd.notna(row.get('model_size_mb')) else "--"
            latency = f"{row.get('avg_inference_time_ms', 0):.1f}" if pd.notna(row.get('avg_inference_time_ms')) else "--"
            ram = f"{row.get('inference_peak_ram_mb', 0):.0f}" if pd.notna(row.get('inference_peak_ram_mb')) else "--"
            power = f"{row.get('inference_avg_power_w', 0):.1f}" if pd.notna(row.get('inference_avg_power_w')) else "--"
            throughput = f"{row.get('throughput_predictions_per_sec', 0):.2f}" if pd.notna(row.get('throughput_predictions_per_sec')) else "--"
            
            latex += f"        {model} & {size} & {latency} & {ram} & {power} & {throughput} \\\\\n"
        
        latex += r"""        \bottomrule
    \end{tabular}%
    }
\end{table*}

\end{document}"""
        
        return latex
    
    def analyze_performance(self) -> Dict[str, float]:
        """Analyze model performance using real experimental data only"""
        print("🔄 Analyzing performance...")
        
        analysis: Dict[str, float] = {
            'total_models': 0.0,
            'avg_latency': 0.0,
            'avg_model_size': 0.0,
            'avg_memory': 0.0,
            'avg_power': 0.0
        }
        
        # Use real data if available
        if self.inference_summary is not None:
            df = self.inference_summary
            analysis['total_models'] = float(len(df))
            
            # Calculate real statistics
            if len(df) > 0:
                latencies = df['avg_inference_time_ms'].dropna()
                sizes = df['model_size_mb'].dropna()
                memory = df['inference_peak_ram_mb'].dropna()
                power = df['inference_avg_power_w'].dropna()
                
                if len(latencies) > 0:
                    analysis['avg_latency'] = float(latencies.mean())
                if len(sizes) > 0:
                    analysis['avg_model_size'] = float(sizes.mean())
                if len(memory) > 0:
                    analysis['avg_memory'] = float(memory.mean())
                if len(power) > 0:
                    analysis['avg_power'] = float(power.mean())
        
        print(f"   ✅ Analysis complete: {int(analysis['total_models'])} models analyzed")
        return analysis
    
    def create_summary_report(self) -> str:
        """Create a comprehensive summary report using real data"""
        print("🔄 Creating summary report...")
        
        analysis = self.analyze_performance()
        
        report = f"""
LLM EFFICIENCY ANALYSIS SUMMARY REPORT
=====================================

Dataset Overview:
- Total models analyzed: {int(analysis['total_models'])}
- Data source: Real experimental results

Performance Metrics (Averages):
- Inference latency: {analysis['avg_latency']:.1f} ms
- Model size: {analysis['avg_model_size']:.0f} MB  
- Peak memory usage: {analysis['avg_memory']:.0f} MB
- Power consumption: {analysis['avg_power']:.1f} W

Analysis Notes:
- Based on real experimental measurements only
- No synthetic or estimated data included
- All metrics from actual model execution
"""
        
        # Save report
        report_file = self.outputs_path / "real_efficiency_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"   ✅ Report saved: {report_file}")
        return report
    
    def run_complete_analysis(self) -> Dict[str, str]:
        """Run complete analysis pipeline using only real experimental data"""
        print("🚀 RUNNING REAL DATA LLM EFFICIENCY ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        # 1. Load data
        data = self.load_experimental_data()
        results['data_loaded'] = f"Loaded {len(data)} data sources"
        
        # 2. Generate LaTeX table
        latex_file = self.generate_inference_latex_table()
        results['latex_table'] = latex_file
        
        # 3. Analyze performance
        analysis = self.analyze_performance()
        results['analysis'] = f"{int(analysis['total_models'])} models analyzed"
        
        # 4. Create report
        report = self.create_summary_report()
        results['report'] = "Summary report generated"
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"LaTeX table: {Path(latex_file).name if latex_file != 'No data available' else 'No data available'}")
        print(f"Models analyzed: {int(analysis['total_models'])}")
        print(f"All outputs in: {self.outputs_path}")
        
        return results


# Quick usage functions
def quick_analysis(base_path: str = "/home/amma/LLM-TIME"):
    """Quick analysis function for testing"""
    analyzer = LLMEfficiencyAnalyzer(base_path)
    data = analyzer.load_experimental_data()
    
    if analyzer.inference_summary is not None:
        print(f"Loaded {len(analyzer.inference_summary)} inference records")
        latex_table = analyzer.generate_inference_latex_table()
        print("\nLaTeX table generated successfully!")
    else:
        print("No inference data found")
    
    return analyzer


def generate_latex_only(base_path: str = "/home/amma/LLM-TIME") -> str:
    """Just generate LaTeX table quickly"""
    analyzer = LLMEfficiencyAnalyzer(base_path)
    analyzer.load_experimental_data()
    return analyzer.generate_inference_latex_table()


def run_full_analysis(base_path: str = "/home/amma/LLM-TIME") -> Dict[str, str]:
    """Run the complete analysis pipeline"""
    analyzer = LLMEfficiencyAnalyzer(base_path)
    return analyzer.run_complete_analysis()