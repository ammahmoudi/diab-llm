"""
Clean, function-based LLM Efficiency Analysis Module
===================================================

This module replaces the messy notebook with clean, reusable functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import re
from typing import Dict, List, Tuple, Optional, Union

class LLMEfficiencyAnalyzer:
    """Clean, organized LLM efficiency analysis with proper functions"""
    
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
        self.edge_df = None
        
        print(f"✅ LLMEfficiencyAnalyzer initialized")
        print(f"   Base path: {self.base_path}")
        print(f"   Output dir: {self.outputs_path}")
    
    def load_experimental_data(self) -> Dict[str, pd.DataFrame]:
        """Load all experimental data from pickle and log files"""
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
    

    
    def generate_inference_latex_table(self, include_gpu: bool = True) -> str:
        """Generate clean LaTeX table for inference metrics"""
        print(f"🔄 Generating LaTeX table (GPU: {include_gpu})...")
        
        # Use real inference summary data if available
        if self.inference_summary is not None:
            inference_df = self.inference_summary.copy()
        else:
            print("   ⚠️  No inference data available")
            return "No data available"
        
        # Create LaTeX content based on real data
        if include_gpu:
            latex_content = self._create_cpu_gpu_latex_real(inference_df)
            filename = "inference_cpu_gpu_clean.tex"
        else:
            latex_content = self._create_cpu_only_latex_real(inference_df)
            filename = "inference_cpu_only_clean.tex"
        
        # Save file
        latex_file = self.latex_dir / filename
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"   ✅ LaTeX saved: {latex_file.name}")
        return str(latex_file)
    
    def _create_cpu_gpu_latex(self, df: pd.DataFrame) -> str:
        """Create LaTeX table with both CPU and GPU metrics"""
        latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{adjustbox}

\begin{document}

\begin{table*}[htbp]
    \centering
    \caption{LLM Inference Performance: CPU vs GPU Acceleration}
    \label{tab:llm_cpu_gpu_performance}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lcccccccc}
        \toprule
        \textbf{Model} & \textbf{Size} & \textbf{CPU Latency} & \textbf{GPU Latency} & \textbf{Peak RAM} & \textbf{Power} & \textbf{Throughput} & \textbf{Edge Ready} \\
        & \textbf{(MB)} & \textbf{(ms)} & \textbf{(ms)} & \textbf{(MB)} & \textbf{(W)} & \textbf{(samples/s)} & \\
        \midrule
"""
        
        for _, row in df.iterrows():
            model = row['Model']
            size = f"{row['Model_Size_MB']:.0f}" if pd.notna(row['Model_Size_MB']) else "--"
            
            cpu_lat = row['CPU_Latency_ms']
            if pd.notna(cpu_lat):
                if cpu_lat >= 10000:
                    cpu_latency = f"{cpu_lat:.0f}"
                elif cpu_lat >= 1000:
                    cpu_latency = f"{cpu_lat:.1f}"
                else:
                    cpu_latency = f"{cpu_lat:.1f}"
            else:
                cpu_latency = "--"
            
            gpu_lat = row.get('GPU_Latency_ms', None)
            gpu_latency = f"{gpu_lat:.0f}" if pd.notna(gpu_lat) else "--"
            
            ram = f"{row['Peak_Memory_MB']:.0f}" if pd.notna(row['Peak_Memory_MB']) else "--"
            power = f"{row['Power_Consumption_W']:.1f}" if pd.notna(row['Power_Consumption_W']) else "--"
            
            throughput_val = row['Throughput_samples_per_sec']
            if pd.notna(throughput_val):
                if throughput_val < 0.001:
                    throughput = f"{throughput_val:.6f}"
                elif throughput_val < 0.01:
                    throughput = f"{throughput_val:.4f}"
                elif throughput_val < 1:
                    throughput = f"{throughput_val:.3f}"
                else:
                    throughput = f"{throughput_val:.2f}"
            else:
                throughput = "--"
            
            # Edge readiness (conservative thresholds)
            best_latency = gpu_lat if pd.notna(gpu_lat) else cpu_lat
            edge_ready = (pd.notna(best_latency) and best_latency < 1000 and
                         pd.notna(row['Peak_Memory_MB']) and row['Peak_Memory_MB'] < 4000 and
                         pd.notna(row['Model_Size_MB']) and row['Model_Size_MB'] < 2000)
            
            edge_status = "\\textcolor{green}{Yes}" if edge_ready else "\\textcolor{red}{No}"
            
            latex += f"        {model} & {size} & {cpu_latency} & {gpu_latency} & {ram} & {power} & {throughput} & {edge_status} \\\\\n"
        
        latex += r"""        \bottomrule
    \end{tabular}%
    }
\end{table*}

\end{document}"""
        
        return latex
    
    def _create_cpu_only_latex(self, df: pd.DataFrame) -> str:
        """Create LaTeX table with CPU metrics only"""
        latex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{adjustbox}

\begin{document}

\begin{table*}[htbp]
    \centering
    \caption{LLM Inference Performance Metrics}
    \label{tab:llm_inference_performance}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Model} & \textbf{Size (MB)} & \textbf{Latency (ms)} & \textbf{Peak RAM (MB)} & \textbf{Power (W)} & \textbf{Throughput} & \textbf{Edge Ready} \\
        & & & & & \textbf{(samples/s)} & \\
        \midrule
"""
        
        for _, row in df.iterrows():
            model = row['Model']
            size = f"{row['Model_Size_MB']:.0f}" if pd.notna(row['Model_Size_MB']) else "--"
            
            cpu_lat = row['CPU_Latency_ms']
            if pd.notna(cpu_lat):
                if cpu_lat >= 10000:
                    latency = f"{cpu_lat:.0f}"
                elif cpu_lat >= 1000:
                    latency = f"{cpu_lat:.1f}"
                else:
                    latency = f"{cpu_lat:.1f}"
            else:
                latency = "--"
            
            ram = f"{row['Peak_Memory_MB']:.0f}" if pd.notna(row['Peak_Memory_MB']) else "--"
            power = f"{row['Power_Consumption_W']:.1f}" if pd.notna(row['Power_Consumption_W']) else "--"
            
            throughput_val = row['Throughput_samples_per_sec']
            if pd.notna(throughput_val):
                if throughput_val < 0.001:
                    throughput = f"{throughput_val:.6f}"
                elif throughput_val < 0.01:
                    throughput = f"{throughput_val:.4f}"
                elif throughput_val < 1:
                    throughput = f"{throughput_val:.3f}"
                else:
                    throughput = f"{throughput_val:.2f}"
            else:
                throughput = "--"
            
            # Edge readiness
            edge_ready = (pd.notna(cpu_lat) and cpu_lat < 5000 and
                         pd.notna(row['Peak_Memory_MB']) and row['Peak_Memory_MB'] < 4000 and
                         pd.notna(row['Model_Size_MB']) and row['Model_Size_MB'] < 2000)
            
            edge_status = "\\textcolor{green}{Yes}" if edge_ready else "\\textcolor{red}{No}"
            
            latex += f"        {model} & {size} & {latency} & {ram} & {power} & {throughput} & {edge_status} \\\\\n"
        
        latex += r"""        \bottomrule
    \end{tabular}%
    }
\end{table*}

\end{document}"""
        
        return latex
    
    def analyze_performance(self) -> Dict[str, any]:
        """Analyze model performance using real experimental data"""
        print("🔄 Analyzing performance...")
        
        analysis = {
            'total_models': 0,
            'avg_latency': 0,
            'avg_model_size': 0,
            'avg_memory': 0,
            'avg_power': 0
        }
        
        # Use real data if available
        if self.inference_summary is not None:
            df = self.inference_summary
            analysis['total_models'] = len(df)
            
            # Calculate real statistics
            if len(df) > 0:
                latencies = df['avg_inference_time_ms'].dropna()
                sizes = df['model_size_mb'].dropna()
                memory = df['inference_peak_ram_mb'].dropna()
                power = df['inference_avg_power_w'].dropna()
                
                if len(latencies) > 0:
                    analysis['avg_latency'] = latencies.mean()
                if len(sizes) > 0:
                    analysis['avg_model_size'] = sizes.mean()
                if len(memory) > 0:
                    analysis['avg_memory'] = memory.mean()
                if len(power) > 0:
                    analysis['avg_power'] = power.mean()
        
        print(f"   ✅ Analysis complete: {analysis['total_models']} models analyzed")
        return analysis
    
    def create_summary_report(self) -> str:
        """Create a comprehensive summary report"""
        print("🔄 Creating summary report...")
        
        analysis = self.analyze_performance()
        
        report = f"""
LLM EFFICIENCY ANALYSIS SUMMARY REPORT
=====================================

Dataset Overview:
- Total models analyzed: {analysis['total_models']}
- Edge-ready models: {analysis['edge_ready_count']}/{analysis['total_models']}
- Edge readiness rate: {analysis['edge_ready_count']/analysis['total_models']*100:.1f}%

Performance Metrics:
- Average CPU latency: {analysis['avg_cpu_latency']:.1f} ms
- Average GPU latency: {analysis['avg_gpu_latency']:.1f} ms
- GPU speedup: {analysis['avg_cpu_latency']/analysis['avg_gpu_latency']:.1f}x faster
- Average model size: {analysis['avg_model_size']:.0f} MB

Model Categories:
- Small models (< 200MB): {analysis['performance_categories']['small']}
- Medium models (200-1000MB): {analysis['performance_categories']['medium']}
- Large models (≥ 1000MB): {analysis['performance_categories']['large']}

Edge Deployment Assessment:
Conservative thresholds used:
- Latency: < 1000ms (GPU preferred, CPU fallback)
- Memory: < 4000MB peak RAM
- Disk space: < 2000MB model size

Recommendations:
1. GPU acceleration significantly improves edge viability
2. Small-medium models best suited for edge deployment
3. Large models require optimization or cloud deployment
"""
        
        # Save report
        report_file = self.outputs_path / "efficiency_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"   ✅ Report saved: {report_file}")
        return report
    
    def run_complete_analysis(self, include_gpu: bool = True) -> Dict[str, str]:
        """Run complete analysis pipeline using real experimental data"""
        print("🚀 RUNNING COMPLETE LLM EFFICIENCY ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Load data
        data = self.load_experimental_data()
        results['data_loaded'] = f"Loaded {len(data)} data sources"
        
        # 2. Generate LaTeX table
        latex_file = self.generate_inference_latex_table(include_gpu=include_gpu)
        results['latex_table'] = latex_file
        
        # 3. Analyze performance
        analysis = self.analyze_performance()
        results['analysis'] = f"{analysis['total_models']} models analyzed"
        
        # 4. Create report
        report = self.create_summary_report()
        results['report'] = "Summary report generated"
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"LaTeX table: {Path(latex_file).name if latex_file != 'No data available' else 'No data available'}")
        print(f"Models analyzed: {analysis['total_models']}")
        print(f"All outputs in: {self.outputs_path}")
        
        return results


# Quick usage functions
def quick_analysis(base_path: str = "/home/amma/LLM-TIME"):
    """
    Quick analysis function for testing
    """
    analyzer = LLMEfficiencyAnalyzer(base_path)
    data = analyzer.load_experimental_data()
    
    if analyzer.inference_summary is not None:
        print(f"Loaded {len(analyzer.inference_summary)} inference records")
        latex_table = analyzer.generate_inference_latex_table()
        print("\nLaTeX table generated successfully!")
    else:
        print("No inference data found")
    
    return analyzer

def generate_latex_table(base_path: str = "/home/amma/LLM-TIME", include_gpu: bool = True) -> str:
    """Just generate LaTeX table quickly"""
    analyzer = LLMEfficiencyAnalyzer(base_path)
    analyzer.load_experimental_data()
    return analyzer.generate_inference_latex_table(include_gpu=include_gpu)