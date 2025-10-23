"""
Enhanced Data Loader for Time-LLM JSON Files

This module provides an enhanced data loader that properly parses
nested Time-LLM JSON structures while maintaining compatibility
with existing Chronos flat JSON format.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class EnhancedEfficiencyDataLoader:
    """Enhanced data loader that properly parses nested Time-LLM JSON files"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def scan_all_files(self) -> Dict[str, List[Path]]:
        """Scan for all JSON files in the experiments directory"""
        all_files = {
            'efficiency_reports': [],
            'comprehensive_reports': [],
            'real_performance_reports': []
        }
        
        # Scan all experiment directories
        experiments_dir = self.base_path / 'efficiency_experiments' / 'experiments'
        if experiments_dir.exists():
            for json_file in experiments_dir.rglob('*.json'):
                filename = json_file.name.lower()
                if 'efficiency_report' in filename:
                    all_files['efficiency_reports'].append(json_file)
                elif 'comprehensive_performance_report' in filename:
                    all_files['comprehensive_reports'].append(json_file)
                elif 'real_performance_report' in filename:
                    all_files['real_performance_reports'].append(json_file)
        
        return all_files
    
    def _extract_model_name_from_path(self, file_path: Path) -> str:
        """Extract model name from file path"""
        path_str = str(file_path)
        
        # Check for Time-LLM models in path
        if 'model_BERT' in path_str:
            return 'BERT'
        elif 'model_GPT2' in path_str:
            return 'GPT2'
        elif 'model_LLAMA' in path_str:
            return 'LLAMA'
        elif 'chronos' in path_str.lower():
            if 't5-tiny' in path_str:
                return 'chronos-t5-tiny'
            elif 't5-base' in path_str:
                return 'chronos-t5-base'
            else:
                return 'chronos-unknown'
        elif 'tinybert' in path_str.lower():
            return 'tinybert'
        
        # Fallback: extract from filename or directory
        parts = file_path.parts
        for part in reversed(parts):
            if any(model in part.upper() for model in ['BERT', 'GPT2', 'LLAMA']):
                if 'BERT' in part.upper():
                    return 'BERT'
                elif 'GPT2' in part.upper():
                    return 'GPT2'
                elif 'LLAMA' in part.upper():
                    return 'LLAMA'
        
        return 'unknown'
    
    def _extract_experiment_type(self, file_path: Path) -> str:
        """Extract experiment type from file path"""
        path_str = str(file_path)
        
        if 'time_llm_inference' in path_str:
            return 'time_llm_inference_ohiot1dm'
        elif 'time_llm_training' in path_str:
            return 'time_llm_training_ohiot1dm'
        elif 'chronos_inference' in path_str:
            return 'chronos_inference_ohiot1dm'
        elif 'chronos_training' in path_str:
            return 'chronos_training_ohiot1dm'
        elif 'distillation_inference' in path_str:
            return 'distillation_inference_ohiot1dm'
        elif 'distillation' in path_str:
            return 'distillation_experiments'
        
        return 'unknown_experiment'
    
    def _extract_report_type(self, file_path: Path) -> str:
        """Extract report type from filename"""
        filename = file_path.name.lower()
        
        if 'efficiency_report' in filename:
            return 'efficiency_reports'
        elif 'comprehensive_performance_report' in filename:
            return 'comprehensive_reports'
        elif 'real_performance_report' in filename:
            return 'real_performance_reports'
        
        return 'unknown_report'
    
    def parse_time_llm_comprehensive(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Time-LLM comprehensive JSON files with nested structure"""
        results = []
        model_name = self._extract_model_name_from_path(file_path)
        
        # Parse performance_summary section
        if 'performance_summary' in data:
            perf_summary = data['performance_summary']
            
            # Parse training_inference data (key section with inference metrics)
            if 'training_inference' in perf_summary:
                inf_data = perf_summary['training_inference']
                record = {
                    'experiment_type': self._extract_experiment_type(file_path),
                    'model_name': model_name,
                    'mode': 'inference',
                    'report_type': self._extract_report_type(file_path),
                    'file_path': str(file_path),
                    
                    # Model characteristics
                    'total_parameters': inf_data.get('parameters_count'),
                    'model_size_mb': inf_data.get('model_size_on_disk_mb'),
                    
                    # Inference performance
                    'avg_inference_time_ms': inf_data.get('average_latency_ms'),
                    'median_latency_ms': inf_data.get('median_latency_ms'),
                    'p95_latency_ms': inf_data.get('p95_latency_ms'),
                    
                    # Memory usage
                    'inference_peak_ram_mb': inf_data.get('process_peak_ram_mb'),
                    'process_average_ram_mb': inf_data.get('process_average_ram_mb'),
                    'system_peak_ram_mb': inf_data.get('system_peak_ram_mb'),
                    'inference_peak_gpu_mb': inf_data.get('peak_gpu_allocated_mb'),
                    'average_gpu_allocated_mb': inf_data.get('average_gpu_allocated_mb'),
                    'peak_gpu_reserved_mb': inf_data.get('peak_gpu_reserved_mb'),
                    
                    # Performance metrics
                    'peak_gpu_utilization_percent': inf_data.get('peak_gpu_utilization_percent'),
                    'average_gpu_utilization_percent': inf_data.get('average_gpu_utilization_percent'),
                    'peak_temperature_celsius': inf_data.get('peak_temperature_celsius'),
                    'inference_avg_power_w': inf_data.get('average_power_usage_watts'),
                    'peak_power_usage_watts': inf_data.get('peak_power_usage_watts'),
                    
                    # Calculate throughput
                    'throughput_predictions_per_sec': (1000.0 / inf_data.get('average_latency_ms', float('inf'))) 
                                                    if inf_data.get('average_latency_ms') else None,
                    
                    # Edge deployment
                    'edge_feasibility': inf_data.get('edge_feasibility', 'unknown'),
                    'total_inferences': inf_data.get('total_inferences')
                }
                results.append(record)
        
        return results
    
    def parse_chronos_flat(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Chronos JSON structure (which is actually nested, not flat)"""
        model_name = self._extract_model_name_from_path(file_path)
        
        # Extract nested data from Chronos structure
        timing_data = {}
        memory_data = {}
        model_data = {}
        
        # Extract timing measurements
        if 'real_performance_measurements' in data:
            real_perf = data['real_performance_measurements']
            if 'timing_measurements' in real_perf:
                timing_data = real_perf['timing_measurements']
            if 'memory_measurements' in real_perf:
                memory_data = real_perf['memory_measurements']
        
        # Extract model characteristics
        if 'model_characteristics' in data:
            model_data = data['model_characteristics']
        
        # Extract theoretical performance for additional metrics
        theoretical_data = data.get('theoretical_performance', {})
        
        record = {
            'experiment_type': self._extract_experiment_type(file_path),
            'model_name': model_name,
            'mode': data.get('mode', 'inference'),
            'report_type': self._extract_report_type(file_path),
            'file_path': str(file_path),
            
            # Model characteristics
            'total_parameters': model_data.get('total_parameters'),
            'model_size_mb': model_data.get('model_size_mb') or theoretical_data.get('model_size_on_disk_mb'),
            'model_dtype': model_data.get('model_dtype'),
            'model_architecture': model_data.get('model_architecture'),
            
            # Timing data from real measurements
            'avg_inference_time_ms': timing_data.get('average_inference_time_ms'),
            'min_inference_time_ms': timing_data.get('min_inference_time_ms'),
            'max_inference_time_ms': timing_data.get('max_inference_time_ms'),
            'measurement_runs': timing_data.get('measurement_runs'),
            
            # Memory data from real measurements
            'inference_peak_ram_mb': memory_data.get('current_ram_usage_mb'),
            'inference_peak_gpu_mb': memory_data.get('gpu_max_memory_allocated_mb'),
            'current_vram_usage_mb': memory_data.get('current_vram_usage_mb'),
            'gpu_memory_reserved_mb': memory_data.get('gpu_memory_reserved_mb'),
            
            # Theoretical estimates
            'estimated_cpu_latency_ms': theoretical_data.get('estimated_cpu_latency_ms'),
            'estimated_gpu_latency_ms': theoretical_data.get('estimated_gpu_latency_ms'),
            'estimated_memory_usage_mb': theoretical_data.get('estimated_memory_usage_mb'),
            'estimated_throughput_samples_per_sec': theoretical_data.get('estimated_throughput_samples_per_sec'),
            
            # Calculate throughput from timing if available
            'throughput_predictions_per_sec': (1000.0 / timing_data.get('average_inference_time_ms', float('inf'))) 
                                            if timing_data.get('average_inference_time_ms') else None,
            
            # Edge deployment
            'edge_feasibility': data.get('edge_deployment_analysis', {}).get('overall_feasibility', 'unknown'),
            'feasible_edge_devices': data.get('edge_deployment_analysis', {}).get('feasible_edge_devices', [])
        }
        
        return [record]
    
    def parse_chronos_comprehensive(self, data: Dict, file_path: Path) -> List[Dict]:
        """Parse Chronos comprehensive JSON files"""
        model_name = self._extract_model_name_from_path(file_path)
        
        results = []
        
        # Parse performance_summary section for Chronos
        if 'performance_summary' in data and 'inference' in data['performance_summary']:
            inf_data = data['performance_summary']['inference']
            
            record = {
                'experiment_type': self._extract_experiment_type(file_path),
                'model_name': model_name,
                'mode': 'inference',
                'report_type': self._extract_report_type(file_path),
                'file_path': str(file_path),
                
                # Model characteristics  
                'total_parameters': inf_data.get('parameters_count'),
                'model_size_mb': inf_data.get('model_size_on_disk_mb'),
                
                # Inference performance
                'avg_inference_time_ms': inf_data.get('average_latency_ms'),
                'median_latency_ms': inf_data.get('median_latency_ms'),
                'p95_latency_ms': inf_data.get('p95_latency_ms'),
                'total_inferences': inf_data.get('total_inferences'),
                
                # Memory usage
                'inference_peak_ram_mb': inf_data.get('process_peak_ram_mb'),
                'process_average_ram_mb': inf_data.get('process_average_ram_mb'),
                'system_peak_ram_mb': inf_data.get('system_peak_ram_mb'),
                'inference_peak_gpu_mb': inf_data.get('peak_gpu_allocated_mb'),
                'average_gpu_allocated_mb': inf_data.get('average_gpu_allocated_mb'),
                'peak_gpu_reserved_mb': inf_data.get('peak_gpu_reserved_mb'),
                
                # Performance metrics
                'peak_gpu_utilization_percent': inf_data.get('peak_gpu_utilization_percent'),
                'average_gpu_utilization_percent': inf_data.get('average_gpu_utilization_percent'),
                'peak_temperature_celsius': inf_data.get('peak_temperature_celsius'),
                'inference_avg_power_w': inf_data.get('average_power_usage_watts'),
                'peak_power_usage_watts': inf_data.get('peak_power_usage_watts'),
                
                # Calculate throughput
                'throughput_predictions_per_sec': (1000.0 / inf_data.get('average_latency_ms', float('inf'))) 
                                                if inf_data.get('average_latency_ms') else None,
                
                # Edge deployment
                'edge_feasibility': inf_data.get('edge_feasibility', 'unknown'),
                'feasible_edge_devices': inf_data.get('feasible_edge_devices', [])
            }
            results.append(record)
        
        return results

    def parse_single_file(self, file_path: Path) -> List[Dict]:
        """Parse a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if this is a comprehensive report with performance_summary
            if isinstance(data, dict) and 'performance_summary' in data:
                # Check if it's Time-LLM format (has training_inference) or Chronos format (has inference)
                if 'training_inference' in data['performance_summary']:
                    return self.parse_time_llm_comprehensive(data, file_path)
                elif 'inference' in data['performance_summary']:
                    return self.parse_chronos_comprehensive(data, file_path)
                else:
                    # Fallback to regular Time-LLM parsing
                    return self.parse_time_llm_comprehensive(data, file_path)
            else:
                # Regular Chronos efficiency report format
                return self.parse_chronos_flat(data, file_path)
                
        except Exception as e:
            print(f"⚠️ Error parsing {file_path}: {e}")
            return []
    
    def parse_all_data(self) -> pd.DataFrame:
        """Parse all JSON files and return a pandas DataFrame"""
        print("🔍 Scanning for experiment files...")
        all_files = self.scan_all_files()
        
        total_files = sum(len(files) for files in all_files.values())
        print(f"📊 Found {total_files} JSON files")
        
        all_records = []
        
        for report_type, files in all_files.items():
            print(f"📊 Processing {len(files)} {report_type}...")
            for file_path in files:
                records = self.parse_single_file(file_path)
                all_records.extend(records)
        
        df = pd.DataFrame(all_records)
        print(f"✅ Loaded {len(df)} total records")
        
        return df