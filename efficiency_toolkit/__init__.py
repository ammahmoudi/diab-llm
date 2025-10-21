"""
LLM-TIME Efficiency Analysis Toolkit

A comprehensive toolkit for analyzing the efficiency of Time-LLM and Chronos models.
Includes training efficiency, inference performance, memory usage analysis, and more.

Main Components:
- Core: Essential efficiency calculation and profiling tools
- Analysis: Legacy analysis scripts and notebooks
- Scripts: Shell scripts for automation
- Results: Generated analysis results and reports

Usage:
    from efficiency_toolkit.core.comprehensive_efficiency_runner import ComprehensiveEfficiencyRunner
    
    runner = ComprehensiveEfficiencyRunner()
    runner.run_all_experiments()
    runner.analyze_all_experiments()
"""

__version__ = "1.0.0"
__author__ = "LLM-TIME Team"

# Make main classes easily accessible
try:
    from .core.comprehensive_efficiency_runner import ComprehensiveEfficiencyRunner
    from .core.efficiency_calculator import EfficiencyCalculator
    from .core.real_time_profiler import RealTimeProfiler
    
    __all__ = [
        'ComprehensiveEfficiencyRunner',
        'EfficiencyCalculator', 
        'RealTimeProfiler'
    ]
except ImportError:
    # Allow imports to fail gracefully during setup
    __all__ = []