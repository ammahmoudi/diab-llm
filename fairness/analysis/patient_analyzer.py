"""
Patient Data Analyzer for Fairness Assessment
============================================

This module analyzes patient demographics and creates subgroups for fairness evaluation.
It loads the patient metadata and provides utilities to group patients by various attributes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path


class PatientAnalyzer:
    """Analyzes patient demographics and creates fairness evaluation groups."""
    
    def __init__(self, data_path: str = "/workspace/LLM-TIME/data/ohiot1dm/data.csv"):
        """Initialize the patient analyzer.
        
        Args:
            data_path: Path to the patient metadata CSV file
        """
        self.data_path = data_path
        self.patient_data = None
        self.load_patient_data()
    
    def load_patient_data(self):
        """Load patient metadata from CSV file."""
        try:
            self.patient_data = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.patient_data)} patient records")
            print("Available columns:", list(self.patient_data.columns))
            print("Patient data preview:")
            print(self.patient_data.head())
        except Exception as e:
            raise FileNotFoundError(f"Could not load patient data from {self.data_path}: {e}")
    
    def get_demographic_summary(self) -> Dict:
        """Get summary statistics of patient demographics.
        
        Returns:
            Dictionary containing demographic summaries
        """
        if self.patient_data is None:
            raise ValueError("Patient data not loaded")
        
        summary = {
            'total_patients': len(self.patient_data),
            'gender_distribution': self.patient_data['Gender'].value_counts().to_dict(),
            'age_distribution': self.patient_data['Age'].value_counts().to_dict(),
            'pump_model_distribution': self.patient_data['Pump Model'].value_counts().to_dict(),
            'sensor_band_distribution': self.patient_data['Sensor Band'].value_counts().to_dict(),
            'cohort_distribution': self.patient_data['Cohort'].value_counts().to_dict()
        }
        
        # Calculate percentages
        for key in ['gender_distribution', 'age_distribution', 'pump_model_distribution', 
                   'sensor_band_distribution', 'cohort_distribution']:
            total = sum(summary[key].values())
            summary[f"{key}_pct"] = {k: round(v/total*100, 2) for k, v in summary[key].items()}
        
        return summary
    
    def create_fairness_groups(self, attribute: str) -> Dict[str, List[int]]:
        """Create patient groups based on a specific attribute for fairness analysis.
        
        Args:
            attribute: The attribute to group by ('Gender', 'Age', 'Pump Model', etc.)
            
        Returns:
            Dictionary mapping attribute values to lists of patient IDs
        """
        if self.patient_data is None:
            raise ValueError("Patient data not loaded")
        
        if attribute not in self.patient_data.columns:
            raise ValueError(f"Attribute '{attribute}' not found in patient data")
        
        groups = {}
        for value in self.patient_data[attribute].unique():
            patient_ids = self.patient_data[self.patient_data[attribute] == value]['ID'].tolist()
            groups[str(value)] = patient_ids
        
        return groups
    
    def get_patient_attribute(self, patient_id: int, attribute: str) -> str:
        """Get a specific attribute value for a patient.
        
        Args:
            patient_id: The patient ID
            attribute: The attribute to retrieve
            
        Returns:
            The attribute value for the patient
        """
        if self.patient_data is None:
            raise ValueError("Patient data not loaded")
        
        patient_row = self.patient_data[self.patient_data['ID'] == patient_id]
        if len(patient_row) == 0:
            raise ValueError(f"Patient ID {patient_id} not found")
        
        return patient_row[attribute].iloc[0]
    
    def get_intersectional_groups(self, attributes: List[str]) -> Dict[str, List[int]]:
        """Create intersectional groups based on multiple attributes.
        
        Args:
            attributes: List of attributes to combine (e.g., ['Gender', 'Age'])
            
        Returns:
            Dictionary mapping combined attribute values to patient ID lists
        """
        if self.patient_data is None:
            raise ValueError("Patient data not loaded")
        
        for attr in attributes:
            if attr not in self.patient_data.columns:
                raise ValueError(f"Attribute '{attr}' not found in patient data")
        
        groups = {}
        # Create combinations of attribute values
        grouped = self.patient_data.groupby(attributes)
        
        for group_name, group_data in grouped:
            if isinstance(group_name, tuple):
                group_key = "_".join(str(x) for x in group_name)
            else:
                group_key = str(group_name)
            
            patient_ids = group_data['ID'].tolist()
            groups[group_key] = patient_ids
        
        return groups
    
    def print_fairness_analysis_summary(self, attribute: str = 'Gender'):
        """Print a summary for fairness analysis preparation.
        
        Args:
            attribute: The primary attribute to analyze for fairness
        """
        print("\n" + "="*60)
        print("FAIRNESS ANALYSIS SUMMARY")
        print("="*60)
        
        summary = self.get_demographic_summary()
        print(f"Total Patients: {summary['total_patients']}")
        
        print(f"\n{attribute} Distribution:")
        groups = self.create_fairness_groups(attribute)
        for group, patients in groups.items():
            percentage = len(patients) / summary['total_patients'] * 100
            print(f"  {group}: {len(patients)} patients ({percentage:.1f}%)")
            print(f"    Patient IDs: {patients}")
        
        print(f"\nRecommended Analysis:")
        print(f"1. Train models on each {attribute.lower()} group separately")
        print(f"2. Evaluate cross-group performance")
        print(f"3. Measure fairness metrics (equalized odds, demographic parity)")
        print(f"4. Compare teacher vs student model fairness")
        print(f"5. Apply fairness-aware loss functions")
        
        # Check for intersectional analysis opportunities
        if len(groups) >= 2:
            print(f"\nIntersectional Analysis Options:")
            for other_attr in ['Age', 'Pump Model', 'Sensor Band', 'Cohort']:
                if other_attr != attribute:
                    intersectional = self.get_intersectional_groups([attribute, other_attr])
                    print(f"  {attribute} × {other_attr}: {len(intersectional)} combinations")


if __name__ == "__main__":
    # Example usage
    analyzer = PatientAnalyzer()
    analyzer.print_fairness_analysis_summary('Gender')
    analyzer.print_fairness_analysis_summary('Age')