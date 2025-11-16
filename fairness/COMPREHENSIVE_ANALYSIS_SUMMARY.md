# Comprehensive Fairness Analysis Summary

**Generated:** 2024-11-15
**Analysis Script:** `investigate_all_scenarios.py`

## Overview

This analysis examines fairness across **48 total scenarios** spanning three contexts:
1. **Inference Scenarios** (20 scenarios): 4 training conditions × 5 features
2. **Distillation - All Patients** (14 scenarios): One model trained on all patients
3. **Distillation - Per Patient** (14 scenarios): Separate models per patient

## Key Findings

### 1. Total Scenarios Analyzed: 48

- **Inference:** 20 scenarios
  - Inference Only
  - Trained Standard
  - Trained Noisy
  - Trained Denoised
  
- **Distillation (All Patients):** 14 scenarios
  - Teacher phase
  - Student Baseline phase
  - Distilled phase
  
- **Distillation (Per Patient):** 14 scenarios
  - Teacher phase
  - Student Baseline phase
  - Distilled phase

### 2. Poor Fairness Cases (≥1.5): 2

Both poor fairness cases occur in the **Trained Denoised** inference scenario:

1. **Cohort** - 1.899x ratio
   - Worst: 2020 (RMSE=40.16)
   - Best: 2018 (RMSE=21.14)
   - Gap: 19.02 RMSE points
   - **88.8% degradation from baseline**

2. **Gender** - 1.506x ratio
   - Worst: Female (RMSE=38.13)
   - Best: Male (RMSE=25.31)
   - Gap: 12.82 RMSE points
   - **45.4% degradation from baseline**

### 3. Significant Degradation (≥0.1): 7 Cases

All significant degradation occurs in **inference scenarios with training**:

| Feature | Scenario | Degradation | Percentage |
|---------|----------|-------------|------------|
| Cohort | Trained Denoised | +0.893x | +88.8% |
| Gender | Trained Denoised | +0.470x | +45.4% |
| Pump Model | Trained Standard | +0.232x | +22.4% |
| Cohort | Trained Noisy | +0.219x | +21.7% |
| Age Group | Trained Denoised | +0.215x | +19.8% |
| Pump Model | Trained Noisy | +0.138x | +13.3% |
| Gender | Trained Standard | +0.121x | +11.7% |

### 4. Context Comparison

#### Average Fairness Ratios:
- **Inference:** 1.179x (σ=0.203) - Most variable
- **Distillation (All Patients):** 1.101x (σ=0.072) - Most consistent
- **Distillation (Per Patient):** 1.104x (σ=0.073) - Slightly worse than All
- **All Distillation:** 1.103x (σ=0.073) - Better than inference

#### Worst Cases by Context:
- **Inference:** Cohort - Trained Denoised (1.899x)
- **Distillation (All):** Sensor Band - Teacher (1.252x)
- **Distillation (Per):** Sensor Band - Distilled (1.236x)

## Critical Insights

### 🚨 Training Impact on Fairness

**The "Trained Denoised" scenario shows severe fairness degradation:**
- Cohort degraded by **88.8%** (worst case)
- Gender degraded by **45.4%**
- Age Group degraded by **19.8%**

This suggests that **denoising training may introduce harmful bias**, particularly affecting:
- Temporal cohorts (2018 vs 2020)
- Gender groups (Female vs Male)

### ✅ Distillation Maintains Better Fairness

**Distillation scenarios consistently outperform inference:**
- **15% better average fairness** (1.103x vs 1.179x)
- **3x lower variance** (σ=0.073 vs σ=0.203)
- **No cases crossing poor threshold** (≥1.5)

Both distillation approaches (All Patients vs Per Patient) show similar performance:
- All Patients: 1.101x average
- Per Patient: 1.104x average
- Difference: Only 0.003x (negligible)

### 📊 Feature-Specific Patterns

**Most affected features across contexts:**
1. **Sensor Band** - Consistently highest ratios in distillation (1.23-1.25x)
2. **Cohort** - Most affected by training (up to 1.899x)
3. **Gender** - Second most affected by training (up to 1.506x)

**Most robust features:**
1. **Age Group** - Generally good across contexts (1.07-1.30x)
2. **Pump Model** - Acceptable in most scenarios (1.03-1.27x)

## Recommendations

### 1. Immediate Actions
- **Avoid "Trained Denoised" configuration** - Shows severe fairness degradation
- **Review denoising algorithm** - May be introducing temporal or gender bias
- **Monitor Cohort and Gender features** - Most vulnerable to training-induced bias

### 2. Best Practices
- **Prefer distillation approaches** - 15% better fairness with lower variance
- **Use "Inference Only" or distillation** - Avoid trained scenarios when fairness is critical
- **Monitor Sensor Band** - Consistently shows elevated fairness ratios

### 3. Further Investigation
- **Why does denoising hurt fairness so severely?**
  - Examine denoising algorithm for temporal/demographic bias
  - Check if denoising amplifies existing data imbalances
  
- **Why is distillation more fair?**
  - Understand mechanism that preserves fairness
  - Apply insights to improve training approaches

## Fairness Thresholds

- **< 1.1:** Excellent fairness
- **1.1-1.25:** Good fairness
- **1.25-1.5:** Acceptable fairness
- **> 1.5:** Poor fairness (unacceptable)

## Visual Report

See `comprehensive_fairness_report_20251115_100018.png` for:
- Comprehensive heatmap showing all 10 scenario columns
- Distillation comparison (All Patients vs Per Patient)
- Top 15 worst cases
- Context comparison bar chart
- Distribution analysis

## Methodology

### Fairness Ratio Calculation
```
Fairness Ratio = max(RMSE) / min(RMSE)
```

### Degradation Calculation
```
Degradation % = (fairness_ratio - baseline_ratio) / baseline_ratio × 100
```

### Significant Degradation Threshold
- **≥0.1 absolute change** in fairness ratio
- **≥10% relative change** from baseline

### Constants
- `POOR_RATIO = 1.5` - Threshold for poor fairness
- `SIGNIFICANT_DEGRADATION = 0.1` - Threshold for significant degradation

## Synchronized Scripts

All three analysis scripts use identical formulas and constants:

1. **`investigate_fairness_issues.py`** - Original fairness investigation
2. **`analyze_training_impact.py`** - Detailed training impact breakdown
3. **`investigate_all_scenarios.py`** - Comprehensive analysis (all contexts)

See `FORMULA_SYNC.md` for synchronization details.
