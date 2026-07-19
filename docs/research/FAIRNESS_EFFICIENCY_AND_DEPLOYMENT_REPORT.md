# Fairness Distillation: Efficiency and Deployment Report

## Purpose and Evidence Scope

This report summarizes the efficiency artifacts available for the OhioT1DM
fairness-distillation study. It distinguishes measured model-size and CPU
microbenchmark evidence from claims that would require a deployment trial.

The primary efficiency evidence comes from saved per-patient inference artifacts
for the same six-step input and nine-step output used in the fairness study. It
does **not** establish clinical-device readiness, battery life, end-to-end system
latency, or prospective clinical safety.

## Model Compression Result

| Model | Parameters | Reported weight size | Reduction versus BERT teacher |
| --- | ---: | ---: | ---: |
| BERT teacher | 140,605,953 | 536.37 MB | Reference |
| BERT-tiny student | 35,017,473 | 133.58 MB | 75.10% |

The compact BERT-tiny backbone is used by Standard KD, Exposure-Balanced Teacher
Distillation (EBTD), and EBTD with Group-Conditional Affine Output Adaptation
(GCOA). EBTD changes the teacher-training sampler only. For the two reported BG
groups, GCOA adds four scalar values in total: one scale and one bias per group.
It therefore does not materially alter the student parameter count or stored
weight size.

## Measured Inference Microbenchmarks

All values below are means of per-report timing averages. Each report measured
five already-loaded float32 CPU inference calls with batch size one, a six-step
input, and a nine-step output. The compact-method results contain 60 reports per
condition: 12 patients across five completed initializations, for 300 timed calls
per condition. Teacher and standalone-student rows contain 12 reports, or 60
timed calls each.

| Model or condition | Reports | Timed calls | Mean latency (ms) | SD (ms) | Median (ms) | Range (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BERT teacher | 12 | 60 | 753.4 | 25.1 | 740.6 | 724.6–801.0 |
| Standalone BERT-tiny student | 12 | 60 | 91.8 | 4.3 | 90.0 | 88.9–103.6 |
| Standard KD student | 60 | 300 | 111.0 | 10.0 | 107.8 | 99.7–153.2 |
| EBTD student | 60 | 300 | 109.5 | 6.4 | 108.0 | 99.6–121.9 |
| EBTD+GCOA student | 60 | 300 | 124.5 | 27.2 | 117.8 | 96.3–241.2 |

### Interpretation

- The static compression result is strong and unambiguous: BERT-tiny has 75.1%
  fewer parameters and reported weights than the BERT teacher.
- All three primary fairness methods retain the same compact backbone. Their
  timing differences should not be interpreted as a controlled comparison of
  fairness-method overhead because the benchmarks were recorded in separate
  runs under ordinary host load.
- The saved reports record `device=cpu`. Their GPU fields must not be used for
  hardware sizing because the generic monitoring path can include unrelated
  process or device state.
- The results are useful as implementation evidence that the compact model has
  a substantially smaller footprint. They are not a claim that any method meets
  a particular real-time or edge-device service-level objective.

## Artifact Provenance

The primary artifacts are stored under:

- `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/phase_1_teacher/per_patient_inference/`
- `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/phase_2_student/per_patient_inference/`
- `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/phase_3_distillation/`

The phase-three folders corresponding to the five-initialization primary
comparison are the Standard KD, fair-teacher, and fair-teacher-plus-output
adaptation variants. Each contains timestamped
`efficiency_report_time_llm_*.json` files. The reports record the model
configuration, model size, parameter count, device, and five-call timing series.

The current measurement implementation is in:

- `efficiency_toolkit/core/efficiency_calculator.py`
- `efficiency_toolkit/core/real_time_profiler.py`

An older broad model-suite analysis is available at:

- `efficiency_toolkit/results/efficiency_analysis_results/analysis_20251021_141856/reports/comprehensive_efficiency_report_20251021_141856.md`

That 2025 suite is useful for historical comparisons among Time-LLM, Chronos,
and distillation configurations, but it uses a different hardware and execution
protocol. It should not be merged numerically with the 2026 fairness-study
microbenchmarks.

## Deployment Assessment

### What the Current Evidence Supports

- A BERT-tiny student is substantially smaller than the BERT teacher.
- The primary fairness methods can be evaluated with the same compact student
  architecture and fixed glucose forecasting window.
- Saved CPU microbenchmarks provide a baseline for a controlled deployment
  benchmark.

### What Remains Unproven

- End-to-end latency, including data ingestion, feature preparation, model
  loading, network transfer, and prediction delivery.
- Memory, latency, and energy performance on a specified target device.
- Cold-start performance, long-running stability, and concurrent-request
  throughput.
- Clinical safety, external validity, usability, privacy, security, and
  regulatory acceptability.

### Group-Conditional Deployment Requirement

GCOA requires the group attribute selected by the study at inference time. A
deployment must therefore establish whether this attribute is reliable,
permissible to use, protected appropriately, and available at the point of
prediction. This is a governance requirement, not a performance optimization.
When that requirement cannot be met, EBTD remains the group-agnostic compact
student option among the primary methods.

## Recommended Controlled Benchmark

Before making an edge or real-time deployment claim, rerun the primary teacher,
Standard KD, EBTD, and EBTD+GCOA checkpoints on one named target device with:

1. Explicit CPU/GPU selection, software versions, precision, batch size, and
   input/output horizon.
2. A fixed warm-up phase followed by at least 100 timed calls per checkpoint.
3. Latency mean, median, p95, p99, throughput, model-load time, process RSS,
   framework-allocated memory, and energy or power where hardware support
   exists.
4. Separate measurements for preprocessing, model inference, and full
   end-to-end request handling.
5. The same checkpoint, device, and workload for all compared methods.

This design will convert the current implementation evidence into a defensible
deployment-efficiency result for the manuscript.
