# Literature Search Strategy

## Review question

Which peer-reviewed and high-value preprint references best support the Introduction and Related Work of a paper on event-level demographic fairness under knowledge distillation for clinical time-series forecasting, with blood-glucose forecasting as the primary task and ECG classification as bounded secondary evidence?

## Evidence clusters

1. Fairness under knowledge distillation
   - KD as a source of bias or fairness change
   - Fairness-aware teacher/student objectives
   - Teacher-side imbalance and fairness transfer
   - Post-distillation or output-side correction
2. Clinical algorithmic fairness
   - Healthcare fairness definitions and reporting guidance
   - Subgroup performance and equal opportunity in clinical prediction
   - Bias in medical AI and clinically meaningful thresholds
3. Blood-glucose and diabetes AI fairness
   - Demographic disparities in diabetes technology or glucose prediction
   - Continuous glucose monitoring subgroup validity
   - Thresholded hypoglycemia prediction and event-level evaluation
4. ECG and cardiac AI fairness
   - Demographic bias in ECG models
   - Subgroup performance, race/sex/age effects, and hidden stratification
   - Fairness-aware ECG or arrhythmia classification
5. Fair training and data exposure
   - Group DRO, JTT, balancing, reweighting, and spurious correlations
6. LLM and agent fairness
   - LLM fairness benchmarks and surveys relevant to the Time-LLM provenance
   - System-level or agent-level fairness only where it clarifies the broader pipeline

## Query families

- `knowledge distillation fairness bias teacher student demographic`
- `model compression fairness knowledge transfer bias`
- `clinical artificial intelligence fairness equal opportunity subgroup performance`
- `healthcare machine learning fairness reporting framework`
- `continuous glucose monitoring machine learning fairness sex gender race hypoglycemia prediction`
- `blood glucose forecasting demographic bias fairness`
- `electrocardiogram deep learning fairness demographic bias sex race age`
- `arrhythmia classification fairness subgroup equal opportunity`
- `group DRO Just Train Twice data balancing worst-group accuracy`
- `large language model fairness benchmark survey agent fairness`

## Selection criteria

Include papers that satisfy at least one of the following:

- Directly evaluate fairness or bias under KD/model compression.
- Establish a fairness definition, reporting framework, or subgroup-evaluation principle used in clinical AI.
- Empirically document demographic performance differences in blood-glucose, diabetes, ECG, cardiac, or adjacent medical AI.
- Provide a strong non-KD comparator explaining data exposure, optimization, or worst-group performance.
- Provide a canonical LLM/agent fairness benchmark or survey needed to contextualize model provenance.

Prioritize peer-reviewed papers, original studies, systematic reviews, and authoritative guidance. Retain preprints only when no peer-reviewed version is verified or when the work is uniquely recent and directly relevant. Exclude papers that use “fairness” only to mean class balance unless they are explicitly labeled as structurally adjacent rather than demographic fairness.

## Verification requirements

For every retained paper:

- Verify exact title, authors, year, venue, DOI, and publication status.
- Prefer DOI/publisher metadata over inferred metadata.
- Obtain PDFs only from lawful open sources.
- Read at least the abstract and relevant full-text sections before mapping a claim.
- Record which manuscript claim or paragraph the paper supports.
- Do not add a citation to the active manuscript unless the cited claim has been checked against the paper.
