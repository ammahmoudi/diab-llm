# Future-Aware Blood Glucose Forecasting Using Knowledge Distillation with Transformer-Based Sequence-to-Sequence Models

- Priority: 32
- Category: Blood-glucose knowledge distillation
- Authors: Xiaoyu Sun; Hongru Li; Xia Yu
- Year: 2026
- Venue/status: Scientific Reports (peer-reviewed)
- DOI: 10.1038/s41598-026-41787-7
- arXiv: none
- Source: [paper page](https://doi.org/10.1038/s41598-026-41787-7)
- Local PDF: ../pdfs/32_sun2026_future_aware_glucose_kd.pdf
- BibTeX key: `sun2026futureaware`

## Why It Matters

A recent Transformer KD precedent for glucose forecasting that reinforces the gap in demographic fairness reporting.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether future disturbance information available during training can be distilled to a history-only Transformer student for multi-step glucose prediction.

## Method and Evidence

Uses a future-aware teacher and history-only sequence-to-sequence student, evaluating clinical CGM data including OhioT1DM and AZT1D.

## Findings Relevant Here

Reports RMSE, MAE, and Clarke error-grid outcomes; it lists the OhioT1DM sex composition but does not report sex-stratified fairness metrics.

## Limitations and Claim Boundary

Future-information privilege, model architecture, datasets, and endpoint definitions differ from the present protocol.

## Relevance to This Article

Keep as a recent direct glucose-KD precedent; do not treat its aggregate gains as fairness evidence.
