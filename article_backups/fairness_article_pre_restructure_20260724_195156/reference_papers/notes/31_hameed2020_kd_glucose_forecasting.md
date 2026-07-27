# Investigating Potentials and Pitfalls of Knowledge Distillation Across Datasets for Blood Glucose Forecasting

- Priority: 31
- Category: Blood-glucose knowledge distillation
- Authors: Hadia Hameed; Samantha Kleinberg
- Year: 2020
- Venue/status: KDH@ECAI 2020 (peer-reviewed workshop)
- DOI: none verified
- arXiv: none
- Source: [paper page](https://ceur-ws.org/Vol-2675/paper14.pdf)
- Local PDF: ../pdfs/31_hameed2020_kd_glucose_forecasting.pdf
- BibTeX key: `hameed2020investigating`

## Why It Matters

The closest prior OhioT1DM KD study, establishing an aggregate-forecasting baseline rather than a fairness precedent.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether large public OAPS data can support OhioT1DM blood-glucose forecasting through pretraining, retraining, or mimic learning.

## Method and Evidence

Compares OhioT1DM-only RNNs, OAPS-pretrained RNNs, retrained models, and an OAPS-teacher/RNN-to-ANN mimic-learning path; uses CGM-only forecasts at 30- and 60-minute horizons.

## Findings Relevant Here

Reports RMSE and MAE by patient; no demographic fairness terms or group-stratified fairness metrics are analyzed.

## Limitations and Claim Boundary

Model family, horizons, data source, and endpoints differ from Time-LLM, and absence of a fairness analysis is a gap rather than evidence of fairness.

## Relevance to This Article

Keep as the primary prior glucose-KD reference and use only for the qualified aggregate-metric comparison.
