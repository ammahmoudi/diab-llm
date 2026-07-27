# Investigating Sex Bias in ECG Classification for Atrial Fibrillation, Sinus Rhythm and Myocardial Infarction

- Priority: 35
- Category: ECG demographic fairness
- Authors: Maria Galanty; Bjorn van der Ster; Alexander P. Vlaar; Clara I. Sanchez
- Year: 2025
- Venue/status: Machine Learning for Biomedical Imaging (peer-reviewed)
- DOI: 10.59275/j.melba.2025-9fe7
- arXiv: none
- Source: [paper page](https://doi.org/10.59275/j.melba.2025-9fe7)
- Local PDF: ../pdfs/35_galanty2025_ecg_sex_bias.pdf
- BibTeX key: `galanty2025sex`

## Why It Matters

Direct ECG sex-bias evidence showing that balanced training composition may not remove all performance differences.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether training sex ratios and model architecture affect sex bias in ECG classification.

## Method and Evidence

Compares CNN, xResNet101, and attention-based residual models for sinus rhythm, atrial fibrillation, and myocardial infarction under varying sex ratios.

## Findings Relevant Here

Reports the attention-based residual model as most equitable for sinus rhythm and atrial fibrillation, while myocardial-infarction classification retains pronounced sex disparities even with balanced training data.

## Limitations and Claim Boundary

It uses 12-lead clinical ECGs and different labels, groups, and models from the MIT-BIH protocol.

## Relevance to This Article

Keep as a targeted ECG fairness reference; use only for the qualified claim that balancing composition may not eliminate sex gaps.
