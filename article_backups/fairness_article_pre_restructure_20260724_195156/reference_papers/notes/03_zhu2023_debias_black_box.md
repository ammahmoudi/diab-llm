# Debias the Black-Box: A Fair Ranking Framework via Knowledge Distillation

- Priority: 3
- Category: KD fairness
- Authors: Zhitao Zhu; Shijing Si; Jianzong Wang; Yaodong Yang; Jing Xiao
- Year: 2023
- Venue/status: ECML PKDD 2022 Workshops (peer-reviewed)
- DOI: 10.1007/978-3-031-20891-1_28
- arXiv: 2208.11628
- Source: [paper page](https://doi.org/10.1007/978-3-031-20891-1_28)
- Local PDF: ../pdfs/03_zhu2023_debias_black_box.pdf
- BibTeX key: `zhu2023debias`

## Why It Matters

A ranking-specific example of fairness-aware KD, useful only as breadth for the wider method landscape.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether a black-box teacher can be distilled into a fairer ranking model.

## Method and Evidence

Uses knowledge distillation to transfer ranking behavior while incorporating a fairness objective for ranked outputs.

## Findings Relevant Here

Illustrates that fairness constraints can be introduced during distillation even when the teacher is treated as a black box.

## Limitations and Claim Boundary

Ranking exposure fairness is not comparable to clinical event detection or regression forecasting.

## Relevance to This Article

Do not cite in the main paper unless a short survey of non-clinical fair-KD methods is needed.
