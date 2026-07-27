# Fairness without Demographics through Knowledge Distillation

- Priority: 16
- Category: KD fairness
- Authors: Junyi Chai; Taeuk Jang; Xiaoqian Wang
- Year: 2022
- Venue/status: Advances in Neural Information Processing Systems 35 (peer-reviewed)
- DOI: 10.52202/068431-1392
- Source: [paper page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/79dc391a2c1067e9ac2b764e31a60377-Abstract-Conference.html)
- Local PDF: ../pdfs/16_chai2022_fairness_without_demographics.pdf
- BibTeX key: `chai2022fairness`

## Why It Matters

A direct fair-KD method that avoids target-domain demographic labels, providing a useful contrast to group-aware GCOA.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether fairness can be transferred through KD without demographic labels in the target domain.

## Method and Evidence

Formulates fairness as transferable teacher knowledge and evaluates a demographic-label-free KD approach.

## Findings Relevant Here

Shows that fair-KD designs need not always consume target demographic labels.

## Limitations and Claim Boundary

The setting and fairness outcome differ from group-conditional clinical output adaptation.

## Relevance to This Article

Keep as a concise contrast: the present GCOA deliberately requires a permissible group attribute at inference time.
