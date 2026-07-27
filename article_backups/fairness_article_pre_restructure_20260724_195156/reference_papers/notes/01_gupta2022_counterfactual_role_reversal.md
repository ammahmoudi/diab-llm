# Mitigating Gender Bias in Distilled Language Models via Counterfactual Role Reversal

- Priority: 1
- Category: KD fairness
- Authors: Umang Gupta; Jwala Dhamala; Varun Kumar; Apurv Verma; Yada Pruksachatkun; Satyapriya Krishna; Rahul Gupta; Kai-Wei Chang; Greg Ver Steeg; Aram Galstyan
- Year: 2022
- Venue/status: Findings of ACL 2022 (peer-reviewed)
- DOI: 10.18653/v1/2022.findings-acl.55
- arXiv: 2203.12574
- Source: [paper page](https://aclanthology.org/2022.findings-acl.55/)
- Local PDF: ../pdfs/01_gupta2022_counterfactual_role_reversal.pdf
- BibTeX key: `gupta2022mitigating`

## Why It Matters

Direct evidence that fairness can be altered during language-model KD, although its generation task and gender-bias setting differ from clinical time-series prediction.

## Evidence Scope

This note records claims at the scope supported by the locally archived source or its verified metadata. It is a literature-tracking record, not a source of quantitative claims beyond the cited paper.

## Research Question

Whether counterfactual role reversal can reduce gender bias transferred from a teacher to a distilled language model.

## Method and Evidence

Uses a teacher--student language-model distillation setting with counterfactual gender-role-swapped examples to modify the teacher signal.

## Findings Relevant Here

Shows that fairness-aware teacher modification can affect gender-bias measures after distillation; it supports auditing the full teacher-to-student path rather than assuming fairness transfers unchanged.

## Limitations and Claim Boundary

It concerns social bias in language generation, not clinical outcomes, protected-group event sensitivity, or continuous forecasts.

## Relevance to This Article

Keep as a direct KD-fairness precedent, but use only for the general claim that teacher behavior and distillation can affect fairness.
