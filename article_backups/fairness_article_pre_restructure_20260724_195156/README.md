# Fairness Under Compression

Private manuscript workspace for a conference paper on fairness in knowledge
distillation for clinical time-series models.

## Scope

The paper studies whether a compressed clinical model can retain average utility
while creating or preserving an event-level subgroup fairness gap. The primary
case study is OhioT1DM blood-glucose forecasting. MIT-BIH ECG classification is
secondary, cross-task evidence and must not be presented as an additional
blood-glucose cohort.

The planned comparison follows the teacher-to-student pipeline:

1. Teacher baseline
2. Student trained without distillation
3. Baseline knowledge distillation
4. Fair-teacher distillation (`T1`)
5. Group-aware output calibration (`O2`)
6. Combined fair-teacher distillation and output calibration (`T1+O2`)

For blood glucose, utility is measured by RMSE and fairness by the absolute
male/female true-positive-rate gap for hypoglycemia at the fixed 70 mg/dL
threshold. Lower EO gap is better.

## Evidence Status

This repository contains a manuscript scaffold, not a statement that the paper
is ready for submission. The parent project's evidence plan remains the
authority for result readiness. Final tables, numerical claims, and citations
are added only after the confirmatory blood-glucose suite and its paired
analysis are validated.

`deep-research-report.md` is planning material. It is not manuscript text:
its internal research citation tokens must be replaced with verified BibTeX
entries before any content is moved into the paper.

## Layout

| Path | Purpose |
| --- | --- |
| `main.tex` | IEEE conference manuscript entry point |
| `sections/` | Modular paper sections |
| `references.bib` | Verified external bibliography, initially empty |
| `deep-research-report.md` | Literature and paper-direction memo |
| `.gitignore` | LaTeX outputs and platform metadata |

## Build

Install a TeX distribution that includes `latexmk` and `IEEEtran`, then run:

```text
latexmk -pdf main.tex
```

The generated PDF and auxiliary files are intentionally ignored by Git.

## Writing Rules

- Treat the OhioT1DM analysis as the primary result.
- Keep ECG results explicitly secondary and endpoint-dependent.
- Use the fixed 70 mg/dL raw EO gap as the primary BG fairness metric.
- Report calibrated EO only as a secondary diagnostic; it is not the O2 method.
- Do not include unresolved `cite` or `filecite` tokens in the manuscript.
- Do not make a final empirical claim until its source artifact is reproduced,
  versioned, and cited in the results table.

## Git Relationship

This directory is an independent private Git repository. The parent
`LLM-TIME` repository tracks it as a submodule and pins the exact manuscript
revision used with each experiment state.
