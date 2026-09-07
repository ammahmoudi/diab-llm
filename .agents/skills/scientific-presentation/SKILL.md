---
name: scientific-presentation
description: "Create rigorous scientific and academic presentations from papers, LaTeX manuscripts, Markdown, PDFs, experiment CSVs, and figures. Use for research talks, thesis defenses, lab meetings, conference decks, paper summaries, DiabLLM, Time-LLM, Chronos, blood-glucose forecasting, knowledge distillation, fairness, and medical time-series presentations. Produces evidence-grounded outlines, slides, speaker notes, citations, and a final quality audit."
argument-hint: "Describe the audience, duration, source files, and desired output (PPTX or HTML)."
user-invocable: true
disable-model-invocation: false
---

# Scientific Presentation

Create accurate, visually clear research decks from this workspace's papers, article sources, figures, and experimental results.

## Required Inputs

Infer available details from the request and workspace. Ask only for missing decisions that materially affect the deck:

1. Audience: supervisor/lab, conference specialists, thesis committee, clinical audience, or general technical audience.
2. Duration or slide count.
3. Delivery mode: speaker-led or reading-first.
4. Output: editable PPTX, self-contained HTML, or both.
5. Claims to emphasize or avoid.

## Source Priority

Use evidence in this order:

1. User-selected files and final article text.
2. Verified experiment outputs and tables.
3. Downloaded canonical papers.
4. Repository documentation.
5. External sources only when the workspace lacks required evidence.

Useful workspace sources include:

- `diab llm paper presention/`: Chronos, Time-LLM, and DiabLLM PDFs and Markdown.
- `diab-llm-article/`: DiabLLM article LaTeX, bibliography, figures, and tables.
- `fairness_article/` and `fairness/`: fairness study and analysis.
- `results/`, `outputs/`, `experiments/`, and `distillation_experiments/`: empirical results.
- `docs/`: implementation and study context.

Never invent metrics, sample sizes, baselines, citations, clinical implications, or statistical significance. Label preliminary, unpublished, author-manuscript, and externally reported results accurately.

## Workflow

### 1. Build an evidence ledger

Before slide design, make a compact internal ledger containing:

- claim;
- exact value or conclusion;
- source file and page, section, table, figure, or row;
- status: project result, prior work, interpretation, or limitation.

Resolve conflicting values before using them. If they cannot be resolved, present the discrepancy explicitly or omit the claim.

### 2. Define one thesis

Write one sentence completing: "After this presentation, the audience should understand that..."

Every slide must advance that thesis. Remove interesting details that do not support it.

### 3. Choose a narrative arc

For a paper presentation, default to:

1. Clinical or technical problem.
2. Why existing forecasting approaches are insufficient.
3. Time-LLM and Chronos foundations.
4. DiabLLM adaptation and study questions.
5. Data and protocol.
6. Architecture or workflow.
7. Main predictive results.
8. Robustness, efficiency, distillation, or fairness evidence as relevant.
9. Limitations and validity boundaries.
10. Takeaways and next steps.

Use assertion-evidence slide titles: state the conclusion rather than naming the topic. Prefer "Distillation preserves accuracy while reducing model cost" over "Distillation results."

### 4. Set information density

- Speaker-led: one claim per slide, 1-3 short text elements, large visuals, and detailed speaker notes.
- Reading-first: self-contained slides with concise annotations, methods details, and source notes.
- Split overloaded slides. Do not shrink body text to fit.

### 5. Select the output skill

- Editable native PowerPoint: load and follow `../pptx/SKILL.md`.
- Distinctive self-contained HTML: load and follow `../frontend-slides/SKILL.md`.
- Bespoke deterministic 1920x1080 HTML with strong design gates: load and follow `../slide-design-skill/SKILL.md`.
- SlideSpeak API generation only when the user explicitly requests it and `SLIDESPEAK_API_KEY` is available: load `../slidespeak/SKILL.md`. Do not upload unpublished papers, patient-level data, private results, or proprietary documents without explicit user approval.

### 6. Design scientific visuals

- Redraw workflows and architectures as clean editable diagrams when feasible.
- Prefer plots over dense result tables; highlight the comparison that supports the slide claim.
- Preserve units, axes, uncertainty, cohort definitions, and prediction horizons.
- Do not imply causality from observational comparisons.
- Do not compare absolute metrics across incompatible tasks, splits, datasets, or protocols.
- Use equations only when the audience needs them; define every symbol and explain the intuition.
- Use restrained transitions and animations to reveal sequence or causality, not as decoration.
- Include a short source footer for external facts and paper-derived figures.

### 7. Write speaker notes

For each substantive slide, include:

- the point to say first;
- evidence interpretation;
- transition to the next slide;
- caveat or likely audience question when relevant.

Do not merely repeat visible slide text.

### 8. Run the final audit

Audit every slide for:

- factual traceability to the evidence ledger;
- readable labels, units, legends, and citations;
- no unsupported superiority, clinical-readiness, fairness, or generalization claims;
- no cross-task metric comparisons;
- consistent terminology: BG, CGM, T1DM, prediction horizon, teacher, student, and distilled model;
- one clear conclusion per slide;
- no overflow, overlap, clipping, placeholder text, or low contrast;
- varied but coherent layouts;
- correct title, authorship, venue, DOI/arXiv identifier, and manuscript status.

Render and inspect all slides at presentation resolution. Fix visible defects and rerun structural validation before delivery.

## Presentation Standards

- Use a 16:9 canvas.
- Use sentence case for titles.
- Keep body text at least 18 pt for live talks unless the user requests a reading deck.
- Treat color as meaning: keep model and cohort colors stable across slides.
- Use color-blind-safe encodings and reinforce color with labels or shapes.
- Keep citations legible and compact; include a final references slide for substantial external literature.
- End with 2-4 concrete takeaways, not a generic "Thank you" slide alone.

## Deliverables

Unless the user requests otherwise, provide:

1. the presentation file;
2. a Markdown outline with slide titles, evidence sources, and speaker-note summary;
3. rendered slide previews or a contact sheet for QA;
4. a short note identifying unresolved evidence limitations.
