# Workspace Presentation Skills

The presentation toolkit is installed project-locally for GitHub Copilot under `.agents/skills/`.

| Skill | Best use | Output | External service |
|---|---|---|---|
| `scientific-presentation` | Research talks, paper presentations, thesis or lab decks, and evidence-grounded DiabLLM presentations | Outline plus PPTX or HTML workflow | No |
| `pptx` | Creating, editing, validating, and rendering native PowerPoint files | Editable `.pptx` | No |
| `frontend-slides` | Distinctive animated presentations and visual style exploration | Self-contained HTML; optional PDF | No for local generation; Vercel only when explicitly deploying |
| `slide-design-skill` | Bespoke, on-brand, data-rich slide design | 1920x1080 HTML | Optional `FAL_KEY` for generated imagery |
| `slidespeak` | API-based PowerPoint generation or editing | Editable `.pptx` download | Yes; requires `SLIDESPEAK_API_KEY` |

## Recommended routing

1. Start scientific work with `/scientific-presentation` so claims, citations, caveats, and narrative are planned before design.
2. Use `/pptx` when an editable PowerPoint is required.
3. Use `/frontend-slides` for a polished browser presentation with motion and style previews.
4. Use `/slide-design-skill` for a bespoke HTML design system and data-heavy layouts.
5. Use `/slidespeak` only when API generation is explicitly requested and an API key is configured.

## Privacy

Do not upload unpublished manuscripts, patient-level data, private experiment results, or proprietary documents to SlideSpeak, FAL, Vercel, or another external service without explicit approval. The local `pptx` and `frontend-slides` workflows are preferred for sensitive research.

## Example request

> Use the scientific presentation skill to create a 15-minute, speaker-led DiabLLM paper presentation from `diab llm paper presention/`. Compare the Time-LLM and Chronos foundations, explain the DiabLLM adaptation, use only traceable metrics, include limitations and speaker notes, and produce an editable PPTX with rendered QA previews.
