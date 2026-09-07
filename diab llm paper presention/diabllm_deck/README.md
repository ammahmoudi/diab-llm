# DiabLLM paper presentation

## Main deliverable

- `DiabLLM_paper_presentation_Amirhossein_Mahmoudi.pptx` - editable 16:9 PowerPoint, 15 slides, with speaker notes on every slide.
- `DiabLLM_animated_presentation.html` - animated, self-contained 20-slide 16:9 browser deck with native HTML/CSS/SVG shapes and charts; it uses no raster slide images.

## Supporting files

- `outline_and_evidence.md` - narrative plan, evidence ledger, source mapping, and claim boundaries.
- `DiabLLM_contact_sheet.png` - contact sheet used for visual QA.
- `qa_previews/` - one wireframe preview per slide.
- `assets/` - high-resolution manuscript figures used in the deck.
- `generate_deck.js` - reproducible deck-generation source.
- `render_wireframes.py` - local fallback preview renderer.

## Presentation profile

- Presenter: Amirhossein Mahmoudi, first author.
- Intended duration: approximately 15 minutes.
- Density: speaker-led.
- Main thesis: Time-LLM provides the strongest accuracy in the tested DiabLLM protocol, while denoising and distillation address input corruption and computational cost; prospective clinical and physical edge validation remain future work.

## HTML controls

- Navigate with arrow keys, Space, Page Up/Down, mouse wheel, or touch swipe.
- Press `N` to show or hide the current slide's speaker notes.
- Press `E` to enable inline editing for selected title text; press `Ctrl+S` to save edits to browser local storage.
- Use Home/End to jump to the first or final slide.
- On the Time-LLM, Chronos, and complete DiabLLM flow slides, use the focus tabs plus `◀`, `▶`, and `AUTO` for step-by-step walkthroughs.
- Introductory concept slides explain transformer next-token prediction, denoising autoencoders, and knowledge distillation before the detailed system graphs.
- Browser printing produces one 1920x1080 slide per page.

## QA status

- PowerPoint package validation: passed.
- Slide count and 16:9 dimensions: verified.
- Speaker-note parts: verified on all 15 slides.
- Visual wireframe inspection: completed for all slides.
- Native charts remain editable in PowerPoint.
- HTML structure and JavaScript syntax: validated.
- HTML slide count, notes, native SVG charts, and image-free rendering: verified.
- Desktop and 390x844 mobile scaling: browser-tested with no page overflow.

LibreOffice is not installed in the current environment, so the deck could not be rendered through the Office-compatible PDF pipeline. The included contact sheet is a local wireframe rendering; final font metrics should be checked once in Microsoft PowerPoint before presenting.
