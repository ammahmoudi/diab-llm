const pptxgen = require('/tmp/diabllm-pptx-tools/node_modules/pptxgenjs');
const path = require('path');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Amirhossein Mahmoudi';
pptx.subject = 'DiabLLM paper presentation';
pptx.title = 'DiabLLM: An LLM-Based Framework for Blood Glucose Prediction in Type 1 Diabetes';
pptx.company = 'Sharif University of Technology';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Cambria',
  bodyFontFace: 'Arial',
  lang: 'en-US',
};
pptx.defineSlideMaster({
  title: 'CONTENT',
  background: { color: 'F7FAFC' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: 'F7FAFC' }, line: { color: 'F7FAFC' } } },
    { text: { text: 'DiabLLM  |  Amirhossein Mahmoudi', options: { x: 0.55, y: 7.12, w: 5.6, h: 0.18, fontFace: 'Arial', fontSize: 7.5, color: '64748B', margin: 0 } } },
    { text: { text: 'IEEE JBHI 2026  |  DOI 10.1109/JBHI.2026.3658588', options: { x: 7.3, y: 7.12, w: 5.45, h: 0.18, fontFace: 'Arial', fontSize: 7.5, color: '64748B', align: 'right', margin: 0 } } },
  ],
  slideNumber: { x: 12.78, y: 0.22, color: '64748B', fontFace: 'Arial', fontSize: 8 },
});

const C = {
  ink: '14213D',
  navy: '0B1F33',
  teal: '007C83',
  mint: '2A9D8F',
  coral: 'E76F51',
  amber: 'E9C46A',
  sky: 'DCEFF4',
  pale: 'EDF4F7',
  white: 'FFFFFF',
  muted: '526477',
  gray: 'D7E1E7',
  darkGray: '334155',
  red: 'C94747',
};

const A = path.join(__dirname, 'assets');
const asset = (name) => path.join(A, name);
const cleanText = (s) => s.replace(/–/g, '-').replace(/→/g, 'to');
const shadow = () => ({ type: 'outer', color: '000000', opacity: 0.12, blur: 1.5, angle: 45, distance: 1 });

function addTitle(slide, title, kicker) {
  if (kicker) slide.addText(kicker.toUpperCase(), { x: 0.6, y: 0.35, w: 4.5, h: 0.22, fontFace: 'Arial', fontSize: 9, bold: true, color: C.teal, charSpacing: 1.2, margin: 0 });
  slide.addText(cleanText(title), { x: 0.6, y: 0.65, w: 11.8, h: 0.68, fontFace: 'Cambria', fontSize: 26, bold: true, color: C.ink, margin: 0, breakLine: false, fit: 'shrink' });
}

function addSource(slide, text) {
  slide.addText(cleanText(text), { x: 0.6, y: 6.82, w: 11.95, h: 0.18, fontFace: 'Arial', fontSize: 7.5, italic: true, color: '66788A', margin: 0, fit: 'shrink' });
}

function addNotes(slide, notes) {
  slide.addNotes(cleanText(notes));
}

function addPill(slide, text, x, y, w, fill, color = C.ink) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h: 0.42, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  slide.addText(text, { x, y: y + 0.08, w, h: 0.18, fontFace: 'Arial', fontSize: 10, bold: true, color, align: 'center', margin: 0, fit: 'shrink' });
}

function addMetricCard(slide, { x, y, w, h = 1.35, value, label, detail, color = C.teal, fill = C.white }) {
  slide.addShape(pptx.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: 'D3E0E5', width: 1 }, shadow: shadow() });
  slide.addText(value, { x: x + 0.2, y: y + 0.18, w: w - 0.4, h: 0.43, fontFace: 'Cambria', fontSize: 26, bold: true, color, margin: 0, fit: 'shrink' });
  slide.addText(label, { x: x + 0.2, y: y + 0.69, w: w - 0.4, h: 0.24, fontFace: 'Arial', fontSize: 11, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
  if (detail) slide.addText(detail, { x: x + 0.2, y: y + 0.99, w: w - 0.4, h: 0.2, fontFace: 'Arial', fontSize: 8.5, color: C.muted, margin: 0, fit: 'shrink' });
}

function addArrow(slide, x, y, w, color = C.teal) {
  slide.addShape(pptx.ShapeType.chevron, { x, y, w, h: 0.4, fill: { color }, line: { color } });
}

function addLineChart(slide, series, x, y, w, h, opts = {}) {
  slide.addChart(pptx.ChartType.line, series, {
    x, y, w, h,
    showLegend: opts.showLegend ?? true,
    legendPos: 'b',
    legendFontFace: 'Arial',
    legendFontSize: 9,
    chartColors: opts.colors || [C.coral, C.teal, C.amber],
    showTitle: false,
    showValue: false,
    showCatName: false,
    catAxisLabelFontFace: 'Arial',
    catAxisLabelFontSize: 9,
    catAxisLabelColor: C.muted,
    valAxisLabelFontFace: 'Arial',
    valAxisLabelFontSize: 9,
    valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 },
    catGridLine: { style: 'none' },
    showCatName: false,
    showValAxisTitle: !!opts.valAxisTitle,
    valAxisTitle: opts.valAxisTitle || '',
    valAxisTitleFontFace: 'Arial',
    valAxisTitleFontSize: 10,
    showCatAxisTitle: false,
    showMarker: true,
    lineSize: 2.5,
    showBorder: false,
  });
}

function addBarChart(slide, categories, values, x, y, w, h, opts = {}) {
  slide.addChart(pptx.ChartType.bar, [{ name: opts.name || 'RMSE', labels: categories, values }], {
    x, y, w, h,
    catAxisLabelFontFace: 'Arial',
    catAxisLabelFontSize: 10,
    catAxisLabelColor: C.ink,
    valAxisLabelFontFace: 'Arial',
    valAxisLabelFontSize: 9,
    valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 },
    catGridLine: { style: 'none' },
    showLegend: false,
    showTitle: false,
    showValue: true,
    dataLabelPosition: 'outEnd',
    dataLabelColor: C.ink,
    dataLabelFormatCode: '0.00',
    chartColors: opts.colors || [C.teal],
    showBorder: false,
    gapWidthPct: 45,
  });
}

// 1. Title
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(pptx.ShapeType.arc, { x: 8.5, y: -1.2, w: 5.8, h: 5.8, adjustPoint: 0.35, rotate: 20, fill: { color: C.teal, transparency: 10 }, line: { color: C.teal, transparency: 100 } });
  slide.addShape(pptx.ShapeType.arc, { x: 9.4, y: 3.5, w: 4.3, h: 4.3, adjustPoint: 0.35, rotate: 205, fill: { color: C.coral, transparency: 10 }, line: { color: C.coral, transparency: 100 } });
  slide.addText('DiabLLM', { x: 0.72, y: 0.72, w: 6.2, h: 0.85, fontFace: 'Cambria', fontSize: 40, bold: true, color: C.white, margin: 0 });
  slide.addText('An LLM-Based Framework for Blood Glucose Prediction in Type 1 Diabetes', { x: 0.75, y: 1.75, w: 8.0, h: 1.2, fontFace: 'Cambria', fontSize: 26, bold: true, color: 'DCEFF4', margin: 0, breakLine: false, fit: 'shrink' });
  slide.addText('Amirhossein Mahmoudi  |  First author and presenter', { x: 0.76, y: 3.45, w: 6.6, h: 0.34, fontFace: 'Arial', fontSize: 15, bold: true, color: C.amber, margin: 0 });
  slide.addText('IEEE Journal of Biomedical and Health Informatics  •  2026', { x: 0.76, y: 3.95, w: 6.8, h: 0.3, fontFace: 'Arial', fontSize: 13, color: C.white, margin: 0 });
  addPill(slide, 'TIME-LLM', 0.76, 5.25, 1.55, C.coral, C.white);
  addPill(slide, 'CHRONOS', 2.5, 5.25, 1.55, C.teal, C.white);
  addPill(slide, 'CGM FORECASTING', 4.25, 5.25, 2.2, C.amber, C.navy);
  slide.addText('DOI 10.1109/JBHI.2026.3658588', { x: 0.76, y: 6.55, w: 4.2, h: 0.22, fontFace: 'Arial', fontSize: 9, color: 'A7BDCA', margin: 0 });
  addNotes(slide, 'Open with the core idea: this work asks whether language-inspired forecasting architectures can model short-term glucose dynamics from CGM alone. Introduce yourself as Amirhossein Mahmoudi, first author. The talk has three parts: architectural choices, evidence, and what remains before deployment.');
}

// 2. Problem
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Thirty minutes of CGM history must anticipate the next 30-45 minutes', 'Clinical forecasting problem');
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.65, y: 1.65, w: 5.65, h: 4.55, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.gray }, shadow: shadow() });
  slide.addText('PAST', { x: 0.95, y: 1.98, w: 1, h: 0.2, fontFace: 'Arial', fontSize: 10, bold: true, color: C.teal, margin: 0 });
  const past = [155, 149, 142, 151, 160, 166];
  const future = [171, 178, 186, 193, 201, 207, 211, 214, 218];
  addLineChart(slide, [{ name: 'Observed CGM', labels: ['-30', '-25', '-20', '-15', '-10', '-5'], values: past }], 0.9, 2.3, 4.95, 2.65, { showLegend: false, colors: [C.teal], valAxisTitle: 'BG (mg/dL)' });
  slide.addShape(pptx.ShapeType.line, { x: 5.25, y: 2.45, w: 0, h: 2.25, line: { color: C.coral, width: 2, dash: 'dash' } });
  slide.addText('NOW', { x: 4.88, y: 5.08, w: 0.8, h: 0.2, fontFace: 'Arial', fontSize: 9, bold: true, color: C.coral, align: 'center', margin: 0 });
  slide.addShape(pptx.ShapeType.roundRect, { x: 6.68, y: 1.65, w: 5.95, h: 4.55, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy }, shadow: shadow() });
  slide.addText('FORECAST', { x: 7.0, y: 1.98, w: 1.3, h: 0.2, fontFace: 'Arial', fontSize: 10, bold: true, color: C.amber, margin: 0 });
  addLineChart(slide, [{ name: 'Predicted BG', labels: ['+5', '+10', '+15', '+20', '+25', '+30', '+35', '+40', '+45'], values: future }], 6.95, 2.3, 5.05, 2.65, { showLegend: false, colors: [C.coral], valAxisTitle: 'BG (mg/dL)' });
  slide.addText('6 readings', { x: 1.02, y: 5.55, w: 1.3, h: 0.35, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.teal, margin: 0 });
  slide.addText('5-minute sampling', { x: 2.4, y: 5.63, w: 1.7, h: 0.2, fontFace: 'Arial', fontSize: 10.5, color: C.muted, margin: 0 });
  slide.addText('6 or 9 targets', { x: 7.03, y: 5.55, w: 1.75, h: 0.35, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.amber, margin: 0 });
  slide.addText('30- or 45-minute horizon', { x: 8.85, y: 5.63, w: 2.25, h: 0.2, fontFace: 'Arial', fontSize: 10.5, color: 'D7E1E7', margin: 0 });
  addSource(slide, 'Source: DiabLLM abstract and experimental protocol. Illustrative glucose trajectory; not patient data.');
  addNotes(slide, 'The operational task is deliberately constrained: six CGM readings, covering only thirty minutes, must generate six or nine future values. Explain why this matters: short context limits feature engineering, while longer horizons amplify uncertainty. The illustrated trajectory is schematic, not a patient record. Transition: the key design question is how to make language-model machinery accept continuous glucose.');
}

// 3. Architectures
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Chronos and Time-LLM bridge continuous signals in fundamentally different ways', 'Two language-inspired strategies');
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.62, y: 1.48, w: 6.03, h: 4.95, rectRadius: 0.08, fill: { color: 'F4EAF5' }, line: { color: 'CFA8D4' }, shadow: shadow() });
  slide.addText('CHRONOS', { x: 0.95, y: 1.78, w: 2, h: 0.3, fontFace: 'Cambria', fontSize: 19, bold: true, color: '663A75', margin: 0 });
  slide.addImage({ path: asset('chronos.png'), x: 0.95, y: 2.2, w: 5.35, h: 2.25 });
  addPill(slide, 'scale', 1.05, 4.82, 1.05, 'EBD8ED');
  addArrow(slide, 2.2, 4.83, 0.38, 'A666B1');
  addPill(slide, 'quantize', 2.7, 4.82, 1.2, 'EBD8ED');
  addArrow(slide, 4.02, 4.83, 0.38, 'A666B1');
  addPill(slide, 'predict tokens', 4.5, 4.82, 1.45, 'EBD8ED');
  slide.addText('Regression via classification', { x: 1.02, y: 5.55, w: 3.3, h: 0.28, fontFace: 'Arial', fontSize: 13, bold: true, color: '663A75', margin: 0 });
  slide.addText('The transformer is trained on a time-series vocabulary.', { x: 1.02, y: 5.91, w: 4.8, h: 0.25, fontFace: 'Arial', fontSize: 10.5, color: C.muted, margin: 0 });

  slide.addShape(pptx.ShapeType.roundRect, { x: 6.92, y: 1.48, w: 5.8, h: 4.95, rectRadius: 0.08, fill: { color: 'EAF5F1' }, line: { color: 'A6D3C7' }, shadow: shadow() });
  slide.addText('TIME-LLM', { x: 7.25, y: 1.78, w: 2, h: 0.3, fontFace: 'Cambria', fontSize: 19, bold: true, color: '176B62', margin: 0 });
  slide.addImage({ path: asset('time-llm.png'), x: 7.25, y: 2.2, w: 5.1, h: 2.14 });
  addPill(slide, 'patch', 7.32, 4.82, 1.0, 'D7EEE7');
  addArrow(slide, 8.43, 4.83, 0.38, C.teal);
  addPill(slide, 'reprogram', 8.92, 4.82, 1.2, 'D7EEE7');
  addArrow(slide, 10.23, 4.83, 0.38, C.teal);
  addPill(slide, 'frozen LLM', 10.73, 4.82, 1.25, 'D7EEE7');
  slide.addText('Modality alignment', { x: 7.32, y: 5.55, w: 2.7, h: 0.28, fontFace: 'Arial', fontSize: 13, bold: true, color: '176B62', margin: 0 });
  slide.addText('Lightweight layers map patches into the LLM embedding space.', { x: 7.32, y: 5.91, w: 4.75, h: 0.25, fontFace: 'Arial', fontSize: 10.5, color: C.muted, margin: 0 });
  addSource(slide, 'Sources: Ansari et al., Chronos, arXiv:2403.07815; Jin et al., Time-LLM, arXiv:2310.01728; DiabLLM methodology.');
  addNotes(slide, 'Chronos discretizes the values and trains a language-model architecture to predict the next token. Time-LLM keeps a pretrained language model frozen and learns how to reprogram numerical patches into its embedding space, adding a textual prompt with dataset and input statistics. Emphasize that both are called language-inspired, but only Time-LLM reuses a frozen general-purpose language backbone in this implementation.');
}

// 4. Study design
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'DiabLLM evaluates adaptation, robustness, transfer, and compression', 'Study design');
  const stages = [
    ['1', 'Forecast', 'Zero-shot and personalized fine-tuning'],
    ['2', 'Stress', 'Noise, calibration bias, and missing CGM'],
    ['3', 'Transfer', 'Cross-patient evaluation'],
    ['4', 'Compress', 'BERT teacher to TinyBERT student'],
    ['5', 'Assess', 'RMSE, MAE, CEG, SEG, and efficiency'],
  ];
  stages.forEach((s, i) => {
    const x = 0.72 + i * 2.48;
    slide.addShape(pptx.ShapeType.roundRect, { x, y: 2.0, w: 2.08, h: 3.45, rectRadius: 0.08, fill: { color: i % 2 === 0 ? C.white : C.pale }, line: { color: i === 4 ? C.coral : C.gray, width: i === 4 ? 2 : 1 }, shadow: shadow() });
    slide.addShape(pptx.ShapeType.ellipse, { x: x + 0.67, y: 2.27, w: 0.72, h: 0.72, fill: { color: [C.teal, C.coral, C.amber, C.mint, C.navy][i] }, line: { color: [C.teal, C.coral, C.amber, C.mint, C.navy][i] } });
    slide.addText(s[0], { x: x + 0.67, y: 2.43, w: 0.72, h: 0.22, fontFace: 'Cambria', fontSize: 15, bold: true, color: i === 2 ? C.navy : C.white, align: 'center', margin: 0 });
    slide.addText(s[1], { x: x + 0.18, y: 3.3, w: 1.72, h: 0.35, fontFace: 'Cambria', fontSize: 18, bold: true, color: C.ink, align: 'center', margin: 0 });
    slide.addText(s[2], { x: x + 0.18, y: 3.93, w: 1.72, h: 0.9, fontFace: 'Arial', fontSize: 11, color: C.muted, align: 'center', valign: 'mid', margin: 0.04, fit: 'shrink' });
    if (i < stages.length - 1) addArrow(slide, x + 2.12, 3.48, 0.28, '9AB1BC');
  });
  slide.addText('The contribution is the complete evidence chain - not a single leaderboard number.', { x: 1.22, y: 5.95, w: 10.85, h: 0.4, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.teal, align: 'center', margin: 0 });
  addSource(slide, 'Source: DiabLLM methodology and experimental sections.');
  addNotes(slide, 'Frame the paper as a structured evaluation. It begins with forecasting, then tests the effect of adaptation, realistic corruption, patient transfer, model compression, and clinical error grids. This helps the audience understand why the later slides are organized as an evidence chain rather than a catalogue of experiments.');
}

// 5. Data
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Two public T1DM cohorts test personalization and external transfer', 'Data and protocol');
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.72, y: 1.55, w: 5.78, h: 4.75, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy }, shadow: shadow() });
  slide.addText('OhioT1DM', { x: 1.05, y: 1.94, w: 2.5, h: 0.4, fontFace: 'Cambria', fontSize: 25, bold: true, color: C.white, margin: 0 });
  addMetricCard(slide, { x: 1.02, y: 2.65, w: 1.45, h: 1.3, value: '12', label: 'adults', detail: 'with T1DM', color: C.amber, fill: '173149' });
  addMetricCard(slide, { x: 2.7, y: 2.65, w: 1.45, h: 1.3, value: '8', label: 'weeks', detail: 'per cohort record', color: C.amber, fill: '173149' });
  addMetricCard(slide, { x: 4.38, y: 2.65, w: 1.45, h: 1.3, value: '5', label: 'minutes', detail: 'CGM sampling', color: C.amber, fill: '173149' });
  slide.addText('Primary personalized and cross-patient evaluation', { x: 1.04, y: 4.65, w: 4.9, h: 0.6, fontFace: 'Arial', fontSize: 15, bold: true, color: C.white, margin: 0, fit: 'shrink' });
  slide.addText('Predefined train/test sets; missing values preserved in the core protocol.', { x: 1.04, y: 5.35, w: 4.85, h: 0.45, fontFace: 'Arial', fontSize: 10.5, color: 'B9CBD5', margin: 0 });

  slide.addShape(pptx.ShapeType.roundRect, { x: 6.82, y: 1.55, w: 5.78, h: 4.75, rectRadius: 0.08, fill: { color: C.white }, line: { color: 'BFD8D3' }, shadow: shadow() });
  slide.addText('D1NAMO', { x: 7.15, y: 1.94, w: 2.5, h: 0.4, fontFace: 'Cambria', fontSize: 25, bold: true, color: C.teal, margin: 0 });
  addMetricCard(slide, { x: 7.12, y: 2.65, w: 1.45, h: 1.3, value: '9', label: 'people', detail: 'in the dataset', color: C.teal, fill: C.pale });
  addMetricCard(slide, { x: 8.8, y: 2.65, w: 1.45, h: 1.3, value: '7', label: 'used', detail: 'patients 1-7', color: C.teal, fill: C.pale });
  addMetricCard(slide, { x: 10.48, y: 2.65, w: 1.45, h: 1.3, value: '4', label: 'weeks', detail: 'collection period', color: C.teal, fill: C.pale });
  slide.addText('Complementary validation cohort', { x: 7.15, y: 4.65, w: 4.7, h: 0.45, fontFace: 'Arial', fontSize: 15, bold: true, color: C.ink, margin: 0 });
  slide.addText('Used to test whether the adaptation pattern extends beyond OhioT1DM.', { x: 7.15, y: 5.35, w: 4.85, h: 0.45, fontFace: 'Arial', fontSize: 10.5, color: C.muted, margin: 0 });
  addSource(slide, 'Source: DiabLLM experiments. OhioT1DM: Marling and Bunescu (2020); D1NAMO: Dubosson et al. (2018).');
  addNotes(slide, 'Describe the evidence base without overselling scale. OhioT1DM is the main cohort: twelve adults, eight weeks, sampled every five minutes. D1NAMO is a complementary cohort; seven of nine individuals are used. The study is retrospective and public-data based. Transition to the central experimental finding: adaptation matters more than model size alone.');
}

// 6. Adaptation
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Task-specific adaptation produces the largest accuracy gain', 'Zero-shot versus fine-tuned');
  const cats = ['Time-LLM\nLLaMA-8', 'Chronos\nBase'];
  const zero = [24.44, 22.29];
  const tuned = [16.10, 20.89];
  slide.addChart(pptx.ChartType.bar, [
    { name: 'Zero-shot', labels: cats, values: zero },
    { name: 'Fine-tuned', labels: cats, values: tuned },
  ], {
    x: 0.75, y: 1.55, w: 7.1, h: 4.75,
    catAxisLabelFontFace: 'Arial', catAxisLabelFontSize: 11, catAxisLabelColor: C.ink,
    valAxisLabelFontFace: 'Arial', valAxisLabelFontSize: 9, valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendFontFace: 'Arial', legendFontSize: 10,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelColor: C.ink,
    chartColors: ['9CB6C3', C.coral], showBorder: false, gapWidthPct: 50,
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 8.25, y: 1.65, w: 4.35, h: 4.45, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy }, shadow: shadow() });
  slide.addText('RMSE reduction after adaptation', { x: 8.62, y: 2.0, w: 3.6, h: 0.5, fontFace: 'Cambria', fontSize: 20, bold: true, color: C.white, margin: 0, align: 'center' });
  slide.addText('34%', { x: 8.75, y: 2.9, w: 1.4, h: 0.65, fontFace: 'Cambria', fontSize: 35, bold: true, color: C.amber, margin: 0, align: 'center' });
  slide.addText('Time-LLM', { x: 8.72, y: 3.62, w: 1.5, h: 0.3, fontFace: 'Arial', fontSize: 11, bold: true, color: C.white, align: 'center', margin: 0 });
  slide.addText('6%', { x: 10.62, y: 2.9, w: 1.25, h: 0.65, fontFace: 'Cambria', fontSize: 35, bold: true, color: C.mint, margin: 0, align: 'center' });
  slide.addText('Chronos', { x: 10.5, y: 3.62, w: 1.5, h: 0.3, fontFace: 'Arial', fontSize: 11, bold: true, color: C.white, align: 'center', margin: 0 });
  slide.addText('Same 30-minute history and 30-minute horizon', { x: 8.72, y: 4.45, w: 3.5, h: 0.48, fontFace: 'Arial', fontSize: 12, color: 'C7D7DF', align: 'center', margin: 0 });
  slide.addText('Fine-tuning the modality bridge is more consequential than simply selecting a larger pretrained model.', { x: 8.7, y: 5.1, w: 3.55, h: 0.58, fontFace: 'Arial', fontSize: 11, bold: true, color: C.white, align: 'center', margin: 0.04, fit: 'shrink' });
  addSource(slide, 'Sources: DiabLLM Tables: zero-shot and fine-tuned 30-minute results. Percentages calculated from table averages.');
  addNotes(slide, 'At the same thirty-minute horizon, Time-LLM LLaMA 8-layer moves from 24.44 to 16.10 RMSE, while Chronos Base moves from 22.29 to 20.89. The exact percentage differs slightly from the manuscript narrative because this slide computes directly from the table averages. The point is not that zero-shot fails; it is that clinical-domain adaptation is decisive.');
}

// 7. Benchmark
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Fine-tuned Time-LLM reaches 16.10 mg/dL average RMSE', '30-minute OhioT1DM benchmark');
  const cats = ['Time-LLM\nLLaMA-8', 'Time-LLM\nGPT-2', 'Deep RL', 'Chronos\nBase', 'LSTM +\nWaveNet + GRU'];
  const vals = [16.10, 16.34, 18.32, 20.89, 21.90];
  slide.addChart(pptx.ChartType.bar, [{ name: 'Average RMSE', labels: cats, values: vals }], {
    x: 0.72, y: 1.5, w: 8.0, h: 4.95,
    catAxisLabelFontFace: 'Arial', catAxisLabelFontSize: 10.5, catAxisLabelColor: C.ink,
    valAxisLabelFontFace: 'Arial', valAxisLabelFontSize: 9, valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 }, catGridLine: { style: 'none' },
    showLegend: false, showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelColor: C.ink,
    chartColors: [C.teal], showBorder: false, gapWidthPct: 35,
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 9.05, y: 1.58, w: 3.52, h: 4.7, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.gray }, shadow: shadow() });
  slide.addText('What the comparison supports', { x: 9.38, y: 1.98, w: 2.88, h: 0.48, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.ink, margin: 0 });
  const bullets = [
    'Time-LLM variants cluster near 16 mg/dL RMSE.',
    'LLaMA-8 improves 12.1% over the matched Deep RL baseline.',
    'Chronos offers a smaller accuracy gain with faster fine-tuning.',
  ];
  bullets.forEach((b, i) => {
    slide.addShape(pptx.ShapeType.ellipse, { x: 9.38, y: 2.78 + i * 0.92, w: 0.35, h: 0.35, fill: { color: [C.teal, C.coral, C.amber][i] }, line: { color: [C.teal, C.coral, C.amber][i] } });
    slide.addText(String(i + 1), { x: 9.38, y: 2.86 + i * 0.92, w: 0.35, h: 0.12, fontFace: 'Arial', fontSize: 8.5, bold: true, color: i === 2 ? C.navy : C.white, align: 'center', margin: 0 });
    slide.addText(b, { x: 9.88, y: 2.72 + i * 0.92, w: 2.28, h: 0.62, fontFace: 'Arial', fontSize: 11, color: C.darkGray, margin: 0, fit: 'shrink' });
  });
  slide.addText('Caution', { x: 9.38, y: 5.55, w: 0.75, h: 0.22, fontFace: 'Arial', fontSize: 9.5, bold: true, color: C.red, margin: 0 });
  slide.addText('Rows from other papers are contextual unless the protocol is explicitly matched.', { x: 10.1, y: 5.5, w: 2.05, h: 0.48, fontFace: 'Arial', fontSize: 9.5, color: C.muted, margin: 0, fit: 'shrink' });
  addSource(slide, 'Source: DiabLLM Table: Model Performance Comparison (30-minute history and horizon). Lower RMSE is better.');
  addNotes(slide, 'Lead with the absolute result: 16.10 milligrams per deciliter average RMSE for the fine-tuned LLaMA 8-layer Time-LLM. The directly comparable Deep RL and fusion baselines are 18.32 and 21.90. State the caveat visibly: the full article table contains other studies with differing protocols, so this slide selects the comparisons the manuscript identifies as matched.');
}

// 8. Clinical grids
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Error-grid analysis favors Time-LLM at the cohort level', 'Clinical error characterization');
  addMetricCard(slide, { x: 0.76, y: 1.55, w: 2.65, h: 1.55, value: '99.58%', label: 'CEG Zone A', detail: 'Time-LLM cohort result', color: C.teal });
  addMetricCard(slide, { x: 3.65, y: 1.55, w: 2.65, h: 1.55, value: '98.43%', label: 'CEG Zone A', detail: 'Chronos cohort result', color: '7E5A87', fill: 'F4EAF5' });
  addMetricCard(slide, { x: 6.78, y: 1.55, w: 2.65, h: 1.55, value: '97.15%', label: 'SEG: None', detail: 'Time-LLM cohort result', color: C.teal });
  addMetricCard(slide, { x: 9.67, y: 1.55, w: 2.65, h: 1.55, value: '92.94%', label: 'SEG: None', detail: 'Chronos cohort result', color: '7E5A87', fill: 'F4EAF5' });
  slide.addImage({ path: asset('ceg-timellm.png'), x: 0.95, y: 3.45, w: 2.8, h: 2.68 });
  slide.addImage({ path: asset('ceg-chronos.png'), x: 3.82, y: 3.45, w: 2.8, h: 2.68 });
  slide.addImage({ path: asset('seg-timellm.png'), x: 6.9, y: 3.55, w: 2.8, h: 2.18 });
  slide.addImage({ path: asset('seg-chronos.png'), x: 9.77, y: 3.55, w: 2.8, h: 2.18 });
  slide.addText('Time-LLM', { x: 1.67, y: 6.17, w: 1.3, h: 0.2, fontFace: 'Arial', fontSize: 9.5, bold: true, color: C.teal, align: 'center', margin: 0 });
  slide.addText('Chronos', { x: 4.55, y: 6.17, w: 1.3, h: 0.2, fontFace: 'Arial', fontSize: 9.5, bold: true, color: '7E5A87', align: 'center', margin: 0 });
  slide.addText('Time-LLM', { x: 7.62, y: 6.0, w: 1.3, h: 0.2, fontFace: 'Arial', fontSize: 9.5, bold: true, color: C.teal, align: 'center', margin: 0 });
  slide.addText('Chronos', { x: 10.5, y: 6.0, w: 1.3, h: 0.2, fontFace: 'Arial', fontSize: 9.5, bold: true, color: '7E5A87', align: 'center', margin: 0 });
  addSource(slide, 'Source: DiabLLM cohort-level CEG and SEG tables and figures, OhioT1DM, 30-minute horizon.');
  addNotes(slide, 'The error-grid results complement RMSE. Time-LLM places 99.58 percent of cohort predictions in Clarke Zone A and 97.15 percent in the SEG None category, compared with 98.43 and 92.94 percent for Chronos. Use careful language: these are retrospective error-grid outcomes, not proof of safety in insulin dosing or prospective clinical use.');
}

// 9. Robustness
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Denoising recovers much of the accuracy lost to corrupted CGM', 'Robustness to sensor imperfections');
  const cats = ['Chronos Base', 'Time-LLM LLaMA-8'];
  slide.addChart(pptx.ChartType.bar, [
    { name: 'Noisy input', labels: cats, values: [47.453, 51.067] },
    { name: 'After denoising', labels: cats, values: [25.963, 19.119] },
  ], {
    x: 0.72, y: 1.55, w: 7.2, h: 4.75,
    catAxisLabelFontFace: 'Arial', catAxisLabelFontSize: 11, catAxisLabelColor: C.ink,
    valAxisLabelFontFace: 'Arial', valAxisLabelFontSize: 9, valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendFontFace: 'Arial', legendFontSize: 10,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0', dataLabelColor: C.ink,
    chartColors: [C.red, C.teal], showBorder: false, gapWidthPct: 50,
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 8.28, y: 1.65, w: 4.25, h: 4.5, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.gray }, shadow: shadow() });
  slide.addText('Corruption model', { x: 8.62, y: 2.02, w: 2.5, h: 0.35, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.ink, margin: 0 });
  const corrupt = [
    ['gain + offset', 'calibration bias'],
    ['AR(1) noise', 'temporally correlated'],
    ['dropout blocks', '~10% cumulative duration'],
  ];
  corrupt.forEach((d, i) => {
    addPill(slide, d[0], 8.62, 2.65 + i * 0.83, 1.55, [C.sky, 'FBE8E3', 'F7EBCB'][i]);
    slide.addText(d[1], { x: 10.35, y: 2.76 + i * 0.83, w: 1.75, h: 0.18, fontFace: 'Arial', fontSize: 9.5, color: C.muted, margin: 0, fit: 'shrink' });
  });
  slide.addText('45-minute horizon', { x: 8.62, y: 5.3, w: 1.8, h: 0.25, fontFace: 'Arial', fontSize: 11, bold: true, color: C.teal, margin: 0 });
  slide.addText('Denoised Time-LLM is ~26% lower RMSE than denoised Chronos.', { x: 8.62, y: 5.62, w: 3.2, h: 0.35, fontFace: 'Arial', fontSize: 10.5, color: C.darkGray, margin: 0, fit: 'shrink' });
  addSource(slide, 'Source: DiabLLM robustness table. Cohort averages across 12 OhioT1DM patients; mean RMSE in mg/dL.');
  addNotes(slide, 'Sensor corruption is simulated with gain and offset calibration error, correlated noise, and missing blocks. The autoencoder reduces average RMSE from 51.067 to 19.119 for Time-LLM and from 47.453 to 25.963 for Chronos. This supports preprocessing as a robustness strategy under the tested simulation, but it does not demonstrate robustness to every real sensor failure mode.');
}

// 10. Cross patient
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Cross-patient transfer shows limited degradation in the tested pairs', 'Generalization across individuals');
  const labels = ['570 to 570', '584 to 570', '570 to 584', '584 to 584'];
  slide.addChart(pptx.ChartType.column, [
    { name: 'Time-LLM', labels, values: [16.81, 17.63, 26.63, 27.23] },
    { name: 'Chronos', labels, values: [23.69, 22.09, 31.68, 30.76] },
  ], {
    x: 0.72, y: 1.55, w: 8.4, h: 4.8,
    catAxisLabelFontFace: 'Arial', catAxisLabelFontSize: 9.5, catAxisLabelColor: C.ink,
    valAxisLabelFontFace: 'Arial', valAxisLabelFontSize: 9, valAxisLabelColor: C.muted,
    valGridLine: { color: 'DDE5EA', width: 1 }, catGridLine: { style: 'none' },
    showLegend: true, legendPos: 'b', legendFontFace: 'Arial', legendFontSize: 10,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00', dataLabelColor: C.ink,
    chartColors: [C.teal, '8E6A98'], showBorder: false, gapWidthPct: 55,
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 9.42, y: 1.65, w: 3.1, h: 4.45, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy }, shadow: shadow() });
  slide.addText('What transfers?', { x: 9.75, y: 2.05, w: 2.45, h: 0.35, fontFace: 'Cambria', fontSize: 20, bold: true, color: C.white, align: 'center', margin: 0 });
  slide.addText('Temporal glucose patterns', { x: 9.8, y: 2.8, w: 2.35, h: 0.45, fontFace: 'Arial', fontSize: 13, bold: true, color: C.amber, align: 'center', margin: 0 });
  slide.addText('Training on patient 584 slightly improves Chronos performance on patient 570.', { x: 9.78, y: 3.55, w: 2.38, h: 0.75, fontFace: 'Arial', fontSize: 11, color: C.white, align: 'center', margin: 0.04, fit: 'shrink' });
  slide.addText('Boundary', { x: 10.52, y: 4.65, w: 0.9, h: 0.25, fontFace: 'Arial', fontSize: 10, bold: true, color: C.coral, align: 'center', margin: 0 });
  slide.addText('Only two patients are used in this transfer matrix; population-level generalization is unresolved.', { x: 9.78, y: 5.0, w: 2.38, h: 0.62, fontFace: 'Arial', fontSize: 10, color: 'CAD9E1', align: 'center', margin: 0.04, fit: 'shrink' });
  addSource(slide, 'Source: DiabLLM cross-patient performance table. Labels are Train ID to Test ID; metric is RMSE in mg/dL.');
  addNotes(slide, 'This experiment uses patients 570 and 584 to create a compact transfer matrix. Time-LLM changes from 16.81 to 17.63 on patient 570 depending on the training patient, and from 27.23 to 26.63 on patient 584. Chronos also shows small changes. The result is encouraging for transfer, but the scope is only two patients and should not be generalized to a new clinical population.');
}

// 11. Distillation accuracy
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Distillation preserves BERT-level accuracy with a 45M-parameter student', 'Knowledge distillation');
  slide.addShape(pptx.ShapeType.roundRect, { x: 0.75, y: 1.65, w: 3.15, h: 3.8, rectRadius: 0.08, fill: { color: C.navy }, line: { color: C.navy }, shadow: shadow() });
  slide.addText('TEACHER', { x: 1.15, y: 2.0, w: 2.35, h: 0.25, fontFace: 'Arial', fontSize: 10, bold: true, color: C.amber, align: 'center', charSpacing: 1, margin: 0 });
  slide.addText('Time-LLM\nBERT', { x: 1.08, y: 2.55, w: 2.5, h: 0.85, fontFace: 'Cambria', fontSize: 27, bold: true, color: C.white, align: 'center', margin: 0, breakLine: true });
  slide.addText('12 layers  •  768 dimensions', { x: 1.08, y: 3.65, w: 2.5, h: 0.28, fontFace: 'Arial', fontSize: 10.5, color: 'BCD0DA', align: 'center', margin: 0 });
  slide.addText('RMSE 20.967', { x: 1.15, y: 4.45, w: 2.35, h: 0.35, fontFace: 'Cambria', fontSize: 20, bold: true, color: C.amber, align: 'center', margin: 0 });
  addArrow(slide, 4.22, 3.15, 1.05, C.coral);
  slide.addText('ground truth + teacher alignment', { x: 3.88, y: 3.78, w: 1.7, h: 0.42, fontFace: 'Arial', fontSize: 9.5, color: C.muted, align: 'center', margin: 0, fit: 'shrink' });
  slide.addShape(pptx.ShapeType.roundRect, { x: 5.6, y: 1.65, w: 3.15, h: 3.8, rectRadius: 0.08, fill: { color: C.white }, line: { color: 'A6D3C7', width: 2 }, shadow: shadow() });
  slide.addText('STUDENT', { x: 6.0, y: 2.0, w: 2.35, h: 0.25, fontFace: 'Arial', fontSize: 10, bold: true, color: C.teal, align: 'center', charSpacing: 1, margin: 0 });
  slide.addText('Time-LLM\nTinyBERT', { x: 5.93, y: 2.55, w: 2.5, h: 0.85, fontFace: 'Cambria', fontSize: 27, bold: true, color: C.ink, align: 'center', margin: 0, breakLine: true });
  slide.addText('4 layers  •  312 dimensions', { x: 5.93, y: 3.65, w: 2.5, h: 0.28, fontFace: 'Arial', fontSize: 10.5, color: C.muted, align: 'center', margin: 0 });
  slide.addText('RMSE 20.953', { x: 6.0, y: 4.45, w: 2.35, h: 0.35, fontFace: 'Cambria', fontSize: 20, bold: true, color: C.teal, align: 'center', margin: 0 });
  slide.addShape(pptx.ShapeType.roundRect, { x: 9.15, y: 1.65, w: 3.4, h: 3.8, rectRadius: 0.08, fill: { color: 'EAF5F1' }, line: { color: 'A6D3C7' }, shadow: shadow() });
  slide.addText('Across 12 patients', { x: 9.55, y: 2.05, w: 2.6, h: 0.35, fontFace: 'Cambria', fontSize: 19, bold: true, color: C.ink, align: 'center', margin: 0 });
  slide.addText('0.014', { x: 9.7, y: 2.75, w: 2.3, h: 0.7, fontFace: 'Cambria', fontSize: 38, bold: true, color: C.teal, align: 'center', margin: 0 });
  slide.addText('mg/dL lower average RMSE', { x: 9.65, y: 3.48, w: 2.4, h: 0.32, fontFace: 'Arial', fontSize: 11, bold: true, color: C.ink, align: 'center', margin: 0 });
  slide.addText('No significance claim; predictive performance is effectively matched at the reported precision.', { x: 9.55, y: 4.25, w: 2.6, h: 0.65, fontFace: 'Arial', fontSize: 10.5, color: C.muted, align: 'center', margin: 0.04, fit: 'shrink' });
  slide.addText('Ldistill = alpha LGT + beta LMSE', { x: 3.35, y: 5.92, w: 6.7, h: 0.4, fontFace: 'Cambria', fontSize: 20, italic: true, color: C.coral, align: 'center', margin: 0 });
  addSource(slide, 'Sources: DiabLLM distillation methodology and 12-patient distillation results table.');
  addNotes(slide, 'The teacher is Time-LLM BERT and the student is TinyBERT. Training combines ground-truth regression loss with alignment to the teacher predictions. Across all twelve patients, average RMSE is 20.967 for the teacher and 20.953 for the distilled model. Do not call the student statistically better; the supported conclusion is preservation of accuracy at the reported precision.');
}

// 12. Efficiency
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Compression changes the deployment trade-off by orders of magnitude', 'Inference efficiency');
  const cats = ['Parameters', 'Disk size', 'Latency'];
  const teacher = [282.4, 1077, 208218.8];
  const student = [45.0, 172, 1271.3];
  const ratios = [6.28, 6.26, 163.78];
  ratios.forEach((r, i) => {
    const x = 0.78 + i * 4.13;
    slide.addShape(pptx.ShapeType.roundRect, { x, y: 1.65, w: 3.65, h: 4.55, rectRadius: 0.08, fill: { color: i === 2 ? C.navy : C.white }, line: { color: i === 2 ? C.navy : C.gray }, shadow: shadow() });
    slide.addText(cats[i], { x: x + 0.3, y: 2.02, w: 3.05, h: 0.35, fontFace: 'Cambria', fontSize: 20, bold: true, color: i === 2 ? C.white : C.ink, align: 'center', margin: 0 });
    slide.addText(`${r.toFixed(i === 2 ? 1 : 1)}x`, { x: x + 0.45, y: 2.72, w: 2.75, h: 0.7, fontFace: 'Cambria', fontSize: 38, bold: true, color: i === 2 ? C.amber : C.teal, align: 'center', margin: 0 });
    slide.addText(i === 2 ? 'lower' : 'smaller', { x: x + 0.7, y: 3.42, w: 2.25, h: 0.25, fontFace: 'Arial', fontSize: 11, bold: true, color: i === 2 ? C.white : C.ink, align: 'center', margin: 0 });
    const t = i === 0 ? '282.4M to 45.0M' : i === 1 ? '1077 MB to 172 MB' : '208218.8 ms to 1271.3 ms';
    slide.addText(t, { x: x + 0.45, y: 4.18, w: 2.75, h: 0.4, fontFace: 'Arial', fontSize: 11, color: i === 2 ? 'C7D7DF' : C.muted, align: 'center', margin: 0, fit: 'shrink' });
    slide.addText('BERT teacher to distilled TinyBERT', { x: x + 0.42, y: 4.9, w: 2.8, h: 0.38, fontFace: 'Arial', fontSize: 9.5, color: i === 2 ? C.white : C.darkGray, align: 'center', margin: 0 });
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 3.3, y: 6.42, w: 6.75, h: 0.38, rectRadius: 0.06, fill: { color: 'FBE8E3' }, line: { color: 'F3C8BE' } });
  slide.addText('Boundary: measured under simulated constraints; physical edge-device deployment was not performed.', { x: 3.52, y: 6.51, w: 6.3, h: 0.18, fontFace: 'Arial', fontSize: 9.5, bold: true, color: C.red, align: 'center', margin: 0, fit: 'shrink' });
  addSource(slide, 'Source: DiabLLM inference metrics and limitations. Ratios calculated from reported values.');
  addNotes(slide, 'This is the practical payoff of distillation. The student has about one sixth the parameters and disk size, and the reported latency is roughly one hundred sixty-four times lower. However, repeat the boundary on the slide: these measurements substantiate feasibility under simulated constraints; the study did not deploy on a physical pump, phone, or Raspberry Pi.');
}

// 13. Limitations
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'Strong retrospective results do not establish clinical readiness', 'Validity boundaries');
  const items = [
    ['DATA', 'Two small public T1DM cohorts', 'Broader demographic and device diversity remains untested.'],
    ['INPUTS', 'CGM-only forecasting', 'Meals, insulin, activity, and uncertainty are not integrated.'],
    ['ROBUSTNESS', 'Simulated corruption', 'Real sensor failures may differ from the gain, noise, and dropout model.'],
    ['DEPLOYMENT', 'No physical edge validation', 'Latency, energy, thermals, and reliability require hardware testing.'],
    ['CLINICAL', 'Retrospective error grids', 'No prospective trial or closed-loop dosing evaluation.'],
    ['TRUST', 'Complex model representations', 'Interpretability, bias, privacy, and calibration remain open.'],
  ];
  items.forEach((it, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.72 + col * 4.18;
    const y = 1.55 + row * 2.35;
    slide.addShape(pptx.ShapeType.roundRect, { x, y, w: 3.72, h: 1.95, rectRadius: 0.08, fill: { color: row === 0 ? C.white : C.pale }, line: { color: i === 4 ? 'E9B3A5' : C.gray, width: i === 4 ? 2 : 1 }, shadow: shadow() });
    slide.addText(it[0], { x: x + 0.25, y: y + 0.22, w: 1.05, h: 0.2, fontFace: 'Arial', fontSize: 8.5, bold: true, color: [C.teal, C.coral, '8E6A98', C.amber, C.red, C.mint][i], charSpacing: 1, margin: 0 });
    slide.addText(it[1], { x: x + 0.25, y: y + 0.62, w: 3.2, h: 0.38, fontFace: 'Cambria', fontSize: 17, bold: true, color: C.ink, margin: 0, fit: 'shrink' });
    slide.addText(it[2], { x: x + 0.25, y: y + 1.13, w: 3.2, h: 0.52, fontFace: 'Arial', fontSize: 10, color: C.muted, margin: 0, fit: 'shrink' });
  });
  slide.addText('The next milestone is not another retrospective benchmark - it is prospective, calibrated, hardware-validated evidence.', { x: 1.25, y: 6.42, w: 10.85, h: 0.35, fontFace: 'Cambria', fontSize: 17, bold: true, color: C.teal, align: 'center', margin: 0, fit: 'shrink' });
  addSource(slide, 'Source: DiabLLM discussion, limitations, and conclusion.');
  addNotes(slide, 'Be explicit about what the paper does not establish. The datasets are small, CGM is the only input, corruption is simulated, edge results are not from physical deployment, and error grids are retrospective. The proper next step is prospective validation with calibrated uncertainty and real hardware, not simply another leaderboard comparison.');
}

// 14. Takeaways
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addText('DiabLLM establishes a promising but bounded foundation', { x: 0.72, y: 0.68, w: 11.85, h: 0.75, fontFace: 'Cambria', fontSize: 30, bold: true, color: C.white, align: 'center', margin: 0, fit: 'shrink' });
  const take = [
    ['1', 'Architecture', 'Chronos tokenizes glucose; Time-LLM reprograms patches for a frozen LLM.'],
    ['2', 'Evidence', 'Fine-tuned Time-LLM leads the tested 30-minute benchmark and error-grid outcomes.'],
    ['3', 'Practicality', 'Denoising and distillation address corrupted inputs and computational cost.'],
  ];
  take.forEach((t, i) => {
    const x = 0.82 + i * 4.15;
    slide.addShape(pptx.ShapeType.roundRect, { x, y: 2.02, w: 3.7, h: 3.4, rectRadius: 0.08, fill: { color: ['173149', '123B42', '3D2E27'][i] }, line: { color: ['2E536B', '2A6F73', '735445'][i] }, shadow: shadow() });
    slide.addShape(pptx.ShapeType.ellipse, { x: x + 1.43, y: 2.42, w: 0.84, h: 0.84, fill: { color: [C.teal, C.coral, C.amber][i] }, line: { color: [C.teal, C.coral, C.amber][i] } });
    slide.addText(t[0], { x: x + 1.43, y: 2.61, w: 0.84, h: 0.24, fontFace: 'Cambria', fontSize: 17, bold: true, color: i === 2 ? C.navy : C.white, align: 'center', margin: 0 });
    slide.addText(t[1], { x: x + 0.38, y: 3.62, w: 2.94, h: 0.38, fontFace: 'Cambria', fontSize: 21, bold: true, color: C.white, align: 'center', margin: 0 });
    slide.addText(t[2], { x: x + 0.38, y: 4.27, w: 2.94, h: 0.75, fontFace: 'Arial', fontSize: 11.5, color: 'D3E0E7', align: 'center', margin: 0.04, fit: 'shrink' });
  });
  slide.addText('Next: prospective evaluation • uncertainty calibration • multimodal context • real edge hardware', { x: 1.15, y: 6.25, w: 11.0, h: 0.35, fontFace: 'Arial', fontSize: 13, bold: true, color: C.amber, align: 'center', margin: 0 });
  slide.addText('Amirhossein Mahmoudi  |  github.com/ammahmoudi/diab-llm', { x: 3.65, y: 6.88, w: 6.05, h: 0.2, fontFace: 'Arial', fontSize: 9, color: 'AFC2CD', align: 'center', margin: 0 });
  addNotes(slide, 'Close with three takeaways. First, the two architectures solve the continuous-to-language interface differently. Second, adaptation makes Time-LLM the strongest tested model in this protocol. Third, denoising and distillation move the system toward practical use. The remaining work is prospective clinical and hardware validation.');
}

// 15. References
{
  const slide = pptx.addSlide('CONTENT');
  addTitle(slide, 'References and discussion', 'Selected sources');
  const refs = [
    ['DiabLLM', 'A. Mahmoudi et al., IEEE Journal of Biomedical and Health Informatics, vol. 30, no. 8, 2026.', 'doi.org/10.1109/JBHI.2026.3658588'],
    ['Time-LLM', 'M. Jin et al., Time Series Forecasting by Reprogramming Large Language Models, ICLR 2024.', 'arXiv:2310.01728'],
    ['Chronos', 'A. F. Ansari et al., Learning the Language of Time Series, TMLR 2024.', 'arXiv:2403.07815'],
    ['OhioT1DM', 'C. Marling and R. Bunescu, The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020.', 'CEUR-WS Vol. 2675'],
    ['D1NAMO', 'F. Dubosson et al., The D1NAMO dataset: a multi-modal dataset for research on non-invasive type 1 diabetes management.', 'Informatics in Medicine Unlocked, 2018'],
    ['Code', 'Official DiabLLM implementation and experiment pipeline.', 'github.com/ammahmoudi/diab-llm'],
  ];
  refs.forEach((r, i) => {
    const y = 1.48 + i * 0.87;
    slide.addShape(pptx.ShapeType.roundRect, { x: 0.72, y, w: 11.85, h: 0.68, rectRadius: 0.05, fill: { color: i % 2 === 0 ? C.white : C.pale }, line: { color: C.gray } });
    slide.addText(r[0], { x: 0.98, y: y + 0.14, w: 1.35, h: 0.22, fontFace: 'Arial', fontSize: 11, bold: true, color: i < 3 ? C.teal : C.coral, margin: 0 });
    slide.addText(r[1], { x: 2.35, y: y + 0.1, w: 7.35, h: 0.35, fontFace: 'Arial', fontSize: 9.5, color: C.darkGray, margin: 0, fit: 'shrink' });
    slide.addText(r[2], { x: 9.85, y: y + 0.14, w: 2.45, h: 0.22, fontFace: 'Arial', fontSize: 8.5, color: C.muted, align: 'right', margin: 0, fit: 'shrink' });
  });
  slide.addShape(pptx.ShapeType.roundRect, { x: 3.95, y: 6.72, w: 5.4, h: 0.4, rectRadius: 0.06, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addText('Questions?', { x: 5.25, y: 6.8, w: 2.8, h: 0.2, fontFace: 'Cambria', fontSize: 15, bold: true, color: C.white, align: 'center', margin: 0 });
  addNotes(slide, 'Use this as the discussion slide. If asked about architecture details, return to slide three. If asked about clinical readiness, return to slide thirteen. If asked about model efficiency, distinguish training time, the reported inference benchmark, and actual hardware deployment.');
}

pptx.writeFile({ fileName: path.join(__dirname, 'DiabLLM_paper_presentation_Amirhossein_Mahmoudi.pptx') });
