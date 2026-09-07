# DiabLLM presentation outline and evidence ledger

## Presentation specification

- Presenter and first author: Amirhossein Mahmoudi
- Format: self-contained editable 1920x1080 HTML (with an existing PPTX companion)
- Intended use: technical, reading-capable 20-slide paper presentation
- Thesis: DiabLLM shows that two language-inspired forecasting strategies can be adapted to CGM-only blood-glucose prediction, with Time-LLM providing the strongest accuracy and distillation recovering practical efficiency, while deployment and clinical validation remain future work.
- Primary source: the published DiabLLM manuscript (`DOI:10.1109/JBHI.2026.3658588`) and repository article source.

## Slide plan

| # | Assertion title | Primary evidence | Speaker-note focus |
| --- | --- | --- | --- |
| 1 | DiabLLM adapts language-inspired models to glucose forecasting | `diab-llm-article/main.tex`; `abstract.tex` | Introduce the paper, presenter, venue, and core question. |
| 2 | A transformer predicts the next token by attending to the context | Standard transformer mechanism; conceptual illustration | Establish tokenization, embeddings, self-attention, and next-token probabilities before time-series adaptations. |
| 2 | Thirty minutes of CGM history must anticipate the next 30-45 minutes | `abstract.tex`; `experiments.tex` | Motivate forecasting for T1DM and state the CGM-only protocol. |
| 3 | Time-LLM aligns numerical patches with a frozen language-model embedding space | `models/time_llm.py`; `methodology.tex`; Jin et al. Fig. 2-4 | Trace Prompt-as-Prefix, vocabulary compression, cross-attention reprogramming, the frozen backbone, and trainable forecast head. |
| 4 | Chronos casts glucose regression as probabilistic next-token generation | `models/chronos/src/chronos/chronos.py`; `scripts/chronos/config_generator.py`; Ansari et al. Sec. 3 | Trace corpus augmentation, mean scaling, uniform quantization, T5 cross-entropy, sampling, and dequantization. |
| 6 | A denoising autoencoder reconstructs clean CGM from corrupted input | `methodology.tex` pp. 4-5 | Explain corrupted-clean training pairs, bottleneck reconstruction, and the deployment boundary. |
| 7 | A compact student learns from both labels and a larger teacher | `methodology.tex`; `results.tex` | Explain the teacher, two-target objective, student, and teacher-free deployment before the full system graph. |
| 5 | DiabLLM cleans corrupted CGM, transfers teacher behavior, then deploys only the compact path | `methodology.tex` pp. 4-5; `results.tex` pp. 8-9 | Join denoising, BERT-to-TinyBERT KD, and teacher-free deployment in one interactive computation graph. |
| 6 | DiabLLM evaluates adaptation, robustness, transfer, and compression | `methodology.tex`; `results.tex` | Define the exact training, corruption, transfer, compression, and evaluation protocols. |
| 6 | Two public T1DM cohorts test personalization and external transfer | `experiments.tex` | OhioT1DM: 12 adults, 8 weeks, 5-minute readings; D1NAMO: patients 1-7 used from a 9-person cohort. |
| 7 | Task-specific adaptation produces the largest accuracy gain | `results.tex`; `tab_models_performance_comparison_30min.tex` | Emphasize matched protocol comparisons and avoid treating all literature rows as identical protocols. |
| 8 | Fine-tuned Time-LLM reaches 16.10 mg/dL average RMSE | `tab_timellm_fine_tuned_30min.tex`; `tab_chronos_fine_tuned_30min.tex` | Compare Time-LLM LLaMA 8-layer, Chronos Base, DRL, and fusion baseline at 30 minutes. |
| 9 | Error-grid analysis favors Time-LLM at the cohort level | `tab_ceg_results.tex`; `tab_seg_results.tex` | CEG Zone A: 99.58% vs 98.43%; SEG None: 97.15% vs 92.94%. Clarify that these metrics do not replace prospective clinical validation. |
| 10 | Denoising recovers much of the accuracy lost to corrupted CGM | `results.tex`; `tab_robustness_comparison.tex` | Average 45-minute RMSE changes: Time-LLM 51.067 to 19.119; Chronos 47.453 to 25.963. |
| 11 | Cross-patient transfer shows limited degradation in the tested pairs | `tab_cross_patient_performance.tex`; `results.tex` | Show all four train/test combinations for patients 570 and 584; avoid extrapolating to unseen populations. |
| 12 | Distillation preserves BERT-level accuracy with a 45M-parameter student | `tab_distillation_results.tex`; `methodology.tex` | Teacher RMSE 20.967 vs distilled 20.953 across 12 patients. Explain ground-truth plus teacher-alignment loss. |
| 13 | Compression changes the deployment trade-off by orders of magnitude | `edge_deployment.tex`; `limitations.tex` | TinyBERT: 45M parameters, 172 MB, 1271.3 ms; BERT: 282.4M, 1077 MB, 208218.8 ms. Measurements were under simulated constraints, not physical edge deployment. |
| 14 | Strong retrospective results do not establish clinical readiness | `limitations.tex`; `conclusion.tex` | Discuss two small public cohorts, CGM-only input, corruption simulation, interpretability, prospective evaluation, and hardware validation. |
| 15 | DiabLLM establishes a promising but bounded foundation | `conclusion.tex`; `abstract.tex` | Close with three takeaways and the next validation steps. |
| 16 | References and discussion | `references.bib`; canonical paper identifiers | Invite questions; retain canonical identifiers and repository link. |

## Evidence ledger

| Claim | Exact evidence | Source/status |
| --- | --- | --- |
| Input and horizons | Six historical CGM values over 30 minutes predict 30- and 45-minute horizons. | `abstract.tex`; project protocol |
| Time-LLM input transformation | RevIN normalization; patch length 6, stride 8, two patches; patch dimension 32; 8-head reprogramming; 1,000 source prototypes; prompt max length 2,048. | `models/time_llm.py`; `scripts/time_llm/config_generator.py`; implementation fact |
| Time-LLM trainable boundary | The pretrained LLM parameters are frozen; patch embedding, vocabulary mapping, cross-attention bridge, and output projection receive gradients. | `models/time_llm.py`; `methodology.tex`; implementation/manuscript fact |
| Time-LLM prompt | Dataset context + task instruction + normalized-window min, max, median, summed-difference trend, and top-five FFT lags. | `models/time_llm.py`; implementation fact |
| Chronos tokenization | Mean absolute scaling over observed context; uniform binning; 4,096 total tokens including special tokens; PAD for NaN and EOS for seq2seq. | `models/chronos/src/chronos/chronos.py`; Ansari et al. Sec. 3 |
| Chronos DiabLLM configuration | Configured tokenizer limits -30 to 30, context 6, horizon 6/9, batch 8, learning rate 0.001; inference experiments use one sample. | `scripts/chronos/config_generator.py`; experiment configs; project configuration |
| Chronos objective | Token-level categorical cross-entropy updates all T5 parameters; the loss is not distance-aware. | Ansari et al. Sec. 3.2; `methodology.tex` |
| OhioT1DM cohort | 12 adults with T1DM, 5-minute CGM readings over 8 weeks. | `experiments.tex`; dataset description |
| D1NAMO validation | Dataset contains 9 individuals with T1DM over 4 weeks; patients 1-7 used in this evaluation. | `experiments.tex`; dataset description |
| Best 30-minute model | Fine-tuned Time-LLM LLaMA 7B (8 layers): average RMSE 16.10 mg/dL and MAE 10.00 mg/dL. | `tab_timellm_fine_tuned_30min.tex`; project result |
| Chronos 30-minute result | Fine-tuned Chronos T5 Base: average RMSE 20.89 mg/dL and MAE 13.05 mg/dL. | `tab_chronos_fine_tuned_30min.tex`; project result |
| Matched baselines | Deep RL RMSE 18.32 mg/dL; LSTM+WaveNet+GRU RMSE 21.90 mg/dL under the paper's stated comparable configuration. | `tab_models_performance_comparison_30min.tex`; external baselines reported in manuscript |
| Improvement framing | Manuscript reports up to 27% lower RMSE and 37% lower MAE relative to state-of-the-art methods. | `abstract.tex`, `conclusion.tex`; author interpretation |
| CEG cohort result | Time-LLM Zone A 99.58%; Chronos Zone A 98.43%; neither has Zone E errors in the table. | `tab_ceg_results.tex`; project result |
| SEG cohort result | Time-LLM None 97.15%; Chronos None 92.94%. | `tab_seg_results.tex`; project result |
| Denoising result | Time-LLM average RMSE 51.067 noisy to 19.119 denoised; Chronos 47.453 to 25.963 at 45 minutes. | `tab_robustness_comparison.tex`; project result |
| Corruption mechanism | Gain alpha ~ N(1, 0.1), offset beta ~ N(0, 6), AR(1) rho=0.5 with innovation sigma=6, and about 10% cumulative dropout in blocks of at most six. | `methodology.tex`; simulated robustness protocol |
| Autoencoder objective and handoff | Corrupted inputs are reconstructed as z; the denoising autoencoder minimizes squared reconstruction error `‖z-x‖²`; z then feeds Time-LLM or Chronos. | `methodology.tex` pp. 4-5; project protocol |
| Cross-patient result | Time-LLM: 570->570 16.81, 570->584 26.63, 584->570 17.63, 584->584 27.23 RMSE. Chronos: 23.69, 31.68, 22.09, 30.76. | `tab_cross_patient_performance.tex`; restricted two-patient transfer experiment |
| Distillation accuracy | Teacher average RMSE 20.967, MAE 13.109; distilled model RMSE 20.953, MAE 13.067 across 12 patients. | `tab_distillation_results.tex`; project result |
| Distillation optimization | BERT 12x768 teacher to TinyBERT 4x312 student; alpha=0.3 and beta=0.3 selected on validation MAE; student trained for 10 epochs. | `methodology.tex`; `results.tex`; project protocol |
| Distillation size and latency | BERT teacher: 282.4M parameters, 1077 MB, 208218.8 ms. Distilled TinyBERT: 45.0M, 172 MB, 1271.3 ms. | `edge_deployment.tex`; simulated-constraint inference measurement |
| Deployment boundary | Actual deployment on physical edge hardware was not conducted. | `limitations.tex`; explicit limitation |

## Claim boundaries

- Do not describe the system as clinically deployed, prospectively validated, or ready for insulin-dosing decisions.
- Do not claim statistical significance unless a specific test is present; none is used in this presentation.
- Do not treat literature results with different histories, cohorts, inputs, or splits as matched comparisons.
- Use "clinical error-grid performance" rather than "clinical safety proven."
- The DiabLLM PDF in this folder is an author-manuscript rendering; the canonical published record is DOI `10.1109/JBHI.2026.3658588`.
- The architecture diagrams are native HTML/SVG reconstructions. They encode the paper and repository computation graphs but do not reuse the source figures as raster images.
- Architecture and objective equations use native MathML so fractions, roots, summations, subscripts, and superscripts render semantically rather than as plain-text approximations.
- Architecture slides include focus controls: Time-LLM can isolate the prompt or training path; Chronos can isolate full-parameter training or probabilistic prediction. Active nodes reveal sequentially and directional arrows animate only on the selected path.
- The architecture visuals are now tangible computation graphs rather than uniform card rows: concrete CGM traces, source datasets, prompt text, patch windows, token IDs, text-prototype anchors, merge nodes, probability histograms, loss feedback loops, and forecast trajectories encode each transformation visually.
- Both architecture slides support mode-aware step walkthroughs (`previous`, `next`, and `auto`). Time-LLM includes separate prompt, training, and prediction explanations; Chronos separates optimization from five-stage probabilistic inference. Each step highlights the current transformation, retains prior context, and dims future stages.
- The complete DiabLLM graph adds `DENOISING`, `KD TRAINING`, and `DEPLOYMENT` focus modes. It explicitly removes the teacher and clean reconstruction target from deployment.
- Numerical traces on the three architecture slides are labeled as illustrative pedagogical examples. They are not patient records and are not paper-reported per-window predictions.
- Chronos supports probabilistic multi-path forecasts, but the reported DiabLLM inference configurations use `num_samples=1`; do not present empirical intervals as evaluated outcomes.
- Time-LLM's prompt statistics are computed after normalization in the repository implementation; visible values are therefore normalized-window statistics, not raw mg/dL summaries.
