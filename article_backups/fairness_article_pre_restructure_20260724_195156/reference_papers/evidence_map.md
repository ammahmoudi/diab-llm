# Evidence Map for the Introduction and Related Work

## Scope and search outcome

The search covered direct fairness-under-KD studies, clinical fairness, diabetes/CGM equity, ECG and biosignal representativeness, fair training, LLM fairness, and agent-level fairness. Searches used the local Paper Search, PubMed Search, arXiv, Crossref, and OpenAlex integrations. Candidate records were retained only after title/author/venue verification; open PDFs were parsed and title-matched before being treated as full-text evidence. The library contains 36 unique paper records, with 30 verified local PDFs. The seven-paper update adds direct OhioT1DM KD precedents, event-aware glucose evaluation, ECG demographic-disparity studies, and recent BERT/LLM compression-fairness work.

The strongest literature story is focused rather than encyclopedic:

1. KD can preserve, amplify, or sometimes reduce bias; fairness transfer is not guaranteed.
2. Clinical models require disaggregated, clinically meaningful endpoint evaluation because average utility can hide subgroup harm.
3. The 70 mg/dL threshold is clinically grounded by CGM consensus guidance.
4. OhioT1DM has direct KD precedents, but they report aggregate forecasting or clinical-accuracy criteria rather than protected-group fairness.
5. Diabetes technology has documented access disparities, while direct fairness research on glucose forecasting remains sparse.
6. ECGs encode demographic information, direct ECG studies find demographic performance disparities, and public biosignal datasets often underreport demographic composition.
7. LLM and agent fairness are useful as brief provenance/system-level context, not as the main related-work axis.

## Claim-to-source mapping

| Manuscript claim | Best references | Evidence and qualification | Suggested location |
| --- | --- | --- | --- |
| KD success on average utility does not guarantee fairness preservation. | `mohammadshahi2025left`, `gupta2022mitigating`, `chai2022fairness` | Mohammadshahi and Ioannou directly compare teacher, distilled student, and non-distilled student fairness; Gupta et al. show gender bias can transfer through language-model KD; Chai et al. treat fairness as transferable knowledge without target demographics. | Introduction opening; Related Work, KD fairness |
| Compression can sometimes reduce measured bias, but results are metric- and task-dependent. | `xu2022compression`, `goncalves2023compression` | Both papers report reductions in selected NLP social-bias or toxicity measures after compression; Goncalves and Strubell also document language-model-score trade-offs and benchmark limitations. | Related Work, KD fairness |
| BERT-family group-fairness KD remains an active but preliminary area. | `fasth2025improving` | The preprint studies BERT-to-DistilBERT on MultiNLI using uncertainty from early exits; it calls for broader datasets, subgroup definitions, and tuning. | Related Work, recent direct method context |
| Fairness-aware KD spans heterogeneous tasks and fairness definitions. | `gupta2022mitigating`, `zhu2023debias`, `liao2024impartial`, `zhao2024improving`, `gao2025maintaining`, `li2025dualteacher`, `masroor2024fair` | These papers cover demographic gender bias, exposure fairness, teacher/data imbalance, robust class fairness, class-incremental balance, graph fairness, and medical imaging. Do not imply that every paper studies protected-group fairness. | Related Work, KD taxonomy |
| Clinical AI should report subgroup outcomes, not only average performance. | `chen2023algorithmic`, `rajkomar2018ensuring`, `pfohl2022comparison`, `seyyedkalantari2021underdiagnosis` | Clinical fairness reviews and empirical studies show that aggregate performance can conceal weak or harmful subgroup behavior. Pfohl et al. also caution that robust optimization is not uniformly superior to standard learning. | Introduction motivation; Related Work, clinical fairness |
| Sex/gender dimensions must be considered explicitly in biomedical AI. | `cirillo2020sex`, `chen2023algorithmic` | Cirillo et al. review sex/gender gaps and warn that ignoring them can produce suboptimal or discriminatory outcomes. | Introduction clinical motivation; limitations |
| The hypoglycemia event threshold of 70 mg/dL is clinically grounded. | `battelino2019clinical` | International consensus defines time below range level 1 as glucose below 70 mg/dL and standardizes CGM interpretation. | Introduction and Methods problem formulation |
| KD has already been evaluated for OhioT1DM forecasting. | `hameed2020investigating`, `sun2026futureaware` | Hameed and Kleinberg compare OpenAPS-to-OhioT1DM mimic learning; Sun et al. distill privileged future disturbances on OhioT1DM and AZT1D. Neither reports protected-group fairness. Novelty must therefore be fairness-aware KD, not KD itself. | Introduction and Related Work positioning |
| RMSE alone can miss clinically critical glycemic-event behavior. | `wolff2025criteria` | The OhioT1DM study shows that the lowest-RMSE predictor can perform poorly on glycemic events and proposes a clinically weighted composite metric. This supports event-aware evaluation but is not demographic fairness evidence. | Introduction motivation; Related Work, glucose context |
| Diabetes technology operates within an inequitable access context. | `venkatesh2023disparities` | In the T1D Exchange, CGM use was lower for non-Hispanic Black and Hispanic women and for groups with lower income, education, and Medicaid coverage. This supports context, not a claim that OhioT1DM itself represents these populations. | Related Work, clinical/domain context |
| Direct demographic-fairness evidence for BG forecasting is limited. | Search result plus `venkatesh2023disparities`, `battelino2019clinical` | Searches found strong CGM threshold and access-equity literature but little direct work on demographic fairness of glucose forecasts under KD. Phrase as a scoped search finding, not proof of absence. | Related Work positioning; Discussion |
| ECG representations contain demographic information. | `attia2019age` | The ECG study estimates age and sex from standard 12-lead ECGs, motivating subgroup auditing even when demographics are not model targets. Full text was not locally available, so rely on verified bibliographic record/abstract unless obtained through institutional access. | Related Work, ECG context |
| ECG models can exhibit demographic and intersectional performance disparities. | `kaur2024disparities`, `galanty2025sex` | Kaur et al. find age/race/sex disparities in heart-failure prediction and limited benefit from balancing; Galanty et al. find sex gaps in ECG classification even with balanced training. Neither uses MIT-BIH or KD. | Related Work, ECG context |
| Fairness claims from public biosignal benchmarks are constrained by demographic reporting. | `sauer2024demographic` | The Lancet Digital Health analysis specifically audits demographic reporting across PhysioNet. Full text was not locally retained because an MCP-proposed repository PDF was a false match and was removed. | Related Work/limitations for MIT-BIH |
| ECG transfer is technically plausible but does not establish demographic fairness. | `weimann2021ecgtransfer`, `moody2001mitbih` | Transfer-learning work supports the technical ECG context; it does not justify extending the primary BG fairness claim. | Secondary ECG protocol and limitations |
| LLM fairness has mature benchmark traditions. | `nadeem2021stereoset`, `dhamala2021bold`, `parrish2022bbq` | These benchmarks cover stereotypical associations, open-ended generation, and biased question answering. Use briefly to contextualize pretrained-LM provenance. | Related Work, one compact sentence |
| Fairness can emerge at the surrounding agent/system level. | `lee2025interactional` | The AIES paper evaluates interactional fairness in LLM multi-agent systems. This is broader context only; it is not directly comparable to event-level clinical fairness. | Optional final Related Work sentence |
| Data exposure and optimization are credible alternative fairness mechanisms. | `sagawa2020distributionally`, `liu2021jtt`, `idrissi2022balancing`, `pfohl2022comparison` | Group DRO, JTT, and balancing show that worst-group outcomes depend strongly on regularization, hard-example identification, and data balancing. Pfohl et al. provide the clinical caution that no method uniformly dominates. | Related Work, fair training and EBTD rationale |

## Recommended Related Work structure

### 1. Fairness under knowledge distillation

Lead with direct evidence that KD changes fairness rather than assuming compression is neutral. Use Mohammadshahi and Ioannou as the closest audit-style precedent, then contrast mitigation approaches: counterfactual teacher/student modification, demographic-free fairness transfer, exposure/ranking fairness, teacher/data imbalance, robust class fairness, and graph/medical applications. End by distinguishing this paper's thresholded regression event and teacher-to-student decomposition.

### 2. Fairness and subgroup robustness in clinical prediction

Use Chen et al. and Rajkomar et al. for the clinical fairness framework. Use Pfohl et al. and Seyyed-Kalantari et al. to establish that average performance can hide weak subgroup outcomes and that mitigation methods require empirical validation. Connect this literature to event-level true-positive rates, not to claims of clinical deployment readiness.

### 3. Blood-glucose and ECG context

Use Hameed and Kleinberg and Sun et al. to acknowledge direct OhioT1DM KD precedents, Wolff et al. for the limitation of RMSE-only evaluation, Battelino et al. for the 70 mg/dL event threshold, and Venkatesh et al. for diabetes-technology disparities. State that the search found sparse direct work on demographic fairness in BG forecasting. For ECG, use Attia et al. to show demographic information is encoded in ECG signals, Kaur et al. and Galanty et al. for direct performance-disparity evidence, Sauer et al. to motivate caution about biosignal demographics, and Weimann and Conrad only for technical transfer-learning context. Preserve the manuscript's bounded-transfer language.

### 4. Broader model and system fairness

If space permits, use one sentence on LLM fairness benchmarks and one on interactional fairness in agents. Do not let this material displace the more relevant clinical and KD literature.

## Important corrections and cautions

- `What is Left After Distillation?` is peer-reviewed in TMLR (March 2025), not merely an arXiv preprint. The library metadata has been corrected.
- `MAPS` is a multilingual agent performance/security benchmark, not an accessibility-fairness benchmark. It should not anchor an agent-fairness claim.
- “Fairness” in the CIL and adversarial-robustness KD papers is class-wise/structural, not necessarily demographic fairness.
- CGM access disparities do not prove algorithmic unfairness in a glucose forecasting model; they establish deployment and representation context.
- The PhysioNet demographic-reporting paper is bibliographically verified, but its full text was not retained locally after a false repository match was detected and removed.
- The ECG age/sex paper is bibliographically verified, but no lawful local full-text copy was obtained in this search.
- The article currently has six pages. A focused Related Work expansion is preferable to adding every adjacent LLM/agent citation.
- `Future-Aware Blood Glucose Forecasting` is a direct 2026 OhioT1DM KD precedent; the manuscript must not claim novelty for applying KD to OhioT1DM itself.
- Hameed and Kleinberg and Sun et al. list participant- or subject-level results but do not report protected-group fairness metrics.
- Kaur et al. and Galanty et al. provide direct ECG disparity evidence but do not use MIT-BIH, BERT/TinyBERT, or KD.
- Fallback downloads for Wolff et al. and Kaur et al. failed title verification and were deleted; their notes use verified publisher/Crossref abstracts only.
