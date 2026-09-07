# DiabLLM: An LLM-Based Framework for Blood Glucose Prediction in Type 1 Diabetes

- Source: [DOI:10.1109/JBHI.2026.3658588](https://doi.org/10.1109/JBHI.2026.3658588)
- PDF: [diabllm.pdf](diabllm.pdf)
- Pages: 17
- Conversion: automatic text extraction; equations, tables, and multi-column reading order may need manual cleanup.

## Page 1

DiabLLM: An LLM-Based Framework for Blood Glucose
 Prediction in Type 1 Diabetes

 Amirhossein Mahmoudi, Ghazal Farahani, Peter Domanski, Bahar Farahani, Farshad Firouzi, and Krishnendu
 Chakrabarty

 Author-manuscript rendering from the repository LaTeX source. Published in IEEE Journal of Biomedical and Health Informatics, vol.
 30, no. 8, pp. 6446-6459, 2026. DOI: 10.1109/JBHI.2026.3658588.

Abstract
Accurate Blood Glucose (BG) prediction is essential for enabling glycemic control in individuals with Type 1 Diabetes
Mellitus (T1DM), particularly within Smart and Connected Health (SCH) systems that integrate Continuous Glucose
Monitoring (CGM) and automated insulin delivery. The adaptability of Large Language Models (LLMs) provides a promising foundation for unified, fine-tunable forecasting models. We introduce DiabLLM, a framework based on two recent LLM-based architectures: Time-LLM, which incorporates a lightweight projection layer and alignment techniques to transform time-series data into embeddings interpretable by pre-trained LLMs, and Chronos, which employs time-series-aware tokenization and quantization to convert continuous inputs into discrete sequences for forecasting.
Both models process 30-minute sequences of six historical BG values and predict 30- and 45-minute horizons.
Experimental results on the OhioT1DM and D1NAMO datasets demonstrate that DiabLLM outperforms state-of-the-art baselines, including a Deep Reinforcement Learning model and an ensemble of LSTM, GRU, and WaveNet, achieving up to 27% improvement in RMSE and 37% in MAE. To enhance robustness to noisy and missing input data, a denoising autoencoder was employed for input reconstruction, yielding improved predictive performance. In addition, knowledge distillation was shown to significantly compress the model, making it a practical candidate for efficient deployment on resource-constrained edge devices without compromising accuracy.

1. Introduction
Diabetes is a significant and growing global health challenge, affecting an estimated 537 million adults as of 2021 - a number projected to rise substantially in the coming decades [ogurtsova2022idf]. It is characterized by chronic hyperglycemia due to impaired insulin secretion, action, or both, and is associated with serious long-term complications involving the cardiovascular, renal, ocular, and nervous systems [american2014diagnosis,deshpande2008epidemiology].
In Type 1 Diabetes Mellitus (T1DM), the autoimmune destruction of pancreatic -cells necessitates lifelong insulin therapy and continuous blood glucose (BG) management [american2014diagnosis]. Smart and Connected Health (SCH) systems, enabled by wearable Internet of Things (IoT) sensors and Artificial Intelligence (AI), have advanced real-time monitoring and personalized disease management. However, while SCH systems excel at data collection and analytics, they often lack intelligent, interactive components that support users more naturally and adaptively. The emergence of
Large Language Models (LLMs) has introduced new opportunities to build Virtual Health Assistants (VHAs) that serve as personal health companions, capable of offering real-time guidance, contextual reasoning, and conversational interaction. Integrating LLMs into SCH frameworks holds the potential to transform diabetes care by enabling proactive, user-centric support that bridges the gap between raw data and meaningful, personalized insights. A key advancement in
T1DM management, supported by SCH technologies, is the development of closed-loop insulin delivery systems, which integrate Continuous Glucose Monitoring (CGM) with automated insulin pumps to deliver insulin based on real-time BG readings [bruttomesso2019toward]. Accurate BG predictions are essential to optimize glycemic control in these systems.
However, achieving reliable prediction remains challenging due to noisy, inconsistent sensor readouts, affected by calibration errors, missing data, and patient non-compliance. Models must also generalize across diverse patient profiles to ensure clinical applicability. BG prediction approaches typically fall into two categories: general models trained on large-scale CGM datasets, and personalized models adapted to individual patients. Personalized models can capture unique glucose dynamics, but often overfit if data is scarce%- a common scenario for newly diagnosed individuals
[seo2020personalized]. Existing approaches frequently employ Deep Learning (DL) for time-series modeling. Long

 DiabLLM author manuscript - page 1

## Page 2

Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks have been widely used in general and personalized BG prediction models due to their ability to capture temporal dependencies [DUDUKCU20211208, bhimireddy2020blood, zhu2020dilated, yang2020multi, freiburghaus2020deep, bevan2020experiments, gu2020neural, hameed2020investigating, cui2021personalised, nemat2022blood, shuvo2023deep]. More recently, Reinforcement
Learning (RL) has been explored for personalized insulin dosing strategies, leveraging sequential decision-making to improve glycemic outcomes [bothe2013use,zhu2021reinforcement,fox2020deep]. While prior research has explored deep and reinforcement learning methods for BG prediction within SCH frameworks, the integration of LLMs into this domain remains mostly unexplored. This study addresses an ongoing question in time-series forecasting: to what extent are expensive LLMs necessary for effective prediction. Recent work has shown that simpler architectures can achieve competitive accuracy on standard benchmarks [tan2024languagemodelsactuallyuseful], raising questions about the trade-off between model size and performance. However, in the context of T1DM, we argue that the motivation to employ advanced models such as LLM extends beyond the accuracy of forecasting. The long-term vision is a interactive health companion. While the full realization of this vision involves interactive systems, a critical and foundational step is to evaluate whether these new model architectures can perform the core task of physiological forecasting. LLMs offer several unique capabilities crucial for this vision: (1) robust zero-shot and few-shot learning, which enables effective performance even in the presence of limited, noisy, or irregular data [jin2023large]; (2) rich pattern representations and prior knowledge that can be effectively transferred to downstream tasks; %making them more robust to noise and imperfections in the data; (3) the ability to generate individualized, real-time predictions %and valuations while providing contextualized explanations linked to lifestyle factors such as food intake, physical activity, and medication; and (4) support for VHAs that can engage patients through natural language dialogue, enhancing user understanding and promoting more informed self-management. Simpler DL models may demonstrate comparable forecasting accuracy but are inherently limited in their ability to support interactive, multi-modal, and agentic functionalities. Therefore, we contend that systematic investigation of LLM-based frameworks is a necessary step toward the next generation of intelligent diabetes management systems. Building on this vision, we present DiabLLM, an approach that integrates LLMs into
SCH-driven CGM systems for BG prediction in T1DM care. To this end, we explore two general-purpose architectures,
Time-LLM and Chronos, which adapt the architectural principles of LLMs for time-series forecasting. In this context,
'LLM-based' does not refer to conversational AI, but rather to an emerging class of time-series forecasters that leverage
LLM backbones and their next-token prediction paradigm. We adapt and evaluate these models using the publicly available clinical datasets: OhioT1DM [marling2020ohiot1dm] and D1NAMO [DUBOSSON201892] [jin2023large]. The two approaches differ in how they handle time series data: one involves fine-tuning the LLM with added numeric tokens to support BG forecasting, while the other learns a lightweight projection layer to align sensor inputs with the model’s internal representations. We evaluate both models under zero-shot and fine-tuned settings and compare their performance against our two main state-of-the-art baselines, a Deep Reinforcement Learning (DRL) approach
[domanski2023blood,DOMANSKI2024481] and a decision-level fusion ensemble combining LSTM, WaveNet, and GRU architectures [DUDUKCU20211208], along with a broader set of other deep learning and classical time series models.
Our results show that fine-tuned Time-LLM demonstrates improved performance compared to these baselines in predictive accuracy. In addition, we analyze the models' robustness under calibration errors, noise injection, and missing-value - conditions that commonly occur in real-world CGM data. We employed an autoencoder-based model, as described in [jain2024selfsupervised] to reconstruct corrupted data, utilizing this as a preprocessing technique to enhance data quality and model performance. Furthermore, to ensure the practicality of our framework for real-world deployment, we use knowledge distillation to create a lightweight version for edge devices, successfully reducing its size without compromising predictive accuracy. Additionally, we relied solely on CGM data for predictions, as prior studies have shown CGM-only models achieve clinically safe performance. Including other signals offers minimal gains and is often impractical due to missing or unreliable real-world data. [vandoorn2021machine] [zecchin2021forecasting] To the best of our knowledge, this is one of the first studies to adapt and systematically evaluate general-purpose LLMs for
CGM-based BG prediction using the OhioT1DM and D1NAMO datasets, contributing to the development of intelligent, personalized diabetes management within SCH systems. In summary, this paper makes the following key contributions:
itemize %It introduces the first LLM-based framework for T1DM care using the OhioT1DM dataset. We introduce
DiabLLM, a framework that implements LLMs into SCH-driven CGM systems for BG prediction in T1DM care. %It addresses sensor noise and missing data, enhancing model robustness in data-scarce settings. We explore two
LLM-based approaches - Time-LLM and Chronos - and adapt them to the task of BG forecasting using real-world CGM data. %It improves generalization across patients. We conduct a comprehensive evaluation under both zero-shot and fine-tuned settings, demonstrating improved predictive accuracy and generalization across patients compared to

 DiabLLM author manuscript - page 2

## Page 3

state-of-the-art baselines. %It highlights the potential of LLMs for enhanced predictive accuracy and practical application in clinical settings. To address data quality, we introduce an autoencoder-based preprocessing step for missing value reconstruction. We show that LLMs exhibit improved stability and generalization under sensor noise and missing data, highlighting their robustness in realistic clinical settings. We introduce a knowledge distillation method to compress
LLMs, creating lightweight versions with high potential for deployment on edge devices while maintaining high predictive accuracy. itemize The rest of the paper is organized as follows: Section [sec:related_work] reviews prior work. Section
[sec:methodology] describes the proposed methodology. Section [sec:experiments] details the experimental setup.
Section [sec:results] presents the results. Section [sec:discussion] discusses the key findings and limitations. Finally,
Section [sec:conclusion] concludes the paper. The source code for DiabLLM are available at https://github.com/ammahmoudi/diab-llm

2. Related Prior Work
Large Language Models for Time Series Forecasting LLMs are experiencing growing adoption in time series analysis, showing promise in general and domain-specific tasks. Zhou et al.[zhou2023fits] proposed One Fits All (GPT4TS), a GPT-2-based model fine-tuned via patch embeddings and selective parameter updates, achieving strong results across time series tasks.
Liu et al.[liu2023calf] introduced CALF, aligning textual and temporal distributions to improve few- and zero-shot forecasting. LLM4TS [chang2023llm4ts] uses a two-stage fine-tuning strategy to adapt LLMs to multi-scale temporal dependencies. PromptCast [xue2023promptcast] frames forecasting as question-answering via template-based text conversion, while LLMTIME
[gruver2024largelanguagemodelszeroshot] highlights zero-shot capabilities using digit-based tokenization of real-valued inputs. In the specific domain of healthcare, Kim et al.
[kim2024healthllmlargelanguagemodels] introduced Health-LLM, a framework evaluating LLMs on health prediction tasks using wearable sensor data. Their approach converts physiological time series into simple text strings within the prompt and leverages fine-tuning and context enhancement strategies, such as including user demographics or health knowledge, to improve performance. The resulting fine-tuned model, HealthAlpaca, achieved state-of-the-art performance in 8 out of 10 consumer health tasks, outperforming even larger models such as
GPT-4. TEST [sun2023test] aligns time series with LLM embeddings via tokenization and embedding mapping, enabling frozen LLMs to handle time series tasks effectively. Chronos
[ansari2024chronos] and Time-LLM [jin2023time] are recent LLM-based frameworks demonstrating strong performance in multi-purpose forecasting. Chronos tokenizes numeric data and trains transformer models for probabilistic forecasting, excelling in zero-shot settings.
Time-LLM reprograms LLMs with text prototypes and Prompt-as-Prefix (PaP), enabling contextual reasoning over time series. In the specific context of BG prediction, Lara-Abelenda et al. [lara2025personalized] conducted a comprehensive evaluation of Time-LLM and other deep learning models for personalized BG forecasting using CGM data. Their results demonstrated that LLM-based models, particularly Time-LLM-GPT, achieved competitive accuracy with low variance.
BG Prediction for T1DM Research on BG prediction for individuals with T1DM has extensively explored various factors influencing BG levels, including insulin administration, carbohydrate intake, physical activity, and stress[zhu2020dilated,yang2020multi,daniels2020personalised,freib urghaus2020deep,bevan2020experiments]. A wide range of machine learning (ML) models have been developed, such as causal dilated convolutional neural networks (CNNs) [zhu2018deep], autoregressive integrated moving average (ARIMA) models [ma2020online], multitask networks, convolutional recurrent neural networks (CRNNs) [daniels2020personalised], and dilated recurrent neural networks (RNNs) [zhu2020dilated]. Further advancements include deep residual networks, latent variable-based statistical models, shallow neural networks, multi-scale LSTMs, attention-based RNNs, neural physiological encoders, sequence-to-sequence models, ensemble

 DiabLLM author manuscript - page 3

## Page 4

methods combining LSTM, GRU, and WaveNet, self-attention networks, and bidirectional LSTM ensembles [rubin2020deep,sun2020prediction,pavan2020personalized,yang2020multi,freiburgha us2020deep,bevan2020experiments]. Recent approaches have introduced deep multi-task learning with stacked LSTMs [shuvo2023deep] and DRL for BG prediction
[domanski2023blood,DOMANSKI2024481]. In addition to real-world datasets, simulation tools such as the Padova T1DM Simulator [man2014uva], Simglucose [xie2021simglucose], and
DMMS.R [ubl2022distributed] have been instrumental in evaluating predictive models under diverse physiological conditions. Acuna et al. [acuna2023predicting] applied a Transformer model to the OhioT1DM dataset, achieving strong results with a 60-minute historical input window.

3. Proposed Methodology
In this section, we present DiabLLM - a LLM-based framework for BG prediction within SCH-driven CGM systems. We leverage two recent general-purpose time series forecasting models, Chronos and Time-LLM, which represent two distinct types of language-inspired approaches. Specifically, Time-LLM transforms continuous time series patches into reprogrammed embeddings, while Chronos leverages discrete temporal embeddings. We adapt both of these models for the BG prediction task.

Problem Formulation The objective is to predict future BG levels using a fixed-length sequence of past CGM readings. Let x = \x_t-w+1, , x_t \ R^w denote the input window of CGM measurements over the past w time steps, and let y = \x_t+1, , x_t+h \ R^h represent the target sequence over a forecast horizon h. The goal is to learn a function f_ : R^w R^h, %such that:
where \( \) represents the model's trainable parameters. The sequence of future predicted BG values is denoted as \( y \).
Training Strategies for BG Forecasting We explore a set of complementary training approaches that reflect different levels of adaptation to CGM data: zero-shot inference, task-specific fine-tuning, preprocessing using denoising autoencoders, robust training on noisy data, and knowledge distillation for model compression.
Zero-shot Forecasting with Pretrained LLMs In this setting, models are used without any task-specific training. Consequently, the pretrained parameters \( _pre \) are applied directly to the CGM prediction task. This setup evaluates the ability of LLMs to generalize to BG forecasting based on their pre-training on external time series data.
Task-Specific Adaptation to CGM Data In this setting, the model is fine-tuned on CGM data to adapt its parameters \( \) for improved BG prediction performance: ^* = _ 1 N _ i=1 ^ N L (f_ (x^ (i)
), y^ (i) ) eq:improve Each training instance consists of a sliding input window \( x^(i) \) and corresponding target sequence \( y^(i) \), with \( N \) total samples. The specific cost function \( L
\) may vary by experiment. We consider two adaptation strategies: itemize Personalized
Fine-Tuning: Training and testing data are drawn from the same patient \( p \), denoted \(
D_train^p \) and \( D_test^p \), allowing the model to capture patient-specific glucose dynamics.
Cross-patient Fine-Tuning: The model is trained on one patient \( p_1 \) and evaluated on another
\( p_2 \), using \( D_train^p_1 \) and \( D_test^p_2 \). This setup tests generalization to unseen individuals. To assess cross-patient generalization, we compute the average prediction error across all test patients: E _ avg = 1 P _ p=1 ^ P E ^ (p) eq:avg_error where \( P \) is the number of test patients and \( E^(p) \) is the prediction error on patient \( p \), computed using common evaluation metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).
itemize The fine-tuning strategy is adapted to each model’s architecture. For Time-LLM, only the lightweight projection layers are trained, with the LLM backbone kept frozen. In contrast,

 DiabLLM author manuscript - page 4

## Page 5

Chronos is fine-tuned by updating all model parameters.
Robust Training and Preprocessing for Missing Data via Denoising Autoencoder Robustness to missing and noisy inputs is critical for real world BG prediction with CGM data, since sensor failures, connectivity issues, and patient noncompliance often produce irregular, noisy, or incomplete sequences. We first modeled and simulated corrupted data, and then used an autoencoder to reconstruct the simulated artifacts, inspired by the approach proposed in
[jain2024selfsupervised]. To simulate CGM artifacts, we modeled calibration error as a linear bias consisting of a multiplicative gain term and an additive offset. The multiplicative factor is sampled as N(1, 0.1) and the additive offset as N(0, 6). Random noise is added using an autoregressive Gaussian process of order one: _t = \, _ t-1 + _t, _t N (0, _ ^2), with \( = 0.5\) and \(
_ = 6\), capturing both noise magnitude and temporal correlation across samples. Finally, missingness is introduced as random gaps, where data points are randomly removed.
[biagi2017modeling, facchinetti2016modeling, vettoretti2019modeling]. The cumulative dropout duration is approximately 10% of the signal with block of maximum 6 consecutive values. The resulting corrupted CGM signal is therefore given by G_ sim (t) = cases \, G_ true (t) + + _t, & if not missing ,\\[2mm] 0, & if dropout active . cases These corrupted inputs x are paired with their clean counterparts x, and the autoencoder is trained to minimize the reconstruction loss: L _
Reconstruct = \| z - x \|^2 eq:reconstruct_loss The model learns to reconstruct a clean representation z from x, which is then used as input to both Time-LLM and Chronos, improving forecasting accuracy in the presence of missing or noisy values. We evaluate both strategies with Time-LLM and Chronos, demonstrating their effectiveness in improving robustness and reliability under real-world deployment conditions.
Knowledge Distillation for Model Compression To reduce the computational cost of large models, we employ a distillation framework [hinton2015distillingknowledgeneuralnetwork], compressing the high-performing Time-LLM (BERT: 12 layers, 768 dims) into a compact student version (TinyBERT [jiao2020tinybertdistillingbertnatural]: 4 layers, 312 dims) while maintaining predictive accuracy. The student minimizes a composite loss
[sanh2020distilbertdistilledversionbert]: L _ Distill = L _ GT + L _ MSE eq:distill_loss where L_GT measures MSE between student output (y_S) and ground truth (y), and L_MSE aligns student predictions with those of the teacher (y_T). The hyperparameters and control their respective contributions. [Figure, table, or algorithm available in the source manuscript.]
Models LLMs generate sequences of discrete tokens, such as words or subwords, by predicting the next token from a fixed vocabulary. While this architecture is well-suited for natural language processing tasks, it presents a fundamental limitation for time series forecasting, where target outputs are continuous, real-valued quantities, e.g., predicting that BG will reach 23.6 mg/dL.
Because LLMs are built around discrete classification rather than continuous regression, they cannot directly generate such values, requiring alternate solutions to bridge this modality gap. In this study, we investigate two techniques that address this challenge using fundamentally different strategies. [Figure, table, or algorithm available in the source manuscript.]
Time-LLM Time-LLM addresses the challenge of forecasting real-valued sequences by reusing a frozen pretrained language model (e.g., GPT-2 or LLaMA) while introducing lightweight, trainable components around it. The input time series is first normalized using instance normalization, then divided into fixed-length segments called patches. These patches are passed into the Patch
Reprogram block - a trainable module that transforms raw patches into a token-compatible representation suitable for language model processing. The Patch Reprogram block first applies a Patch Embedder, which maps each patch into a dense vector. These patch embeddings are then processed through a multi-head cross-attention mechanism over a set of learned text

 DiabLLM author manuscript - page 5

## Page 6

prototypes - vectors derived from pretrained word embeddings and refined through a trainable linear transformation. These prototypes encode semantic temporal patterns, such as “short up” and “steady down”, enabling the model to align numerical time series patterns with language-like embeddings. The output of this attention-based alignment is then passed through a linear projection layer, producing the final Reprogrammed Patch Embeddings. In parallel,
Time-LLM constructs a Prompt-as-Prefix composed of three components: the dataset context, the task instruction, and a summary of input statistics such as minimum, maximum, median values, trend direction, and top lag features. This prompt is tokenized and passed through a frozen LLM embedder to generate prompt embeddings, which are concatenated with the
Reprogrammed Patch Embeddings to form the full input sequence. This sequence is then fed into the frozen LLM body, and the resulting hidden states are mapped to real-valued forecasts via a trainable Output Projection layer. The complete Time-LLM architecture is illustrated in Fig.
[fig:time_llm_flow]. This modular design enables Time-LLM to harness the reasoning capabilities of large language models without modifying their internal parameters. For the Time-LLM implementation, we primarily utilize pre-trained LLM backbones such as BERT, GPT-2, and
LLaMA. In our knowledge distillation experiments, we also explore a compressed version,
TinyBERT. The specific parameter counts for these models are detailed in Table
[tab:comprehensive_standardized_metrics].
Chronos Chronos trains language model based architectures from scratch by transforming continuous time series into a specialized “language” composed of discrete symbols. This is accomplished through a two-step process of scaling and quantization, wherein real-valued observations are normalized and discretized into one of B fixed bins - for instance, a value such as 23.6 might be mapped to bin 867 out of 1024. Each bin serves as a unique token within the model’s vocabulary. Chronos subsequently trains standard LLM architectures, such as
T5[2020t5] and GPT-2[Radford2019LanguageMA], on these tokenized sequences using a conventional cross-entropy objective, effectively casting forecasting as a token prediction task.
The full Chronos pipeline is illustrated in Fig. [fig:chronos_flow]. During inference, the model autoregressively generates a sequence of token predictions, which are dequantized afterward to recover real-valued forecasts. This strategy, referred to as regression via classification, enables
Chronos to fully exploit language modeling frameworks without requiring architectural modifications. In the following experiments, we utilize various Chronos models built on the T5 architecture (Tiny, and Base). The parameter counts for the models used are listed in Table
[tab:comprehensive_standardized_metrics].

4. Experimental Setting
This section outlines the experimental setup for evaluating the LLM-based approaches to BG prediction.

Datasets Our experimental evaluation is conducted on two publicly available datasets for T1DM management: enumerate The OhioT1DM dataset contains CGM data from 12 adults with T1DM, with readings recorded every 5 minutes over an 8-week period [marling2020ohiot1dm]. The
D1NAMO dataset provides data from 9 individuals with T1DM, collected over a 4-week period.
For our evaluation, we utilized the data from patient 1 to patient 7 as a complementary validation cohort [DUBOSSON201892]. enumerate Each subject's data is split into separate training and test sets, with the test set reserved for final evaluation. Missing CGM values remain unaltered to reflect real-world conditions.
Evaluation Metrics To evaluate the performance of the models, we use common analytical and clinical evaluation metrics. Analytical metrics quantify the error of the prediction compared to the ground truth, ensuring accuracy. Clinical metrics ensure that the predictions are robust and

 DiabLLM author manuscript - page 6

## Page 7

reliable in a clinical setting.
Analytical Metrics We evaluate the models using two metrics: RMSE and MAE. RMSE penalizes larger errors by squaring the difference between predicted and actual BG values, making it more sensitive to outliers. In contrast, MAE measures the average magnitude of errors, ignoring direction, making it less sensitive to large deviations than RMSE.
Clinical Metrics We assess the clinical safety %and practical utility of predictions using two established metrics: Surveillance Error Grid (SEG) and Clarke Error Grid (CEG). The SEG categorizes glucose prediction errors into clinical risk levels based on their potential impact on patient safety. Each prediction is assigned a risk category. %(e.g., None, Slight, Moderate, Great,
Extreme) These categories reflect the likelihood and severity of incorrect treatment decisions.
%resulting from prediction errors. The proportion of predictions in low-risk categories % such as
'None' and 'Slight' is a key measure of clinical acceptability [klonoff2014SEG]. The CEG is a clinical metric that classifies prediction errors into zones. Zone A indicates accurate predictions for correct treatment decisions. Zones B to E represent progressively higher risks, where treatment decisions may be impacted. The distribution across these zones evaluates the model’s clinical safety and effectiveness.

5. Results
Pre-trained Models (Zero-shot) tables/tab_chronos_zero_shot_30min tables/tab_chronos_zero_shot_45min We conducted a zero-shot evaluation of pre-trained
Chronos and TimeILLM models on the OhioT1DM dataset to assess their generalization capability for BG forecasting without task-specific fine-tuning. For Chronos, we evaluated two model variants: (1) Chronos Base and (2) Chronos Tiny. The prediction performance for
30-minute and 45-minute horizons is summarized in Tables [tab:chronos_zero_shot_30min] and
[tab:chronos_zero_shot_45min]. Similarly, for TimeILLM, we performed zero-shot evaluation using four transformer-based configurations: (1) BERT
[devlin2019bertpretrainingdeepbidirectional], (2) GPTI2, (3) LLaMA 7B with 8 layers, and (4)
LLaMA 7B with 16 layers. Their forecasting performance is reported in Tables
[tab:timellm_zero_shot_30min] and [tab:timellm_zero_shot_45min]. Based on these zero-shot results, the best performance at the 45-minute prediction horizon was achieved by Chronos Base and TimeILLM (GPT-2), which outperformed the decision-level fusion baseline by approximately
1–3% in RMSE and 8–15% in MAE. This demonstrates that large pre-trained models can deliver clinically meaningful improvements even without task-specific adaptation.
tables/tab_timellm_zero_shot_30min tables/tab_timellm_zero_shot_45min tables/tab_chronos_zero_shot_45min_d1namo We also conducted zero-shot evaluations for
Chronos and Time-LLM models on the D1NAMO dataset. For the 30-minute horizon, the models demonstrated consistent performance, with an average RMSE of 27.95 for Time-LLM and 27.80 for Chronos across all patients. Detailed results for the 45-minute horizon are presented in Table
[tab:chronos_zero_shot_45min_d1namo] for Chronos and in Table
[tab:timellm_zero_shot_45min_d1namo] for Time-LLM.
tables/tab_timellm_zero_shot_45min_d1namo
Fine-tuned Performance Comparison To build upon the zero-shot results, we evaluated the impact of task-specific adaptation. For TimeILLM, all model variants were fine-tuned for 10 epochs, while for Chronos, we fine-tuned all parameters after initial explorations with
LoRA-based fine-tuning [hu2021loralowrankadaptationlarge] were not competitive. On the
OhioT1DM dataset, fine-tuning demonstrated clear benefits. As shown in Tables
[tab:chronos_fine_tuned_30min] and [tab:chronos_fine_tuned_45min], the Chronos Base model

 DiabLLM author manuscript - page 7

## Page 8

achieved modest gains over its zero-shot configuration (approx. 5% RMSE reduction). The improvement for TimeILLM was more significant; the LLaMA 7B model with 8 layers improved upon its zero-shot results by 26% in RMSE and 32% in MAE (45-min horizon), highlighting the effectiveness of fine-tuning. tables/tab_chronos_fine_tuned_30min tables/tab_chronos_fine_tuned_45min tables/tab_timellm_fine_tuned_30min tables/tab_timellm_fine_tuned_45min A comprehensive performance comparison is presented in
Table [tab:models_performance_comparison_30min]. This table benchmarks DiabLLM against key baselines. A direct comparison is made with models sharing our identical configuration, such as Deep RL [DOMANSKI2024481] and LSTM+WaveNet+GRU [DUDUKCU20211208].
Additionally, results from other models are included for a comprehensive overview, even if their experimental parameters are not completely identical. As the results demonstrate, the fine-tuned
Time-LLM models achieved the strongest performance, with the LLaMA 7B (8 layers) variant achieving the lowest RMSE of 16.1 mg/dL. Notably, this model outperformed the DRL baseline by approximately 12.12% (30-min horizon) and a decision-level fusion baseline
(LSTM+WaveNet+GRU) by 26.5% (30-min) and 26.95% (45-min). In contrast, the fine-tuned
Chronos models showed moderate gains, surpassing the fusion baseline by 4.6%.
tables/tab_models_performance_comparison_30min To further validate our models, we fine-tuned them on the D1NAMO dataset. This resulted in an average RMSE of 14.34 for
Time-LLM variants and a 3.8% RMSE reduction for Chronos models on the 30-minute horizon.
Detailed results for the 45-minute horizon are provided in Tables
[tab:chronos_fine_tuned_45min_d1namo] and [tab:timellm_fine_tuned_45min_d1namo]. It surpasses that of standard regression models, such as Linear Regression (RMSE 29.0) and
Ridge Regression (28.7) [BASILE2025103681]. Furthermore, it is comparable to specialized approaches, including various RNN-LSTM models with reported RMSEs ranging from 6.42 to 15.5
[s21165273, s20143896]. Despite methodological differences between studies, the performance of Chronos and Time-LLM is noteworthy, as it was achieved with the significant constraint of using only a 30-minute blood glucose history. tables/tab_chronos_fine_tuned_45min_d1namo tables/tab_timellm_fine_tuned_45min_d1namo All fine-tuning was conducted on a single NVIDIA
L40S GPU (48GB VRAM), where we observed a significant difference in training efficiency.
Chronos proved to be highly efficient, with fine-tuning runs averaging 5 minutes per patient. In contrast, training a Time-LLM (LLaMA 7B, 16 layers) model took approximately 45 minutes per patient, highlighting a practical trade-off between peak performance and computational cost.
Model Compression via Knowledge Distillation To assess the effectiveness of the knowledge distillation framework, we applied it to compress our fine-tuned Time-LLM (BERT) on the
OhioT1DM dataset. A grid search identified optimal hyperparameters of =0.3 and =0.3, which minimize the MAE on a held-out validation set. The student model (Time-LLM-TinyBERT) was subsequently trained for 10 epochs. Table [tab:distillation_results] compares the predictive performance of the original Teacher (Time-LLM BERT) and the Distilled Student (Time-LLM
TinyBERT) across all 12 patients of the OhioT1DM dataset. The results show that the distilled student achieves nearly identical predictive performance to the larger teacher model. The overall data indicate that distillation successfully regularizes the smaller student model, preserving generalization capabilities without a significant loss in predictive accuracy.
tables/tab_distillation_results To evaluate the model's suitability for edge deployment, we measured key inference metrics, with results presented in Table
[tab:comprehensive_standardized_metrics]. The distilled student demonstrates substantial efficiency gains over its teacher. It achieves significant reductions in latency, requires considerably less RAM and VRAM, and operates with lower power consumption. These measurements confirm that the distilled model is a viable candidate for resource-constrained

 DiabLLM author manuscript - page 8

## Page 9

environments, offering a compelling balance between high predictive accuracy and computational efficiency.
Clinical Metrics Evaluation To assess clinical accuracy, we evaluated fine-tuned models on the
OhioT1DM dataset using CEG and SEG metrics, which measure the acceptability and safety of
BG forecasts. The analysis focuses on a 30-minute prediction horizon at the cohort level and for two individual patients, 570 and 584. We compare fine-tuned Chronos Base with TimeILLM
(LLaMA 7B, 8 layers), which has shown the best BG prediction performance. CEG results (Table
[tab:ceg_results], Fig. [fig:ceg_comparison]) show that although both models achieve clinically reliable performance, Time-LLM delivers higher accuracy and greater stability. At the cohort level, Time-LLM surpassed Chronos in Zone A, reaching 99.58% accuracy compared to 98.43%, while also exhibiting substantially lower variance (0.0002% vs. 0.0183%). The 95% confidence interval for Time-LLM is [99.58--99.59%], notably narrower than that of Chronos [98.39--98.46%].
As a result, Time-LLM also produced fewer errors in non-A zones (0.42% vs. 1.57%). Similarly,
SEG results (Table [tab:seg_results], Fig. [fig:seg_comparison]) confirm the improved safety profile of Time-LLM. At the cohort level, it achieved a significantly higher rate of "None" (safe) predictions, 97.15% (variance 0.0016%) compared to Chronos's 92.94% (variance 0.2691%), and fewer "Mild" errors, 2.38% (variance 0.0030%) versus 5.42% (variance 0.3947%). The 95% confidence interval for the "None" category is 97.14%--97.16% for Time-LLM versus
92.81%--93.07% for Chronos. tables/tab_ceg_results [Figure, table, or algorithm available in the source manuscript.] tables/tab_seg_results [Figure, table, or algorithm available in the source manuscript.]
Sensitivity Analysis of LLMs to Missing and Noisy Inputs We evaluated the impact of autoencoder-based preprocessing on the predictive accuracy of Time-LLM and Chronos across
12 patients from the OhioT1DM dataset, comparing model performance on corrupted and noisy inputs against the same inputs reconstructed by the denoising autoencoder. Table
[tab:robustness_comparison] shows the performance for all patients on a 45-minute prediction horizon on noisy and denoised data. When operating on the noisy inputs, Chronos Base achieved an average RMSE of 47.453 1.603, whereas Time-LLM demonstrated superior MAE resilience (20.568 0.910) despite a higher RMSE (51.067 3.301). The application of autoencoder-based preprocessing yielded substantial error reduction for both models.
Time-LLM achieved an average RMSE reduction, decreasing the average RMSE from 51.067
3.301 (Noisy) to 19.119 0.231 (Denoised). Similarly, Chronos Base achieved an average RMSE reduction, lowering the average RMSE from 47.453 1.603 (Noisy) to 25.963 0.551 (Denoised). After preprocessing, Time-LLM maintained superior accuracy, achieving a 26% lower average RMSE than Chronos. This finding confirms that the Time-LLM architecture, when combined with
Autoencoder denoising, provides the most accurate and stable predictions in data corruption scenarios. tables/tab_robustness_comparison
Analysis of Prediction Windows and Steps To examine how forecast accuracy evolves within the prediction window, we analyzed step-wise performance over the 45-minute horizon (nine future
BG values) for OhioT1DM patient 584, who exhibited the highest overall prediction error. As shown in Fig. [fig:MAE_Comparison], both models exhibit increasing MAE as the prediction horizon extends. Chronos Base shows a steep increase in MAE - from 5 mg/dL at step 1 to 29 mg/dL at step 9 (a 480% increase) - and large variability between steps. In contrast, TimeLLM (
LLaMA 7B, 8 layers) shows less variability in performance. However, MAE increases from 4 mg/dL to 25 mg/dL (a 525% increase), highlighting the importance of recent input data for predictive accuracy and demonstrating the superior temporal stability of Chronos for longer-term forecasts. [Figure, table, or algorithm available in the source manuscript.]

 DiabLLM author manuscript - page 9

## Page 10

Cross-patient Generalization Evaluation To assess each model's ability to generalize across individuals, we conducted a cross-patient evaluation on the OhioT1DM dataset, training on one patient’s data and testing on another’s. This setup reflects real-world clinical scenarios where patient-specific data may be limited, and generalizability is essential for deployment. Table
[tab:cross_patient_performance] summarizes results for patients 570 and 584. Both models exhibit strong generalization. Time-LLM shows minimal degradation, with average RMSE increasing slightly from 22.02 (same-patient) to 22.13 (cross-patient). Chronos exhibits a slight improvement, with RMSE decreasing from 27.23 to 26.89. Notably, performance on patient 570 improved when trained on data from patient 584, suggesting that the challenging patterns in patient 584’s data contributed to a more robust model. These findings indicate that both models effectively transfer temporal patterns across individuals, supporting their potential for broader clinical deployment without extensive patient-specific fine-tuning.
tables/tab_cross_patient_performance

6. Discussion
Enhanced Robustness in BG Forecasting Although DiabLLM is primarily focused on integrating
LLMs into BG forecasting, we showed that LLM-based model performance is susceptible to noisy and missing inputs - highlighting robustness as a valuable direction for future research.
To address this, we introduced a noise-aware training approach and demonstrated its potential to improve tolerance to data imperfections. %Nevertheless, robustness was not the central objective of this work and remains a critical area for future exploration. In CGM systems, where sensor dropouts and noisy measurements are common, further advancements - such as diffusion-based imputation, uncertainty-aware masking, and noise-adaptive training - could significantly enhance DiabLLM’s reliability and generalization in clinical and real-world environments.
Computational Complexity and Real-Time Constraints tables/edge_deployment A primary consideration in deploying LLMs is the trade-off between model size and predictive performance.
Our findings align with emerging evidence that larger models are not always superior to simpler models for time series forecasting [tan2024languagemodelsactuallyuseful], underscoring the importance of selecting model size carefully to balance computational cost and performance, particularly in real-time clinical systems. While Time-LLM and Chronos demonstrate strong BG prediction performance, real-time deployment in closed-loop systems raises computational efficiency challenges. TimeILLM models, such as Llama 7B, may require optimization for resource-constrained devices such as CGM systems or insulin pumps. However, optimized inference engines like llama.cpp [ggml2023llama] enable efficient deployment on edge devices—including laptops and Raspberry Pi—via quantization and memory optimizations.
Smaller Time-LLM base models such as BERT, GPT-2, and their distilled versions further enhance edge suitability. The inference analysis, detailed in Table
[tab:comprehensive_standardized_metrics], validates these findings. The distilled
Time-LLM-TinyBERT model exhibits a disk size over 6.2 times smaller and achieves a significant
163.8-fold reduction in latency compared to its Time-LLM-BERT teacher model. While these metrics were measured under simulated constraints to substantiate feasibility, actual deployment on physical edge hardware was not conducted in this study and remains a subject for future validation. Such optimization strategies are consistent with prior work; For example,
EdgeBERT [tambe2021edgebertsentencelevelenergyoptimizations] achieved low-latency on-chip inference through pruning, adaptive attention, and quantization. TinyBERT
[jiao2020tinybertdistillingbertnatural] and DistilBERT [sanh2020distilbertdistilledversionbert] offer reduced size and faster inference, ideal for smartphones. Similarly, GPT-2 can be quantized

 DiabLLM author manuscript - page 10

## Page 11

to minimize memory use and boost inference speed with minimal accuracy loss. Chronos includes smaller variants such as T5-base (200M) and T5-tiny (8M), making it well-suited for edge deployment due to the low parameter count. Techniques such as post-training quantization [xiao
2024smoothquantaccurateefficientposttraining,yao2022zeroquantefficientaffordableposttraining
] and runtime optimization [microsoft2025onxruntime,ki6an2024fastT5] can further reduce T5's model size and accelerate inference, facilitating deployment on edge platforms including micro-controllers and mobile devices [ abdelali2025tiny]. Given that CGM devices report every five minutes, low-latency inference is critical for timely insulin dosing. Advances in model compression (e.g., pruning and quantization) help minimize memory and compute demands, while emerging AI/LLM accelerators increase the viability of on-device inference. Offloading strategies, such as model partitioning with early layers on-device and deeper layers in the cloud, offer a practical balance between latency, privacy, and performance for deployment. While the initial pretraining of LLMs is resource-intensive, the practical burden lies in fine-tuning. Our results, supported by recent advances [dettmers2023qloraefficientfinetuningquantized], show this cost is computationally comparable to training simpler models from scratch. This presents a key trade-off: leveraging a powerful pretrained model for superior accuracy with minimal feature engineering, versus using a traditional, less resource-intensive model that may require more extensive data preparation to achieve competitive performance.
Interpretability and Clinical Trustworthiness Interpretability is a critical concern when applying
LLM-based frameworks such as Time-LLM and Chronos to BG prediction. Despite their strong predictive performance, these models are highly complex and often perceived as black boxes. In
T1DM care, where BG forecasts directly inform insulin dosing, explainability is essential to build trust among clinicians and patients [Contreras2018AIDiabetes]. The adaptation of LLMs to time series forecasting further compounds interpretability challenges. However, LLMs also offer a unique opportunity to enhance transparency by generating natural language explanations of their predictions. Ethical issues, including algorithmic bias and data security, further underscore the need for explainable AI to ensure fairness, trust, and clinical reliability [Ellahham2020]. The proposed DiabLLM framework demonstrates that LLMs can be effectively adapted for physiological time-series forecasting, providing clinically meaningful improvements in BG prediction. Its ability to anticipate glycemic trends and generalize across patients supports earlier intervention and more adaptive diabetes management. Beyond diabetes, DiabLLM highlights the broader applicability of LLMs to other biosignals, such as heart rate, respiration, or sleep, contributing to personalized health monitoring and the development of digital twins for health. DiabLLM can also be integrated into VHAs, combining continuous biosignal forecasting with human-computer interaction principles to improve usability and support decision-making.
The integration of LLM-based tools is particularly relevant for diverse user populations, including older adults, by offering accessible, context-aware feedback that enhances engagement with digital health tools.

7. Conclusion
This study investigated the applicability of LLM-based architectures for short-term blood glucose prediction in individuals with T1DM. The proposed framework, DiabLLM, integrates two architectures, Time-LLM and Chronos, to model blood glucose dynamics from historical measurements. The evaluated framework demonstrated strong predictive performance, indicating the potential of adapting LLM-based architectures to physiological data, as reflected in reductions of up to 27% in RMSE and 37% in MAE compared with state-of-the-art methods. To address practical deployment considerations, knowledge distillation was employed to obtain a compact and computationally efficient model suitable for resource-constrained edge environments. Moreover, to address imperfect and noisy input signals, a denoising autoencoder was employed to reconstruct noisy and missing input data. Potential directions for future research include

 DiabLLM author manuscript - page 11

## Page 12

the integration of multimodal data, such as activity and nutrition, as well as training on larger and more diverse datasets.
The strong performance of Time-LLM under limited data conditions suggests its suitability for real-world and data-scarce applications. Nonetheless, improving the sample efficiency of LLM training and fine-tuning, particularly in scenarios with limited sensor sampling frequency or missing data, remains an important area for future research to enable robust and scalable deployment.

 DiabLLM author manuscript - page 12

## Page 13

References
American Diabetes Association. Diagnosis and classification of diabetes mellitus. 2014

A. D. Deshpande and M. Harris-Hayes and M. Schootman. Epidemiology of diabetes and diabetes-related complications.

K. Ogurtsova and L. Guariguata and N. C. Barengo and P. L. Ruiz and J. W. Sacre and S. Karuranga and H. Sun and E.
J. Boyko and D. J. Magliano. IDF diabetes Atlas: Global estimates of undiagnosed diabetes in adults for 2021. 2022

Elsayed, Nelly and ElSayed, Zag and Ozer, Murat. SoutheastCon 2022. 2022

D. Bruttomesso. Toward automated insulin delivery. 2019

Rodbard, David. Continuous glucose monitoring: A review of successes, challenges, and opportunities. 2016

Bhimireddy, Akhil and Sinha, Priyansh and Oluwalade, Bolu and Gichoya, Judy W and Purkayastha, Saptarshi. Blood glucose level prediction as time-series modeling using sequence-to-sequence neural networks. 2020

Yang, Tao and Wu, Rui and Tao, Rui and Wen, Shuang and Ma, Ning and Zhao, Yiming and others. Multi-scale long short-term memory network with multi-lag structure for blood glucose prediction. 2020

Freiburghaus, Julian and Rizzotti, Andrea and Albertetti, Fabio. A deep learning approach for blood glucose prediction of type 1 diabetes. 2020

Bevan, Ryan and Coenen, Frans. Experiments in non-personalized future blood glucose level prediction. 2020

Gu, Kevin and Dang, Ruochen and Prioleau, Temiloluwa. Neural physiological model: A simple module for blood glucose prediction. 2020

Hameed, Hira and Kleinberg, Samantha. Investigating potentials and pitfalls of knowledge distillation across datasets for blood glucose forecasting. 2020

Cui, Runzhou and Hettiarachchi, Chathuri and Nolan, Christopher J and Daskalaki, Elena and Suominen, Hanna.
Personalised short-term glucose prediction via recurrent self-attention network. 2021

Nemat, Hamed and Khadem, Hadi and Eissa, Mohammad Reza and Elliott, Jackie and Benaissa, Mohammad. Blood glucose level prediction: advanced deep-ensemble learning approach. 2022

Shuvo, Md Maruf Hossain and Islam, Sheikh Khaled. Deep multitask learning by stacked long short-term memory for predicting personalized blood glucose concentration. 2023

Jin, Ming and Wen, Qingsong and Liang, Yuxuan and Zhang, Chaoli and Xue, Siqiao and Wang, Xue and Zhang, James and Wang, Yi and Chen, Haifeng and Li, Xiaoli and others. Large models for time series and spatio-temporal data: A survey and outlook. 2023

W. Seo and S. W. Park and N. Kim and S. M. Jin and S. M. Park. A personalized blood glucose level prediction model with a fine-tuning strategy: A proof-of-concept study. 2021

Tom Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared D. Kaplan and Prafulla Dhariwal and
Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell. Language Models are Few-Shot
Learners. 2020

Collaboration NCDRF. Worldwide trends in diabetes since 1980: a pooled analysis of 751 population-based studies with
4.4 million participants. 2016

Tian Zhou and Peisong Niu and Xue Wang and Liang Sun and Rong Jin. One Fits All: Power General Time Series
Analysis by Pretrained LM. 2023

Peiyuan Liu and Hang Guo and Tao Dai and Naiqi Li and Jigang Bao and Xudong Ren and Yong Jiang and Shu-Tao Xia.
CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning. 2024

 DiabLLM author manuscript - page 13

## Page 14

Cao, Defu and Jia, Furong and Arik, Sercan O and Pfister, Tomas and Zheng, Yixiang and Ye, Wen and Liu, Yan.
TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting. 2023

Ching Chang and Wen-Chih Peng and Tien-Fu Chen. LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with
Pre-Trained LLMs. 2023

Chenxi Sun and Hongyan Li and Yaliang Li and Shenda Hong. TEST: Text Prototype Aligned Embedding to Activate
LLM's Ability for Time Series. 2024

Hao Xue and Flora D. Salim. PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting. 2023

Jin, Ming and Zhang, Yifan and Chen, Wei and Zhang, Kexin and Liang, Yuxuan and Yang, Bin and Wang, Jindong and
Pan, Shirui and Wen, Qingsong. Position: What Can Large Language Models Tell Us about Time Series Analysis. 2023

Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Arango, Sebastian Pineda and Kapoor, Shubham and others. Chronos: Learning the language of time series. 2024

Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and others. Time-llm: Time series forecasting by reprogramming large language models. 2023

Marling, Cindy and Bunescu, Razvan. The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. 2020

Klonoff, David C and Lias, Courtney and Vigersky, Robert and Clarke, William and Parkes, Joan Lee and Sacks, David B and Kirkman, M Sue and Kovatchev, Boris and others. The Surveillance Error Grid. 2014

Peter Domanski and Aritra Ray and Kyle Lafata and Farshad Firouzi and Krishnendu Chakrabarty and Dirk Pflüger.
Advancing blood glucose prediction with neural architecture search and deep reinforcement learning for type 1 diabetics.

Domanski, Peter and Ray, Aritra and Firouzi, Farshad and Lafata, Kyle and Chakrabarty, Krishnendu and Pfl\"uger, Dirk.
Blood glucose prediction for type-1 diabetics using deep reinforcement learning. 2023

Hatice Vildan Dudukcu and Murat Taskiran and T\"ulay YIldIrIm. Blood glucose prediction with deep neural networks using weighted decision level fusion. 2021

Braem, Carlijn I. R. and Yavuz, Utku S. and Hermens, Hermie J. and Veltink, Peter H.. Missing Data Statistics Provide
Causal Insights into Data Loss in Diabetes Health Monitoring by Wearable Sensors. 2024

Tang, Hua and Zhang, Chong and Jin, Mingyu and Yu, Qinkai and Wang, Zhenting and Jin, Xiaobo and Zhang,
Yongfeng and Du, Mengnan. Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities.

Papadopoulos, Panagiotis and Sofianos, Emmanouil and Papadopoulos, Theodoros and Chatzimichail, Theodoros. The
Impact of Missing Continuous Blood Glucose Samples on Machine Learning Models for Predicting Postprandial
Hypoglycemia: An Experimental Analysis. 2024

Nate Gruver and Marc Finzi and Shikai Qiu and Andrew Gordon Wilson. Large Language Models Are Zero-Shot Time
Series Forecasters. 2024

Hugo Touvron and Thibaut Lavril and Gautier Izacard and Xavier Martinet and Marie-Anne Lachaux and Timothée
Lacroix and Baptiste Rozière and Naman Goyal and Eric Hambro and Faisal Azhar and others. LLaMA: Open and
Efficient Foundation Language Models. 2023

Alec Radford and Jeff Wu and Rewon Child and David Luan and Dario Amodei and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. 2019

Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant. A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers. 2023

 DiabLLM author manuscript - page 14

## Page 15

Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and
Yanqi Zhou and Wei Li and Peter J. Liu. Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer. 2020

Jacob Devlin and Ming-Wei Chang and Kenton Lee and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. 2019

Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. 2021

Zhou, Liang et al.. Interpretability and fidelity of glucose signal modeling in time series forecasts. 2023

Singh, Aarav and Wu, Meiling. When Accuracy Isn’t Enough: The Cost of Low Interpretability in Glucose Forecasting.

Martinez, Raul and Chen, Yu. Challenges in deploying time series forecasting models for real-world clinical data: a case study in glucose prediction. 2023

Aymen Rayane Khouas and Mohamed Reda Bouadjenek and Hakim Hacid and Sunil Aryal. Training Machine Learning models at the Edge: A Survey. 2024

H. Thabit and G. Rayman. Technology in the Management of Diabetes in Hospitalised Adults. 2024

Yue Zheng and Yuhao Chen and Bin Qian and Xiufang Shi and Yuanchao Shu and Jiming Chen. A Review on Edge
Large Language Models: Design, Execution, and Applications. 2025

Reza Rawassizadeh and Yi Rong. ODSearch: Fast and Resource Efficient On-device Natural Language Search for
Fitness Trackers' Data. 2022

Hao Wen and Yuanchun Li and Guohong Liu and Shanhui Zhao and Tao Yu and Toby Jia-Jun Li and Shiqi Jiang and
Yunhao Liu and Yaqin Zhang and Yunxin Liu. AutoDroid: LLM-powered Task Automation in Android. 2024

Ivan Contreras and Josep Vehi. Artificial Intelligence for Diabetes Management and Decision Support: Literature Review.

Ellahham, Samer. Artificial Intelligence: The Future for Diabetes Care. 2020

Zhiqing Sun and Hongkun Yu and Xiaodan Song and Renjie Liu and Yiming Yang and Denny Zhou. MobileBERT: a
Compact Task-Agnostic BERT for Resource-Limited Devices. 2020

Guanqiao Qu and Qiyuan Chen and Wei Wei and Zheng Lin and Xianhao Chen and Kaibin Huang. Mobile Edge
Intelligence for Large Language Models: A Contemporary Survey. 2025

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. 2019

Bothe, Melanie K and Dickens, Luke and Reichel, Katrin and Tellmann, Arn and Ellger, Bj\"orn and Westphal, Martin and
Faisal, Ahmed A. The use of reinforcement learning algorithms to meet the challenges of an artificial pancreas. 2013

Zhu, Jinhao and Zhang, Yinjia and Rao, Weixiong and Zhao, Qinpei and Li, Jiangfeng and Wang, Congrong.
Reinforcement learning for diabetes blood glucose control with meal information. 2021

Fox, Ian and Lee, Joyce and Pop-Busui, Rodica and Wiens, Jenna. Deep reinforcement learning for closed-loop blood glucose control. 2020

Zhu, Taiyu and Li, Kezhi and Herrero, Pau and Chen, Jianwei and Georgiou, Pantelis. A Deep Learning Algorithm for
Personalized Blood Glucose Prediction.. 2018

Ma, Ning and Zhao, Yuhang and Wen, Shuang and Yang, Tao and Wu, Ruikun and Tao, Rui and Yu, Xia and Li, Hongru.
Online Blood Glucose Prediction Using Autoregressive Moving Average Model with Residual Compensation Network..

Daniels, John and Herrero, Pau and Georgiou, Pantelis. Personalised Glucose Prediction via Deep Multitask Networks..

 DiabLLM author manuscript - page 15

## Page 16

Zhu, Taiyu and Li, Kezhi and Chen, Jianwei and Herrero, Pau and Georgiou, Pantelis. Dilated recurrent neural networks for glucose forecasting in type 1 diabetes. 2020

Rubin-Falcone, Harry and Fox, Ian and Wiens, Jenna. Deep Residual Time-Series Forecasting: Application to Blood
Glucose Prediction.. 2020

Sun, Xiaoyu and Rashid, Mudassir M and Sevil, Mert and Hobbs, Nicole and Brandt, Rachel and Askari,
Mohammad-Reza and Shahidehpour, Andrew and Cinar, Ali. Prediction of Blood Glucose Levels for People with Type 1
Diabetes using Latent-Variable-based Model.. 2020

Pavan, Jacopo and Prendin, Francesco and Meneghetti, Lorenzo and Cappon, Giacomo and Sparacino, Giovanni and
Facchinetti, Andrea and Del Favero, Simone and others. Personalized Machine Learning Algorithm based on Shallow
Network and Error Imputation Module for an Improved Blood Glucose Prediction.. 2020

Man, Chiara Dalla and Micheletto, Francesco and Lv, Dayu and Breton, Marc and Kovatchev, Boris and Cobelli, Claudio.
The UVA/PADOVA type 1 diabetes simulator: new features. 2014

Xie, J.. Simglucose. 2021

Ubl, Martin and Koutny, Tomas and Della Cioppa, Antonio and De Falco, Ivanoe and Tarantino, Ernesto and Scafuri,
Umberto. Distributed assessment of virtual insulin-pump settings using smartcgms and dmms. r for diabetes treatment.

Jain, Pulkit and Ding, Cheng and Rudin, Cynthia and Hu, Xi. A Self-Supervised Algorithm for Denoising
Photoplethysmography Signals for Heart Rate Estimation From Wearables. 2024

Geoffrey Hinton and Oriol Vinyals and Jeff Dean. Distilling the Knowledge in a Neural Network. 2015

Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf. DistilBERT, a distilled version of BERT:
smaller, faster, cheaper and lighter. 2020

Xiaoqi Jiao and Yichun Yin and Lifeng Shang and Xin Jiang and Xiao Chen and Linlin Li and Fang Wang and Qun Liu.
TinyBERT: Distilling BERT for Natural Language Understanding. 2020

Fabien Dubosson and Jean-Eudes Ranvier and Stefano Bromuri and Jean-Paul Calbimonte and Juan Ruiz and Michael
Schumacher. The open D1NAMO dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management. 2018
 ggerganov. Llama.cpp. 2023

Thierry Tambe and Coleman Hooper and Lillian Pentecost and Tianyu Jia and En-Yu Yang and Marco Donato and Victor
Sanh and Paul N. Whatmough and Alexander M. Rush and David Brooks and Gu-Yeon Wei. EdgeBERT:
Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference. 2021

Ki6an. FastT5. 2024

Microsoft. ONNX Runtime. 2025

Abdelali, Taha and Allouche, M’Hammed and Mezghani, Hassen. Tiny Language Models for Automation and Control:
Overview, Potential Applications, and Future Research Directions. 2025

Yubin Kim and Xuhai Xu and Daniel McDuff and Cynthia Breazeal and Hae Won Park. Health-LLM: Large Language
Models for Health Prediction via Wearable Sensor Data. 2024

Zhewei Yao and Reza Yazdani Aminabadi and Minjia Zhang and Xiaoxia Wu and Conglong Li and Yuxiong He.
ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers. 2022

Guangxuan Xiao and Ji Lin and Mickael Seznec and Hao Wu and Julien Demouth and Song Han. SmoothQuant:
Accurate and Efficient Post-Training Quantization for Large Language Models. 2024

Francisco J. Lara-Abelenda and David Chushig-Muzo and Pablo Peiro-Corbacho and Ana M. Wägner and Conceição
Granja and Cristina Soguero-Ruiz. Personalized glucose forecasting for people with type 1 diabetes using large

 DiabLLM author manuscript - page 16

## Page 17

language models. 2025

Xinye Chen and Stefan Güttel. An efficient aggregation method for the symbolic representation of temporal data. 2022

Acuna, Edgar and Aparicio, Roxana. 2023 International Conference on Computational Science and Computational
Intelligence (CSCI). 2023

Martínez-Delgado, Laura and Munoz-Organero, Mario and Queipo-Alvarez, Paula. Using Absorption Models for Insulin and Carbohydrates and Deep Leaning to Improve Glucose Level Predictions. 2021

Ilaria Basile and Giovanna Sannino. Blood glucose level prediction in type 1 diabetes: A comparative analysis of interpretable artificial intelligence approaches. 2025

Munoz-Organero, Mario. Deep Physiological Model for Blood Glucose Prediction in T1DM Patients. 2020

Zhu, Taiyu and Yao, Xi and Li, Kezhi and Herrero, Pau and Georgiou, Pantelis. Blood glucose prediction for type 1 diabetes using generative adversarial networks. 2020

McShinsky, Richard and Marshall, Brandon. Comparison of Forecasting Algorithms for Type 1 Diabetic Glucose
Prediction on 30 and 60-Minute Prediction Horizons.. 2020

Dudukcu, Hatice Vildan and Taskiran, Murat and Yildirim, Tulay. 2021 International Conference on INnovations in
Intelligent SysTems and Applications (INISTA). 2021

Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer. QLoRA: Efficient Finetuning of Quantized
LLMs. 2023

Mingtian Tan and Mike A. Merrill and Vinayak Gupta and Tim Althoff and Thomas Hartvigsen. Are Language Models
Actually Useful for Time Series Forecasting?. 2024

A. Facchinetti and S. Del Favero and G. Sparacino and C. Cobelli. Modeling Transient Disconnections and Compression
Artifacts of Continuous Glucose Sensors. 2016

M. Vettoretti and S. Del Favero and G. Sparacino and A. Facchinetti. Modeling the Error of Factory-Calibrated
Continuous Glucose Monitoring Sensors: Application to Dexcom G6 Sensor Data. 2019

W. P. T. M. van Doorn and B. J. A. Mertens and C. J. H. van der Kallen and R. M. A. Henry and M. M. J. van
Greevenbroek and N. C. Schaper and M. T. Schram and A. Koster. Machine Learning–Based Glucose Prediction with
Use of Continuous Glucose and Physical Activity Monitoring Data: The Maastricht Study. 2021

C. Zecchin and A. Facchinetti and G. Sparacino and C. Cobelli. Forecasting of Glucose Levels and Hypoglycemic
Events: Head-to-Head Comparison of Linear and Nonlinear Data-Driven Algorithms Based on Continuous Glucose
Monitoring Data Only. 2021

L. Biagi and C. M. Ramkissoon and A. Facchinetti and Y. Leal and J. Vehi. Modeling the Error of the Medtronic Paradigm
Veo Enlite Glucose Sensor. 2017

 DiabLLM author manuscript - page 17
