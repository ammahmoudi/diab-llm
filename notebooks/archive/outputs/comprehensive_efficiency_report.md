# LLM Efficiency Analysis Report\nGenerated: 2025-10-23 14:34:56.443306\n\n## Efficiency Metrics\n                                                                     Power (W)  Energy (kWh)  Time (s)  Train Eff  Overall Score  Power Eff
model_name                                             mode                                                                                
BERT_dim_768_seq_6_context_6_pred_6_patch_6_epochs_0   inference        15.000         0.000      24.0        NaN         75.304     96.392
BERT_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10  training         45.000         0.000       5.0        NaN         71.368     89.175
GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_0   inference        30.000         0.000      26.0        NaN         73.336     92.783
GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10  training         87.500         0.016     656.0      0.922         88.519     78.951
LLAMA_dim_4096_seq_6_context_6_pred_6_patch_6_epochs_0 inference       155.885         0.008     187.0        NaN         56.818     62.500
LLAMA_dim_4096_seq_6_context_6_pred_6_patch_6_epochs_1 training        415.692         0.113     981.0      0.000          0.000      0.000
amazon                                                 inference        15.000         0.000       8.5        NaN         75.304     96.392
                                                       training         45.000         0.004     287.5        NaN         71.368     89.175
bert-base-uncased                                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_540_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_544_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_552_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_559_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_563_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_567_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_570_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_575_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_584_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_588_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_591_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
bert_to_tinybert_596_summary.json                      distillation        NaN           NaN       NaN        NaN         50.000        NaN
distillation_summary                                   distillation        NaN           NaN       NaN        NaN         50.000        NaN
distilled_bert_tiny                                    inference        15.000           NaN       0.0        NaN         75.304     96.392
time_llm                                               distillation        NaN           NaN       NaN        NaN         50.000        NaN
time_llm_comprehensive                                 distillation        NaN           NaN       NaN        NaN         50.000        NaN
time_llm_training_inference                            distillation        NaN           NaN       NaN        NaN         50.000        NaN
time_llm_training_mode                                 distillation        NaN           NaN       NaN        NaN         50.000        NaN
tinybert                                               distillation        NaN           NaN       NaN        NaN         50.000        NaN
                                                       inference        15.000         0.000      20.0        NaN         75.304     96.392\n\n## Resource Requirements\n                                                       estimated_power_watts                   estimated_energy_kwh       
                                                                         min      max     mean                  min    max
model_name                                                                                                                
BERT_dim_768_seq_6_context_6_pred_6_patch_6_epochs_0                  15.000   15.000   15.000                0.000  0.000
BERT_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10                 45.000   45.000   45.000                0.000  0.000
GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_0                  30.000   30.000   30.000                0.000  0.000
GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10                 87.500   87.500   87.500                0.016  0.016
LLAMA_dim_4096_seq_6_context_6_pred_6_patch_6_epochs_0               155.885  155.885  155.885                0.008  0.008
LLAMA_dim_4096_seq_6_context_6_pred_6_patch_6_epochs_1               415.692  415.692  415.692                0.113  0.113
amazon                                                                15.000   45.000   30.000                0.000  0.005
bert-base-uncased                                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_540_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_544_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_552_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_559_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_563_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_567_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_570_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_575_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_584_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_588_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_591_summary.json                                        NaN      NaN      NaN                  NaN    NaN
bert_to_tinybert_596_summary.json                                        NaN      NaN      NaN                  NaN    NaN
distillation_summary                                                     NaN      NaN      NaN                  NaN    NaN
distilled_bert_tiny                                                   15.000   15.000   15.000                  NaN    NaN
time_llm                                                                 NaN      NaN      NaN                  NaN    NaN
time_llm_comprehensive                                                   NaN      NaN      NaN                  NaN    NaN
time_llm_training_inference                                              NaN      NaN      NaN                  NaN    NaN
time_llm_training_mode                                                   NaN      NaN      NaN                  NaN    NaN
tinybert                                                              15.000   15.000   15.000                0.000  0.000