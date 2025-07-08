# # GPT2
# python distillation_driver.py --student_model GPT2 --student_layers 12 --student_dim 768

# # BERT
# python distillation_driver.py --student_model BERT --student_layers 12 --student_dim 768

# # DistilBERT
# python distillation_driver.py --student_model DistilBERT --student_layers 6 --student_dim 768

# # MiniLM
# python distillation_driver.py --student_model MiniLM --student_layers 6 --student_dim 384

# # TinyBERT
python distillation_driver.py --student_model TinyBERT --student_layers 4 --student_dim 312

# # MobileBERT
# python distillation_driver.py --student_model MobileBERT --student_layers 24 --student_dim 512

# # ALBERT
# python distillation_driver.py --student_model ALBERT --student_layers 12 --student_dim 768

# # BERT-tiny
# python distillation_driver.py --student_model BERT-tiny --student_layers 2 --student_dim 128

# # OPT-125M
# python distillation_driver.py --student_model OPT-125M --student_layers 12 --student_dim 768

# # Chronos
# python distillation_driver.py --student_model Chronos --student_layers 4 --student_dim 512
