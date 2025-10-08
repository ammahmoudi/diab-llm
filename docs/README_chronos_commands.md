# Chronos Training & Inference Command Guide

This guide provides all commands needed to generate Chronos training and inference configs, and to run experiments for the OhioT1DM dataset. It covers all scenarios and window configurations (6_6 and 6_9).

---

## 1. Generate Training Configs (1 seed)

### Train on Standardized Raw
```bash
python scripts/chronos/config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Missing Periodic
```bash
python scripts/chronos/config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Missing Random
```bash
python scripts/chronos/config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Noisy
```bash
python scripts/chronos/config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

### Train on Denoised
```bash
python scripts/chronos/config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363
```

---

## 2. Run Training Experiments
```bash
python scripts/chronos/run_all_chronos_experiments.py --modes training --datasets ohiot1dm
```

---

## 3. Generate Inference Configs (all 5 seeds)

### Trained on Raw (standardized), Tested on:
#### Raw
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6
```
#### Denoised
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario denoised --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6
```
#### Noisy
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6
```
#### Missing Periodic
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6
```
#### Missing Random
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6
```

### Trained on Noisy, Tested on:
#### Noisy
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```
#### Raw
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario noisy
```

### Trained on Missing Periodic, Tested on:
#### Missing Periodic
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_periodic --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_periodic
```

### Trained on Missing Random, Tested on:
#### Missing Random
```bash
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --patients 540,544,552,559,563,567,570,575,584,588,591,596 --models amazon/chronos-t5-base --seeds 831363,809906,427368,238822,247659 --window_config 6_6 --train_scenario missing_random
```

---

## 4. Run Inference Experiments
```bash
python scripts/chronos/run_all_chronos_experiments.py --modes trained_inference --datasets ohiot1dm
```

---

## Repeat All Inference Commands for 6_9 Window
Replace `--window_config 6_6` with `--window_config 6_9` in all inference config generator commands above.

---

## Notes
- All commands assume you are in the project root directory.
- Patient IDs: 540,544,552,559,563,567,570,575,584,588,591,596
- Seeds: 831363,809906,427368,238822,247659
- Model: amazon/chronos-t5-base
- For cross-scenario inference, use `--train_scenario` to specify the training scenario.
