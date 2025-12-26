# Missing Data and Noise Utilities

Core utilities for applying data corruption scenarios to datasets.

## Contents

- **`apply_noise.py`** - Apply Gaussian noise to glucose measurements
- **`apply_missings.py`** - Apply missing data patterns (periodic or random)

## Usage

These utilities are used to create corrupted versions of datasets for robustness testing.

### Apply Noise

```bash
python scripts/missing_and_noise/apply_noise.py --input data/standardized/ --output data/noisy/
```

### Apply Missing Data

```bash
# Random missing pattern
python scripts/missing_and_noise/apply_missings.py --input data/standardized/ --output data/missing_random/ --pattern random

# Periodic missing pattern
python scripts/missing_and_noise/apply_missings.py --input data/standardized/ --output data/missing_periodic/ --pattern periodic
```

## Integration

These utilities are integrated with the unified config generators:
- `scripts/chronos/config_generator.py --data_scenario noisy|missing_periodic|missing_random`
- `scripts/time_llm/config_generator.py --data_scenario noisy|missing_periodic|missing_random`
