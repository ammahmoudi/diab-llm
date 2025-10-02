#!/bin/bash

# ==============================================================================
# Archive Old Missing/Noise Config Generators
# ==============================================================================
# This script moves the old individual missing/noise config generators to an 
# archived folder since they're now consolidated into the unified generator.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MISSING_NOISE_DIR="${SCRIPT_DIR}/../missing_and_noise"
ARCHIVE_DIR="${MISSING_NOISE_DIR}/archived_configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Create archive directory
print_header "ARCHIVING OLD CONFIG GENERATORS"

if [ ! -d "$ARCHIVE_DIR" ]; then
    mkdir -p "$ARCHIVE_DIR"
    print_success "Created archive directory: $ARCHIVE_DIR"
fi

# List of files to archive (chronos-related missing/noise generators)
files_to_archive=(
    "config_generator_chronos_train_missing_periodic.py"
    "config_generator_chronos_train_missing_random.py"
    "config_generator_chronos_train_noisy.py"
    "config_generator_chronos_trained_inference_missing_periodic.py"
    "config_generator_chronos_trained_inference_missing_random.py"
    "config_generator_chronos_trained_inference_noisy.py"
    "config_generator_chronos_trained_inference_denoised.py"
    "config_generator_chronos_trained_missing_periodic_inference_missing_periodic.py"
    "config_generator_chronos_trained_missing_random_inference_missing_random.py"
    "config_generator_chronos_trained_noisy_inference_noisy.py"
)

echo "Moving old chronos config generators to archive..."
echo

archived_count=0
for file in "${files_to_archive[@]}"; do
    if [ -f "$MISSING_NOISE_DIR/$file" ]; then
        mv "$MISSING_NOISE_DIR/$file" "$ARCHIVE_DIR/"
        print_success "Archived: $file"
        ((archived_count++))
    else
        print_warning "File not found: $file"
    fi
done

echo
print_success "Archived $archived_count chronos config generator files"

# Create README in archive directory
cat > "$ARCHIVE_DIR/README.md" << 'EOF'
# Archived Config Generators

These files have been archived because their functionality has been consolidated into the unified config generator system.

## Replacement

All these individual config generators have been replaced by:
- `/scripts/chronos/config_generator_chronos.py` - Unified chronos config generator with `--data_scenario` parameter

## Data Scenario Mapping

Old files → New unified command:

### Missing Periodic Data
- `config_generator_chronos_train_missing_periodic.py` → `--mode train --data_scenario missing_periodic`
- `config_generator_chronos_trained_inference_missing_periodic.py` → `--mode trained_inference --data_scenario missing_periodic`

### Missing Random Data  
- `config_generator_chronos_train_missing_random.py` → `--mode train --data_scenario missing_random`
- `config_generator_chronos_trained_inference_missing_random.py` → `--mode trained_inference --data_scenario missing_random`

### Noisy Data
- `config_generator_chronos_train_noisy.py` → `--mode train --data_scenario noisy` 
- `config_generator_chronos_trained_inference_noisy.py` → `--mode trained_inference --data_scenario noisy`

### Denoised Data
- `config_generator_chronos_trained_inference_denoised.py` → `--mode trained_inference --data_scenario denoised`

## Usage Examples

```bash
# Training on noisy data
python config_generator_chronos.py --mode train --data_scenario noisy --patients 570,584

# Inference on missing periodic data  
python config_generator_chronos.py --mode trained_inference --data_scenario missing_periodic --patients 570

# See all data scenario examples
./chronos_config_examples.sh data_scenarios
```

## Archive Date
Archived on: $(date)

Files archived: $archived_count
EOF

print_success "Created README.md in archive directory"

echo
print_header "SUMMARY"
echo "✨ Successfully consolidated chronos missing/noise config generators!"
echo "📁 Old files moved to: $ARCHIVE_DIR"
echo "🎯 Use unified generator: scripts/chronos/config_generator_chronos.py"
echo "📖 Usage examples: ./chronos_config_examples.sh data_scenarios"
echo
echo "🔧 Example commands:"
echo "  python3 config_generator_chronos.py --mode train --data_scenario noisy --patients 570"
echo "  python3 config_generator_chronos.py --mode train --data_scenario missing_periodic --patients 570,584"
echo "  python3 config_generator_chronos.py --mode trained_inference --data_scenario missing_random --patients 570"