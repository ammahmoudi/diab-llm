#!/bin/bash
# Archive Old Time-LLM Configuration Generators
# This script moves all old Time-LLM config generators to an archive folder

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Time-LLM Config Generator Archival Tool${NC}"
echo "============================================"
echo ""

# Create archive directory
ARCHIVE_DIR="/home/amma/LLM-TIME/scripts/time_llm/archive_old_generators"
mkdir -p "$ARCHIVE_DIR"

echo -e "${BLUE}📁 Created archive directory: $ARCHIVE_DIR${NC}"
echo ""

# List of old Time-LLM config generators to archive
OLD_GENERATORS=(
    # Main time_llm directory
    "/home/amma/LLM-TIME/scripts/time_llm/config_generator_time_llm.py"
    
    # Missing and noise directory  
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_noisy_train.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_missing_periodic.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_noised.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_missing_random.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_denoised.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_missing_random_train.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_noisy.py"
    "/home/amma/LLM-TIME/scripts/missing_and_noise/config_generator_time_llm_missing_periodic_train.py"
    
    # D1NAMO specific
    "/home/amma/LLM-TIME/scripts/d1namo_scripts/config_d1namo_generator_time_llm.py"
    
    # Distil scripts  
    "/home/amma/LLM-TIME/scripts/distil_scripts/config_generator.py"
)

# Files to archive (not delete, as they might be needed for running experiments)
RUN_SCRIPTS=(
    "/home/amma/LLM-TIME/scripts/time_llm/run_configs_time_llm_training.py"
    "/home/amma/LLM-TIME/scripts/time_llm/run_configs_time_llm_inference.py" 
    "/home/amma/LLM-TIME/scripts/time_llm/run_configs_time_llm_training_n.py"
    "/home/amma/LLM-TIME/scripts/time_llm/time_llm_test.sh"
    "/home/amma/LLM-TIME/scripts/time_llm/time_llm_train.sh"
    "/home/amma/LLM-TIME/scripts/time_llm/time_llm_train_test.sh"
    "/home/amma/LLM-TIME/scripts/d1namo_scripts/run_configs_time_llm_training.py"
    "/home/amma/LLM-TIME/scripts/d1namo_scripts/run_configs_time_llm_inference.py"
    "/home/amma/LLM-TIME/scripts/distil_scripts/run_configs_time_llm_training.py"
    "/home/amma/LLM-TIME/scripts/distil_scripts/run_configs_time_llm_inference.py"
)

echo -e "${YELLOW}🗂️  Archiving old Time-LLM config generators...${NC}"
echo ""

archived_count=0

# Archive config generators
for file in "${OLD_GENERATORS[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo -e "Moving: ${BLUE}$filename${NC}"
        mv "$file" "$ARCHIVE_DIR/"
        ((archived_count++))
    else
        echo -e "Not found: ${RED}$file${NC}"
    fi
done

echo ""
echo -e "${YELLOW}📋 Copying run scripts (keeping originals)...${NC}"
echo ""

copied_count=0

# Copy (don't move) run scripts as they might be used by existing experiments
for file in "${RUN_SCRIPTS[@]}"; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        dirname=$(basename $(dirname "$file"))
        target_name="${dirname}_${filename}"
        echo -e "Copying: ${BLUE}$filename${NC} → ${BLUE}$target_name${NC}"
        cp "$file" "$ARCHIVE_DIR/$target_name"
        ((copied_count++))
    else
        echo -e "Not found: ${RED}$file${NC}"
    fi
done

echo ""
echo -e "${GREEN}✅ Archival Complete!${NC}"
echo ""
echo -e "${BLUE}Summary:${NC}"
echo "• Archived generators: $archived_count"
echo "• Copied run scripts: $copied_count"
echo "• Archive location: $ARCHIVE_DIR"
echo ""
echo -e "${YELLOW}What was archived:${NC}"
echo "• All individual Time-LLM config generators"
echo "• Scenario-specific generators (noisy, missing, denoised)"
echo "• Dataset-specific generators (d1namo)"
echo "• Copies of run scripts (originals kept for existing experiments)"
echo ""
echo -e "${GREEN}What to use now:${NC}"
echo "• config_generator_time_llm_unified.py (main generator)"
echo "• quick_config_time_llm.sh (interactive commands)"
echo "• run_common_time_llm_configs.sh (auto-run common configs)"
echo "• TIME_LLM_USAGE_GUIDE.md (documentation)"
echo ""
echo -e "${BLUE}The unified generator replaces ALL archived generators with more features!${NC}"