#!/bin/bash
# Perform knowledge distillation
cd /home/amma/LLM-TIME
python scripts/distill_students.py "$@"
