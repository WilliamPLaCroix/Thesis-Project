#!/bin/sh
cd /nethome/wlacroix/miniconda3/bin/
source activate thesis
cd /nethome/wlacroix/Thesis-Project/
pwd
# conda activate thesis
# cd /nethome/wlacroix/Thesis-Project/
HF_HOME=./.config python3 lora_baseline_adapter.py