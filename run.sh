#!/bin/sh
. /nethome/wlacroix/miniconda3/envs/thesis/bin/activate
# source activate thesis
cd /nethome/wlacroix/Thesis-Project/
# conda activate thesis
# cd /nethome/wlacroix/Thesis-Project/
HF_HOME=./.config python3 lora_baseline_adapter.py