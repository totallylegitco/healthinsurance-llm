#!/bin/bash

set -ex

source setup.sh

export PATH=$PATH:~/.local/bin:/usr/lib/x86_64-linux-gnu
MODEL=${MODEL:-"TotallyLegitCo/appeal-alpaca"}


pip install huggingface_hub

if [ ! -d lit-parrot ]; then
  git clone https://github.com/Lightning-AI/lit-parrot.git
fi

cd lit-parrot

if [ ! -f ".firstrun" ]; then
  pip3 install -U --pre -r ../requirements.txt -r requirements.txt  --extra-index-url "${extra_url}"
  pip3 install -U --index-url "${extra_url}" --pre 'torch>=2.1.0dev'
  python scripts/download.py --repo_id ${MODEL}
  touch .firstrun
fi

python generate/adapter_v2.py --prompt "Generate a health insurance appeal for a babies cancer treatment." --adapter_path ./checkpoints/TotallyLegitCo/appeal-alpaca/adv2_ft/lit_model_adapter_finetuned.pth --checkpoint_dir ./checkpoints/TotallyLegitCo/appeal-alpaca/checkpoints/tiiuae/falcon-7b
