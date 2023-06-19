#!/bin/bash

set -ex

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
  touch .firstrun
fi

python scripts/download.py --repo_id ${MODEL}



