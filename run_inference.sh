#!/bin/bash

export PATH=$PATH:~/.local/bin:/usr/lib/x86_64-linux-gnu
INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}


pip install huggingface_hub

if [ ! -d lit-parrot ]; then
  git clone https://github.com/Lightning-AI/lit-parrot.git
fi

cd lit-parrot

python scripts/download.py --repo_id TotallyLegitCo/appeal-alpaca

