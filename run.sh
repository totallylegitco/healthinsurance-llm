#!/bin/bash

set -ex

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-3b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}
# On Jetson AGX we need to add /usr/local/cuda/bin to our path
if [ -f /usr/local/cuda/bin/nvcc ]; then
  export PATH=$PATH:/usr/local/cuda/bin
fi

if [ ! -d dolly ]; then
  git clone https://github.com/databrickslabs/dolly.git
  pip install -r ./dolly/requirements.txt
fi

# Check bits and bytes, it needs to be compiled from source for the jetson (and some others)
python -m bitsandbytes || ./setup_bits_and_bytes.sh
if [ ! -d out ]; then
  mkdir out
  python dataset_generator.py
fi

cd dolly

rm -rf out
cp -af ../out ./

python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 1
