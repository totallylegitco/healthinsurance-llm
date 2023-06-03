#!/bin/bash

set -ex

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-3b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}
# Only in holden's branch and even then it its kind of funky.
#QLORA=${QLORA:-"--qlora-4bit true"}
EPOCHS=${EPOCHS:-"10"}

if [ $(uname -m) == "aarch64" ]; then
  # On ARM for bits and bytes we need neon
  if [ ! -d sse2neon ]; then
    git clone https://github.com/DLTcollab/sse2neon.git
    make
    sudo cp sse2neon.h /usr/include/
  fi
fi
if [ ! -d dolly ]; then
  git clone https://github.com/databrickslabs/dolly.git
  pip install -r ./dolly/requirements.txt
fi

if [ ! -d "appeals-llm-data" ]; then
  git clone https://github.com/totallylegitco/appeals-llm-data.git
fi

# Check bits and bytes, it needs to be compiled from source for the jetson (and some others)
if [ ! -z "$QLORA" ]; then
  python -m bitsandbytes || ./setup_bits_and_bytes.sh
fi
if [ ! -d out ]; then
  mkdir out
  python dataset_generator.py
fi

cd dolly

rm -rf out
cp -af ../out ./

# TODO: Select 4bit qlora based on GPU memory available.
python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 1 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS}
cd ..
python test_new_model.py
