#!/bin/bash

set -ex

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-3b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}

if [ ! -d dolly ]; then
  git clone https://github.com/databrickslabs/dolly.git
fi
if [ ! -d out ]; then
  mkdir out
  python dataset_generator.py
fi

cd dolly

rm -rf out
cp -af ../out ./

python ./trainer_main.py --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 1
