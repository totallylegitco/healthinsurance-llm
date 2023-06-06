#!/bin/bash

set -ex

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}
# Only in holden's branch and even then it its kind of funky.
#QLORA=${QLORA:-"--qlora-4bit true"}
EPOCHS=${EPOCHS:-"10"}

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | cut -f 1 -d " ")

pip install -r requirements.txt
if [ ${gpu_memory} < 40564 ]; then
  if [ $(uname -m) == "aarch64" ]; then
    # On ARM for bits and bytes we need neon
    if [ ! -d sse2neon ]; then
      git clone https://github.com/DLTcollab/sse2neon.git
      make
      sudo cp sse2neon.h /usr/include/
    fi
  else
    pip install bitsandbytes
    python -m bitsandbytes || ./setup_bits_and_bytes.sh
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
fi

if [ ! -d data_sources ]; then
  mkdir -p data_sources
  if [ ! -f "./data_sources/ca-independent-medical-review-imr-determinations-trends.csv" ]; then
    # From https://data.chhs.ca.gov/dataset/independent-medical-review-imr-determinations-trend/resource/3340c5d7-4054-4d03-90e0-5f44290ed095
    # From https://data.chhs.ca.gov/dataset/independent-medical-review-imr-determinations-trend
    wget https://data.chhs.ca.gov/dataset/b79b3447-4c10-4ae6-84e2-1076f83bb24e/resource/3340c5d7-4054-4d03-90e0-5f44290ed095/download/independent-medical-review-imr-determinations-trends.csv -O \
	 ./data_sources/ca-independent-medical-review-imr-determinations-trends.csv
    iconv -c -t utf-8 ./data_sources/ca-independent-medical-review-imr-determinations-trends.csv  > ./data_sources/ca-independent-medical-review-imr-determinations-trends-utf8.csv
    # If we don't have much memory and bitsandbytes works then we go for it
    if [ ${gpu_memory} < 40564 ]; then
      (python -m bitsandbytes && python -m dataset_tools.ca_data --small-gpu) || \
	(python -m dataset_tools.ca_data)
    else
      python -m dataset_tools.ca_data
    fi
  fi
fi

if [ ! -d combined-llm-data ]; then
  mkdir -p combined-llm-data
  ln -s $(pwd)/appeals-llm-data/* $(pwd)/combined-llm-data/
  ln -s $(pwd)/generated-llm-data/* $(pwd)/combined-llm-data/
fi


if [ ! -f "out/out.jsonl" ]; then
  python -m dataset_tools.final
fi



cd dolly

rm -rf out
cp -af ../out ./

# TODO: Select 4bit qlora based on GPU memory available.
python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 1 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS}
cd ..
python test_new_model.py
