#!/bin/bash

set -ex

export PATH=$PATH:~/.local/bin:/usr/lib/x86_64-linux-gnu
INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}
# Only in holden's branch and even then it its kind of funky.
#QLORA=${QLORA:-"--qlora-4bit true"}
EPOCHS=${EPOCHS:-"10"}

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | cut -f 1 -d " ")

python3 -m pip install --upgrade pip

pip install -U -r requirements.txt
if [ -z "${LD_LIBRARY_PATH}" ]; then
  export LD_LIBRARY_PATH=$PATH
fi

if [ "${gpu_memory}" -lt 49564 ]; then
  if [ $(uname -m) == "aarch64" ]; then
    # On ARM for bits and bytes we need neon
    if [ ! -d sse2neon ]; then
      git clone https://github.com/DLTcollab/sse2neon.git
      make
      sudo cp sse2neon.h /usr/include/
    fi
  else
    pip install -U bitsandbytes
    python -m bitsandbytes || ./setup_bits_and_bytes.sh
  fi
fi

if [ ! -d dolly ]; then
  git clone https://github.com/databrickslabs/dolly.git
#  pip install -r ./dolly/requirements.txt
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
  if [ ! -f "./data_sources/wpath_soc7.pdf"]; then
    wget https://www.wpath.org/media/cms/Documents/SOC%20v7/SOC%20V7_English2012.pdf?_t=1613669341 -O \
	 ./data_sources/wpath_soc7.pdf
  fi
  if [ ! -f "./data_sources/wpath_soc8.pdf"]; then
    wget https://www.tandfonline.com/doi/pdf/10.1080/26895269.2022.2100644 -O \
	 ./data_sources/wpath_soc8.pdf
  fi
  if [ ! -f "./data_sources/hiv_prep_soc.pdf"]; then
    wget https://www.cdc.gov/hiv/pdf/risk/prep/cdc-hiv-prep-guidelines-2021.pdf -O \
	 ./data_sources/hiv_prep_soc.pdf
  fi
  if [ ! -f "./data_sources/erisa.pdf" ]; then
    wget https://www.govinfo.gov/content/pkg/COMPS-896/pdf/COMPS-896.pdf -O \
	 ./data_sources/erisa.pdf
  fi
  if [! -f "./data_sources/ppacacon.pdf" ]; then
    wget http://housedocs.house.gov/energycommerce/ppacacon.pdf -O \
	 ./data_sources/ppacacon.pdf
  fi
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

mkdir -p out_oa

if [ ! -f "out/train.jsonl" ]; then
  if [ ! -d combined-llm-data ]; then
    mkdir -p combined-llm-data
    # Generated file list can be too long to pass through the shell as an argument.
    for i in ./generated-llm-data*/*.txt; do cp "$i" ./combined-llm-data/; done
    # Manual not so much
    cp $(pwd)/appeals-llm-data/* $(pwd)/combined-llm-data/
  fi

  python -m dataset_tools.final
fi



if [ "${INPUT_MODEL}" == "databricks/dolly-v2-7b" ]; then
# dolly
  cd dolly
  if nvcc --version |grep -q 11.8; then
    pip install -r ../requirements.txt -r requirements.txt  --extra-index-url https://download.pytorch.org/whl/cu118
    pip3 install -U "torch<2" --index-url https://download.pytorch.org/whl/cu118
  else
    pip install -r ../requirements.txt -r requirements.txt
  fi
   if [ "$gpu_memory" == "40960" ]; then
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 100 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS} --deepspeed ./config/a100_config.json --bf16
   elif [ "$gpu_memory" == "23028" ]; then
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 100 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS} --deepspeed ./config/a10_config.json --per-device-eval-batch-size 3 --per-device-train-batch-size 3 --bf16 false
   else
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 2000 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS}
   fi
else
  # falcon
  if nvcc --version |grep -q 11.8; then
    pip3 install -U "torch>2" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  else
    pip3 install -U "torch>2" torchvision torchaudio
  fi
  python train.py --input-model ${INPUT_MODEL} --training-dataset out_oa
  python test_new_model.py
fi
