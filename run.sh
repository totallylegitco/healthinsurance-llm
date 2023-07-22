#!/bin/bash

set -ex

source setup.sh

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | cut -f 1 -d " ")

if [ ! -f ".firstrun" ]; then
  ./firstrun.sh
fi


if [ ! -d data_sources ]; then
  mkdir -p data_sources
  if [ ! -f "./data_sources/drugs.csv" ]; then
    wget https://seer.cancer.gov/seertools/seerrx/download-rx/?type=drug -O \
	 ./data_sources/drugs.csv
  fi
  if [ ! -f "./data_sources/mtsamples2.csv" ]; then
    wget "https://raw.githubusercontent.com/mcoplan11/service_denial/master/dataset/mtsamples%202.csv" -O \
	 ./data_sources/mtsamples2.csv
  fi
  if [ ! -f "./data_sources/wpath_soc7.pdf"]; then
    wget https://www.wpath.org/media/cms/Documents/SOC%20v7/SOC%20V7_English2012.pdf?_t=1613669341 -O \
	 ./data_sources/wpath_soc7.pdf
  fi
  # TBD: Should we include this?
  if [ ! -f "./data_sources/ic10k.csv"] ; then
    wget https://drive.google.com/u/0/uc?id=1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA&export=download -O \
	./data_sources/ic10k.csv						    
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

mkdir -p ${TR_DATA}/oa

if [ ! -f "${TR_DATA}/train.jsonl" ]; then
  if [ ! -d combined-llm-data ]; then
    mkdir -p combined-llm-data
    # Generated file list can be too long to pass through the shell as an argument.
    for i in ./generated-llm-data*/*.txt; do cp "$i" ./combined-llm-data/; done
    # Manual not so much
    cp $(pwd)/appeals-llm-data/* $(pwd)/combined-llm-data/
  fi

  python -m dataset_tools.final
fi

if [ ! -d lit-parrot ]; then
  git clone https://github.com/Lightning-AI/lit-parrot.git
fi


# Different models need different love for fine tuning.
if [ "${INPUT_MODEL}" == "databricks/dolly-v2-7b" ]; then
# dolly
  cd dolly
  mkdir -p ${TR_DATA}
  cp "../${TR_DATA}/train.jsonl" ./${TR_DATA}
  pip3 install -r ../requirements.txt -r requirements.txt  --extra-index-url ${extra_url}
  pip3 install -U "torch<2" --extra-index-url ${extra_url}
  if [ "$gpu_memory" == "81920" ]; then
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 100 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS} --deepspeed ./config/a100_config.json --bf16 true
   elif [ "$gpu_memory" == "40960" ]; then
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 100 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS} --deepspeed ./config/a100_config.json --bf16 true
   elif [ "$gpu_memory" == "23028" ]; then
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 100 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS} --deepspeed ./config/a10_config.json --per-device-eval-batch-size 3 --per-device-train-batch-size 3 --bf16 false
   else
     python -m training.trainer --input-model ${INPUT_MODEL} --training-dataset ${TR_DATA} --local-output-dir ${OUTDIR} --test-size 2000 --warmup-steps 1 ${QLORA} --epochs ${EPOCHS}
  fi
elif [ "${INPUT_MODEL}" == "NOPEtiiuae/falcon-7b-instruct" ];  then
  # falcontune has some issues right now (seems to ignore our dataset param) and I'm lazy so we'll skip this one for now.
  cd falcontune
  pip install -r requirements.txt
  # Protobufs are compiled before 3.20
  pip install -U protobuf<=3.19 numexpr>2.7.3
  sudo python setup.py install
  cd ..
  export WANDB_MODE=offline
  falcontune finetune \
    --model=falcon-7b-instruct \
    --weights=tiiuae/falcon-7b-instruct \
    --dataset=./out/train_alpaca.jsonl \
    --data_type=alpaca \
    --lora_out_dir=./falcon-7b-instruct-alpaca/ \
    --mbatch_size=1 \
    --batch_size=2 \
    --epochs=3 \
    --lr=3e-4 \
    --cutoff_len=256 \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --warmup_steps=5 \
    --save_steps=50 \
    --save_total_limit=3 \
    --logging_steps=5 \
    --target_modules='["query_key_value"]'
  falcontune generate \
    --interactive \
    --model falcon-7b-instruct \
    --weights mosaicml/falcon-7b-instruct \
    --lora_apply_dir falcon-7b-instruct-alpaca \
    --max_new_tokens 50 \
    --use_cache \
    --do_sample \
    --instruction "Generate a health insurance appeal"
elif [ "${INPUT_MODEL}" == "meta-llama/Llama-2-7b-hf" ]; then
  deepspeed python llamav2_finetune_from_databricks.py --local-output-dir ./llamav2-updated
else
  # lit-parrot seems the happiest
  # falcon
  if [ -z "$QLORA" ]; then 
    mkdir -p lit-parrot/data/alpaca
    cp ${TR_DATA}/*_alpaca.jsonl lit-parrot/data/alpaca/
    cd lit-parrot
    if [ ! -f ".firstrun" ]; then
      pip3 install -U --pre -r ../requirements.txt -r requirements.txt  --extra-index-url "${extra_url}"
      pip3 install -U --index-url "${extra_url}" --pre 'torch>=2.1.0dev'
      touch .firstrun
    fi
    if [ ! -d checkpoints/${INPUT_MODEL} ]; then
      python scripts/download.py --repo_id ${INPUT_MODEL}
      python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/${INPUT_MODEL}
      python ./scripts/prepare_alpaca.py --data_file_name train_alpaca.jsonl  --checkpoint_dir ./checkpoints/${INPUT_MODEL}
    fi
    python generate/base.py --prompt "Hello, my name is" --checkpoint_dir checkpoints/${INPUT_MODEL}
    pip install -U deepspeed
    echo "You may need to edit adapter_v2 to change the hard coded number of devices."
    echo "Training part 1"
    time python finetune/adapter_v2.py --checkpoint_dir checkpoints/${INPUT_MODEL} --out_dir adv2_ft --data_dir data/alpaca/ --precision bf16-mixed
  else
    # We can run our own sketchy script too! But the result does not seem to produce a fully functioning model out of the box
    # We might be able to copy some stuff from the src model and magic it but idk.
    pip install -q -U bitsandbytes
    pip install -q -U git+https://github.com/huggingface/transformers.git 
    pip install -q -U git+https://github.com/huggingface/peft.git
    pip install -q -U git+https://github.com/huggingface/accelerate.git
    pip install -q datasets
    python train.py --input-model ${INPUT_MODEL} --training-dataset out_oa ${QLORA}
    python test_new_model.py --input-model ${INPUT_MODEL} --fine-tuned results/finetuned
  fi
fi
