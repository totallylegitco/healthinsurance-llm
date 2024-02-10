#!/bin/bash

set -ex

source setup.sh

cd "$(dirname "$0")"

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | cut -f 1 -d " " |  awk '{s+=$1} END {print s}')

if [ ! -f ".firstrun" ]; then
  ./firstrun.sh
fi

if [ ! -f ".fetched_data" ]; then
  ./fetch_data.sh
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
if [ "$INPUT_MODEL" == "mistral" ]; then
  mkdir -p out
  mkdir -p mistral_fine_out
  if [ "${TR_DATA}" != "out" ]; then
    cp ./${TR_DATA}/train_alpaca.jsonl ./out/
  fi
  if [ ${gpu_memory} -gt 70000 ]; then
    accelerate launch -m axolotl.cli.train mistral_config.yml
  else
    accelerate launch -m axolotl.cli.train mistral_config_qlora.yml
  fi
elif [ "${INPUT_MODEL}" == "databricks/dolly-v2-7b" ]; then
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
  mkdir -p llamav2-updated
  mkdir -p llama-input
  if [ ! -f ./llama-input/train_alpaca.jsonl ]; then
    cp ./out/train_alpaca.jsonl ./llama-input/
  fi
  pip install -r ./llama-recipes/requirements.txt
  # deepspeed python llamav2_finetune_from_databricks.py --local-output-dir ./llamav2-updated | tee -a ~/train.log ||
  # python llamav2_finetune_from_databricks.py --local-output-dir ./llamav2-updated --disable-deepspeed true --training-dataset llama-input | tee -a ~/train_nods.log
  python llama-recipes/llama_finetuning.py --use_peft --peft_method lora --quantization --model_name ${INPUT_MODEL} --output_dir llamav2-updated | tee -a ~/train_fb.log
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
