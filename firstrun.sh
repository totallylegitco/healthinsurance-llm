#!/bin/bash

set -ex

if [ ! -f ".firstrun" ]; then
  # See https://askubuntu.com/questions/272248/processing-triggers-for-man-db
  # echo "set man-db/auto-update false" | sudo debconf-communicate; sudo dpkg-reconfigure man-db

  sudo apt-get update
  sudo apt-get install -y libaio-dev python3-pybind11

  python3 -m pip install --upgrade pip
  # We need to install pybind11 before deepspeed because it is not listed as a depdency.
  pip install pybind11[global]
  pip install packaging
  # We need good pytorch v soon
  CU_MINOR=$(nvcc --version |grep "cuda_" |cut -d "_" -f 2 |cut -d "." -f 2)
  pip install "torch==2.0.1" --index-url https://download.pytorch.org/whl/cu11${CU_MINOR} || pip install torch

  if [ ! -d axolotl ]; then
    git clone https://github.com/OpenAccess-AI-Collective/axolotl
  fi
  cd axolotl
  pip install -e '.[flash-attn,deepspeed]'
  pip install -U git+https://github.com/huggingface/peft.git
  cd ..
  if [ ! -d apex ]; then
    git clone https://github.com/NVIDIA/apex
  fi
  cd apex
  sudo pip install -U pip
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--deprecated_fused_adam" ./
  cd ..

  # deepspeed
  pip install ninja hjson py-cpuinfo
  DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_FUSED_ADAM=1 pip install "git+https://github.com/microsoft/deepspeed.git#" --global-option="build_ext" --global-option="-j16"
  pip3 install -U -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu11${CU_MINOR}
  ds_report
  # Setup bits and bytes if we are likely to need it.
  if [ ${gpu_memory} -lt 495640 ]; then
    if [ $(uname -m) == "aarch64" ]; then
      # On ARM for bits and bytes we need neon
      if [ ! -d sse2neon ]; then
	git clone https://github.com/DLTcollab/sse2neon.git
	make
	sudo cp sse2neon.h /usr/include/
      fi
    else
      python -m bitsandbytes || pip3 install -U bitsandbytes
      python -m bitsandbytes || ./setup_bits_and_bytes.sh
      python -m bitsandbytes | grep "The installed version of bitsandbytes was compiled without GPU support." || ./setup_bits_and_bytes.sh
    fi
  fi

  if [ ! -d dolly ]; then
    git clone https://github.com/databrickslabs/dolly.git
  fi

  if [ ! -d "appeals-llm-data" ]; then
    git clone https://github.com/totallylegitco/appeals-llm-data.git
  fi

  if [ ! -d out ]; then
    mkdir out
  fi

  if [ ! -d falcontune ]; then
    git clone https://github.com/rmihaylov/falcontune.git
  fi

  if [ ! -d llama-recipes ]; then
    git clone https://github.com/facebookresearch/llama-recipes.git
  fi

  touch .firstrun
fi
