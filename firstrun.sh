#!/bin/bash

set -ex

cd "$(dirname "$0")"

source setup.sh

gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv | tail -n 1 | cut -f 1 -d " " |  awk '{s+=$1} END {print s}')

if [ ! -f ".firstrun" ]; then
  # See https://askubuntu.com/questions/272248/processing-triggers-for-man-db
  # echo "set man-db/auto-update false" | sudo debconf-communicate; sudo dpkg-reconfigure man-db
  if ! command -v sudo &> /dev/null
  then
    apt-get update
    apt-get install -y sudo
  fi

  sudo apt-get update
  sudo apt-get install -y libaio-dev python3-pybind11 screen nano emacs lbzip2
  # Use parallel bzip2
  sudo rm $(which bzip2)
  sudo ln -s $(which lbzip2) /bin/bzip2
  sudo rm $(which bunzip2)
  sudo ln -s $(which lbunzip2) /bin/bunzip2
  nvcc --version || sudo apt install nvidia-cuda-toolkit

  python3 -m pip install --upgrade pip
  # We need to install pybind11 before deepspeed because it is not listed as a depdency.
  pip install pybind11[global]
  pip install packaging
  pip install ninja
  # We need good pytorch v soon
  CU_MINOR=$(nvcc --version |grep "cuda_" |cut -d "_" -f 2 |cut -d "." -f 2)
  #pip install -U "torch>=2.1.1" --index-url https://download.pytorch.org/whl/cu11${CU_MINOR} || pip install -U "torch>=2.1.1"
  # We need to install xformers at the same time so we get a compatible torch version.
  pip install -U "xformers<=0.0.26.post1" "torch>=2.1.1" --index-url https://download.pytorch.org/whl/cu11${CU_MINOR} || pip install -U "xformers<=0.0.26.post1" "torch>=2.1.1"

  if [ ! -d axolotl ]; then
    git clone https://github.com/OpenAccess-AI-Collective/axolotl
  fi
  cd axolotl
  pip install packaging
  pip install ninja
  # flash-attn hack version magic
  #pip install -e '.' "flash-attn==2.3.6"
  pip install -U -e '.[flash-attn,deepspeed]'
  pip install -U git+https://github.com/huggingface/peft.git
  cd ..
  if [ ! -d apex ]; then
    git clone https://github.com/NVIDIA/apex
  fi
  cd apex
  sudo pip install -U pip
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--deprecated_fused_adam" ./ || pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ || echo "skipping apex."
  cd ..

  # deepspeed
  pip install ninja hjson py-cpuinfo
  DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_FUSED_ADAM=1 pip install "git+https://github.com/microsoft/deepspeed.git#" --global-option="build_ext" --global-option="-j16"
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
      python -m bitsandbytes | grep "The installed version of bitsandbytes was compiled without GPU support." || ./setup_bits_and_bytes.sh || echo "No b and b for you."
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
