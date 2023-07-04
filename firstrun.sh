#!/bin/bash

set -ex

if [ ! -f ".firstrun" ]; then
  touch .firstrun
  python3 -m pip install --upgrade pip
  pip3 install -U -r requirements.txt
  # Setup bits and bytes if we are likely to need it.
  if [ ${gpu_memory} -lt 49564 ]; then
    if [ $(uname -m) == "aarch64" ]; then
      # On ARM for bits and bytes we need neon
      if [ ! -d sse2neon ]; then
	git clone https://github.com/DLTcollab/sse2neon.git
	make
	sudo cp sse2neon.h /usr/include/
      fi
    else
      pip3 install -U bitsandbytes
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
fi
