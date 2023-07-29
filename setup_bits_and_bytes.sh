#!/bin/bash
set -ex

sudo apt-get install locate
sudo updatedb

if [ ! -d bitsandbytes ]; then
  if [ $(uname -m) == "x86_64" ]; then
    git clone https://github.com/TimDettmers/bitsandbytes.git
  else
    git clone https://github.com/g588928812/bitsandbytes_jetsonX.git
  fi
fi
cd bitsandbytes
BUILD_COMMAND=$(python -m bitsandbytes 2>&1 |grep "make" |grep "CUDA_VERSION" |grep -v rm |grep -v "for example" |tail -n 1)
if [ -z "${BUILD_COMMAND}" ]; then
  sudo make CUDA_VERSION=116
else
  bash -c "${BUILD_COMMAND}"
fi
sudo python setup.py install
cd ..
