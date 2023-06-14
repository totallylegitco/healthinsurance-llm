#!/bin/bash
set -ex

sudo apt-get install locate
sudo updatedb

# On Jetson AGX we need to add /usr/local/cuda/bin to our path
if [ -f /usr/local/cuda/bin/nvcc ]; then
  export PATH=$PATH:/usr/local/cuda/bin
fi


if [ ! -d bitsandbytes ]; then
  git clone https://github.com/TimDettmers/bitsandbytes.git
fi
cd bitsandbytes
BUILD_COMMAND=$(python -m bitsandbytes 2>&1 |grep "make" |grep "CUDA_VERSION" |grep -v rm |grep -v "for example" |tail -n 1)
bash -c "${BUILD_COMMAND}"
sudo python setup.py install
cd ..
