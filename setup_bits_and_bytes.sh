#!/bin/bash
set -ex

if [ ! -d bitsandbytes ]; then
  git clone https://github.com/TimDettmers/bitsandbytes.git
fi
cd bitsandbytes
BUILD_COMMAND=$(python -m bitsandbytes 2>&1 |grep "make" |grep "CUDA_VERSION" |grep -v rm |tail -n 1)
`$BUILD_COMMAND`
python setup.py install
cd ..
