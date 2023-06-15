#!/bin/bash

set -ex

if [ -z "$1" ]; then
  echo "Usage: ./run_remote.sh [remotehost] (e.g. ubuntu@farts.com)"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


temp_dir=$(mktemp -d -t "fartsXXXXXXXXXXX")
filename=upload.tbz2
target="${temp_dir}/${filename}"

tar --exclude="dolly" --exclude "combined-llm-data*" --exclude "generated-llm-data*" -cjf "${target}" .

scp ${target} $1:~/
scp ~/.ssh/authorized_keys  $1:~/.ssh/
ssh $1 "tar -C ./ -xjf ${filename}"
ssh -t $1 "screen ./run.sh"

rm -rf ${temp_dir}
