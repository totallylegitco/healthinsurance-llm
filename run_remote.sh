#!/bin/bash

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}

set -ex

if [ -z "$1" ]; then
  echo "Usage: ./run_remote.sh [remotehost] (e.g. ubuntu@farts.com)"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


temp_dir=$(mktemp -d -t "fartsXXXXXXXXXXX")
filename=upload.tbz2
target="${temp_dir}/${filename}"

tar --exclude lit-parrot --exclude falcontune --exclude "*.tbz2" --exclude wandb --exclude "results" --exclude="dolly" --exclude "combined-llm-data*" --exclude "generated-llm-data*" -cjf "${target}" .

# Lambda labs seems to be having isssues with ssh not coming up quickly so retry.
(scp ${target} $1:~/ || (sleep 120 && scp ${target} $1:~/ ))
# Put the passwordless https config there
scp ~/.ssh/authorized_keys  $1:~/.ssh/
# ssh -t $1 "sudo apt-get update && sudo apt-get upgrade -y" &
ssh $1 "tar -C ./ -xjf ${filename}"
scp remote_git $1:~/.git/config
ssh -t $1 "QLORA=\"${QLORA}\" INPUT_MODEL=\"${INPUT_MODEL}\" screen ./run.sh"

rm -rf ${temp_dir}
