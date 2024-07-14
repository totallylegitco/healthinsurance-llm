#!/bin/bash

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}

HOST=${1:-${HOST}}
PORT=${2:-${PORT:-22}}
TARGET_DIR=${3:-${TARGET_DIR:-"~/"}}

TARGET_DIR=$(echo $TARGET_DIR | sed 's![^/]$!&/!')

set -ex

if [ -z "$HOST" ]; then
  echo "Usage: ./run_remote.sh [remotehost] (e.g. ubuntu@farts.com)"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


temp_dir=$(mktemp -d -t "fartsXXXXXXXXXXX")
filename=upload.tbz2
target="${temp_dir}/${filename}"

tar --exclude apex --exclude axolotl --exclude lit-parrot --exclude falcontune --exclude "*.tbz2" --exclude wandb --exclude "results" --exclude="dolly" --exclude "out_*" --exclude "combined-llm-data*" --exclude "generated-llm-data*" --exclude "*_out" --exclude "falcon-7b-instruct-alpaca" --exclude "*_model_*" --exclude "*_out_*" --exclude "apex" --exclude "other" --exclude "storage" --exclude "*/runs/*" --exclude "backup-generated" --exclude "miniconda" --exclude "last_run_prepared" -cjf "${target}" . &
tpid=$!

# Copy over firstrun.sh so we can get the system setup a bit while we transfer all of ourdata.
# Lambda labs seems to be having isssues with ssh not coming up quickly so retry.
(scp -P $PORT ./firstrun.sh $HOST:${TARGET_DIR} || (sleep 120 && scp -P $PORT ./firstrun.sh $HOST:${TARGET_DIR} ) || (sleep 120 && scp -P $PORT ./firstrun.sh $HOST:${TARGET_DIR} ))
scp -P $PORT ./setup.sh $HOST:${TARGET_DIR}
scp -P $PORT ./requirements.txt $HOST:${TARGET_DIR}
ssh -p $PORT $HOST "${TARGET_DIR}/firstrun.sh | tee -a fr.log" &
frpid=$!
# Put the passwordless https config there
# scp -P $PORT ~/.ssh/authorized_keys  $HOST:~/.ssh/
# ssh -p $PORT -t $HOST "sudo apt-get update && sudo apt-get upgrade -y" &
wait ${tpid}
# Race condition with tbz2 file not being written all the way
sync
sleep 1
scp -P $PORT ${target} $HOST:${TARGET_DIR}
echo "Preparing to decompress ${TARGET_DIR}${filename}"
ssh -p $PORT $HOST "tar -C ${TARGET_DIR} -xjf ${TARGET_DIR}${filename}" &
wait $!
ssh -p $PORT $HOST "mkdir -p ~/.git"
scp -P $PORT remote_git $HOST:~/.git/config
# Copy git credentials for huggingface access.
scp -P $PORT ~/.gitconfig $HOST:~/.gitconfig
scp -P $PORT ~/.config/git/credentials $HOST:~/.config/git/credentials || scp -P $PORT ~/.git-credentials $HOST:~/
ssh -p $PORT $HOST mkdir -p "~/.cache/huggingface"
scp -P $PORT ~/.cache/huggingface/token $HOST:~/.cache/huggingface/token
wait ${frpid}
ssh -p $PORT -t $HOST "QLORA=\"${QLORA}\" INPUT_MODEL=\"${INPUT_MODEL}\" screen ${TARGET_DIR}/run.sh"
rm -rf ${temp_dir}
