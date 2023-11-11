#!/bin/bash

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}

HOST=${1:-${HOST}}
PORT=${2:-${PORT:-22}}

set -ex

if [ -z "$HOST" ]; then
  echo "Usage: ./run_remote.sh [remotehost] (e.g. ubuntu@farts.com)"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


temp_dir=$(mktemp -d -t "fartsXXXXXXXXXXX")
filename=upload.tbz2
target="${temp_dir}/${filename}"

tar --exclude lit-parrot --exclude falcontune --exclude "*.tbz2" --exclude wandb --exclude "results" --exclude="dolly" --exclude "combined-llm-data*" --exclude "generated-llm-data*" -cjf "${target}" . &
tpid=$!

# Copy over firstrun.sh so we can get the system setup a bit while we transfer all of ourdata.
# Lambda labs seems to be having isssues with ssh not coming up quickly so retry.
(scp -P $PORT ./firstrun.sh $HOST:~/ || (sleep 120 && scp -P $PORT ./firstrun.sh $HOST:~/ ))
scp -P $PORT ./requirements.txt $HOST:~/
ssh -p $PORT $HOST "./firstrun.sh | tee -a fr.log" &
frpid=$!
# Put the passwordless https config there
scp -P $PORT ~/.ssh/authorized_keys  $HOST:~/.ssh/
# ssh -p $PORT -t $HOST "sudo apt-get update && sudo apt-get upgrade -y" &
wait ${tpid}
# Race condition with tbz2 file not being written all the way
sync
sleep 1
scp -P $PORT ${target} $HOST:~/
ssh -p $PORT $HOST "tar -C ./ -xjf ${filename}" &
wait $!
scp -P $PORT remote_git $HOST:~/.git/config
# Copy git credentials for huggingface access.
scp -P $PORT ~/.gitconfig $HOST:~/.gitconfig
scp -P $PORT ~/.config/git/credentials $HOST:~/.config/git/credentials || scp -P $PORT ~/.git-credentials $HOST:~/
ssh -p $PORT $HOST mkdir -p "~/.cache/huggingface"
scp -P $PORT ~/.cache/huggingface/token $HOST:~/.cache/huggingface/token
wait ${frpid}
ssh -p $PORT -t $HOST "QLORA=\"${QLORA}\" INPUT_MODEL=\"${INPUT_MODEL}\" screen ./run.sh"
rm -rf ${temp_dir}
