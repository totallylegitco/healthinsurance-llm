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

tar --exclude lit-parrot --exclude falcontune --exclude "*.tbz2" --exclude wandb --exclude "results" --exclude="dolly" --exclude "combined-llm-data*" --exclude "generated-llm-data*" -cjf "${target}" . &
tpid=$!

# Copy over firstrun.sh so we can get the system setup a bit while we transfer all of ourdata.
# Lambda labs seems to be having isssues with ssh not coming up quickly so retry.
(scp ./firstrun.sh $1:~/ || (sleep 120 && scp ./firstrun.sh $1:~/ ))
scp ./requirements.txt $1:~/
ssh $1 "./firstrun.sh" &
frpid=$!
# Put the passwordless https config there
scp ~/.ssh/authorized_keys  $1:~/.ssh/
# ssh -t $1 "sudo apt-get update && sudo apt-get upgrade -y" &
wait ${tpid}
# Race condition with tbz2 file not being written all the way
sync
sleep 1
scp ${target} $1:~/
ssh $1 "tar -C ./ -xjf ${filename}" &
wait
scp remote_git $1:~/.git/config
# Copy git credentials for huggingface access.
scp ~/.gitconfig $1:~/.gitconfig
scp ~/.config/git/credentials $1:~/.config/git/credentials || scp ~/.git-credentials $1:~/
ssh $1 mkdir -p "~/.cache/huggingface"
scp ~/.cache/huggingface/token $1:~/.cache/huggingface/token
ssh -t $1 "QLORA=\"${QLORA}\" INPUT_MODEL=\"${INPUT_MODEL}\" screen ./run.sh"
rm -rf ${temp_dir}
