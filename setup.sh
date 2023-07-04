export PATH=$PATH:~/.local/bin:/usr/lib/x86_64-linux-gnu
if [ -z "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH=$PATH
fi

INPUT_MODEL=${INPUT_MODEL:-"databricks/dolly-v2-7b"}
TR_DATA=${TR_DATA:-"out"}
OUTDIR=${OUTDIR:-"new_model"}
# Only in holden's branch and even then it its kind of funky.
#QLORA=${QLORA:-"--qlora-4bit true"}
EPOCHS=${EPOCHS:-"10"}


# Figure out our version of CUDA so we can install the right package
if nvcc --version |grep -q 11.8; then
  extra_url=https://download.pytorch.org/whl/nightly/cu118
elif nvidia-smi  |grep "CUDA Version" |grep -q "11.7"; then
  extra_url=https://download.pytorch.org/whl/nightly/cu117
elif nvcc --version |grep -q 11.6; then
  extra_url=https://download.pytorch.org/whl/nightly/cu116
else
  extra_url=https://download.pytorch.org/whl/nightly
fi

