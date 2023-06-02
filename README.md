# healthinsurance-llm
LLM for Generating Health Insurance Appeals

## Jetson AGX Notes:

Install torch from https://developer.download.nvidia.cn/compute/redist/jp/v51/pytorch/torch-1.14.0a0+44dac51c.nv23.02-cp38-cp38-linux_aarch64.whl (needs torch < 2 >=1.13) otherwise no GPU support.
If you get `module 'torch.distributed' has no attribute 'ReduceOp'` uninstall `deepspeed` since it does not play nicely with the NVIDIA torch fork. idk why but similar issues exist on OSX (see https://github.com/microsoft/DeepSpeed/issues/2830).
