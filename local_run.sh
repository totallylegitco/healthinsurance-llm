PORT=${PORT:-8000}
docker run -d --gpus all     -p ${PORT}:8000     -v  /home/holden/repos/healthinsurance-llm/fighthealthinsurance_model_v0.5:/TotallyLegitCo/fighthealthinsurance_model_v0.5 vllm/vllm-openai:latest     --host 0.0.0.0  --gpu-memory-utilization=0.9 --max-model-len=26280 --enforce-eager   --model="/TotallyLegitCo/fighthealthinsurance_model_v0.5"
