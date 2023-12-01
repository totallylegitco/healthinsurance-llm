PORT=${PORT:-8000}
docker run -d --gpus all     -p ${PORT}:8000     -v  /home/holden/repos/healthinsurance-llm/fighthealthinsurance_model_v0.2:/fighthealthinsurance_model_v0.2 ghcr.io/mistralai/mistral-src/vllm:latest     --host 0.0.0.0     --model="/fighthealthinsurance_model_v0.2"
