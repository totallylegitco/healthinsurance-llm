#!/bin/bash

export OPENAI_API_KEY="EMPTY"
export LOCAL_MODEL="/TotallyLegitCo/fighthealthinsurance_model_v0.3"
export OPENAI_BASE_API="http://localhost:8000/v1"

set -ex

python -m rag.initial
