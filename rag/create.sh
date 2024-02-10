#!/bin/bash

export OPENAI_API_KEY="fake"
export LOCAL_MODEL="/TotallyLegitCo/fighthealthinsurance_model_v0.3"
export OPENAI_BASE_API="http://localhost:8000"

set -ex

python -m rag.initial
