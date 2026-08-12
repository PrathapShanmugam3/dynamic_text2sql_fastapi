#!/bin/bash
set -e
export ADAPTER_PATH="${ADAPTER_PATH:-$(pwd)/models/texttosql/results/checkpoint-3480}"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
