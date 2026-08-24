#!/bin/bash
# Runs HumanEval+ and MBPP+ (EvalPlus) against the vLLM server whose
# host:port is advertised in scripts/server_endpoint.txt by serve_vllm.sbatch.
set -euo pipefail

PROJECT_DIR="/scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl"
ENDPOINT_FILE="$PROJECT_DIR/scripts/server_endpoint.txt"
RESULTS_DIR="$PROJECT_DIR/eval_results"
MODEL_NAME="qwen3.6-35b-a3b-fp8"

source /etc/profile.d/modules.sh
module load Anaconda3/2025.12-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-eval

if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "No server_endpoint.txt found — is the vLLM sbatch job running?" >&2
    exit 1
fi

ENDPOINT=$(cat "$ENDPOINT_FILE")
BASE_URL="http://${ENDPOINT}/v1"

echo "Waiting for vLLM server at $BASE_URL to become healthy..."
until curl -sf "http://${ENDPOINT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "Server is up. Running EvalPlus."

for DATASET in humaneval mbpp; do
    echo "=== ${DATASET}+ generation ==="
    python -m evalplus.codegen \
        --model "$MODEL_NAME" \
        --dataset "$DATASET" \
        --backend openai \
        --base_url "$BASE_URL" \
        --greedy \
        --root "$RESULTS_DIR"

    echo "=== ${DATASET}+ evaluation ==="
    SAMPLE_FILE="$RESULTS_DIR/${DATASET}/${MODEL_NAME}_openai_temp_0.0.jsonl"
    python -m evalplus.evaluate \
        --dataset "$DATASET" \
        --samples "$SAMPLE_FILE" \
        | tee "$RESULTS_DIR/${DATASET}_plus_report.txt"
done

echo "Done. Reports in $RESULTS_DIR/{humaneval,mbpp}_plus_report.txt"
