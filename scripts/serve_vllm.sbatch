#!/bin/bash
#SBATCH --job-name=qwen3.6-vllm
#SBATCH --partition=def
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="/scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl"
MODEL_DIR="$PROJECT_DIR/models/Qwen3.6-35B-A3B-FP8"
PORT=8000

source /etc/profile.d/modules.sh
module load Anaconda3/2025.12-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-eval

# Both DeepGEMM and flashinfer JIT-compile CUDA kernels at runtime via nvcc.
# That fails here: the pip-installed nvidia-cuda-nvcc (13.3.73) and
# nvidia-cuda-runtime (13.0.96) are mismatched versions, and CCCL's
# cuda_toolkit.h compatibility check correctly rejects the combination
# ("CUDA compiler and CUDA toolkit headers are incompatible"). Rather than
# chase matching pip package versions, skip both JIT paths and use vLLM's
# precompiled kernels instead.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_HOME="$(python -c "
import nvidia, os
for p in nvidia.__path__:
    cu13 = os.path.join(p, 'cu13')
    if os.path.isdir(cu13):
        print(cu13)
")"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# Advertise where this server is running so client scripts can find it.
echo "$(hostname):$PORT" > "$PROJECT_DIR/scripts/server_endpoint.txt"

# SWE-bench phase config: thinking mode ON (default chat template, no
# override — unlike the EvalPlus run, our own agent scaffold controls its
# own max_tokens per turn, so there's no hardcoded budget forcing thinking
# off) and a 128K context ceiling (up from EvalPlus's 32K) since multi-turn
# agent trajectories — repo exploration, edits, test output — accumulate
# far more context than a single-shot code-gen prompt. See
# scripts/chat_template_no_think.jinja / git history for the EvalPlus config
# this replaced.
exec vllm serve "$MODEL_DIR" \
    --served-model-name qwen3.6-35b-a3b-fp8 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --tensor-parallel-size 1
