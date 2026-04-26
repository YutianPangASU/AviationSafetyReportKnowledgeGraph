#!/usr/bin/env bash
# Launch an OpenAI-compatible vLLM server hosting Qwen3.6-35B-A3B.
#
# Hardware profile: 4x RTX 6000 Ada (48GB each).
#   TP=2, leaving 2 GPUs free for a second replica or other jobs.
#   Pass GPUS=0,1 or GPUS=2,3 to pick which cards this replica uses.
#
# Two-replica (DP=2) pattern for max throughput: run this script twice,
# first with GPUS=0,1 PORT=8000, then GPUS=2,3 PORT=8001. The extraction
# client round-robins across ports.
#
# Thinking mode is DISABLED for structured extraction. The default Qwen3.x
# prompt template enables thinking, which wastes output tokens on reasoning
# traces we would just strip. vLLM forwards `chat_template_kwargs` from the
# client, so we flip `enable_thinking: false` at request time — see
# extract_vllm.py.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/yp6443/.cache/huggingface/hub/Qwen--Qwen3.6-35B-A3B}"
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"       # keep modest; pilot narratives << 32K
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
LOG_DIR="${LOG_DIR:-$(dirname "$0")/../logs}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "Serving $SERVED_NAME on port $PORT (GPUS=$GPUS, TP=$TP)"
echo "Log: $LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPUS" \
/home/yp6443/miniconda3/envs/qwen-vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --enable-prefix-caching \
    2>&1 | tee "$LOG_FILE"
