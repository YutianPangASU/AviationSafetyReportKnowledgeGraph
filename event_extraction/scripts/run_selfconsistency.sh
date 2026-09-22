#!/usr/bin/env bash
# Self-consistency of the extraction (review 2026-09-14, comment 5).
#
# Re-extracts the 2,000-report evaluation sample, narrative-only, three
# times at the production temperature (0.2) and once at 0.7, on two vLLM
# replicas, then scores every run with the same judge as the main
# evaluation. score_selfconsistency.py aggregates the node-set agreement
# across runs and the spread of recall.
#
# Usage:  nohup bash event_extraction/scripts/run_selfconsistency.sh &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUTD=event_extraction/out/selfconsistency
LOGD=event_extraction/logs
mkdir -p "$OUTD" "$LOGD"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGD/selfconsistency_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] launching two vLLM replicas"

GPUS=0,1 PORT=8000 nohup bash event_extraction/scripts/serve_qwen.sh \
    > "$LOGD/vllm_sc_8000_$STAMP.log" 2>&1 &
GPUS=2,3 PORT=8001 nohup bash event_extraction/scripts/serve_qwen.sh \
    > "$LOGD/vllm_sc_8001_$STAMP.log" 2>&1 &

for i in $(seq 1 180); do
  if curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 \
     && curl -s -m 3 http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "[$(date)] both replicas ready after ~$((i*10)) s"; break
  fi
  sleep 10
done
curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: :8000 never came up"; exit 1; }

run_one () {  # tag temperature
  local tag=$1 temp=$2
  echo "[$(date)] extraction $tag (temperature $temp)"
  python3 event_extraction/scripts/extract_vllm.py \
      --input-records event_extraction/out/calibration_2k_input.jsonl \
      --out "$OUTD/calibration_2k_v4_narrative_$tag.jsonl" \
      --supervision none --temperature "$temp" \
      --endpoints http://localhost:8000/v1 --endpoints http://localhost:8001/v1 \
      --concurrency 16 --resume
  echo "[$(date)] judge $tag"
  python3 event_extraction/scripts/semantic_eval.py \
      --extraction "$OUTD/calibration_2k_v4_narrative_$tag.jsonl" \
      --tag "sc-$tag" --out "$OUTD/semantic_eval_$tag.jsonl" \
      --endpoint http://localhost:8001/v1
}

run_one run1 0.2
run_one run2 0.2
run_one run3 0.2
run_one t07 0.7

echo "[$(date)] stopping replicas"
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "[$(date)] done"
