#!/usr/bin/env bash
# Session-independent Stage-3 v4 pipeline (Phase 5 of the v4 plan).
# Launch detached:  nohup setsid bash event_extraction/scripts/run_stage3_v4_pipeline.sh \
#                     >> event_extraction/logs/stage3_v4_pipeline.log 2>&1 &
#
# Expects two vLLM replicas coming up on :8000/:8001 (only piece 2 needs them;
# they are shut down right after to free the GPUs).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root
PY=/home/yp6443/miniconda3/envs/qwen-vllm/bin/python

echo "=== $(date) waiting for vLLM replica on :8000 (piece 2 needs it) ==="
for i in $(seq 1 90); do
  curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "replica :8000 ready (~${i}0s)"; break; }
  sleep 10
done
curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1 || { echo "FATAL: :8000 never came up"; exit 1; }

echo "=== $(date) S3v4 piece 1: precedence matrices (47-factor vocab) ==="
$PY event_extraction/scripts/stage3/build_precedence_matrices.py || exit 1
echo "=== $(date) S3v4 piece 2: LLM causal order ==="
$PY event_extraction/scripts/stage3/llm_causal_order.py --concurrency 16 || exit 1

echo "=== $(date) piece 2 done — shutting down vLLM servers to free GPUs ==="
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=== $(date) S3v4 piece 3: laplacian prior ==="
$PY event_extraction/scripts/stage3/laplacian_prior.py || exit 1
echo "=== $(date) S3v4 piece 4: PC per category ==="
$PY event_extraction/scripts/stage3/run_pc_per_category.py || exit 1
echo "=== $(date) S3v4 piece 4-ablation: PC no-prior ==="
$PY event_extraction/scripts/stage3/run_pc_per_category.py --no-laplacian --no-llm-order \
    --out-dir event_extraction/out/aggregate_kg/per_category_dag_noprior || exit 1
echo "=== $(date) S3v4 piece 4.5: reorient ==="
$PY event_extraction/scripts/stage3/reorient_dag.py || exit 1
echo "=== $(date) S3v4 piece 6: viz ==="
$PY event_extraction/scripts/stage3/viz_dag.py || exit 1
echo "=== $(date) S3v4 bootstrap (key categories, 30 resamples) ==="
$PY event_extraction/scripts/stage3/bootstrap_stability.py --bootstraps 30 \
    --categories LOC-I CFIT SCF-PP LOC-G ICE ARC WSTRW USOS MAC RE WX RI || exit 1
echo "=== $(date) S3v4 BN do() smoke test on LOC-I ==="
$PY event_extraction/scripts/stage3/bn_do_smoke_test.py --category LOC-I || exit 1
echo "=== $(date) STAGE-3 v4 PIPELINE COMPLETE ==="
