#!/usr/bin/env bash
set -euo pipefail

# H100 smoke benchmark for DFlash compile-cache isolation by speculative depth.
# Usage:
#   TARGET_MODEL=... DRAFT_MODEL=... ./benchmarks/bench_dflash_depth_cache_key.sh

: "${TARGET_MODEL:?set TARGET_MODEL to the target checkpoint}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the DFlash checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/vllm-dflash-depth-bench}"
mkdir -p "${OUTPUT_DIR}"

for depth in 8 9; do
  VLLM_USE_PRECOMPILED=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  vllm bench latency \
    --model "${TARGET_MODEL}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --language-model-only \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_MODEL}\",\"num_speculative_tokens\":${depth}}" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}' \
    --batch-size 1 \
    --input-len 256 \
    --output-len 128 \
    --num-iters-warmup "${WARMUP_ITERS:-2}" \
    --num-iters "${BENCH_ITERS:-3}" \
    --seed 1234 \
    --output-json "${OUTPUT_DIR}/depth-${depth}.json"
done

echo "Results written to ${OUTPUT_DIR}/depth-{8,9}.json"
