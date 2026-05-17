#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash benchmark/unicycle_open/inference/run_t2t_inference.sh <backend> <input> <output> <out_image_dir> [extra args...]

Backends:
  bagel
  janus
  showo

Examples:
  bash benchmark/unicycle_open/inference/run_t2t_inference.sh \
    bagel \
    benchmark/unicycle_open/data/T2T_qa_V1_cleaned.jsonl \
    benchmark/unicycle_open/output/v1_bagel_predictions.jsonl \
    benchmark/unicycle_open/output/v1_bagel_images \
    --model_dir path/to/BAGEL-7B-MoT

  bash benchmark/unicycle_open/inference/run_t2t_inference.sh \
    janus \
    benchmark/unicycle_open/data/T2T_qa_V1_cleaned.jsonl \
    benchmark/unicycle_open/output/v1_janus_predictions.jsonl \
    benchmark/unicycle_open/output/v1_janus_images \
    --model_path deepseek-ai/Janus-1.3B

  bash benchmark/unicycle_open/inference/run_t2t_inference.sh \
    showo \
    benchmark/unicycle_open/data/T2T_qa_V1_cleaned.jsonl \
    benchmark/unicycle_open/output/v1_showo_predictions.jsonl \
    benchmark/unicycle_open/output/v1_showo_images \
    --config path/to/showo_eval.yaml \
    --batch_size 1 \
    --guidance_scale 4.0 \
    --num_inference_steps 50
EOF
  exit 1
fi

BACKEND="$1"
INPUT="$2"
OUTPUT="$3"
OUT_IMAGE_DIR="$4"
shift 4

python "${SCRIPT_DIR}/run_t2t_inference.py" \
  --backend "${BACKEND}" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --out_image_dir "${OUT_IMAGE_DIR}" \
  "$@"
