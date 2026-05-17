#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash benchmark/unicycle_open/evaluation/run_t2t_eval.sh --backend <backend> --bench <bench_jsonl> --output_dir <output_dir> [extra args...]

Examples:
  bash benchmark/unicycle_open/evaluation/run_t2t_eval.sh \
    --backend bagel \
    --bench benchmark/unicycle_open/data/tiif_style.jsonl \
    --output_dir benchmark/unicycle_open/output \
    --model_dir path/to/BAGEL-7B-MoT \
    --api_key "$OPENAI_API_KEY"

  bash benchmark/unicycle_open/evaluation/run_t2t_eval.sh \
    --backend janus \
    --bench benchmark/unicycle_open/data/self_built.jsonl \
    --output_dir benchmark/unicycle_open/output \
    --model_path deepseek-ai/Janus-1.3B \
    --api_key "$OPENAI_API_KEY"

  bash benchmark/unicycle_open/evaluation/run_t2t_eval.sh \
    --backend showo \
    --bench benchmark/unicycle_open/data/tiif_style.jsonl \
    --output_dir benchmark/unicycle_open/output \
    --config path/to/showo_eval.yaml \
    --api_key "$OPENAI_API_KEY"
EOF
  exit 1
fi

python "${SCRIPT_DIR}/run_t2t_eval.py" "$@"
