#!/bin/bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_SANITY="${RUN_SANITY:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

AUDIT_OUTPUT_DIR="${AUDIT_OUTPUT_DIR:-outputs/safesim_target_audit}"
SANITY_BASE_DIR="${SANITY_BASE_DIR:-./safesim_logs_transfer_sanity}"
STAGE1_LOG_DIR="${STAGE1_LOG_DIR:-./safesim_logs_stage1}"
STAGE2_LOG_DIR="${STAGE2_LOG_DIR:-./safesim_logs_stage2}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/stage2_eval}"

TF_D_MODEL="${TF_D_MODEL:-1024}"
INIT_MODE="${INIT_MODE:-fm_head_conservative}"
TARGET_POLICY="${TARGET_POLICY:-}"

if [ "${RUN_AUDIT}" = "1" ]; then
    python scripts/analysis/audit_safesim_filtered_targets.py \
        --output_dir "${AUDIT_OUTPUT_DIR}"
fi

if [ -z "${TARGET_POLICY}" ]; then
    AUDIT_JSON="${AUDIT_OUTPUT_DIR}/audit_summary.json"
    if [ ! -f "${AUDIT_JSON}" ]; then
        echo "Missing ${AUDIT_JSON}. Run audit first or set TARGET_POLICY explicitly." >&2
        exit 1
    fi
    TARGET_POLICY=$(python - <<PY
import json
from pathlib import Path
path = Path("${AUDIT_JSON}")
data = json.loads(path.read_text())
print(data["recommended_policy"])
PY
)
fi

if [ "${RUN_SANITY}" = "1" ]; then
    BASE_DIR="${SANITY_BASE_DIR}" TF_D_MODEL="${TF_D_MODEL}" sh scripts/training/run_safesim_transfer_sanity.sh
fi

if [ "${RUN_STAGE1}" = "1" ]; then
    LOG_DIR="${STAGE1_LOG_DIR}" TF_D_MODEL="${TF_D_MODEL}" INIT_MODE="${INIT_MODE}" sh scripts/training/run_safesim_stage1.sh
fi

STAGE1_BEST=""
if [ "${RUN_STAGE2}" = "1" ]; then
    STAGE1_BEST=$(python - <<PY
from pathlib import Path
import sys
base = Path("${STAGE1_LOG_DIR}") / "checkpoints"
matches = sorted(base.glob("best-val-*.ckpt"))
if not matches:
    sys.exit(1)
print(matches[-1])
PY
    ) || {
        echo "Failed to locate Stage 1 best checkpoint under ${STAGE1_LOG_DIR}/checkpoints" >&2
        exit 1
    }
fi

if [ "${RUN_STAGE2}" = "1" ]; then
    LOG_DIR="${STAGE2_LOG_DIR}" TF_D_MODEL="${TF_D_MODEL}" TARGET_POLICY="${TARGET_POLICY}" MODEL_CHECKPOINT="${STAGE1_BEST}" sh scripts/training/run_safesim_stage2.sh
fi

if [ "${RUN_EVAL}" = "1" ]; then
    python scripts/analysis/evaluate_safesim_dangerous.py \
        --checkpoint_dir "${STAGE2_LOG_DIR}/checkpoints" \
        --output_dir "${EVAL_OUTPUT_DIR}" \
        --tf_d_model "${TF_D_MODEL}"
fi

echo "Workflow complete."
echo "target_policy=${TARGET_POLICY}"
if [ -n "${STAGE1_BEST}" ]; then
    echo "stage1_best=${STAGE1_BEST}"
fi
