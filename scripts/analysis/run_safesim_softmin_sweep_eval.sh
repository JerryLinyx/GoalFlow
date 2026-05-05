#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

LOG_ROOT="${LOG_ROOT:-safesim_logs_stage2_softmin_sweep_goal_action}"
OUT_ROOT="${OUT_ROOT:-outputs/current_goal_action/softmin_eval}"

find_best_ckpt() {
  run_dir="$1"
  best_ckpt=$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'best-val-*.ckpt' | sort | tail -n 1)
  if [ -z "${best_ckpt}" ]; then
    echo "No best checkpoint found in ${run_dir}" >&2
    exit 1
  fi
  printf "%s" "${best_ckpt}"
}

for run_dir in "${LOG_ROOT}"/softmin_*; do
  if [ ! -d "${run_dir}" ]; then
    continue
  fi
  name=$(basename "${run_dir}")
  ckpt=$(find_best_ckpt "${run_dir}")
  out_dir="${OUT_ROOT}/${name}_protocol64"
  echo "[Softmin Eval] ${name} -> ${out_dir}"
  python scripts/analysis/evaluate_safesim_dangerous.py \
    --model_checkpoint "${ckpt}" \
    --output_dir "${out_dir}" \
    --cfg_scales 1.0 \
    --anchor_size 16 \
    --infer_steps 25 \
    --batch_size 8 \
    --val_split 0.1 \
    --seed 0 \
    --tf_d_model 1024 \
    --max_val_samples 64 \
    --target_policy action \
    --model_label "${name}" || exit 1
done
