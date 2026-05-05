#!/bin/bash

if command -v conda >/dev/null 2>&1; then
    if [ "${CONDA_DEFAULT_ENV}" != "goalflow" ]; then
        source ~/.zshrc >/dev/null 2>&1 || true
        conda activate goalflow >/dev/null 2>&1 || true
    fi
fi

export NAVSIM_DEVKIT_ROOT=$(pwd)

LOG_ROOT="${LOG_ROOT:-safesim_logs_stage2_terminal_sweep_goal_action}"
OUT_ROOT="${OUT_ROOT:-outputs/current_goal_action/terminal_eval}"

find_best_ckpt() {
  run_dir="$1"
  best_ckpt=$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name 'best-val-*.ckpt' | sort | tail -n 1)
  if [ -z "${best_ckpt}" ]; then
    echo "No best checkpoint found in ${run_dir}" >&2
    exit 1
  fi
  printf "%s" "${best_ckpt}"
}

for tag in \
  termxy_0p25_termyaw_0p05 \
  termxy_0p25_termyaw_0p10 \
  termxy_0p50_termyaw_0p05 \
  termxy_0p50_termyaw_0p10
do
  run_dir="${LOG_ROOT}/${tag}"
  if [ ! -d "${run_dir}" ]; then
    echo "[Terminal Eval] Skip missing run ${run_dir}"
    continue
  fi
  ckpt=$(find_best_ckpt "${run_dir}")
  out_dir="${OUT_ROOT}/${tag}_protocol64"
  echo "[Terminal Eval] ${tag} -> ${out_dir}"
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
    --model_label "${tag}" || exit 1
done
