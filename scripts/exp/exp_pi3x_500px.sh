#!/usr/bin/env bash
set -euo pipefail

kitti_sequences_dir="./Datasets/kitti-odometry/dataset/sequences"
logs_root_dir="${LOGS_DIR:-./logs}"
run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
logs_dir="${logs_root_dir}/${run_id}"

mkdir -p "${logs_dir}"
echo "Logs: ${logs_dir}"

run_group() {
  local gpu="$1"
  local config="$2"
  local exp_folder_name="$3"

  for seq in {00..10}; do
    CUDA_VISIBLE_DEVICES="${gpu}" python pi_long.py \
      --image_dir "${kitti_sequences_dir}/${seq}/image_2" \
      --config "${config}" \
      --exp_folder_name "${exp_folder_name}"
  done
}

pids=()

run_group 0 configs/pi3x-se3-120.yaml ./exps_pi3x_se3_120 >"${logs_dir}/gpu0_pi3x-se3-120.log" 2>&1 & pids+=("$!")
run_group 1 configs/pi3x-se3-90.yaml ./exps_pi3x_se3_90 >"${logs_dir}/gpu1_pi3x-se3-90.log" 2>&1 & pids+=("$!")
run_group 2 configs/pi3x-se3-60.yaml ./exps_pi3x_se3_60 >"${logs_dir}/gpu2_pi3x-se3-60.log" 2>&1 & pids+=("$!")
run_group 3 configs/pi3x-se3-30.yaml ./exps_pi3x_se3_30 >"${logs_dir}/gpu3_pi3x-se3-30.log" 2>&1 & pids+=("$!")
run_group 4 configs/pi3x-sim3-120.yaml ./exps_pi3x_sim3_120 >"${logs_dir}/gpu4_pi3x-sim3-120.log" 2>&1 & pids+=("$!")
run_group 5 configs/pi3x-sim3-90.yaml ./exps_pi3x_sim3_90 >"${logs_dir}/gpu5_pi3x-sim3-90.log" 2>&1 & pids+=("$!")
run_group 6 configs/pi3x-sim3-60.yaml ./exps_pi3x_sim3_60 >"${logs_dir}/gpu6_pi3x-sim3-60.log" 2>&1 & pids+=("$!")
run_group 7 configs/pi3x-sim3-30.yaml ./exps_pi3x_sim3_30 >"${logs_dir}/gpu7_pi3x-sim3-30.log" 2>&1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"