#!/usr/bin/env bash
set -euo pipefail

# Samples one representative fused sweep launch.  It intentionally does not
# profile the full matrix because Nsight Compute serializes kernel execution.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out_dir=${PROFILE_OUT_DIR:-"$root_dir/results/si_rsi_parallel_optimization_data/profiles"}
cells=${PROFILE_CELLS:-"$root_dir/results/si_rsi_parallel_optimization_data/meshes/10k/cells.csv"}
faces=${PROFILE_FACES:-"$root_dir/results/si_rsi_parallel_optimization_data/meshes/10k/faces.csv"}
sn=${PROFILE_SN:-70}
samples=${PROFILE_SAMPLES:-1000}

mkdir -p "$out_dir"
env RSI_CUDA_STREAMING_PLAN_THRESHOLD_MB=0 \
    RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000 \
    RSI_CUDA_SI_HOST_PLAN_CACHE_MB=4096 \
    ncu --csv --target-processes all --set full \
    --kernel-name regex:sweepLevelFusedKernel --launch-skip 0 --launch-count 1 \
    "$root_dir/rsi_unstructured_gpu" --gpu --source-shape circle --only figure5 \
    --figure5-fine-sn "$sn" --figure5-samples "$samples" \
    --figure5-dir "$out_dir/fields" "$cells" "$faces" \
    >"$out_dir/ncu_s${sn}_r${samples}.csv" 2>"$out_dir/ncu_s${sn}_r${samples}.stderr"
