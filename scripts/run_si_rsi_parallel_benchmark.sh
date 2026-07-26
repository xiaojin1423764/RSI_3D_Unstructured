#!/usr/bin/env bash
set -euo pipefail

# Generates independent tetrahedral meshes and runs the Figure 5 CUDA path.
# Override comma-separated lists to resume a subset, for example:
#   BENCH_GRIDS=100k BENCH_SN=224,316 BENCH_SAMPLES=1000,5000 \
#   BENCH_BATCH_SIZES=32,64,128,256 bash $0
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
results_dir=${BENCH_RESULTS_DIR:-"$root_dir/results/si_rsi_parallel_optimization_data"}
mesh_root=${BENCH_MESH_ROOT:-"$results_dir/meshes"}
plan_cache_dir=${BENCH_PLAN_CACHE_DIR:-"$results_dir/cache"}
reference_case=${BENCH_REFERENCE_CASE:-}
grids=${BENCH_GRIDS:-10k,30k,100k}
sn_values=${BENCH_SN:-10,20,30,70,100,224,316}
samples=${BENCH_SAMPLES:-100,500,1000,5000,10000,50000,100000}
batch_sizes=${BENCH_BATCH_SIZES:-32,64,128,256}
repetitions=${BENCH_REPETITIONS:-1}

declare -A mesh_sizes=( [10k]=0.094 [30k]=0.061 [100k]=0.0407 )

mkdir -p "$mesh_root" "$plan_cache_dir" "$results_dir/cases"
make -C "$root_dir" gpu

IFS=',' read -r -a grid_list <<< "$grids"
IFS=',' read -r -a sn_list <<< "$sn_values"
IFS=',' read -r -a sample_list <<< "$samples"
IFS=',' read -r -a batch_list <<< "$batch_sizes"

for grid in "${grid_list[@]}"; do
    mesh_size=${mesh_sizes[$grid]:-}
    if [[ -z "$mesh_size" ]]; then
        echo "Unknown BENCH_GRIDS entry: $grid" >&2
        exit 2
    fi
    mesh_dir="$mesh_root/$grid"
    mkdir -p "$mesh_dir"
    if [[ ! -s "$mesh_dir/cells.csv" || ! -s "$mesh_dir/faces.csv" ]]; then
        gmsh -3 -format msh2 -setnumber meshSize "$mesh_size" \
            -o "$mesh_dir/mesh.msh" "$root_dir/gmsh_work/example1.geo"
        python3 "$root_dir/scripts/msh_to_rsi_csv.py" "$mesh_dir/mesh.msh" \
            "$mesh_dir/cells.csv" "$mesh_dir/faces.csv"
    fi

    for sn in "${sn_list[@]}"; do
        for sample_count in "${sample_list[@]}"; do
            for batch_size in "${batch_list[@]}"; do
                for repetition in $(seq 1 "$repetitions"); do
                    case_name="${grid}_s${sn}_r${sample_count}_b${batch_size}_rep${repetition}"
                    case_dir="$results_dir/cases/$case_name"
                    mkdir -p "$case_dir"
                    if grep -q "CUDA streaming Figure 5 complete:" "$case_dir/run.log" 2>/dev/null; then
                        echo "Skipping completed case $case_name"
                        continue
                    fi
                    echo "Running $case_name"
                    env RSI_SWEEP_PLAN_CACHE_DIR="$plan_cache_dir" \
                        RSI_CUDA_STREAMING_PLAN_THRESHOLD_MB=0 \
                        RSI_CUDA_SI_DEVICE_PLAN_CACHE_MB=10000 \
                        RSI_CUDA_SI_HOST_PLAN_CACHE_MB=4096 \
                        RSI_CUDA_FIXED_PLAN_PARALLEL_LOAD=1 \
                        RSI_CUDA_RSI_PLAN_PREWARM=1 \
                        RSI_CUDA_RSI_MAX_SAMPLES_PER_BATCH="$batch_size" \
                        "$root_dir/rsi_unstructured_gpu" --gpu --source-shape circle --only figure5 \
                        --figure5-fine-sn "$sn" --figure5-samples "$sample_count" \
                        --figure5-dir "$case_dir" "$mesh_dir/cells.csv" "$mesh_dir/faces.csv" \
                        >"$case_dir/run.log" 2>&1
                done
            done
        done
    done
done

collect_args=(--root "$results_dir" --out "$results_dir/summary.csv")
if [[ -n "$reference_case" ]]; then
    collect_args+=(--reference-case "$reference_case")
fi
python3 "$root_dir/scripts/collect_si_rsi_parallel_benchmark.py" "${collect_args[@]}"
