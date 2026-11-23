#!/usr/bin/env bash
set -euo pipefail

runtime=/root/autodl-tmp/embodied-runtime
upstream=/root/autodl-tmp/robomme-upstream
vulkan="$runtime/vulkan"
stage="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
run_name="${1:?usage: run_phase1.sh RUN_NAME [--gpu-only]}"
shift
output="$runtime/runs/$run_name"
if [[ -e "$output" ]]; then
    printf 'Output already exists: %s\n' "$output" >&2
    exit 2
fi

export SAPIEN_RENDER_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
export XDG_RUNTIME_DIR="$vulkan/runtime"
export VK_ICD_FILENAMES="$vulkan/nvidia-egl.json"
export LD_LIBRARY_PATH="$vulkan/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$upstream"
git rev-parse HEAD
"$upstream/.venv/bin/python" "$stage/gpu_pick_place.py" "$output" "$@"
