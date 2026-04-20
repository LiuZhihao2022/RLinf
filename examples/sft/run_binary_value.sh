#! /bin/bash

# Run ARM + ReWiND binary value model training
# Usage: bash run_binary_value_test.sh [CONFIG_NAME] [EXTRA_ARGS...]
# Example: bash run_binary_value_test.sh rewind_arm_value_test
# Example: bash run_binary_value_test.sh rewind_arm_value_model
# Example: bash run_binary_value_test.sh rewind_arm_value_test data.tag=my_tag

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_binary_value.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export FFREPORT=""

export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

# Reduces allocator fragmentation under FSDP — important when peak
# allocations (e.g. per-FSDP-unit gathered bf16 forward copy) compete with
# already-allocated fp32 master params + optimizer state.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="rewind_arm_value_test"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/binary_value/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
LOG_FILE="${LOG_DIR}/run_binary_value.log"
mkdir -p "${LOG_DIR}"
HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} ${EXTRA_ARGS}" > ${LOG_FILE}
${CMD_BASE} "${HYDRA_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | grep -v "libdav1d" | tee -a ${LOG_FILE}
