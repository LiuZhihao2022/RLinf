#! /bin/bash

# Run frame-level Success/Fail Classifier SFT training.
# Usage: bash run_success_fail_classifier.sh [CONFIG_NAME] [EXTRA_ARGS...]
# Example:
#   bash run_success_fail_classifier.sh libero_sft_success_fail_classifier
#   bash run_success_fail_classifier.sh libero_sft_success_fail_classifier \
#     data.pos_weight_mode=batch_adaptive
#
# Smoke test (10 steps, no eval, no save):
#   bash run_success_fail_classifier.sh libero_sft_success_fail_classifier \
#     runner.max_epochs=10 runner.max_steps=10 \
#     runner.val_check_interval=-1 runner.save_interval=-1 \
#     cluster.component_placement.actor,env,rollout=0-0
#
# Required env vars (override by exporting before invocation):
#   SUCCESS_FAIL_DATA_ROOT  — LeRobot data root (absolute path)
#   LIBERO_NORM_STATS_DIR   — libero norm_stats.json directory
#   DINOV2_PATH             — local DINOv2-s checkpoint; empty ⇒ HF download

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_success_fail_classifier.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d / ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export FFREPORT=""

export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

# Reduce allocator fragmentation under FSDP mixed precision.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# SUCCESS_FAIL_DATA_ROOT / LIBERO_NORM_STATS_DIR are required by the yaml's
# `${oc.env:...}` interpolations; export them in your shell before running.
# DINOV2_PATH is optional — when unset, model/success_fail_classifier.yaml
# falls through to its `facebook/dinov2-small` default.
: "${SUCCESS_FAIL_DATA_ROOT:?must be set to the LeRobot data root}"
: "${LIBERO_NORM_STATS_DIR:?must be set to the libero norm_stats.json directory}"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="libero_sft_success_fail_classifier"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/binary_value/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
LOG_FILE="${LOG_DIR}/run_success_fail_classifier.log"
mkdir -p "${LOG_DIR}"
HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} ${EXTRA_ARGS}" > ${LOG_FILE}
${CMD_BASE} "${HYDRA_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | grep -v "libdav1d" | tee -a ${LOG_FILE}
