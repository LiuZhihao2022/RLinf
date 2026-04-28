#!/bin/bash
# Per-episode visualization for the Success/Fail Classifier.
#
# Usage:
#   bash run_visualize_classifier.sh [FLAGS...] [HYDRA_OVERRIDES...]
#
# All defaults live in `examples/process/config/visualize_classifier.yaml`
# (and `model/success_fail_classifier.yaml`). Flags below only inject a
# Hydra override when explicitly given — omitted flags fall through to the
# yaml.
#
# Common flags:
#   --ckpt PATH              checkpoint dir (.../global_step_N/actor)
#   --dataset PATH           LeRobot dataset root
#   --episodes LIST          explicit episode ids, e.g. "0,7,42" or "[0,7,42]"
#   --num-episodes N         random-sample size (ignored if --episodes given)
#   --output PATH            output dir
#   --threshold FLOAT        P(fail) threshold
#   --batch-size N           inference batch size
#   --num-workers N          DataLoader workers
#   --no-video               skip MP4 generation (PNG only)
#   --fps N                  video fps
#   --smooth-window N        trailing-mean window
#   --only-success           include only success episodes
#   --only-fail              include only fail episodes
#   --min-episode-length N   drop episodes shorter than N frames
#   --seed N                 random-selection seed
#   --device STR             "cuda" / "cpu"
#   --dinov2 PATH            DINOv2-s backbone path / repo id
#   --camera-keys LIST       e.g. "[image,wrist_image]"
#
# Any other args are forwarded verbatim to Hydra. Examples:
#
#   # Use yaml defaults (you must at least provide a checkpoint + dataset
#   # via the yaml or these flags, otherwise the run fails fast):
#   bash run_visualize_classifier.sh \
#       --ckpt /path/to/ckpt/global_step_N/actor \
#       --dataset /path/to/dataset
#
#   # Pick specific episodes and skip video:
#   bash run_visualize_classifier.sh --episodes 0,7,42 --no-video
#
#   # Extra Hydra overrides still work:
#   bash run_visualize_classifier.sh --num-episodes 5 \
#       data.robot_type=libero visualize.fps=20

set -e

source switch_env openpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

EPISODES=""
HYDRA_ARGS=()
EXTRA_HYDRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt|--checkpoint|--checkpoint-dir)
            HYDRA_ARGS+=("visualize.checkpoint_dir=$2"); shift 2 ;;
        --dataset|--dataset-path)
            HYDRA_ARGS+=("data.train_data_paths.0.dataset_path=$2"); shift 2 ;;
        --episodes|--eps)
            EPISODES="$2"; shift 2 ;;
        --num-episodes|--n)
            HYDRA_ARGS+=("visualize.num_episodes=$2"); shift 2 ;;
        --output|--output-dir|-o)
            HYDRA_ARGS+=("visualize.output_dir=$2"); shift 2 ;;
        --threshold|--decision-threshold)
            HYDRA_ARGS+=("visualize.decision_threshold=$2"); shift 2 ;;
        --batch-size|--bs)
            HYDRA_ARGS+=("visualize.batch_size=$2"); shift 2 ;;
        --num-workers|--workers)
            HYDRA_ARGS+=("visualize.num_workers=$2"); shift 2 ;;
        --no-video)
            HYDRA_ARGS+=("visualize.no_video=true"); shift 1 ;;
        --fps)
            HYDRA_ARGS+=("visualize.fps=$2"); shift 2 ;;
        --smooth-window)
            HYDRA_ARGS+=("visualize.smooth_window=$2"); shift 2 ;;
        --only-success)
            HYDRA_ARGS+=("data.include_success=true" "data.include_fail=false"); shift 1 ;;
        --only-fail)
            HYDRA_ARGS+=("data.include_success=false" "data.include_fail=true"); shift 1 ;;
        --min-episode-length)
            HYDRA_ARGS+=("data.min_episode_length=$2"); shift 2 ;;
        --seed)
            HYDRA_ARGS+=("visualize.seed=$2"); shift 2 ;;
        --device)
            HYDRA_ARGS+=("visualize.device=$2"); shift 2 ;;
        --dinov2|--vision-repo-id)
            HYDRA_ARGS+=("model.vision_repo_id=$2"); shift 2 ;;
        --camera-keys)
            HYDRA_ARGS+=("data.camera_keys=$2"); shift 2 ;;
        -h|--help)
            sed -n '2,46p' "$0"; exit 0 ;;
        *)
            EXTRA_HYDRA+=("$1"); shift 1 ;;
    esac
done

# Normalize --episodes: accept either "0,7,42" or "[0,7,42]".
if [ -n "$EPISODES" ]; then
    if [[ "$EPISODES" != \[* ]]; then
        EPISODES="[${EPISODES}]"
    fi
    HYDRA_ARGS+=("visualize.episodes=$EPISODES")
fi

python visualize_episodes_with_classifier.py "${HYDRA_ARGS[@]}" "${EXTRA_HYDRA[@]}"
