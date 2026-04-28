# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-episode visualization for the frame-level Success/Fail Classifier.

Parallel to :mod:`visualize_episodes_with_binary_value`, but for
:class:`SuccessFailClassifier` (DINOv2 + optional proprio MLP + binary
head). The classifier scores every **single** frame with one scalar
``P(fail) = sigmoid(logit) ∈ [0, 1]`` — higher = more fail-like — so
there is no pair axis, no stride ``k``, no ensemble / bin distribution.

For each selected episode we reuse the canonical training-inference path
(:class:`FrameClassifierDataset` + :class:`FrameClassifierDataCollator`)
and run the model once per frame. Rendered per episode:

    1. Sampled camera views at representative timesteps. Border + title
       highlight turns **red** when the model predicts fail
       (``P(fail) >= threshold``) and **green** when it predicts success.
    2. Per-frame ``P(fail)`` curve with the decision threshold drawn
       through it and the episode's ground-truth label as the target
       line (``0.0`` for success, ``1.0`` for fail episodes).
    3. Raw logit curve (no sigmoid) — useful for spotting saturation.
    4. Trailing-window mean ``P(fail)`` — a quick look at how stable the
       decision is over time.

Usage:
    cd examples/process
    python visualize_episodes_with_classifier.py \\
        visualize.checkpoint_dir=/path/to/success_fail_ckpt/global_step_N/actor \\
        data.train_data_paths.0.dataset_path=/path/to/dataset \\
        model.vision_repo_id=/path/to/dinov2-small \\
        +visualize.num_episodes=5 \\
        +visualize.output_dir=./viz_classifier_out

    # Explicit episodes:
    ... +visualize.episodes="[0,7,12]"

    # Plots only:
    ... +visualize.no_video=true
"""

import gc
import logging
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent))
sys.path.insert(0, str(_THIS_DIR))

from rlinf.data.datasets.cfg.success_fail_dataset import (  # noqa: E402
    FrameClassifierDataset,
)
from rlinf.models.embodiment.success_fail_classifier import (  # noqa: E402
    FrameClassifierDataCollator,
    SuccessFailClassifier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image-key detection (inlined from the binary-value viz — same contract)
# ---------------------------------------------------------------------------


def detect_image_keys(sample: dict) -> list[str]:
    """Auto-detect image-like keys in a LeRobot sample."""
    keys: list[str] = []
    for key in sample.keys():
        if key.startswith("observation.images.") or key in (
            "image",
            "wrist_image",
            "front_image",
            "left_image",
            "right_image",
        ):
            val = sample[key]
            arr = (
                val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
            )
            if arr.ndim >= 3:
                keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------


def _select_episodes(total_episodes: int, viz_cfg: DictConfig) -> list[int]:
    """Pick episode indices from explicit list or random sample."""
    episodes = viz_cfg.get("episodes", None)
    if episodes is not None:
        ep_list = [int(e) for e in episodes]
        bad = [e for e in ep_list if e < 0 or e >= total_episodes]
        if bad:
            raise ValueError(
                f"Requested episodes {bad} out of range "
                f"(dataset has {total_episodes} episodes)."
            )
        return sorted(set(ep_list))

    num_episodes = int(viz_cfg.get("num_episodes", 5))
    seed = int(viz_cfg.get("seed", 42))
    if num_episodes <= 0 or num_episodes >= total_episodes:
        return list(range(total_episodes))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_episodes, num_episodes, replace=False).tolist())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _build_episode_index(ds: FrameClassifierDataset) -> dict[int, list[int]]:
    """Group :attr:`FrameClassifierDataset._frame_index` by episode.

    Returns ``{episode -> [flat_index, ...]}`` where the flat indices are
    the positions to feed into :class:`torch.utils.data.Subset`. The lists
    are naturally sorted by frame-within-episode because
    :meth:`FrameClassifierDataset._build_frame_index` walks frames in
    order.
    """
    by_ep: dict[int, list[int]] = {}
    for flat_idx, (ep, _, _) in enumerate(ds._frame_index):
        by_ep.setdefault(int(ep), []).append(flat_idx)
    return by_ep


@torch.no_grad()
def _score_episode(
    model: SuccessFailClassifier,
    ds: FrameClassifierDataset,
    collator: FrameClassifierDataCollator,
    flat_indices: list[int],
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[int, dict[str, float]]:
    """Run the classifier on every frame of one episode.

    Returns a mapping ``frame_idx -> {"prob_fail": float, "logit": float}``
    with one entry per frame in the episode.
    """
    if not flat_indices:
        return {}

    subset = torch.utils.data.Subset(ds, flat_indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    def _to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _to_device(v) for k, v in x.items()}
        return x

    out: dict[int, dict[str, float]] = {}
    for batch in tqdm(loader, desc="scoring frames", leave=False, unit="batch"):
        observation = _to_device(batch["observation"])
        result = model.predict(observation)
        probs = result.probs.detach().float().cpu().numpy()
        logits = result.logits.squeeze(-1).detach().float().cpu().numpy()
        frame_idx = batch["frame_index"].cpu().numpy().tolist()
        if len(probs) != len(frame_idx):
            raise RuntimeError(
                f"Predicted {len(probs)} values for {len(frame_idx)} frames"
            )
        for f, p, lo in zip(frame_idx, probs, logits):
            out[int(f)] = {"prob_fail": float(p), "logit": float(lo)}
    return out


# ---------------------------------------------------------------------------
# Animation writer resolution — tolerate missing system ffmpeg.
# ---------------------------------------------------------------------------

_FFMPEG_PATH_CONFIGURED = False


def _ensure_ffmpeg_on_matplotlib() -> None:
    """Point ``animation.ffmpeg_path`` at a usable binary, once per process."""
    global _FFMPEG_PATH_CONFIGURED
    if _FFMPEG_PATH_CONFIGURED:
        return

    import shutil

    if shutil.which("ffmpeg"):
        _FFMPEG_PATH_CONFIGURED = True
        return

    try:
        import imageio_ffmpeg

        ff_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "No ffmpeg binary found. Install either (a) system ffmpeg "
            "(e.g. `apt-get install ffmpeg`) or (b) the `imageio-ffmpeg` "
            "pip package, which ships a static binary. "
            "Alternatively pass visualize.no_video=true to skip video "
            f"generation. Underlying error: {e}"
        ) from e

    matplotlib.rcParams["animation.ffmpeg_path"] = ff_path
    _FFMPEG_PATH_CONFIGURED = True
    logger.info("Using imageio_ffmpeg ffmpeg binary: %s", ff_path)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def _to_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def _trailing_mean(series: np.ndarray, window: int) -> np.ndarray:
    """Cumulative trailing-mean of a 1D array.

    At position ``t`` returns the mean of ``series[max(0, t-w+1):t+1]``.
    Window ``<= 1`` passes the raw series through.
    """
    if window <= 1 or series.size == 0:
        return series.astype(np.float64, copy=False)
    w = int(min(window, series.size))
    padded = np.concatenate(
        [np.zeros(w - 1, dtype=np.float64), series.astype(np.float64)]
    )
    kernel = np.ones(w, dtype=np.float64)
    sums = np.convolve(padded, kernel, mode="valid")  # length == len(series)
    # Divide by effective window (grows from 1..w)
    effective = np.minimum(np.arange(1, series.size + 1), w).astype(np.float64)
    return sums / effective


def _collect_episode_frames(
    lerobot_ds,
    episode_index: int,
    ep_starts: list[int],
    ep_ends: list[int],
    tasks: dict,
    image_keys: list[str],
    scores_by_frame: dict[int, dict[str, float]],
    gt_is_success: bool,
) -> dict:
    """Gather images + P(fail)/logit per frame for one episode."""
    start = int(ep_starts[episode_index])
    end = int(ep_ends[episode_index])

    data = {
        "frames": [],
        "images": {k: [] for k in image_keys},
        "prob_fail": [],
        "logit": [],
        "task": "",
        "episode_index": episode_index,
        "gt_is_success": bool(gt_is_success),
    }

    for idx in tqdm(range(start, end), desc=f"Episode {episode_index}", leave=False):
        sample = lerobot_ds[idx]
        frame_idx = int(_to_scalar(sample["frame_index"]))
        data["frames"].append(frame_idx)

        for key in image_keys:
            if key in sample:
                img = _to_numpy(sample[key])
                if img.ndim == 4:
                    img = img[0]
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                data["images"][key].append(img)

        # Frame may not have a score only if the dataset filtered it — in
        # practice FrameClassifierDataset keeps every frame, so this is
        # just a safe fallback.
        if frame_idx in scores_by_frame:
            bundle = scores_by_frame[frame_idx]
            data["prob_fail"].append(float(bundle["prob_fail"]))
            data["logit"].append(float(bundle["logit"]))
        elif data["prob_fail"]:
            data["prob_fail"].append(data["prob_fail"][-1])
            data["logit"].append(data["logit"][-1])
        else:
            data["prob_fail"].append(0.5)
            data["logit"].append(0.0)

        if not data["task"]:
            if "task" in sample:
                data["task"] = str(_to_scalar(sample["task"]))
            elif "task_index" in sample and tasks:
                t_idx = int(_to_scalar(sample["task_index"]))
                data["task"] = tasks.get(t_idx, f"Task {t_idx}")

    return data


def _create_episode_summary_plot(
    ep_data: dict,
    output_path: Path,
    decision_threshold: float = 0.5,
    smooth_window: int = 10,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Sampled frames + per-frame P(fail) + logit + trailing-mean curves."""
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        return

    sample_indices = sorted({
        i
        for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
        if i < n_frames
    })

    prob_fail = np.asarray(ep_data["prob_fail"], dtype=np.float64)
    logit = np.asarray(ep_data["logit"], dtype=np.float64)
    trailing = _trailing_mean(prob_fail, smooth_window)
    gt_is_success = bool(ep_data.get("gt_is_success", True))
    target_line = 0.0 if gt_is_success else 1.0

    # Camera rows + 3 curve rows (P(fail), logit, trailing mean)
    curve_rows = 3
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cameras + curve_rows,
        len(sample_indices),
        figure=fig,
        height_ratios=[1] * n_cameras + [1] * curve_rows,
    )

    pred_fail_mask = prob_fail >= decision_threshold

    for cam_idx, key in enumerate(image_keys):
        cam_name = key.replace("observation.images.", "").replace("_", " ").title()
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[cam_idx, col_idx])
            ax.imshow(ep_data["images"][key][frame_idx])
            ax.axis("off")
            pred_fail = bool(pred_fail_mask[frame_idx])
            border_color = "red" if pred_fail else "lime"
            rect = mpatches.FancyBboxPatch(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                boxstyle="round,pad=0",
                linewidth=4,
                edgecolor=border_color,
                facecolor="none",
                zorder=10,
            )
            ax.add_patch(rect)
            if cam_idx == 0:
                title_str = (
                    f"t={frames[frame_idx]} "
                    f"(P_fail={prob_fail[frame_idx]:.2f})"
                )
                if pred_fail:
                    title_str = "[FAIL] " + title_str
                ax.set_title(
                    title_str,
                    fontsize=9,
                    color="red" if pred_fail else "green",
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.text(
                    -0.1,
                    0.5,
                    cam_name,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="center",
                    ha="right",
                    rotation=90,
                )

    # Row 1: P(fail)
    ax_p = fig.add_subplot(gs[n_cameras, :])
    ax_p.plot(frames, prob_fail, color="black", linewidth=2.0, label="P(fail)")
    ax_p.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Threshold={decision_threshold:.2f}",
    )
    ax_p.axhline(
        y=target_line,
        color="tab:green" if gt_is_success else "tab:red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label=f"GT target={target_line:.0f} ({'success' if gt_is_success else 'fail'})",
    )
    ax_p.fill_between(
        frames,
        prob_fail,
        decision_threshold,
        where=pred_fail_mask,
        alpha=0.18,
        color="red",
        label="Predicted fail",
    )
    ax_p.set_ylabel("P(fail)")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.legend(loc="upper left", fontsize=7)
    ax_p.grid(True, alpha=0.3)
    ax_p.tick_params(labelbottom=False)

    # Row 2: raw logit
    ax_l = fig.add_subplot(gs[n_cameras + 1, :], sharex=ax_p)
    ax_l.plot(frames, logit, color="tab:blue", linewidth=1.5, label="logit")
    ax_l.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_l.set_ylabel("logit")
    ax_l.set_xlim(frames[0], frames[-1])
    ax_l.legend(loc="upper left", fontsize=7)
    ax_l.grid(True, alpha=0.3)
    ax_l.tick_params(labelbottom=False)

    # Row 3: trailing-mean P(fail)
    ax_s = fig.add_subplot(gs[n_cameras + 2, :], sharex=ax_p)
    ax_s.plot(
        frames,
        trailing,
        color="tab:purple",
        linewidth=1.8,
        label=f"Trailing mean P(fail) (w={smooth_window})",
    )
    ax_s.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.2,
        alpha=0.8,
    )
    ax_s.axhline(
        y=target_line,
        color="tab:green" if gt_is_success else "tab:red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    ax_s.set_xlabel("Frame")
    ax_s.set_ylabel("Trailing mean")
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_xlim(frames[0], frames[-1])
    ax_s.legend(loc="upper left", fontsize=7)
    ax_s.grid(True, alpha=0.3)

    task_text = ep_data.get("task", "")[:80]
    gt_tag = "SUCCESS" if gt_is_success else "FAIL"
    pred_frac = float(pred_fail_mask.mean())
    final_pfail = float(prob_fail[-1]) if prob_fail.size else float("nan")
    suptitle_head = (
        f"Episode {ep_data['episode_index']} "
        f"[GT: {gt_tag}, threshold={decision_threshold:.2f}]: {task_text}"
    )
    suptitle_body = (
        f"P(fail) mean={prob_fail.mean():.3f}  |  "
        f"fraction above threshold={pred_frac:.2f}  |  "
        f"final P(fail)={final_pfail:.3f}"
    )
    fig.suptitle(f"{suptitle_head}\n{suptitle_body}", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_episode_video(
    ep_data: dict,
    output_path: Path,
    decision_threshold: float = 0.5,
    smooth_window: int = 10,
    fps: int = 10,
    figsize: tuple[int, int] = (14, 9),
    dpi: int = 100,
) -> None:
    """Animated episode with camera views and P(fail) / logit / trailing curves."""
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        logger.warning("No images to render for episode %s", ep_data["episode_index"])
        return

    prob_fail = np.asarray(ep_data["prob_fail"], dtype=np.float64)
    logit = np.asarray(ep_data["logit"], dtype=np.float64)
    trailing = _trailing_mean(prob_fail, smooth_window)
    gt_is_success = bool(ep_data.get("gt_is_success", True))
    target_line = 0.0 if gt_is_success else 1.0

    grid_rows = 4  # 1 for cameras + 3 curve rows
    height_ratios = [2, 1, 1, 1]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        grid_rows,
        max(n_cameras, 1),
        figure=fig,
        height_ratios=height_ratios,
    )

    camera_axes = []
    for i, key in enumerate(image_keys):
        ax = fig.add_subplot(gs[0, i])
        display_name = key.replace("observation.images.", "").replace("_", " ").title()
        ax.set_title(display_name, fontsize=10)
        ax.axis("off")
        camera_axes.append(ax)

    # P(fail) panel
    ax_p = fig.add_subplot(gs[1, :])
    ax_p.plot(frames, prob_fail, color="black", linewidth=1.6, label="P(fail)")
    ax_p.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Threshold={decision_threshold:.2f}",
    )
    ax_p.axhline(
        y=target_line,
        color="tab:green" if gt_is_success else "tab:red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label=f"GT={target_line:.0f}",
    )
    ax_p.set_title("P(fail) per frame", fontsize=10)
    ax_p.set_ylabel("P(fail)")
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper left", fontsize=7)
    ax_p.tick_params(labelbottom=False)

    # Logit panel
    ax_l = fig.add_subplot(gs[2, :], sharex=ax_p)
    ax_l.plot(frames, logit, color="tab:blue", linewidth=1.3, label="logit")
    ax_l.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_l.set_title("Raw logit", fontsize=10)
    ax_l.set_ylabel("logit")
    ax_l.set_xlim(frames[0], frames[-1])
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(loc="upper left", fontsize=8)
    ax_l.tick_params(labelbottom=False)

    # Trailing mean panel
    ax_s = fig.add_subplot(gs[3, :], sharex=ax_p)
    ax_s.plot(
        frames,
        trailing,
        color="tab:purple",
        linewidth=1.5,
        label=f"Trailing mean P(fail) (w={smooth_window})",
    )
    ax_s.axhline(
        y=decision_threshold,
        color="orange",
        linestyle="-",
        linewidth=1.2,
        alpha=0.8,
    )
    ax_s.axhline(
        y=target_line,
        color="tab:green" if gt_is_success else "tab:red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )
    ax_s.set_title("Trailing-mean P(fail)", fontsize=10)
    ax_s.set_xlabel("Frame")
    ax_s.set_ylabel("Trailing mean")
    ax_s.set_xlim(frames[0], frames[-1])
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper left", fontsize=7)

    camera_ims = [
        ax.imshow(ep_data["images"][key][0]) for ax, key in zip(camera_axes, image_keys)
    ]

    border_patches_success: list[Rectangle] = []
    border_patches_fail: list[Rectangle] = []
    overlay_patches_fail: list[Rectangle] = []
    for ax in camera_axes:
        border_s = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=8,
            edgecolor="lime",
            facecolor="none",
            visible=True,
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(border_s)
        border_patches_success.append(border_s)

        border_f = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=8,
            edgecolor="red",
            facecolor="none",
            visible=False,
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(border_f)
        border_patches_fail.append(border_f)

        overlay_f = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=0,
            edgecolor="none",
            facecolor="red",
            alpha=0.15,
            visible=False,
            zorder=9,
        )
        ax.add_patch(overlay_f)
        overlay_patches_fail.append(overlay_f)

    (marker_p,) = ax_p.plot(
        [frames[0]],
        [prob_fail[0]],
        "o",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9,
        markeredgewidth=1.5,
    )
    (marker_l,) = ax_l.plot(
        [frames[0]], [logit[0]], "o", color="tab:blue", markersize=7
    )
    (marker_s,) = ax_s.plot(
        [frames[0]], [trailing[0]], "o", color="tab:purple", markersize=8
    )

    task_text = ep_data.get("task", "")[:60]
    gt_tag = "SUCCESS" if gt_is_success else "FAIL"
    title_head = (
        f"Episode {ep_data['episode_index']} [GT: {gt_tag}, "
        f"threshold={decision_threshold:.2f}] - {task_text}"
    )
    title = fig.suptitle(
        f"{title_head}\n"
        f"Frame: {frames[0]}  P(fail)={prob_fail[0]:.3f}  "
        f"logit={logit[0]:+.3f}  trailing={trailing[0]:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    def update(frame_num):
        for im, key in zip(camera_ims, image_keys):
            im.set_array(ep_data["images"][key][frame_num])
        marker_p.set_data([frames[frame_num]], [prob_fail[frame_num]])
        marker_l.set_data([frames[frame_num]], [logit[frame_num]])
        marker_s.set_data([frames[frame_num]], [trailing[frame_num]])

        pred_fail = prob_fail[frame_num] >= decision_threshold
        for rect in border_patches_success:
            rect.set_visible(not pred_fail)
        for rect in border_patches_fail:
            rect.set_visible(pred_fail)
        for overlay in overlay_patches_fail:
            overlay.set_visible(pred_fail)

        indicator = " [FAIL]" if pred_fail else " [SUCCESS]"
        title.set_text(
            f"{title_head}\n"
            f"Frame: {frames[frame_num]}  P(fail)={prob_fail[frame_num]:.3f}  "
            f"logit={logit[frame_num]:+.3f}  "
            f"trailing={trailing[frame_num]:.3f}{indicator}"
        )
        return (
            camera_ims
            + [marker_p, marker_l, marker_s, title]
            + border_patches_success
            + border_patches_fail
            + overlay_patches_fail
        )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    _ensure_ffmpeg_on_matplotlib()
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _opt_cast(value, cast):
    """Cast ``value`` to ``cast`` or pass through ``None``.

    Keeps user-left-null YAML fields as ``None`` so
    :meth:`SuccessFailClassifier.from_checkpoint` falls back to the value
    persisted in the checkpoint's ``config.json``.
    """
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    return cast(value)


def _parse_from_checkpoint_kwargs(cfg: DictConfig) -> dict:
    """Map Hydra config onto SuccessFailClassifier.from_checkpoint kwargs."""
    ckpt = cfg.visualize.checkpoint_dir
    if ckpt is None:
        raise ValueError("visualize.checkpoint_dir must be set")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    m = cfg.model
    d = cfg.data
    return {
        "checkpoint_dir": ckpt,
        "env_type": str(d.get("robot_type", "libero")),
        "model_type": str(d.get("model_type", "pi05")),
        "default_prompt": d.get("default_prompt", None),
        "vision_repo_id": _opt_cast(m.get("vision_repo_id"), str),
        "use_proprio": _opt_cast(m.get("use_proprio"), bool),
        "proprio_dim": _opt_cast(m.get("proprio_dim"), int),
        "image_size": _opt_cast(m.get("image_size"), int),
        "label_smoothing": _opt_cast(m.get("label_smoothing"), float),
        "precision": _opt_cast(m.get("precision"), str),
        "max_state_dim": _opt_cast(m.get("max_state_dim"), int),
        "action_dim": int(d.get("action_dim", 32)),
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="visualize_classifier",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    device = cfg.visualize.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    viz_cfg = cfg.visualize
    output_dir = Path(viz_cfg.get("output_dir", "./viz_classifier_episodes"))
    output_dir.mkdir(parents=True, exist_ok=True)
    no_video = bool(viz_cfg.get("no_video", False))
    fps = int(viz_cfg.get("fps", 10))
    ds_index = int(viz_cfg.get("dataset_index", 0))
    batch_size = int(viz_cfg.get("batch_size", 32))
    num_workers = int(viz_cfg.get("num_workers", 4))
    decision_threshold = float(viz_cfg.get("decision_threshold", 0.5))
    smooth_window = int(viz_cfg.get("smooth_window", 10))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")

    # --- Load the success/fail classifier ---
    model = SuccessFailClassifier.from_checkpoint(
        **_parse_from_checkpoint_kwargs(cfg), device=device
    )

    # Align the processor's image_keys with the dataset's camera_keys. The
    # saved preprocessor_config.json already carries the right keys (set at
    # training time by the worker), but the visualize YAML lets the user
    # override them — honour that override.
    camera_keys = tuple(cfg.data.get("camera_keys", ("image", "wrist_image")))
    if tuple(model.processor.image_keys) != camera_keys:
        logger.info(
            "Overriding processor.image_keys: %s -> %s",
            model.processor.image_keys,
            camera_keys,
        )
        model.processor.image_keys = camera_keys

    use_proprio = bool(getattr(model.config, "use_proprio", False))
    max_state_dim = int(getattr(model.config, "max_state_dim", 32))

    # --- Build the frame classifier dataset ---
    ds_entry = cfg.data.train_data_paths[ds_index]
    ds_path = str(ds_entry.dataset_path)
    ds_type = str(ds_entry.get("type", "rollout"))
    robot_type = str(ds_entry.get("robot_type", cfg.data.get("robot_type", "libero")))
    state_key = str(cfg.data.get("state_key", "state"))
    norm_stats_dir = cfg.data.get("norm_stats_dir", None)
    asset_id = cfg.data.get("asset_id", None)
    include_success = bool(cfg.data.get("include_success", True))
    include_fail = bool(cfg.data.get("include_fail", True))
    min_episode_length = int(cfg.data.get("min_episode_length", 1))

    logger.info(
        "Building FrameClassifierDataset: path=%s type=%s use_proprio=%s",
        ds_path,
        ds_type,
        use_proprio,
    )

    ds = FrameClassifierDataset(
        dataset_path=ds_path,
        dataset_type=ds_type,
        camera_keys=camera_keys,
        include_state=use_proprio,
        state_key=state_key,
        max_state_dim=max_state_dim,
        robot_type=robot_type,
        model_type=str(cfg.data.get("model_type", "pi05")),
        action_dim=int(cfg.data.get("action_dim", 32)),
        default_prompt=cfg.data.get("default_prompt", None),
        norm_stats_dir=norm_stats_dir,
        asset_id=asset_id,
        include_success=include_success,
        include_fail=include_fail,
        min_episode_length=min_episode_length,
    )

    collator = FrameClassifierDataCollator(
        processor=model.processor,
        use_proprio=use_proprio,
    )

    lerobot_ds = ds.base
    ep_starts = ds._ep_starts
    ep_ends = ds._ep_ends
    total_episodes = ds._num_episodes
    tasks = ds._tasks
    ep_success = ds._ep_success

    by_ep = _build_episode_index(ds)
    eligible = sorted(by_ep.keys())
    logger.info(
        f"Dataset: {total_episodes} episodes, {len(eligible)} eligible "
        f"(post filter include_success={include_success}, include_fail={include_fail}, "
        f"min_episode_length={min_episode_length})."
    )
    if not eligible:
        raise RuntimeError(
            "No episodes remained after filtering — relax include_success / "
            "include_fail / min_episode_length in the data config."
        )

    raw_selection = _select_episodes(total_episodes, viz_cfg)
    eligible_set = set(eligible)
    episode_indices = [e for e in raw_selection if e in eligible_set]
    skipped = [e for e in raw_selection if e not in eligible_set]
    if skipped:
        logger.warning(
            f"Skipping filtered-out episodes: {skipped[:10]}"
            f"{'...' if len(skipped) > 10 else ''}"
        )
    if not episode_indices:
        raise RuntimeError(
            "No eligible episodes selected — loosen visualize.num_episodes or "
            "check include_success / include_fail."
        )
    preview = episode_indices[:20]
    logger.info(
        f"Rendering {len(episode_indices)}/{len(eligible)} eligible episodes: "
        f"{preview}{'...' if len(episode_indices) > 20 else ''}"
    )

    image_keys = detect_image_keys(lerobot_ds[int(ep_starts[episode_indices[0]])])
    logger.info(f"Image keys for rendering: {image_keys}")
    if not image_keys:
        logger.warning(
            "No image keys detected — videos/plots will be empty. "
            "Check the dataset schema."
        )

    # --- Run inference per episode and render ---
    model.eval()
    for ep in tqdm(episode_indices, desc="Rendering"):
        flat_indices = by_ep.get(int(ep), [])
        if not flat_indices:
            logger.warning(f"No frames indexed for episode {ep}; skipping.")
            continue
        scores = _score_episode(
            model=model,
            ds=ds,
            collator=collator,
            flat_indices=flat_indices,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if not scores:
            logger.warning(f"No scores produced for episode {ep}; skipping.")
            continue

        ep_data = _collect_episode_frames(
            lerobot_ds=lerobot_ds,
            episode_index=ep,
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            tasks=tasks,
            image_keys=image_keys,
            scores_by_frame=scores,
            gt_is_success=bool(ep_success[ep]),
        )

        gt_tag = "success" if ep_data["gt_is_success"] else "fail"
        summary_path = output_dir / f"episode_{ep:04d}_{gt_tag}_summary.png"
        _create_episode_summary_plot(
            ep_data,
            summary_path,
            decision_threshold=decision_threshold,
            smooth_window=smooth_window,
        )
        logger.info(f"  Wrote {summary_path.name}")

        if not no_video:
            video_path = output_dir / f"episode_{ep:04d}_{gt_tag}.mp4"
            _create_episode_video(
                ep_data,
                video_path,
                decision_threshold=decision_threshold,
                smooth_window=smooth_window,
                fps=fps,
            )
            logger.info(f"  Wrote {video_path.name}")

    # Free model before returning
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
