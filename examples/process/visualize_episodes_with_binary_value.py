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

"""Per-episode visualization for the ARM + ReWiND binary value model.

Parallel to :mod:`visualize_episodes_with_value`, but for
:class:`BinaryValueCriticModel` / :class:`EnsembleBinaryValueCriticModel`
(rewind_arm). The binary critic scores **pairs** ``(frame_t, frame_{t+k})``
with a single scalar ``P(progress)`` in ``[0, 1]``, rather than a
continuous value ``V(o_t)`` — there is no advantage formula or return
normalization here.

Works with both single-model and ensemble checkpoints via the same
:meth:`BinaryValueCriticModel.from_checkpoint` entry point:

    * **Single** (``ensemble_size=1``): one P(progress) per pair.
    * **Ensemble** (``ensemble_size>1``): each of the ``E`` members emits
      its own P(progress); the wrapper aggregates those into a single
      ``predicted_values`` score using ``inference_mode``
      (``mo`` = member mean, ``wco`` = worst-case over members,
      ``uwo`` = mean − λ·variance). The renderer draws every member
      curve in its own color AND the aggregated ("total") curve as a
      bold overlay so you can see both the per-member spread and the
      aggregated decision simultaneously.

For each selected episode we reuse the canonical training-inference path
(:class:`PairDataset` + :class:`BinaryPairDataCollator`), restricted to
the **positive** pair samples (flat index ``2 * pair_position``), and
run the model once per frame. The resulting per-frame scalar is the
probability that the stride-``k`` pair (clamped at the episode boundary)
looks like forward progress; frame ``T-1`` has no valid pair.

Rendered per episode:

    1. Per-pair forward confidence — each ensemble member in its own
       color, plus the ``inference_mode``-aggregated score as a bold
       overlay. *High = local forward motion; ~0.5 = undecided; low =
       looks reversed.* For a single model the two curves coincide.
    2. Member variance (ensemble only) — raw ``var_i P_i`` across members.
    3. ``S(t) = Σ_{i<=t} (2·P_i − 1)`` — cumulative signed progress,
       drawn for both the aggregated score AND each member in matching
       colors. *Each step +1 when the model is fully confident it's
       forward, −1 when fully regressing. A successful trajectory grows
       approximately linearly; a stalled / failing one plateaus or dips.*

The stride ``k`` is pulled from ``data.k`` and **must match the training
config** — the model was only trained on that fixed stride and will not
generalize well to other values.

By default every ``model.*`` knob in the YAML is ``null``, which means
"read the value from the checkpoint's ``config.json``". Override on the
CLI (e.g. ``model.inference_mode=mo``) only when you want to deviate
from the trained setting.

Usage:
    cd examples/process
    python visualize_episodes_with_binary_value.py \\
        visualize.checkpoint_dir=/path/to/binary_value_ckpt \\
        data.train_data_paths.0.dataset_path=/path/to/dataset \\
        model.tokenizer_path=/path/to/gemma-3-270m \\
        model.vision_repo_id=/path/to/siglip-so400m-patch14-384 \\
        model.language_repo_id=/path/to/gemma-3-270m \\
        +visualize.num_episodes=5 \\
        +visualize.output_dir=./viz_binary_out

    # Explicit episodes:
    ... +visualize.episodes="[0,7,12]"

    # Plots only:
    ... +visualize.no_video=true

    # 4-member ensemble checkpoint — ensemble_size/inference_mode are read
    # from the checkpoint by default; override only to test other modes:
    python visualize_episodes_with_binary_value.py \\
        visualize.checkpoint_dir=/path/to/ensemble_ckpt \\
        data.train_data_paths.0.dataset_path=/path/to/dataset \\
        model.tokenizer_path=/path/to/gemma-3-270m \\
        model.vision_repo_id=/path/to/siglip-so400m-patch14-384 \\
        model.language_repo_id=/path/to/gemma-3-270m \\
        data.k=8 \\
        'data.camera_keys=[image,wrist_image]' \\
        visualize.num_episodes=10 \\
        visualize.output_dir=./viz_binary_ensemble_out
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

from rlinf.data.datasets.cfg.rewind import (  # noqa: E402
    BinaryPairDataCollator,
    PairDataset,
)
from rlinf.models.embodiment.value_model_rewind_arm.modeling_critic import (  # noqa: E402
    BinaryValueCriticModel,
)


def detect_image_keys(sample: dict) -> list[str]:
    """Auto-detect image-like keys in a LeRobot sample (inlined from
    ``visualize_advantage_dataset.detect_image_keys`` to avoid pulling
    in that module's newer ``lerobot.common.datasets`` import path)."""
    keys: list[str] = []
    for key in sample.keys():
        if key.startswith("observation.images.") or key in (
            "image", "wrist_image", "front_image", "left_image", "right_image",
        ):
            val = sample[key]
            arr = val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
            if arr.ndim >= 3:
                keys.append(key)
    return keys

logger = logging.getLogger(__name__)


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
# Pair-position bookkeeping (maps episodes ↔ flat PairDataset indices)
# ---------------------------------------------------------------------------


def _episode_to_pair_range(
    pair_ds: PairDataset, episode_index: int
) -> tuple[int, int]:
    """Return the ``[pair_position_start, pair_position_end)`` slice for an episode.

    ``PairDataset.__len__`` is ``2 * num_pair_positions`` (positive + negative per
    anchor). The flat positive-sample index for pair position ``p`` is ``2 * p``,
    which is what we use for inference.
    """
    eligible = pair_ds.eligible_episodes
    try:
        slot = eligible.index(int(episode_index))
    except ValueError as err:
        raise ValueError(
            f"Episode {episode_index} is not in the PairDataset's eligible set "
            f"(eligible={eligible[:10]}{'...' if len(eligible) > 10 else ''}). "
            "This usually means the episode is shorter than "
            f"min_episode_length={pair_ds._min_episode_length} or was filtered "
            "by the only_success flag."
        ) from err

    pair_position_end = int(pair_ds._pair_position_ends[slot])
    pair_position_start = (
        int(pair_ds._pair_position_ends[slot - 1]) if slot > 0 else 0
    )
    return pair_position_start, pair_position_end


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_episode_pairs(
    model: BinaryValueCriticModel,
    pair_ds: PairDataset,
    collator: BinaryPairDataCollator,
    episode_index: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict[int, dict[str, object]]:
    """Run the binary critic on every positive pair of one episode.

    Returns a mapping ``frame_idx_t -> score bundle``.
    """
    pp_start, pp_end = _episode_to_pair_range(pair_ds, episode_index)
    # Positive pair at pair_position ``p`` lives at flat index ``2 * p``.
    flat_indices = [2 * p for p in range(pp_start, pp_end)]
    if not flat_indices:
        return {}

    subset = torch.utils.data.Subset(pair_ds, flat_indices)
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

    out: dict[int, dict[str, object]] = {}
    for batch in tqdm(
        loader, desc=f"P(progress) ep={episode_index}", leave=False, unit="batch"
    ):
        observation = _to_device(batch["observation"])
        result = model.predict(observation)
        probs = result.predicted_values.detach().float().cpu().numpy()
        member_values = result.member_predicted_values.detach().float().cpu().numpy()
        value_mean = result.prediction_mean.detach().float().cpu().numpy()
        value_min = result.prediction_min.detach().float().cpu().numpy()
        value_variance = result.prediction_variance.detach().float().cpu().numpy()
        frame_idx_t = batch["frame_idx_t"].cpu().numpy().tolist()
        if len(probs) != len(frame_idx_t):
            raise RuntimeError(
                f"Predicted {len(probs)} values for {len(frame_idx_t)} pairs"
            )
        for batch_idx, (ft, p) in enumerate(zip(frame_idx_t, probs)):
            out[int(ft)] = {
                "value": float(p),
                "member_values": member_values[:, batch_idx].tolist(),
                "value_mean": float(value_mean[batch_idx]),
                "value_min": float(value_min[batch_idx]),
                "value_variance": float(value_variance[batch_idx]),
            }

    return out


# ---------------------------------------------------------------------------
# Animation writer resolution — tolerate missing system ffmpeg.
# ---------------------------------------------------------------------------

_FFMPEG_PATH_CONFIGURED = False


def _ensure_ffmpeg_on_matplotlib() -> None:
    """Point ``animation.ffmpeg_path`` at a usable binary, once per process.

    matplotlib's default ``animation.ffmpeg_path`` is the literal string
    ``"ffmpeg"``, which only resolves when a system-wide ffmpeg is on
    ``$PATH``. In several of our envs (openpi, openvla) ffmpeg is **not**
    installed at the system level — but the ``imageio-ffmpeg`` pip package
    ships a static binary that we can register with matplotlib instead.

    If neither is available, we raise early rather than produce a cryptic
    ``FileNotFoundError: 'ffmpeg'`` deep inside ``anim.save``.
    """
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
# Progress accumulator
# ---------------------------------------------------------------------------


def _cumulative_progress(progress) -> np.ndarray:
    """Signed forward-progress integrator: ``S(t) = Σ_{i<=t} (2·P_i - 1)``.

    Rationale: ``P_t`` is a per-pair local judgment — at frame ``t`` the
    model is asked whether the stride-k pair looks forward. A single
    ``P_t`` alone is not a progress signal; but summing ``2·P_t - 1``
    across frames gives an "expected net forward steps" curve that
    climbs for a successful demo and plateaus/drops for a stalled one.

    Accepts either a 1D sequence (aggregated progress) or a 2D array of
    shape ``[T, E]`` (per-member progress). Cumulative sum is along axis 0.
    """
    p = np.asarray(progress, dtype=np.float64)
    return np.cumsum(2.0 * p - 1.0, axis=0)


def _member_colors(num_members: int) -> list:
    """Distinct color per ensemble member — used consistently across panels."""
    if num_members <= 10:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        return [cmap(i) for i in range(num_members)]
    cmap = matplotlib.colormaps.get_cmap("viridis")
    return [cmap(i / max(1, num_members - 1)) for i in range(num_members)]


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


def _compute_value_axis_limits(
    progress: np.ndarray,
    member_progress: np.ndarray,
) -> tuple[float, float]:
    """Choose y-limits that do not clip UWO or per-member traces."""
    arrays = [progress.reshape(-1)]
    if member_progress.size > 0:
        arrays.append(member_progress.reshape(-1))

    lower = min(float(arr.min()) for arr in arrays)
    upper = max(float(arr.max()) for arr in arrays)
    if lower >= 0.0 and upper <= 1.0:
        return -0.02, 1.02

    span = max(upper - lower, 1e-6)
    pad = max(0.05, 0.08 * span)
    return lower - pad, upper + pad


def _collect_episode_frames(
    lerobot_ds,
    episode_index: int,
    ep_starts: list[int],
    ep_ends: list[int],
    tasks: dict,
    image_keys: list[str],
    progress_by_frame: dict[int, dict[str, object]],
) -> dict:
    """Gather images + P(progress) per frame for one episode."""
    start = int(ep_starts[episode_index])
    end = int(ep_ends[episode_index])

    data = {
        "frames": [],
        "images": {k: [] for k in image_keys},
        "progress": [],
        "progress_mean": [],
        "progress_min": [],
        "progress_variance": [],
        "member_progress": [],
        "task": "",
        "episode_index": episode_index,
    }
    default_member_count = 1
    if progress_by_frame:
        first_score = next(iter(progress_by_frame.values()))
        default_member_count = max(1, len(first_score.get("member_values", [])))

    for idx in tqdm(
        range(start, end), desc=f"Episode {episode_index}", leave=False
    ):
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

        # Last frame of each episode has no valid pair; default to the previous
        # value (or 0.5 if this is the only frame).
        if frame_idx in progress_by_frame:
            score_bundle = progress_by_frame[frame_idx]
            data["progress"].append(float(score_bundle["value"]))
            data["progress_mean"].append(float(score_bundle["value_mean"]))
            data["progress_min"].append(float(score_bundle["value_min"]))
            data["progress_variance"].append(float(score_bundle["value_variance"]))
            data["member_progress"].append(
                list(score_bundle.get("member_values", [float(score_bundle["value"])]))
            )
        elif data["progress"]:
            data["progress"].append(data["progress"][-1])
            data["progress_mean"].append(data["progress_mean"][-1])
            data["progress_min"].append(data["progress_min"][-1])
            data["progress_variance"].append(data["progress_variance"][-1])
            data["member_progress"].append(data["member_progress"][-1].copy())
        else:
            data["progress"].append(0.5)
            data["progress_mean"].append(0.5)
            data["progress_min"].append(0.5)
            data["progress_variance"].append(0.0)
            data["member_progress"].append([0.5] * default_member_count)

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
    stride_k: int = 0,
    inference_mode: str = "mo",
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Sampled frames + per-pair P_t + cumulative signed progress."""
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        return

    sample_indices = sorted(
        {i for i in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
         if i < n_frames}
    )

    member_progress = np.asarray(ep_data["member_progress"], dtype=np.float64)
    has_ensemble = member_progress.ndim == 2 and member_progress.shape[1] > 1
    curve_rows = 3 if has_ensemble else 2

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cameras + curve_rows,
        len(sample_indices),
        figure=fig,
        height_ratios=[1] * n_cameras + [1] * curve_rows,
    )

    progress = ep_data["progress"]
    p_arr = np.asarray(progress, dtype=np.float64)
    value_variance = np.asarray(ep_data["progress_variance"], dtype=np.float64)
    cum_progress = _cumulative_progress(progress)
    num_members = int(member_progress.shape[1]) if member_progress.ndim == 2 else 1
    member_cum_progress = (
        _cumulative_progress(member_progress) if has_ensemble else None
    )
    member_colors = _member_colors(num_members) if has_ensemble else []
    value_ymin, value_ymax = _compute_value_axis_limits(p_arr, member_progress)

    for cam_idx, key in enumerate(image_keys):
        cam_name = key.replace("observation.images.", "").replace("_", " ").title()
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[cam_idx, col_idx])
            ax.imshow(ep_data["images"][key][frame_idx])
            ax.axis("off")
            above = progress[frame_idx] >= decision_threshold
            if above:
                rect = mpatches.FancyBboxPatch(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    boxstyle="round,pad=0",
                    linewidth=4,
                    edgecolor="lime",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(rect)
            if cam_idx == 0:
                title_str = f"t={frames[frame_idx]}"
                title_str += f" (P={progress[frame_idx]:.2f})"
                if above:
                    title_str = "* " + title_str
                ax.set_title(
                    title_str,
                    fontsize=9,
                    color="green" if above else "black",
                    fontweight="bold" if above else "normal",
                )
            if col_idx == 0:
                ax.text(
                    -0.1, 0.5, cam_name,
                    transform=ax.transAxes,
                    fontsize=9, va="center", ha="right", rotation=90,
                )

    # Row 1: per-pair forward confidence — each ensemble member gets its
    # own color, plus the inference_mode-aggregated ("total") score as a
    # bold black overlay so it's clearly distinguishable.
    ax_p = fig.add_subplot(gs[n_cameras, :])
    if has_ensemble:
        for member_idx in range(num_members):
            ax_p.plot(
                frames,
                member_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.75,
                linewidth=1.0,
                label=f"Member {member_idx}",
            )
    aggregated_label = (
        f"Aggregated ({inference_mode.upper()}, stride k={stride_k})"
        if has_ensemble
        else f"{inference_mode.upper()} score (stride k={stride_k})"
    )
    ax_p.plot(
        frames,
        progress,
        color="black",
        linewidth=2.2,
        label=aggregated_label,
    )
    ax_p.axhline(
        y=decision_threshold,
        color="orange", linestyle="-", linewidth=1.5, alpha=0.8,
        label=f"Threshold={decision_threshold:.2f}",
    )
    ax_p.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    above_mask = p_arr >= decision_threshold
    ax_p.fill_between(
        frames, p_arr, decision_threshold,
        where=above_mask, alpha=0.18, color="lime", label="Above threshold",
    )
    ax_p.set_ylabel(f"{inference_mode.upper()} score")
    ax_p.set_ylim(value_ymin, value_ymax)
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_p.grid(True, alpha=0.3)
    ax_p.tick_params(labelbottom=False)

    cumulative_row = n_cameras + 1
    if has_ensemble:
        ax_v = fig.add_subplot(gs[n_cameras + 1, :], sharex=ax_p)
        ax_v.plot(
            frames,
            value_variance,
            color="tab:red",
            linewidth=1.5,
            label="Member variance",
        )
        ax_v.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_v.set_ylabel("Variance")
        ax_v.set_xlim(frames[0], frames[-1])
        ax_v.legend(loc="upper left", fontsize=8)
        ax_v.grid(True, alpha=0.3)
        ax_v.tick_params(labelbottom=False)
        cumulative_row = n_cameras + 2

    # Final row: cumulative signed progress Σ (2·P_t - 1). Members share
    # colors with the top panel so each trace is easy to follow across
    # both panels; the aggregated (inference_mode) curve is bold purple.
    ax_c = fig.add_subplot(gs[cumulative_row, :], sharex=ax_p)
    if has_ensemble and member_cum_progress is not None:
        for member_idx in range(num_members):
            ax_c.plot(
                frames,
                member_cum_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.55,
                linewidth=0.9,
                label=f"Member {member_idx}",
            )
    ax_c.plot(
        frames, cum_progress,
        color="tab:purple", linewidth=2.0,
        label=f"Aggregated ({inference_mode.upper()}) Σ (2·P − 1)"
        if has_ensemble else "Σ (2·P − 1)",
    )
    ax_c.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    # Ideal "always forward" line for reference (slope = +1).
    ideal = np.arange(len(frames), dtype=np.float64)
    ax_c.plot(frames, ideal, color="lightgray", linestyle=":", linewidth=1.0,
              label="Ideal always-forward")
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Cumulative signed\nprogress")
    ax_c.set_xlim(frames[0], frames[-1])
    ax_c.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_c.grid(True, alpha=0.3)

    task_text = ep_data.get("task", "")[:80]
    final_cum = float(cum_progress[-1]) if len(cum_progress) else 0.0
    fig.suptitle(
        f"Episode {ep_data['episode_index']} (k={stride_k}, mode={inference_mode.upper()}): {task_text}\n"
        f"score mean={p_arr.mean():.3f}, "
        f"above-thresh frac={float(above_mask.mean()):.2f}  |  "
        f"variance mean={value_variance.mean():.4f}  |  "
        f"final cumulative progress={final_cum:.1f} / max {len(frames) - 1}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_episode_video(
    ep_data: dict,
    output_path: Path,
    decision_threshold: float = 0.5,
    stride_k: int = 0,
    inference_mode: str = "mo",
    fps: int = 10,
    figsize: tuple[int, int] = (14, 9),
    dpi: int = 100,
) -> None:
    """Animated episode with camera views, P_t and cumulative progress curves."""
    frames = ep_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in ep_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        logger.warning("No images to render for episode %s", ep_data["episode_index"])
        return

    member_progress = np.asarray(ep_data["member_progress"], dtype=np.float64)
    has_ensemble = member_progress.ndim == 2 and member_progress.shape[1] > 1
    grid_rows = 4 if has_ensemble else 3
    height_ratios = [2] + [1] * (grid_rows - 1)

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

    progress = ep_data["progress"]
    value_variance = np.asarray(ep_data["progress_variance"], dtype=np.float64)
    cum_progress = _cumulative_progress(progress)
    num_members = int(member_progress.shape[1]) if member_progress.ndim == 2 else 1
    member_colors = _member_colors(num_members) if has_ensemble else []
    value_ymin, value_ymax = _compute_value_axis_limits(
        np.asarray(progress, dtype=np.float64),
        member_progress,
    )

    ax_p = fig.add_subplot(gs[1, :])
    if has_ensemble:
        for member_idx in range(num_members):
            ax_p.plot(
                frames,
                member_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.7,
                linewidth=0.9,
                label=f"Member {member_idx}",
            )
    aggregated_label = (
        f"Aggregated ({inference_mode.upper()})"
        if has_ensemble
        else f"{inference_mode.upper()} score"
    )
    ax_p.plot(
        frames, progress,
        color="black", linewidth=1.8,
        label=aggregated_label,
    )
    ax_p.axhline(
        y=decision_threshold,
        color="orange", linestyle="-", linewidth=1.5, alpha=0.8,
        label=f"Threshold={decision_threshold:.2f}",
    )
    ax_p.set_title(
        f"{inference_mode.upper()} score (stride k={stride_k})",
        fontsize=10,
    )
    ax_p.set_ylabel("Score")
    ax_p.set_xlim(frames[0], frames[-1])
    ax_p.set_ylim(value_ymin, value_ymax)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)
    ax_p.tick_params(labelbottom=False)

    variance_row = 2 if has_ensemble else None
    cumulative_row = 3 if has_ensemble else 2
    ax_v = None
    if has_ensemble:
        ax_v = fig.add_subplot(gs[variance_row, :], sharex=ax_p)
        ax_v.plot(
            frames,
            value_variance,
            color="tab:red",
            linewidth=1.2,
            label="Member variance",
        )
        ax_v.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax_v.set_title("Member variance", fontsize=10)
        ax_v.set_ylabel("Var")
        ax_v.set_xlim(frames[0], frames[-1])
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(loc="upper left", fontsize=8)
        ax_v.tick_params(labelbottom=False)

    ax_c = fig.add_subplot(gs[cumulative_row, :], sharex=ax_p)
    if has_ensemble:
        member_cum_progress = _cumulative_progress(member_progress)
        for member_idx in range(num_members):
            ax_c.plot(
                frames,
                member_cum_progress[:, member_idx],
                color=member_colors[member_idx],
                alpha=0.5,
                linewidth=0.8,
                label=f"Member {member_idx}",
            )
    ax_c.plot(
        frames, cum_progress,
        color="tab:purple", linewidth=1.4,
        label=f"Aggregated ({inference_mode.upper()}) Σ (2·P − 1)"
        if has_ensemble else "Σ (2·P − 1)",
    )
    ideal = np.arange(len(frames), dtype=np.float64)
    ax_c.plot(frames, ideal, color="lightgray", linestyle=":", linewidth=1.0,
              label="Ideal always-forward")
    ax_c.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_c.set_title("Cumulative signed progress", fontsize=10)
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Σ (2·P − 1)")
    ax_c.set_xlim(frames[0], frames[-1])
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper left", fontsize=7, ncol=2 if has_ensemble else 1)

    camera_ims = [
        ax.imshow(ep_data["images"][key][0])
        for ax, key in zip(camera_axes, image_keys)
    ]

    border_patches = []
    overlay_patches = []
    for ax in camera_axes:
        rect = Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            linewidth=8, edgecolor="lime", facecolor="none",
            visible=False, zorder=10, clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)
        overlay = Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            linewidth=0, edgecolor="none", facecolor="lime",
            alpha=0.15, visible=False, zorder=9,
        )
        ax.add_patch(overlay)
        overlay_patches.append(overlay)

    # Marker tracks the aggregated ("total") progress. Red-on-black edge so
    # it stands out against member 0's tab:blue line in ensemble renders.
    (marker_p,) = ax_p.plot(
        [frames[0]], [progress[0]], "o",
        markerfacecolor="red", markeredgecolor="black",
        markersize=9, markeredgewidth=1.5,
    )
    marker_v = None
    if ax_v is not None:
        (marker_v,) = ax_v.plot(
            [frames[0]], [value_variance[0]], "o", color="tab:red", markersize=7
        )
    (marker_c,) = ax_c.plot(
        [frames[0]], [cum_progress[0]], "o", color="tab:purple", markersize=8
    )

    task_text = ep_data.get("task", "")[:60]
    title = fig.suptitle(
        f"Episode {ep_data['episode_index']} (k={stride_k}, mode={inference_mode.upper()}) - {task_text}\n"
        f"Frame: {frames[0]}  Score={progress[0]:.3f}  "
        f"Var={value_variance[0]:.4f}  Cum={cum_progress[0]:.1f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    def update(frame_num):
        for im, key in zip(camera_ims, image_keys):
            im.set_array(ep_data["images"][key][frame_num])
        marker_p.set_data([frames[frame_num]], [progress[frame_num]])
        if marker_v is not None:
            marker_v.set_data([frames[frame_num]], [value_variance[frame_num]])
        marker_c.set_data([frames[frame_num]], [cum_progress[frame_num]])

        above = progress[frame_num] >= decision_threshold
        for rect in border_patches:
            rect.set_visible(above)
        for overlay in overlay_patches:
            overlay.set_visible(above)

        indicator = " [FORWARD]" if above else ""
        title.set_text(
            f"Episode {ep_data['episode_index']} (k={stride_k}, mode={inference_mode.upper()}) - {task_text}\n"
            f"Frame: {frames[frame_num]}  Score={progress[frame_num]:.3f}  "
            f"Var={value_variance[frame_num]:.4f}  "
            f"Cum={cum_progress[frame_num]:.1f}{indicator}"
        )
        animated_items = camera_ims + [marker_p, marker_c]
        if marker_v is not None:
            animated_items.append(marker_v)
        animated_items += border_patches + overlay_patches + [title]
        return animated_items

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

    Keeps user-left-null YAML fields as ``None`` so that
    :meth:`BinaryValueCriticModel.from_checkpoint` falls back to the value
    persisted in the checkpoint's ``config.json`` instead of silently
    clamping it to the YAML default (e.g. ``ensemble_size=1`` used to
    overwrite a 4-member checkpoint → random weights).
    """
    return None if value is None else cast(value)


def _parse_from_checkpoint_kwargs(cfg: DictConfig) -> dict:
    """Map Hydra config onto BinaryValueCriticModel.from_checkpoint kwargs.

    Leaves ``None``-valued YAML fields as ``None`` — that lets the
    ensemble / aggregation knobs ride on the checkpoint's saved config
    rather than being overwritten by the visualize-side defaults.
    """
    ckpt = cfg.visualize.checkpoint_dir
    if ckpt is None:
        raise ValueError("visualize.checkpoint_dir must be set")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    m = cfg.model
    return {
        "checkpoint_dir": ckpt,
        "tokenizer_path": m.get("tokenizer_path", None),
        "vision_repo_id": m.get("vision_repo_id", None),
        "language_repo_id": m.get("language_repo_id", None),
        "fusion_hidden_dim": _opt_cast(m.get("fusion_hidden_dim"), int),
        "dropout": _opt_cast(m.get("dropout"), float),
        "label_smoothing": _opt_cast(m.get("label_smoothing"), float),
        "num_frames_per_pair": _opt_cast(m.get("num_frames_per_pair"), int),
        "ensemble_size": _opt_cast(m.get("ensemble_size"), int),
        "inference_mode": _opt_cast(m.get("inference_mode"), str),
        "uwo_lambda": _opt_cast(m.get("uwo_lambda"), float),
        "include_state_in_prompt": _opt_cast(
            m.get("include_state_in_prompt"), bool
        ),
        "max_state_dim": _opt_cast(m.get("max_state_dim"), int),
        "state_discretization_bins": _opt_cast(
            m.get("state_discretization_bins"), int
        ),
        "max_token_len": _opt_cast(m.get("max_token_len"), int),
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="visualize_binary_value",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    device = cfg.visualize.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    viz_cfg = cfg.visualize
    output_dir = Path(viz_cfg.get("output_dir", "./viz_binary_episodes"))
    output_dir.mkdir(parents=True, exist_ok=True)
    no_video = bool(viz_cfg.get("no_video", False))
    fps = int(viz_cfg.get("fps", 10))
    ds_index = int(viz_cfg.get("dataset_index", 0))
    batch_size = int(viz_cfg.get("batch_size", 32))
    num_workers = int(viz_cfg.get("num_workers", 4))
    decision_threshold = float(viz_cfg.get("decision_threshold", 0.5))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")

    # --- Load the binary value critic ---
    model = BinaryValueCriticModel.from_checkpoint(
        **_parse_from_checkpoint_kwargs(cfg), device=device
    )
    inference_mode = str(getattr(model.config, "inference_mode", "mo"))

    # The processor created inside from_checkpoint uses the default
    # evorl image_keys (base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb).
    # For datasets that use different camera keys (e.g. libero's
    # "image" / "wrist_image") we MUST align the processor's keys with
    # the dataset's, otherwise every real camera is silently dropped.
    camera_keys = tuple(cfg.data.get("camera_keys", ("image", "wrist_image")))
    model.processor.image_processor.image_keys = camera_keys
    logger.info(f"Processor image_keys set to: {camera_keys}")

    # --- Build the pair dataset + collator ---
    # Visualization does not need to know whether the dataset is sft or
    # rollout: we are just rendering trajectories, not training. Hardcode
    # dataset_type="sft" so PairDataset treats every episode as successful
    # and skips the is_success column scan — all episodes long enough to
    # form a stride-k pair become eligible for visualization.
    ds_entry = cfg.data.train_data_paths[ds_index]
    ds_path = ds_entry.dataset_path
    k = int(cfg.data.get("k", 4))
    state_key = str(cfg.data.get("state_key", "state"))
    # Prefer the CLI override, fall back to whatever the just-loaded model was
    # trained with. A null YAML here means "use the checkpoint's value".
    include_state_override = cfg.model.get("include_state_in_prompt", None)
    include_state = (
        bool(include_state_override)
        if include_state_override is not None
        else bool(getattr(model.config, "include_state_in_prompt", False))
    )
    max_state_dim_override = cfg.model.get("max_state_dim", None)
    max_state_dim = (
        int(max_state_dim_override)
        if max_state_dim_override is not None
        else int(getattr(model.config, "max_state_dim", 32))
    )

    logger.info(
        "Building PairDataset: path=%s  stride_k=%d", ds_path, k,
    )
    logger.info(
        "  Every frame t will be scored against frame (t+%d, clamped to T-1). "
        "This stride MUST match the value-SFT training config; the model "
        "generalizes poorly to other strides.",
        k,
    )
    pair_ds = PairDataset(
        dataset_path=str(ds_path),
        camera_keys=camera_keys,
        k=k,
        include_state=include_state,
        state_max_dim=max_state_dim,
        state_key=state_key,
        dataset_type="sft",
        only_success=True,
    )

    # Same fall-through rule: null in YAML → use the checkpoint's value.
    max_token_len_override = cfg.model.get("max_token_len", None)
    max_length = (
        int(max_token_len_override)
        if max_token_len_override is not None
        else int(getattr(model.config, "max_token_len", 200))
    )
    collator = BinaryPairDataCollator(
        processor=model.processor,
        max_length=max_length,
        train=False,
    )

    lerobot_ds = pair_ds.source.base
    ep_starts = pair_ds.source._ep_starts
    ep_ends = pair_ds.source._ep_ends
    total_episodes = len(ep_starts)
    tasks = pair_ds.source._tasks

    # Restrict episode selection to eligible episodes only — ineligible ones
    # can't form any pair and we'd immediately raise on lookup.
    eligible = pair_ds.eligible_episodes
    logger.info(
        f"Dataset: {total_episodes} episodes, {len(eligible)} eligible for pairs."
    )

    # Filter the random/explicit selection down to eligible episodes.
    raw_selection = _select_episodes(total_episodes, viz_cfg)
    eligible_set = set(eligible)
    episode_indices = [e for e in raw_selection if e in eligible_set]
    skipped = [e for e in raw_selection if e not in eligible_set]
    if skipped:
        logger.warning(
            f"Skipping ineligible episodes (too short / filtered): {skipped[:10]}"
            f"{'...' if len(skipped) > 10 else ''}"
        )
    if not episode_indices:
        raise RuntimeError(
            "No eligible episodes selected — loosen --num-episodes or check "
            "min_episode_length / only_success."
        )
    preview = episode_indices[:20]
    logger.info(
        f"Rendering {len(episode_indices)}/{len(eligible)} eligible episodes: "
        f"{preview}{'...' if len(episode_indices) > 20 else ''}"
    )

    # --- Detect image keys for rendering (from the raw LeRobot sample, not
    # the pair sample, which has a different schema). ---
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
        progress_by_frame = _score_episode_pairs(
            model=model,
            pair_ds=pair_ds,
            collator=collator,
            episode_index=ep,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        if not progress_by_frame:
            logger.warning(f"No pairs scored for episode {ep}; skipping.")
            continue

        ep_data = _collect_episode_frames(
            lerobot_ds=lerobot_ds,
            episode_index=ep,
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            tasks=tasks,
            image_keys=image_keys,
            progress_by_frame=progress_by_frame,
        )

        summary_path = output_dir / f"episode_{ep:04d}_summary.png"
        _create_episode_summary_plot(
            ep_data, summary_path,
            decision_threshold=decision_threshold,
            stride_k=k,
            inference_mode=inference_mode,
        )
        logger.info(f"  Wrote {summary_path.name}")

        if not no_video:
            video_path = output_dir / f"episode_{ep:04d}.mp4"
            _create_episode_video(
                ep_data, video_path,
                decision_threshold=decision_threshold,
                stride_k=k,
                inference_mode=inference_mode,
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
