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

"""Visualize advantage distribution and episode videos for datasets with pre-computed advantages.

This script is designed for datasets that already have advantage_continuous computed,
supporting various observation key naming conventions (e.g., 'image' vs 'observation.images.*').

Usage:
    # Generate distribution plot and episode videos
    python visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --num-episodes 10
        --tag <tag>

    # Distribution plot only (no videos)
    python visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --no-video \
        --tag <tag>

    # Fast mode: subsample + lower resolution
    python visualize_advantage_dataset.py \
        --dataset /path/to/your/dataset \
        --output /path/to/output \
        --num-episodes 10 \
        --subsample 3 \
        --video-max-frames 500
"""

import argparse
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import hf_transform_to_torch
from PIL import Image as PILImage
from tqdm import tqdm


def _hf_transform_decode_images(batch: dict) -> dict:
    """Decode HF Image-feature struct dicts to PIL before hf_transform_to_torch."""
    for key in list(batch.keys()):
        vals = batch[key]
        if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
            batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
    return hf_transform_to_torch(batch)


def to_numpy(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def detect_image_keys_from_meta(dataset_path: Path) -> list[str]:
    """Read image/video keys directly from meta/info.json without loading data."""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return []
    with open(info_path, "r") as f:
        info = json.load(f)
    features = info.get("features", {})
    image_keys = []
    for key, spec in features.items():
        dtype = spec.get("dtype", "")
        if dtype in ("video", "image"):
            image_keys.append(key)
        elif "shape" in spec:
            shape = spec["shape"]
            if len(shape) == 3 and shape[-1] == 3:
                image_keys.append(key)
    return image_keys


def load_dataset(
    dataset_path: Path,
) -> tuple[LeRobotDataset, LeRobotDatasetMetadata, dict]:
    """Load LeRobot dataset with metadata."""
    meta = LeRobotDatasetMetadata(dataset_path.name, root=dataset_path)

    dataset = LeRobotDataset(
        dataset_path.name,
        root=dataset_path,
        delta_timestamps=None,
    )
    dataset.hf_dataset.set_transform(_hf_transform_decode_images)

    tasks = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "task_index" in entry and "task" in entry:
                    tasks[entry["task_index"]] = entry["task"]

    return dataset, meta, tasks


def get_episode_indices(dataset: LeRobotDataset, episode_index: int) -> list[int]:
    """Get all sample indices for a given episode."""
    if (
        hasattr(dataset, "episode_data_index")
        and dataset.episode_data_index is not None
    ):
        ep_data = dataset.episode_data_index
        if episode_index < len(ep_data["from"]):
            start = int(ep_data["from"][episode_index].item())
            end = int(ep_data["to"][episode_index].item())
            return list(range(start, end))

    indices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(to_scalar(sample["episode_index"])) == episode_index:
            indices.append(idx)
    return sorted(indices)


def create_advantage_distribution_plot(
    dataset_path: Path,
    output_path: Path,
    threshold: float | None = None,
    tag: str | None = None,
    advantage_key: str = "advantage_continuous",
    figsize: tuple = (14, 10),
):
    """Create comprehensive advantage distribution plots from meta/advantages.parquet."""
    adv_filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    adv_parquet = dataset_path / "meta" / adv_filename
    if not adv_parquet.exists():
        if tag:
            raise FileNotFoundError(
                f"Advantages file for tag '{tag}' not found: {adv_parquet}. "
                f"Available files: {sorted(p.name for p in (dataset_path / 'meta').glob('advantages*.parquet'))}"
            )
        raise FileNotFoundError(f"Advantages file not found: {adv_parquet}")
    print(f"Loading advantage data from {adv_parquet}...")

    df = pd.read_parquet(adv_parquet)

    if "advantage_continuous" in df.columns and "advantage" in df.columns:
        df["advantage_value"] = df["advantage_continuous"]
    elif "advantage_continuous" in df.columns:
        df["advantage_value"] = df["advantage_continuous"]
    elif "advantage" in df.columns:
        df["advantage_value"] = df["advantage"].astype(float)
    else:
        print(f"Warning: no advantage column found in {adv_parquet}")
        return None

    print(f"Loaded {len(df)} samples from {df['episode_index'].nunique()} episodes")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"Advantage Analysis: {dataset_path.name}", fontsize=14, fontweight="bold"
    )

    adv = df["advantage_value"].values

    # 1. Advantage histogram
    ax = axes[0, 0]
    p1, p99 = np.percentile(adv, [1, 99])
    margin = (p99 - p1) * 0.05
    xlim_lo, xlim_hi = p1 - margin, p99 + margin
    if threshold is not None:
        xlim_lo = min(xlim_lo, threshold - margin)
        xlim_hi = max(xlim_hi, threshold + margin)
    xlim_lo = min(xlim_lo, -margin)
    ax.hist(
        adv,
        bins=100,
        range=(xlim_lo, xlim_hi),
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(
        x=np.mean(adv),
        color="green",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean={np.mean(adv):.4f}",
    )
    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="orange",
            linestyle="-",
            linewidth=2.0,
            label=f"Threshold={threshold:.4f}",
        )
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Density")
    ax.set_title(f"Advantage Distribution (n={len(adv)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Value prediction histogram
    ax = axes[0, 1]
    if len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        ax.hist(
            v_curr,
            bins=100,
            density=True,
            alpha=0.7,
            color="forestgreen",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("V(o_t)")
        ax.set_ylabel("Density")
        ax.set_title(f"Value Predictions (mean={np.mean(v_curr):.4f})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, "No value_current data",
            ha="center", va="center", transform=ax.transAxes,
        )

    # 3. Value vs Advantage scatter
    ax = axes[0, 2]
    if "value_current" in df.columns and len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        ax.scatter(v_curr, adv, alpha=0.3, s=5, c="purple")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Adv=0")
        if threshold is not None:
            ax.axhline(
                y=threshold, color="orange", linestyle="-",
                linewidth=1.5, label=f"Thresh={threshold:.4f}",
            )
        ax.set_xlabel("V(o_t)")
        ax.set_ylabel("Advantage")
        ax.set_title("Value vs Advantage")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, "No value_current data",
            ha="center", va="center", transform=ax.transAxes,
        )

    # 4. Positive rate per episode
    ax = axes[1, 0]
    if threshold is not None:
        df["_positive"] = df["advantage_value"] >= threshold
        ep_pos = df.groupby("episode_index")["_positive"].mean().reset_index()
        ep_pos.columns = ["episode_index", "positive_rate"]
        ax.bar(
            ep_pos["episode_index"],
            ep_pos["positive_rate"] * 100,
            color="coral", alpha=0.7, edgecolor="black", linewidth=0.3,
        )
        ax.axhline(
            y=ep_pos["positive_rate"].mean() * 100,
            color="red", linestyle="--", linewidth=1.5,
            label=f"Mean={ep_pos['positive_rate'].mean() * 100:.1f}%",
        )
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Positive Rate (%)")
        ax.set_title(f"Positive Rate by Episode (thresh={threshold:.4f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        df.drop(columns=["_positive"], inplace=True)
    else:
        ax.text(
            0.5, 0.5, "No threshold set",
            ha="center", va="center", transform=ax.transAxes,
        )

    # 5. Advantage by episode
    ax = axes[1, 1]
    ep_stats = (
        df.groupby("episode_index")["advantage_value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ax.errorbar(
        ep_stats["episode_index"], ep_stats["mean"],
        yerr=ep_stats["std"], fmt="o", markersize=3,
        alpha=0.7, capsize=2, color="teal",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Advantage (mean +/- std)")
    ax.set_title(f"Advantage by Episode ({len(ep_stats)} episodes)")
    ax.grid(True, alpha=0.3)

    # 6. Statistics summary
    ax = axes[1, 2]
    ax.axis("off")
    stats_text = [
        f"Samples: {len(df):,}",
        f"Episodes: {df['episode_index'].nunique()}",
        "",
        "Advantage:",
        f"  Mean: {adv.mean():.5f}",
        f"  Std: {adv.std():.5f}",
        f"  Min: {adv.min():.5f}",
        f"  Max: {adv.max():.5f}",
        f"  Median: {np.median(adv):.5f}",
    ]
    if threshold is not None:
        positive_count = int((adv >= threshold).sum())
        positive_pct = positive_count / len(adv) * 100
        stats_text.extend([
            "",
            f"Threshold: {threshold:.5f}",
            f"  Positive: {positive_count:,} ({positive_pct:.1f}%)",
        ])
    if len(df["value_current"]) > 0:
        v_curr = df["value_current"].values
        stats_text.extend([
            "",
            "Value V(o_t):",
            f"  Mean: {v_curr.mean():.5f}",
            f"  Std: {v_curr.std():.5f}",
            f"  Range: [{v_curr.min():.4f}, {v_curr.max():.4f}]",
        ])
    ax.text(
        0.1, 0.95, "\n".join(stats_text),
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.3},
    )
    ax.set_title("Statistics Summary")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot: {output_path}")

    return df


def get_episode_data(
    dataset: LeRobotDataset,
    episode_index: int,
    tasks: dict,
    image_keys: list[str],
    adv_df: pd.DataFrame,
    subsample: int = 1,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Extract all data for an episode with optional subsampling.

    Args:
        subsample: Take every Nth frame (1 = all frames, 2 = every other, etc.)
        max_frames: Cap total frames loaded (0 = no cap). Applied after subsampling.
    """
    indices = get_episode_indices(dataset, episode_index)

    if not indices:
        return None

    # Subsample indices
    if subsample > 1:
        indices = indices[::subsample]
    if max_frames > 0 and len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    ep_adv = adv_df[adv_df["episode_index"] == episode_index].sort_values("frame_index")

    data = {
        "frames": [],
        "images": {key: [] for key in image_keys},
        "values": [],
        "advantages": [],
        "task": "",
        "episode_index": episode_index,
    }

    adv_lookup = {}
    for _, row in ep_adv.iterrows():
        adv_lookup[int(row["frame_index"])] = row

    t0 = time.time()
    for i, idx in enumerate(
        tqdm(indices, desc=f"  Ep {episode_index} ({len(indices)} frames)", leave=True, position=1)
    ):
        sample = dataset[idx]
        frame_idx = int(to_scalar(sample["frame_index"]))
        data["frames"].append(frame_idx)

        for key in image_keys:
            if key in sample:
                img = to_numpy(sample[key])
                if img.ndim == 4:
                    img = img[0]
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                data["images"][key].append(img)

        if frame_idx in adv_lookup:
            row = adv_lookup[frame_idx]
            data["advantages"].append(float(row.get("advantage_continuous", 0.0)))
            data["values"].append(float(row.get("value_current", 0.0)))
        else:
            data["advantages"].append(0.0)
            data["values"].append(0.0)

        if not data["task"] and "task_index" in sample and tasks:
            task_idx = int(to_scalar(sample["task_index"]))
            data["task"] = tasks.get(task_idx, f"Task {task_idx}")

    elapsed = time.time() - t0
    print(f"    Loaded {len(indices)} frames in {elapsed:.1f}s ({len(indices)/max(elapsed,0.01):.1f} fps)")

    return data


def _render_plot_bg(
    frames: list[int],
    values: list[float],
    width: int,
    height: int,
    color: str,
    ylabel: str,
    threshold: float | None = None,
    title: str = "",
) -> tuple[np.ndarray, callable]:
    """Pre-render a static curve plot as a BGR image, return it and a coord mapper.

    Returns (bg_bgr, frame_to_xy) where frame_to_xy(i) -> (px_x, px_y) for the dot.
    """
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.plot(frames, values, color=color, alpha=0.8, linewidth=1.2)
    ax.set_xlim(frames[0], frames[-1])
    if values:
        v_min, v_max = min(values), max(values)
        margin = (v_max - v_min) * 0.1 + 0.001
        ax.set_ylim(v_min - margin, v_max + margin)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    if threshold is not None:
        ax.axhline(y=threshold, color="orange", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.5)

    # Render to numpy
    fig.canvas.draw()
    import cv2
    buf = np.asarray(fig.canvas.buffer_rgba())
    canvas_h, canvas_w = buf.shape[:2]
    bg_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    # Resize to exact target dimensions if needed
    if canvas_w != width or canvas_h != height:
        bg_bgr = cv2.resize(bg_bgr, (width, height), interpolation=cv2.INTER_AREA)

    # Build coordinate mapper: data coords -> pixel coords
    x_data = np.array(frames, dtype=float)
    y_data = np.array(values, dtype=float)
    display_coords = ax.transData.transform(np.column_stack([x_data, y_data]))
    # matplotlib pixel coords have origin at bottom-left, flip y
    scale_x = width / canvas_w
    scale_y = height / canvas_h
    px_x = (display_coords[:, 0] * scale_x).astype(int)
    px_y = ((canvas_h - display_coords[:, 1]) * scale_y).astype(int)
    px_x = np.clip(px_x, 0, width - 1)
    px_y = np.clip(px_y, 0, height - 1)

    plt.close(fig)

    def frame_to_xy(i: int) -> tuple[int, int]:
        return int(px_x[i]), int(px_y[i])

    return bg_bgr, frame_to_xy


def create_episode_video_cv2(
    episode_data: dict[str, Any],
    output_path: Path,
    threshold: float | None = None,
    fps: int = 10,
    video_width: int = 960,
):
    """Create episode video using OpenCV with layout:
    Row 1: camera views side by side
    Row 2: value curve with moving dot
    Row 3: advantage curve with moving dot
    """
    import cv2

    frames = episode_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in episode_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        print(f"    No images for episode {episode_data['episode_index']}, skipping video")
        return

    sample_img = episode_data["images"][image_keys[0]][0]
    img_h, img_w = sample_img.shape[:2]

    cam_w = video_width // n_cameras
    scale = cam_w / img_w
    cam_h = int(img_h * scale)

    plot_h = 150
    text_height = 28
    total_w = cam_w * n_cameras
    total_h = cam_h + plot_h * 2 + text_height

    # Pre-render static plot backgrounds
    has_values = any(v != 0 for v in episode_data["values"])
    has_advantages = any(a != 0 for a in episode_data["advantages"])

    value_bg, value_xy = None, None
    if has_values:
        value_bg, value_xy = _render_plot_bg(
            frames, episode_data["values"], total_w, plot_h,
            color="green", ylabel="V(o_t)", title="Value",
        )

    adv_bg, adv_xy = None, None
    if has_advantages:
        adv_bg, adv_xy = _render_plot_bg(
            frames, episode_data["advantages"], total_w, plot_h,
            color="red", ylabel="Advantage", title="Advantage",
            threshold=threshold,
        )

    # If one is missing, create a blank
    blank_plot = np.full((plot_h, total_w, 3), 30, dtype=np.uint8)
    if value_bg is None:
        value_bg = blank_plot
    if adv_bg is None:
        adv_bg = blank_plot

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = str(output_path) + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (total_w, total_h))

    if not writer.isOpened():
        print(f"    Failed to open VideoWriter for episode {episode_data['episode_index']}")
        return

    advantages = episode_data["advantages"]
    dot_radius = 6
    dot_color_value = (0, 255, 0)  # BGR green
    dot_color_adv = (0, 0, 255)  # BGR red

    for i in tqdm(range(n_frames), desc=f"  Ep {episode_data['episode_index']} video", leave=True, position=1):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Row 1: camera images
        for cam_idx, key in enumerate(image_keys):
            img = episode_data["images"][key][i]
            if img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            img_resized = cv2.resize(img_bgr, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

            above = threshold is not None and advantages[i] >= threshold
            if above:
                cv2.rectangle(img_resized, (0, 0), (cam_w - 1, cam_h - 1), (0, 255, 0), 3)

            x_off = cam_idx * cam_w
            canvas[0:cam_h, x_off:x_off + cam_w] = img_resized

        # Row 2: value plot with dot
        row2_y = cam_h
        canvas[row2_y:row2_y + plot_h, :] = value_bg.copy()
        if value_xy is not None:
            dx, dy = value_xy(i)
            cv2.circle(canvas[row2_y:row2_y + plot_h], (dx, dy), dot_radius, dot_color_value, -1)
            cv2.circle(canvas[row2_y:row2_y + plot_h], (dx, dy), dot_radius, (255, 255, 255), 1)

        # Row 3: advantage plot with dot
        row3_y = cam_h + plot_h
        canvas[row3_y:row3_y + plot_h, :] = adv_bg.copy()
        if adv_xy is not None:
            dx, dy = adv_xy(i)
            cv2.circle(canvas[row3_y:row3_y + plot_h], (dx, dy), dot_radius, dot_color_adv, -1)
            cv2.circle(canvas[row3_y:row3_y + plot_h], (dx, dy), dot_radius, (255, 255, 255), 1)

        # Text row
        text_y = cam_h + plot_h * 2
        canvas[text_y:text_y + text_height, :] = (30, 30, 30)
        adv_val = advantages[i]
        above_str = " [ABOVE THRESHOLD]" if (threshold is not None and adv_val >= threshold) else ""
        text = f"Episode {episode_data['episode_index']}  Frame {frames[i]}  Adv={adv_val:.4f}{above_str}"
        cv2.putText(canvas, text, (10, text_y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(canvas)

    writer.release()

    final_path = str(output_path)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-loglevel", "error", final_path],
            check=True, capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        Path(tmp_path).rename(final_path)


def create_episode_video_mpl(
    episode_data: dict[str, Any],
    output_path: Path,
    threshold: float | None = None,
    fps: int = 10,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = 72,
):
    """Create a video using matplotlib (slower, higher quality plots)."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.patches import Rectangle

    frames = episode_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in episode_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        print(f"    No images for episode {episode_data['episode_index']}, skipping video")
        return

    has_values = any(v != 0 for v in episode_data["values"])
    has_advantages = any(a != 0 for a in episode_data["advantages"])

    n_plots = 0
    if has_values:
        n_plots += 1
    if has_advantages:
        n_plots += 1
    if n_plots == 0:
        n_plots = 1

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, max(n_cameras, n_plots), figure=fig, height_ratios=[2, 1])

    camera_axes = []
    for i, key in enumerate(image_keys):
        ax = fig.add_subplot(gs[0, i])
        display_name = key.replace("observation.images.", "").replace("_", " ").title()
        ax.set_title(display_name, fontsize=9)
        ax.axis("off")
        camera_axes.append(ax)

    plot_axes = []
    plot_data = []
    plot_idx = 0
    if has_values:
        ax_value = fig.add_subplot(gs[1, plot_idx])
        ax_value.set_title("Value V(o_t)", fontsize=9)
        ax_value.set_xlabel("Frame")
        ax_value.grid(True, alpha=0.3)
        plot_axes.append(ax_value)
        plot_data.append(("values", episode_data["values"], "tab:green"))
        plot_idx += 1

    if has_advantages:
        ax_adv = fig.add_subplot(gs[1, plot_idx])
        ax_adv.set_title("Advantage", fontsize=9)
        ax_adv.set_xlabel("Frame")
        ax_adv.grid(True, alpha=0.3)
        ax_adv.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if threshold is not None:
            ax_adv.axhline(
                y=threshold, color="orange", linestyle="-",
                linewidth=1.5, alpha=0.8, label=f"Thresh={threshold:.4f}",
            )
            ax_adv.legend(fontsize=7, loc="upper right")
        plot_axes.append(ax_adv)
        plot_data.append(("advantages", episode_data["advantages"], "tab:red"))

    for ax, (name, values, color) in zip(plot_axes, plot_data):
        ax.plot(frames, values, color=color, alpha=0.7, linewidth=1)
        ax.set_xlim(frames[0], frames[-1])
        if values:
            margin = (max(values) - min(values)) * 0.1 + 0.01
            ax.set_ylim(min(values) - margin, max(values) + margin)

    camera_ims = []
    for ax, key in zip(camera_axes, image_keys):
        im = ax.imshow(episode_data["images"][key][0])
        camera_ims.append(im)

    border_patches = []
    overlay_patches = []
    for ax in camera_axes:
        rect = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            linewidth=6, edgecolor="lime", facecolor="none",
            visible=False, zorder=10, clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)
        overlay = Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            linewidth=0, edgecolor="none", facecolor="lime",
            alpha=0.15, visible=False, zorder=9,
        )
        ax.add_patch(overlay)
        overlay_patches.append(overlay)

    plot_markers = []
    for ax, (name, values, color) in zip(plot_axes, plot_data):
        (marker,) = ax.plot([frames[0]], [values[0]], "o", color=color, markersize=6)
        plot_markers.append(marker)

    advantages = episode_data["advantages"]
    task_text = episode_data.get("task", "")[:50]
    title = fig.suptitle(
        f"Ep {episode_data['episode_index']} - {task_text}\nFrame: {frames[0]}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    pbar = tqdm(total=n_frames, desc=f"  Ep {episode_data['episode_index']} video", leave=True, position=1)

    def update(frame_num):
        for im, key in zip(camera_ims, image_keys):
            im.set_array(episode_data["images"][key][frame_num])
        for marker, (name, values, _) in zip(plot_markers, plot_data):
            marker.set_data([frames[frame_num]], [values[frame_num]])
        above = threshold is not None and advantages[frame_num] >= threshold
        for rect in border_patches:
            rect.set_visible(above)
        for ov in overlay_patches:
            ov.set_visible(above)
        adv_val = advantages[frame_num]
        indicator = " [+]" if above else ""
        title.set_text(
            f"Ep {episode_data['episode_index']} - {task_text}\n"
            f"Frame: {frames[frame_num]}  Adv: {adv_val:.4f}{indicator}"
        )
        pbar.update(1)
        return camera_ims + plot_markers + border_patches + overlay_patches + [title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=1500)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    pbar.close()


def create_episode_summary_plot(
    episode_data: dict[str, Any],
    output_path: Path,
    threshold: float | None = None,
    figsize: tuple[int, int] = (14, 8),
):
    """Create a static summary plot for an episode."""
    frames = episode_data["frames"]
    n_frames = len(frames)
    if n_frames == 0:
        return

    image_keys = [k for k, v in episode_data["images"].items() if len(v) > 0]
    n_cameras = len(image_keys)
    if n_cameras == 0:
        return

    sample_indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
    sample_indices = sorted({i for i in sample_indices if i < n_frames})

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_cameras + 2, len(sample_indices), figure=fig,
        height_ratios=[1] * n_cameras + [1, 1],
    )

    advantages = episode_data["advantages"]
    for cam_idx, key in enumerate(image_keys):
        cam_name = key.replace("observation.images.", "").replace("_", " ").title()
        for col_idx, frame_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[cam_idx, col_idx])
            ax.imshow(episode_data["images"][key][frame_idx])
            ax.axis("off")
            above = threshold is not None and advantages[frame_idx] >= threshold
            if above:
                import matplotlib.patches as mpatches
                rect = mpatches.FancyBboxPatch(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    boxstyle="round,pad=0", linewidth=4,
                    edgecolor="lime", facecolor="none", zorder=10,
                )
                ax.add_patch(rect)
            if cam_idx == 0:
                title_str = f"t={frames[frame_idx]}"
                if above:
                    title_str += " *"
                ax.set_title(
                    title_str, fontsize=9,
                    color="green" if above else "black",
                    fontweight="bold" if above else "normal",
                )
            if col_idx == 0:
                ax.text(
                    -0.1, 0.5, cam_name, transform=ax.transAxes,
                    fontsize=9, va="center", ha="right", rotation=90,
                )

    ax_value = fig.add_subplot(gs[n_cameras, :])
    values = episode_data["values"]
    if any(v != 0 for v in values):
        ax_value.plot(frames, values, "g-", label="V(o_t)", linewidth=1.5)
    ax_value.set_ylabel("Value")
    ax_value.legend(loc="upper right", fontsize=8)
    ax_value.grid(True, alpha=0.3)
    ax_value.set_xlim(frames[0], frames[-1])

    ax_adv = fig.add_subplot(gs[n_cameras + 1, :])
    adv_values = episode_data["advantages"]
    ax_adv.plot(frames, adv_values, "r-", label="Advantage", linewidth=1.5)
    ax_adv.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    if threshold is not None:
        ax_adv.axhline(
            y=threshold, color="orange", linestyle="-",
            linewidth=1.5, alpha=0.8, label=f"Threshold={threshold:.4f}",
        )
        adv_arr_plot = np.array(adv_values)
        above_mask = adv_arr_plot >= threshold
        ax_adv.fill_between(
            frames, adv_arr_plot, threshold,
            where=above_mask, alpha=0.2, color="lime", label="Above threshold",
        )
    ax_adv.set_xlabel("Frame")
    ax_adv.set_ylabel("Advantage")
    ax_adv.legend(loc="upper right", fontsize=8)
    ax_adv.grid(True, alpha=0.3)
    ax_adv.set_xlim(frames[0], frames[-1])

    task_text = episode_data.get("task", "")[:80]
    adv_arr = np.array(episode_data["advantages"])
    fig.suptitle(
        f"Episode {episode_data['episode_index']}: {task_text}\n"
        f"Advantage: mean={adv_arr.mean():.4f}, std={adv_arr.std():.4f}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize advantage distribution and episodes"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/advantage_viz", help="Output directory"
    )
    parser.add_argument(
        "--episodes", type=int, nargs="+", help="Specific episode indices to visualize"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of episodes (0=all)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument(
        "--no-distribution", action="store_true", help="Skip distribution plot"
    )
    parser.add_argument(
        "--advantage-key", type=str, default="advantage_continuous",
        help="Column name for advantage values",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Advantage threshold (auto-detected from mixture_config.yaml if not set)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Advantage tag: loads meta/advantages_{tag}.parquet",
    )
    parser.add_argument(
        "--subsample", type=int, default=2,
        help="Frame subsampling factor (1=all, 2=every other, 3=every 3rd, etc.)",
    )
    parser.add_argument(
        "--video-max-frames", type=int, default=0,
        help="Max frames per video (0=no limit). Useful for very long episodes.",
    )
    parser.add_argument(
        "--video-width", type=int, default=960,
        help="Video output width in pixels (cameras arranged side by side)",
    )
    parser.add_argument(
        "--video-backend", type=str, default="cv2", choices=["cv2", "matplotlib"],
        help="Video rendering backend: cv2 (fast) or matplotlib (pretty but slow)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    adv_filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
    adv_parquet = dataset_path / "meta" / adv_filename
    if not adv_parquet.exists():
        if tag:
            raise FileNotFoundError(
                f"Advantages file for tag '{tag}' not found: {adv_parquet}. "
                f"Available: {sorted(p.name for p in (dataset_path / 'meta').glob('advantages*.parquet'))}"
            )
        raise FileNotFoundError(f"Advantages file not found: {adv_parquet}")

    print(f"Using advantages parquet: {adv_parquet}")

    # Detect threshold
    threshold = args.threshold
    if threshold is None:
        for config_dir in [dataset_path, dataset_path.parent]:
            mixture_path = config_dir / "mixture_config.yaml"
            if mixture_path.exists():
                import yaml
                with open(mixture_path, "r") as f:
                    mixture_cfg = yaml.safe_load(f) or {}
                if tag and "tags" in mixture_cfg and tag in mixture_cfg["tags"]:
                    tag_cfg = mixture_cfg["tags"][tag]
                    if "unified_threshold" in tag_cfg:
                        threshold = float(tag_cfg["unified_threshold"])
                        print(f"Auto-detected threshold from {mixture_path} [tags.{tag}]: {threshold:.4f}")
                        break
                elif "unified_threshold" in mixture_cfg:
                    threshold = float(mixture_cfg["unified_threshold"])
                    print(f"Auto-detected threshold from {mixture_path}: {threshold:.4f}")
                    break

    if threshold is None:
        if adv_parquet.exists():
            _df = pd.read_parquet(adv_parquet, columns=["advantage", "advantage_continuous"])
            if "advantage" in _df.columns and "advantage_continuous" in _df.columns:
                positive_mask = _df["advantage"].astype(bool)
                if positive_mask.any():
                    threshold = float(_df.loc[positive_mask, "advantage_continuous"].min())
                    print(f"Inferred threshold from data: {threshold:.4f}")
            del _df

    if threshold is not None:
        print(f"Using threshold: {threshold:.4f}")
    else:
        print("Warning: no threshold detected, threshold lines will not be drawn")

    adv_df = pd.read_parquet(adv_parquet) if adv_parquet.exists() else pd.DataFrame()

    # Step 1: Distribution plot
    if not args.no_distribution:
        dist_path = output_dir / "advantage_distribution.png"
        create_advantage_distribution_plot(
            dataset_path, dist_path, threshold=threshold,
            tag=tag, advantage_key=args.advantage_key,
        )

    # Step 2: Detect image keys from metadata (fast, no data loading needed)
    image_keys = detect_image_keys_from_meta(dataset_path)
    print(f"\nImage keys from meta/info.json: {image_keys}")

    # Step 3: Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset, meta, tasks = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples, {meta.total_episodes} episodes")

    # Fallback: detect from sample if meta didn't find any
    if not image_keys:
        sample = dataset[0]
        skip_keys = {
            "index", "frame_index", "episode_index", "timestamp",
            "task_index", "state", "actions", "action",
            "is_success", "reward", "return", "prompt",
        }
        for key in sample.keys():
            if key in skip_keys:
                continue
            try:
                arr = to_numpy(sample[key])
                if arr.ndim >= 3:
                    image_keys.append(key)
            except (TypeError, ValueError):
                continue
        print(f"Detected image keys from sample: {image_keys}")

    # Determine episodes
    if args.episodes:
        episode_indices = args.episodes
    else:
        all_episodes = list(range(meta.total_episodes))
        if args.num_episodes <= 0:
            episode_indices = all_episodes
        else:
            np.random.seed(args.seed)
            episode_indices = np.random.choice(
                all_episodes, min(args.num_episodes, len(all_episodes)), replace=False
            )
            episode_indices = sorted(episode_indices)

    print(f"\nProcessing {len(episode_indices)} episodes: {episode_indices[:20]}{'...' if len(episode_indices) > 20 else ''}")
    print(f"Settings: subsample={args.subsample}, max_frames={args.video_max_frames}, "
          f"video_width={args.video_width}, backend={args.video_backend}, fps={args.fps}")

    total_t0 = time.time()
    for ep_i, ep_idx in enumerate(
        tqdm(episode_indices, desc="Episodes", position=0)
    ):
        ep_data = get_episode_data(
            dataset=dataset,
            episode_index=ep_idx,
            tasks=tasks,
            image_keys=image_keys,
            adv_df=adv_df,
            subsample=args.subsample,
            max_frames=args.video_max_frames,
        )

        if ep_data is None:
            print(f"  No data for episode {ep_idx}")
            continue

        # Summary plot
        plot_path = output_dir / f"episode_{ep_idx:04d}_summary.png"
        create_episode_summary_plot(ep_data, plot_path, threshold=threshold)

        # Video
        if not args.no_video:
            video_path = output_dir / f"episode_{ep_idx:04d}.mp4"
            if args.video_backend == "cv2":
                create_episode_video_cv2(
                    ep_data, video_path,
                    threshold=threshold, fps=args.fps, video_width=args.video_width,
                )
            else:
                create_episode_video_mpl(
                    ep_data, video_path,
                    threshold=threshold, fps=args.fps,
                    figsize=(10, 6), dpi=72,
                )

        tqdm.write(f"  Done episode {ep_idx} [{ep_i+1}/{len(episode_indices)}]")

    elapsed = time.time() - total_t0
    print(f"\nVisualization complete! {len(episode_indices)} episodes in {elapsed:.1f}s")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
