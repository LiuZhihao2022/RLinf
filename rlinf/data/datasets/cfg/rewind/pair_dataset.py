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

"""Pair dataset + collator for ARM + ReWiND binary value learning.

Dataset contract (``__getitem__``):

    {
        "image_t":  {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t
        "image_tk": {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t+k
        "image_mask_t":  {cam_name: bool, ...},
        "image_mask_tk": {cam_name: bool, ...},
        "prompt": str,
        "state":    Optional[np.ndarray],  # proprio at t (for state-in-prompt)
        "state_tk": Optional[np.ndarray],  # proprio at t+k (reserved)
        "label": float,                    # +1 = progress, -1 = regress
        "episode": int,
        "frame_idx_t": int,
        "frame_idx_tk": int,
    }

Each ``cam_name`` is a camera **view** (e.g. ``"base_0_rgb"``,
``"left_wrist_0_rgb"``). The time axis — frame_t vs frame_{t+k} — is a
separate structural axis: the collator runs the evorl
``Pistar06ValueProcessor`` once for frame_t and once for frame_{t+k}, then
stacks the two per-camera image tensors along a new ``num_frames`` dim so
the backbone receives a ``[B, num_cameras, num_frames, 3, H, W]`` tensor
per camera key.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Camera-view aliases tried against raw LeRobot sample dicts. Callers pass
# a plain camera key (e.g. ``image``) and the dataset probes the standard
# LeRobot path templates. Unlike the camera axis, state only has a single
# canonical field, so its alias list is shorter.
_IMAGE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
    "observation.images.{key}",
    "observation/images/{key}",
)
_STATE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
    "observation.state",
    "observation/state",
)


def _resolve_alias(sample: dict, key: str, aliases: Sequence[str]) -> Any:
    """Return ``sample[alias]`` for the first alias template that matches."""
    for template in aliases:
        resolved = template.format(key=key)
        if resolved in sample:
            return sample[resolved]
    raise KeyError(
        f"Could not resolve key={key!r} in sample. Tried: "
        f"{[t.format(key=key) for t in aliases]}. "
        f"Available: {sorted(sample.keys())}"
    )


def _to_uint8_hwc(frame: Any) -> np.ndarray:
    """Normalise a frame to ``(H, W, 3)`` uint8 numpy."""
    if hasattr(frame, "convert"):  # PIL.Image duck-typed
        frame = np.asarray(frame)
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(f"expected a rank-3 frame, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


def _to_float32_1d(state: Any, *, max_dim: Optional[int] = None) -> np.ndarray:
    """Normalise state to a rank-1 float32 array, optionally truncated/padded."""
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    else:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if max_dim is None:
        return arr
    if arr.shape[0] > max_dim:
        return arr[:max_dim]
    if arr.shape[0] < max_dim:
        padded = np.zeros((max_dim,), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr


# ---------------------------------------------------------------------------
# Trajectory sources
# ---------------------------------------------------------------------------


class TrajectorySource:
    """Minimal interface that any concrete trajectory source must satisfy."""

    def num_episodes(self) -> int:
        raise NotImplementedError

    def episode_length(self, episode: int) -> int:
        raise NotImplementedError

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        """Return a ``(H, W, 3)`` uint8 frame, or ``None`` if the camera is absent."""
        raise NotImplementedError

    def get_state(self, episode: int, frame: int, state_key: str) -> np.ndarray:
        raise NotImplementedError

    def get_prompt(self, episode: int, frame: int) -> Optional[str]:
        """Return the task / language instruction for a given frame.

        Implementations return ``None`` if no per-sample instruction is
        available; the caller is expected to fall back to a default.
        """
        return None

    def episode_is_success(self, episode: int) -> bool:
        """Return whether the episode should be treated as successful."""
        raise NotImplementedError


class _LeRobotSource(TrajectorySource):
    """LeRobot-backed source with lazy per-frame access."""

    def __init__(
        self,
        dataset_path: str,
        *,
        only_success: bool = True,
        dataset_type: str,
    ) -> None:
        try:
            from lerobot.common.datasets.lerobot_dataset import (  # noqa: E501
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
            from lerobot.common.datasets.utils import hf_transform_to_torch
        except ImportError:  # pragma: no cover — older lerobot layout
            from lerobot.common.datasets.lerobot_dataset import (  # noqa: E501
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
            from lerobot.common.datasets.utils import hf_transform_to_torch
        from PIL import Image as PILImage

        local_path = Path(dataset_path).absolute()
        self._dataset_label = str(local_path)
        self.meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        self.base = LeRobotDataset(
            local_path.name, root=local_path, download_videos=False
        )
        self._only_success = bool(only_success)
        self.dataset_type = dataset_type

        eps = self.base.episode_data_index
        self._ep_starts = [int(x) for x in eps["from"].tolist()]
        self._ep_ends = [int(x) for x in eps["to"].tolist()]
        if self.dataset_type == "sft":
            self._episode_success = None
        else:
            self._episode_success = (
                self._scan_episode_successes() if only_success else None
            )

        def _decoding_transform(batch: dict) -> dict:
            for key in list(batch.keys()):
                vals = batch[key]
                if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
                    batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
            return hf_transform_to_torch(batch)

        self.base.hf_dataset.set_transform(_decoding_transform)

        self._tasks: dict[int, str] = self._load_tasks(local_path)

    @staticmethod
    def _load_tasks(dataset_path: Path) -> dict[int, str]:
        """Load task_index → instruction mapping from LeRobot meta."""
        meta = dataset_path / "meta"
        jsonl = meta / "tasks.jsonl"
        if jsonl.exists():
            tasks: dict[int, str] = {}
            with open(jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    tasks[int(d.get("task_index", len(tasks)))] = str(d.get("task", ""))
            return tasks
        parquet = meta / "tasks.parquet"
        if parquet.exists():
            import pandas as pd

            df = pd.read_parquet(parquet)
            if "task_index" in df.columns and "task" in df.columns:
                return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        return {}

    def num_episodes(self) -> int:
        return len(self._ep_starts)

    def episode_length(self, episode: int) -> int:
        return self._ep_ends[episode] - self._ep_starts[episode]

    def _sample(self, episode: int, frame: int) -> dict:
        global_idx = self._ep_starts[episode] + int(frame)
        return self.base[global_idx]

    @staticmethod
    def _coerce_success_flag(raw: Any) -> bool:
        """Normalise a raw per-frame success flag to Python bool."""
        if isinstance(raw, torch.Tensor):
            raw = raw.item()
        if isinstance(raw, np.ndarray):
            raw = raw.reshape(-1)[0].item()
        if isinstance(raw, (list, tuple)):
            if len(raw) != 1:
                raise ValueError(f"Expected scalar success flag, got {raw!r}")
            raw = raw[0]
        return bool(raw)

    def _scan_episode_successes(self) -> list[bool]:
        """Read one representative frame row per episode from ``is_success``."""
        raw_dataset = self.base.hf_dataset
        if "is_success" not in raw_dataset.column_names:
            raise ValueError(
                "PairDataset(dataset_type='rollout', only_success=True) "
                "requires the LeRobot dataset to contain an 'is_success' "
                "column on representative frame rows."
            )

        # Read is_success directly from the Arrow table, bypassing
        # hf_dataset.set_transform. The transform loads every column and runs
        # hf_transform_to_torch, which crashes on non-image dict columns
        # (e.g. lerobot Video struct dicts that lack a "bytes" key).
        is_success_column = raw_dataset.data.column("is_success")

        episode_success: list[bool] = [
            self._coerce_success_flag(is_success_column[int(start)].as_py())
            for start, _end in zip(self._ep_starts, self._ep_ends)
        ]

        num_success = sum(bool(v) for v in episode_success)
        logger.info(
            "Scanned %d episode(s) in %s via one frame per episode from is_success; "
            "%d marked successful",
            len(episode_success),
            self._dataset_label,
            num_success,
        )
        return episode_success

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        sample = self._sample(episode, frame)
        try:
            raw = _resolve_alias(sample, camera_key, _IMAGE_KEY_ALIASES)
        except KeyError:
            return None
        return _to_uint8_hwc(raw)

    def get_state(self, episode: int, frame: int, state_key: str) -> np.ndarray:
        sample = self._sample(episode, frame)
        raw = _resolve_alias(sample, state_key, _STATE_KEY_ALIASES)
        return _to_float32_1d(raw)

    def get_prompt(self, episode: int, frame: int) -> str:
        sample = self._sample(episode, frame)
        task = sample.get("task")
        if isinstance(task, str) and task:
            return task
        ti = sample.get("task_index")
        if ti is None:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has no 'task' string and no "
                "'task_index' field; cannot resolve per-episode task instruction."
            )
        ti_int = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
        if not self._tasks:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but the dataset "
                "has no meta/tasks.jsonl (or meta/tasks.parquet) to resolve the "
                "instruction."
            )
        prompt = self._tasks.get(int(ti_int))
        if not prompt:
            raise RuntimeError(
                f"PairDataset: episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but it is not "
                f"present in meta/tasks.jsonl "
                f"(available indices: {sorted(self._tasks.keys())})."
            )
        return prompt

    def episode_is_success(self, episode: int) -> bool:
        if self.dataset_type == "sft":
            return True
        if self._episode_success is None:
            return True
        return bool(self._episode_success[episode])


# ---------------------------------------------------------------------------
# Pair dataset
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    """Yields ``(frame_t, frame_{t+k})`` pairs with multi-view per frame.

    Args:
        dataset_path: LeRobot dataset path.
        camera_keys: Camera view names to load per frame. These match the
            processor's ``image_keys`` — the collator feeds images under
            exactly these keys, and the processor fills any missing ones
            with zero placeholders (mask=False). Default follows the
            evorl convention: ``("base_0_rgb", "left_wrist_0_rgb",
            "right_wrist_0_rgb")``.
        k: Forward pair stride.
        include_state: If ``True``, samples carry ``state`` (proprio at
            ``t``) and ``state_tk`` (reserved).
        state_max_dim: Pad / truncate state to this dim.
        state_key: Fuzzy LeRobot state alias.
        dataset_type: Must be explicitly provided and be either ``"sft"``
            or ``"rollout"``. ``sft`` datasets are treated as all-success
            episodes, so they do not require an ``is_success`` column.
        only_success: Must be explicitly provided and currently must be
            ``True``. Keeps only episodes whose per-frame ``is_success``
            column marks the episode as successful. For LeRobot datasets
            this checks one representative frame row per episode rather
            than relying on episode-level metadata files.
        min_episode_length: Optional override for the minimum-length
            floor (default ``k + 1``).
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        camera_keys: Sequence[str] = (
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        k: int = 4,
        include_state: bool = False,
        state_max_dim: Optional[int] = None,
        state_key: str = "state",
        dataset_type: Optional[str] = None,
        only_success: Optional[bool] = None,
        min_episode_length: Optional[int] = None,
    ) -> None:
        self.camera_keys: tuple[str, ...] = tuple(camera_keys)
        if not self.camera_keys:
            raise ValueError("camera_keys must be non-empty")
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        self.include_state = bool(include_state)
        self.state_max_dim = state_max_dim
        self.state_key = state_key
        self.source_name = str(dataset_path)
        if dataset_type is None:
            raise ValueError(
                "PairDataset requires an explicit dataset_type argument "
                "('sft' or 'rollout')."
            )
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("sft", "rollout"):
            raise ValueError(
                f"PairDataset dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}."
            )
        if only_success is None:
            raise ValueError(
                "PairDataset requires an explicit only_success argument. "
                "Set only_success=true."
            )
        self.only_success = bool(only_success)
        if not self.only_success:
            raise ValueError(
                "PairDataset currently only supports only_success=True. "
                "Please remove the override or set only_success=true."
            )

        self._source = _LeRobotSource(
            dataset_path,
            only_success=self.only_success,
            dataset_type=self.dataset_type,
        )

        # Default floor: any episode with at least 2 frames can form a pair
        # (t=0, t+k clamped to T-1). Yamls can raise this if they want to
        # exclude episodes that can't supply at least one full stride-k pair.
        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)
        total_eps = self._source.num_episodes()
        self._eligible = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
            and (not self.only_success or self._source.episode_is_success(ep))
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes found with length >= {self._min_episode_length} "
                f"and only_success={self.only_success} "
                f"(dataset has {total_eps} episodes)."
            )

        # Per-eligible-episode count of valid pair start positions
        # t ∈ [0, T_ep - 1). For t in [0, T_ep - k) the pair is the regular
        # (t, t+k); for t in [T_ep - k, T_ep - 1) the second slot is clamped
        # to T_ep - 1 (boundary pair, stride < k). Cumulative sum lets
        # __getitem__ map a flat index to (eligible-slot, t) in
        # O(log |eligible|) via searchsorted.
        pair_positions_per_episode = np.array(
            [self._source.episode_length(ep) - 1 for ep in self._eligible],
            dtype=np.int64,
        )
        self._pair_position_ends = np.cumsum(pair_positions_per_episode)
        # Total number of distinct temporal anchors before we duplicate each
        # anchor into its positive and negative training samples.
        self._num_pair_positions = int(self._pair_position_ends[-1])

        logger.info(
            "PairDataset: dataset_path=%s, episodes=%d eligible=%d, k=%d, "
            "total_positions=%d, include_state=%s, dataset_type=%s, "
            "only_success=%s, camera_keys=%s",
            self.source_name,
            total_eps,
            len(self._eligible),
            self.k,
            self._num_pair_positions,
            self.include_state,
            self.dataset_type,
            self.only_success,
            self.camera_keys,
        )

    # --- Public accessors (used by the time-counter diagnostic) ---

    @property
    def source(self) -> TrajectorySource:
        return self._source

    @property
    def eligible_episodes(self) -> list[int]:
        return list(self._eligible)

    def set_epoch(self, epoch: int) -> None:
        del epoch  # no RNG state, retained for DataLoader wrapper compat

    @property
    def num_pair_positions(self) -> int:
        """Number of distinct ``(episode, t)`` anchors before label duplication."""
        return self._num_pair_positions

    def __len__(self) -> int:
        # Each temporal anchor contributes two labeled samples:
        #   positive: (t, t+k)
        #   negative: (t+k, t)
        return 2 * self._num_pair_positions

    def _decode_sample_index(self, idx: int) -> tuple[int, bool]:
        """Map a flat dataset index to ``(pair_position, is_positive)``."""
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)

        pair_position = idx // 2
        is_positive = (idx % 2) == 0
        return pair_position, is_positive

    def _resolve_pair_position(self, pair_position: int) -> tuple[int, int, int]:
        """Map a pair-position index to ``(episode, t, t_plus_k)``."""
        episode_slot = int(
            np.searchsorted(self._pair_position_ends, pair_position, side="right")
        )
        prev_episode_end = (
            int(self._pair_position_ends[episode_slot - 1]) if episode_slot > 0 else 0
        )
        episode = int(self._eligible[episode_slot])
        t = int(pair_position - prev_episode_end)
        # Boundary clamp: when t+k overruns the episode, use the last
        # available frame as the second slot. Stride degrades to T-1-t < k.
        t_plus_k = min(t + self.k, self._source.episode_length(episode) - 1)
        return episode, t, t_plus_k

    def _resolve_prompt(self, episode: int, frame_idx: int) -> str:
        """Return the per-sample task instruction; raises if missing."""
        return self._source.get_prompt(episode, frame_idx)

    def _build_sample(
        self,
        *,
        episode: int,
        frame_idx_t: int,
        frame_idx_tk: int,
        prompt: str,
        label: float,
    ) -> dict[str, Any]:
        """Assemble the sample dict for a single labeled frame pair."""
        views_t, mask_t = self._load_views(episode, frame_idx_t)
        views_tk, mask_tk = self._load_views(episode, frame_idx_tk)

        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": float(label),
            "episode": int(episode),
            "frame_idx_t": int(frame_idx_t),
            "frame_idx_tk": int(frame_idx_tk),
            "source_name": self.source_name,
        }

        if self.include_state:
            state_t = _to_float32_1d(
                self._source.get_state(episode, frame_idx_t, self.state_key),
                max_dim=self.state_max_dim,
            )
            state_tk = _to_float32_1d(
                self._source.get_state(episode, frame_idx_tk, self.state_key),
                max_dim=self.state_max_dim,
            )
            sample["state"] = state_t  # consumed by state-in-prompt branch
            sample["state_tk"] = state_tk  # reserved for future extensions

        return sample

    def _load_views(
        self, episode: int, frame_idx: int
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for camera_key in self.camera_keys:
            view = self._source.get_view(episode, frame_idx, camera_key)
            if view is None:
                masks[camera_key] = False
            else:
                views[camera_key] = view
                masks[camera_key] = True
        return views, masks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_position, is_positive = self._decode_sample_index(idx)
        episode, t, t_plus_k = self._resolve_pair_position(pair_position)
        prompt = self._resolve_prompt(episode, t)
        # Positive: (t, t+k). Negative: swap the two slots so the "later"
        # frame in original time occupies image_t — the model sees what
        # looks like a forward pair but the motion is reversed.
        if is_positive:
            frame_idx_t, frame_idx_tk = t, t_plus_k
            label = 1.0
        else:
            frame_idx_t, frame_idx_tk = t_plus_k, t
            label = -1.0

        return self._build_sample(
            episode=episode,
            frame_idx_t=frame_idx_t,
            frame_idx_tk=frame_idx_tk,
            prompt=prompt,
            label=label,
        )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class BinaryPairDataCollator:
    """Collator that produces the backbone's observation dict for binary pairs.

    Parallel to :class:`~rlinf.models.embodiment.value_model.data_collator.\
ValueDataCollator`. Runs the evorl :class:`Pistar06ValueProcessor` **twice**
    — once for frame_t's multi-view images, once for frame_{t+k}'s — then
    stacks the per-camera outputs along a new ``num_frames`` axis. The
    backbone receives per-camera tensors of shape
    ``[B, num_frames, 3, H, W]``.

    Attributes:
        processor: :class:`Pistar06ValueProcessor` with ``image_keys``
            matching the dataset's ``camera_keys``.
        max_length: Token padding length.
        train: If ``True``, the processor's image augmentations fire.
    """

    processor: Any
    max_length: int = 200
    train: bool = True

    def _collect_per_camera(
        self,
        examples: list[dict[str, Any]],
        images_key: str,
        masks_key: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Gather per-camera image tensors at a single timestamp.

        Missing camera entries for a sample turn into zero placeholders so
        the processor always sees a rectangular camera dict; the returned
        mask records which samples actually had the view.
        """
        camera_keys = set()
        for ex in examples:
            camera_keys.update(ex[images_key].keys())
        if not camera_keys:
            return {}, {}

        bsize = len(examples)
        images_out: dict[str, torch.Tensor] = {}
        masks_out: dict[str, torch.Tensor] = {}
        for cam in sorted(camera_keys):
            frames: list[np.ndarray] = []
            mask_vec: list[bool] = []
            shapes: list[tuple[int, ...]] = []
            for ex in examples:
                v = ex[images_key].get(cam)
                if v is None:
                    frames.append(None)  # type: ignore[arg-type]
                    mask_vec.append(False)
                else:
                    frames.append(v)
                    shapes.append(tuple(int(dim) for dim in v.shape))
                    mask_vec.append(bool(ex[masks_key].get(cam, True)))

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 1:
                shape_examples = [
                    {
                        "source": ex.get("source_name", "unknown"),
                        "episode": ex.get("episode"),
                        "frame_idx_t": ex.get("frame_idx_t"),
                        "frame_idx_tk": ex.get("frame_idx_tk"),
                        "shape": None
                        if ex[images_key].get(cam) is None
                        else tuple(int(dim) for dim in ex[images_key][cam].shape),
                    }
                    for ex in examples
                ]
                raise ValueError(
                    "BinaryPairDataCollator saw incompatible raw image shapes "
                    f"for camera={cam!r} at {images_key!r}: {unique_shapes}. "
                    "PairDataset assumes camera tensors are already shape-aligned; "
                    "this usually means your train batch mixed datasets with "
                    "different raw resolutions for the same camera key. "
                    f"Examples: {shape_examples}"
                )

            if unique_shapes:
                h, w = unique_shapes[0][:2]
            else:
                h, w = 1, 1

            # Replace None entries with zero placeholders matching the first
            # real frame's spatial size. If no real frame exists for this
            # camera across the whole batch, fall back to 1x1.
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            stacked = torch.from_numpy(
                np.stack([f if f is not None else placeholder for f in frames])
            )
            images_out[cam] = stacked
            masks_out[cam] = torch.tensor(mask_vec, dtype=torch.bool)

        # Ensure every example yields the same bsize (torch stack invariant).
        for cam, t in images_out.items():
            if t.shape[0] != bsize:
                raise RuntimeError(f"Unexpected batch shape for cam={cam}: {t.shape}")
        return images_out, masks_out

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("BinaryPairDataCollator received an empty batch")

        prompts: list[str] = [ex["prompt"] for ex in examples]
        states_list: list[Optional[np.ndarray]] = [ex.get("state") for ex in examples]

        # Frame-t and frame-tk run through the processor independently so
        # image augmentations are sampled per frame and so the per-camera
        # masks can differ between the two timestamps (e.g. a wrist view
        # that blinks on/off mid-episode).
        images_t, masks_t = self._collect_per_camera(
            examples, "image_t", "image_mask_t"
        )
        images_tk, masks_tk = self._collect_per_camera(
            examples, "image_tk", "image_mask_tk"
        )

        processed_t = self.processor.image_processor(
            images=images_t,
            image_masks=masks_t,
            return_tensors="pt",
            train=self.train,
        )
        processed_tk = self.processor.image_processor(
            images=images_tk,
            image_masks=masks_tk,
            return_tensors="pt",
            train=self.train,
        )

        # After process_images, pixel_values is a dict[cam → [B, 3, H, W]]
        # covering the **processor's** image_keys (missing camera keys get
        # zero-filled with mask=False at that stage). Stacking along a new
        # dim=1 gives [B, num_frames, 3, H, W] per camera.
        pixel_values_t = processed_t["pixel_values"]
        pixel_values_tk = processed_tk["pixel_values"]
        camera_keys_out = sorted(set(pixel_values_t) | set(pixel_values_tk))
        if not camera_keys_out:
            raise RuntimeError(
                "Processor returned no camera views — check image_keys / "
                "camera_keys alignment between dataset and processor."
            )

        def _stack_over_time(cam: str) -> tuple[torch.Tensor, torch.Tensor]:
            v_t = pixel_values_t.get(cam)
            v_tk = pixel_values_tk.get(cam)
            if v_t is None or v_tk is None:
                raise RuntimeError(
                    f"Camera {cam!r} missing from one of the two per-frame "
                    "processor outputs — this indicates inconsistent batch "
                    "shapes; investigate the dataset sample schema."
                )
            img_stacked = torch.stack([v_t, v_tk], dim=1)  # [B, 2, 3, H, W]
            m_t = processed_t["image_masks"][cam]
            m_tk = processed_tk["image_masks"][cam]
            mask_stacked = torch.stack([m_t, m_tk], dim=1).to(torch.bool)
            return img_stacked, mask_stacked

        images_observation: dict[str, torch.Tensor] = {}
        masks_observation: dict[str, torch.Tensor] = {}
        for cam in camera_keys_out:
            img, mask = _stack_over_time(cam)
            images_observation[cam] = img
            masks_observation[cam] = mask

        any_state = any(s is not None for s in states_list)
        state_batch: Any = None
        if any_state:
            template = next((s for s in states_list if s is not None), None)
            state_dim = int(template.shape[0])
            state_batch = np.stack(
                [
                    (
                        np.asarray(s, dtype=np.float32).reshape(-1)
                        if s is not None
                        else np.zeros(state_dim, dtype=np.float32)
                    )
                    for s in states_list
                ]
            )

        processed_txt = self.processor.process_text(
            prompts=prompts,
            states=state_batch,
            max_length=self.max_length,
            return_tensors="pt",
        )

        observation = {
            "images": images_observation,
            "image_masks": masks_observation,
            "tokenized_prompt": processed_txt["input_ids"],
            "tokenized_prompt_mask": processed_txt["attention_mask"].bool(),
        }

        labels = torch.tensor(
            [float(ex["label"]) for ex in examples], dtype=torch.float32
        )
        return {
            "observation": observation,
            "labels": labels,
            "episode": torch.tensor(
                [int(ex["episode"]) for ex in examples], dtype=torch.long
            ),
            "frame_idx_t": torch.tensor(
                [int(ex["frame_idx_t"]) for ex in examples], dtype=torch.long
            ),
            "frame_idx_tk": torch.tensor(
                [int(ex["frame_idx_tk"]) for ex in examples], dtype=torch.long
            ),
        }


# ---------------------------------------------------------------------------
# Time-counter-shortcut diagnostic
# ---------------------------------------------------------------------------


def sample_time_counter_diagnosis_batch(
    dataset: "PairDataset",
    num_samples: int,
    *,
    num_anchors: Optional[int] = None,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Build the (normal, shuffled) sample lists for the time-counter test.

    Both lists follow the same schema as :meth:`PairDataset.__getitem__`.
    For each anchor ``(episode_A, t, t+k)``:

        * ``normal``:   frame_A[t], frame_A[t+k]           (label +1)
        * ``shuffled``: frame_A[t], frame_B[t+k]           (label +1 — ignored)

    ``episode_B`` is a different eligible episode that covers ``t+k``.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    k = dataset.k
    source = dataset.source
    eligible = dataset.eligible_episodes
    if len(eligible) < 2:
        raise ValueError(
            "Need at least 2 eligible episodes to construct a shuffled pair"
        )

    rng = np.random.default_rng(seed)
    anchors: list[tuple[int, int]] = []
    max_anchors = num_samples if num_anchors is None else int(num_anchors)
    while len(anchors) < max_anchors:
        ep_a = int(eligible[rng.integers(low=0, high=len(eligible))])
        length_a = source.episode_length(ep_a)
        if length_a <= k:
            continue
        t = int(rng.integers(low=0, high=length_a - k))
        anchors.append((ep_a, t))

    def _collect_views(episode: int, frame_idx: int) -> tuple[dict, dict]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for cam in dataset.camera_keys:
            v = source.get_view(episode, frame_idx, cam)
            if v is None:
                masks[cam] = False
            else:
                views[cam] = v
                masks[cam] = True
        return views, masks

    normal: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []

    for i in range(num_samples):
        ep_a, t = anchors[i % len(anchors)]
        for _ in range(64):
            ep_b = int(eligible[rng.integers(low=0, high=len(eligible))])
            if ep_b != ep_a and source.episode_length(ep_b) > t + k:
                break
        else:
            continue

        views_a_t, mask_a_t = _collect_views(ep_a, t)
        views_a_tk, mask_a_tk = _collect_views(ep_a, t + k)
        views_b_tk, mask_b_tk = _collect_views(ep_b, t + k)

        prompt_a = source.get_prompt(ep_a, t)
        base = {
            "prompt": prompt_a,
            "label": 1.0,
            "episode": ep_a,
            "frame_idx_t": t,
            "frame_idx_tk": t + k,
        }
        normal.append(
            {
                **base,
                "image_t": views_a_t,
                "image_mask_t": mask_a_t,
                "image_tk": views_a_tk,
                "image_mask_tk": mask_a_tk,
            }
        )
        shuffled.append(
            {
                **base,
                "image_t": views_a_t,
                "image_mask_t": mask_a_t,
                "image_tk": views_b_tk,
                "image_mask_tk": mask_b_tk,
            }
        )

        if dataset.include_state:
            sa_t = _to_float32_1d(
                source.get_state(ep_a, t, dataset.state_key),
                max_dim=dataset.state_max_dim,
            )
            sa_tk = _to_float32_1d(
                source.get_state(ep_a, t + k, dataset.state_key),
                max_dim=dataset.state_max_dim,
            )
            sb_tk = _to_float32_1d(
                source.get_state(ep_b, t + k, dataset.state_key),
                max_dim=dataset.state_max_dim,
            )
            normal[-1]["state"] = sa_t
            normal[-1]["state_tk"] = sa_tk
            shuffled[-1]["state"] = sa_t
            shuffled[-1]["state_tk"] = sb_tk

    return {"normal": normal, "shuffled": shuffled}


__all__ = [
    "BinaryPairDataCollator",
    "PairDataset",
    "TrajectorySource",
    "sample_time_counter_diagnosis_batch",
]
