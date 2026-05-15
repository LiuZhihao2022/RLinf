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

"""Frame-level success/fail classifier dataset + mixture + loader.

Flat, single-frame dataset over one or more LeRobot sources. Each frame is
labelled by its episode's ``is_success`` flag (read from one representative
frame per episode — identical reading strategy to
``compute_returns.py:147-166`` and ``pair_dataset.py``'s
``_scan_episode_successes``):

    * success episode → all frames get ``label = 0``
    * fail episode    → all frames get ``label = 1``

This is **episode-level supervision broadcast to frames**, not per-frame
annotation. The downstream classifier tolerates this via label smoothing
(``label_smoothing=0.1``) and relies on fail-ep typical poses recurring at
success-ep bad frames.

Proprio state is threaded through the same openpi transform pipeline that
the inference-side ``build_input_transforms`` uses
(``RepackTransform → policy.Inputs → Normalize(quantiles=True) →
PadStatesAndActions``), so training-time and inference-time state share an
identical [-1, 1] / padded-to-``max_state_dim`` representation. Mirrors
``ValueDataset._build_transform`` (see ``value_dataset.py:250-298``).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.datasets.cfg.rewind.pair_dataset import (
    _IMAGE_KEY_ALIASES,
    _resolve_alias,
    _to_float32_1d,
    _to_uint8_hwc,
)

from .mixture_datasets import _MixtureBase

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

POS_WEIGHT_MODES = frozenset({"auto", "none", "batch_adaptive"})


def _coerce_success_flag(raw: Any) -> bool:
    """Normalise a raw per-frame success flag to Python ``bool``.

    Duplicated from :meth:`_LeRobotSource._coerce_success_flag` so this
    module does not have to import a private symbol; the logic is small
    and stable.
    """
    if isinstance(raw, torch.Tensor):
        raw = raw.item()
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0].item()
    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError(f"Expected scalar success flag, got {raw!r}")
        raw = raw[0]
    return bool(raw)


class FrameClassifierDataset(Dataset):
    """Flat LeRobot frame dataset keyed by per-episode ``is_success``.

    Each sample emits raw HWC uint8 camera frames + (optional) proprio
    state + binary label — the collator handles resize / normalize on the
    image side, and the state is already in the inference-aligned
    ``[-1, 1]`` / padded space when a norm stats dir is supplied.

    Args:
        dataset_path: LeRobot dataset root (Parquet / videos under
            ``<dataset_path>/data`` and ``<dataset_path>/videos``).
        dataset_type: ``"sft"`` treats every episode as success (no
            ``is_success`` column required). ``"rollout"`` scans
            ``is_success`` per episode and assigns frame labels
            accordingly.
        camera_keys: Camera view names to request per frame. Missing
            cameras are tolerated (they become placeholders with
            ``image_mask=False`` at collator stage).
        include_state: If ``True``, each sample carries normalized
            proprio ``state`` (shape ``[max_state_dim]``). When
            ``False``, the ``state`` key is omitted.
        state_key: Fuzzy alias for the LeRobot state column
            (see ``_STATE_KEY_ALIASES``).
        max_state_dim: Padded state dim (matches
            ``PadStatesAndActions`` used at inference time).
        robot_type: Passed to openpi ``build_input_transforms``; must
            match the value used by compute_advantages.py at inference.
        model_type: openpi model family (``"pi0"``, ``"pi05"``, ...).
        action_dim: Padded action dim for ``PadStatesAndActions`` (the
            classifier does not consume actions, but the transform still
            expects this arg). Defaults to 32.
        default_prompt: Passed to ``InjectDefaultPrompt``.
        norm_stats_dir: Optional path to norm stats directory. When
            provided, the ``Normalize(use_quantiles=True)`` step is
            inserted so training-side state matches inference-side
            state. **Required** for ``use_proprio=True`` to produce
            anything meaningful.
        asset_id: Optional asset id under ``norm_stats_dir`` (defaults
            to ``robot_type.lower()``).
        include_success: If ``False``, success episodes are excluded —
            useful when rebalancing with another dataset that is
            success-only. Defaults to ``True``.
        include_fail: If ``False``, fail episodes are excluded.
            Defaults to ``True``. At least one of ``include_success``
            / ``include_fail`` must be ``True``.
        min_episode_length: Drop episodes shorter than this (default
            1, i.e. keep everything).
        source_name: Optional human-readable label for logging (defaults
            to ``dataset_path``).
        inference_mode: If ``True``, relax two training-only checks so
            the dataset can drive Step-3 classifier inference on any
            parquet LeRobot corpus:

            * skip the "need at least one success AND one fail" raise
              (labels are not consumed by inference);
            * for ``dataset_type='rollout'``, tolerate a missing
              ``is_success`` column by dummy-filling ``_ep_success`` with
              ``False`` (labels become 1 but the inference pass ignores
              them). ``'sft'`` mode is unaffected — it never reads the
              column.

            Has no effect on training paths (``inference_mode=False``
            preserves byte-for-byte behaviour).
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        dataset_type: str = "rollout",
        camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
        include_state: bool = True,
        state_key: str = "state",
        max_state_dim: int = 32,
        robot_type: str = "libero",
        model_type: str = "pi05",
        action_dim: int = 32,
        default_prompt: Optional[str] = None,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        include_success: bool = True,
        include_fail: bool = True,
        min_episode_length: int = 1,
        source_name: Optional[str] = None,
        inference_mode: bool = False,
    ) -> None:
        if not include_success and not include_fail:
            raise ValueError(
                "FrameClassifierDataset: at least one of include_success / "
                "include_fail must be True."
            )
        self.dataset_path = str(dataset_path)
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("sft", "rollout"):
            raise ValueError(
                f"dataset_type must be 'sft' or 'rollout', got {dataset_type!r}"
            )
        self.camera_keys = tuple(camera_keys)
        if not self.camera_keys:
            raise ValueError("camera_keys must be non-empty")
        self.include_state = bool(include_state)
        self.state_key = str(state_key)
        self.max_state_dim = int(max_state_dim)
        self.include_success = bool(include_success)
        self.include_fail = bool(include_fail)
        self.min_episode_length = max(1, int(min_episode_length))
        self.source_name = source_name or self.dataset_path
        self.inference_mode = bool(inference_mode)

        self._setup_lerobot(dataset_path)

        self._ep_success = self._scan_episode_success()
        self._tasks = self._load_tasks(Path(dataset_path))

        self._frame_index = self._build_frame_index()
        if not self._frame_index:
            raise ValueError(
                f"FrameClassifierDataset at {self.source_name!r} yielded zero "
                "frames after filtering. Check include_success / include_fail / "
                "min_episode_length and the dataset's is_success column."
            )

        # Proprio transform pipeline (mirrors value_dataset for parity).
        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=int(action_dim),
            default_prompt=default_prompt,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
        )

        n_success = sum(1 for _, _, lbl in self._frame_index if lbl == 0)
        n_fail = len(self._frame_index) - n_success
        self._n_success_frames = n_success
        self._n_fail_frames = n_fail
        if self.include_success and self.include_fail and not self.inference_mode:
            if n_success == 0:
                raise ValueError(
                    "FrameClassifierDataset(include_success=True) requires at "
                    f"least one success episode in {self.source_name!r}; got 0."
                )
            if n_fail == 0:
                raise ValueError(
                    "FrameClassifierDataset(include_fail=True) requires at "
                    f"least one fail episode in {self.source_name!r}; got 0. "
                    "Cannot train a fail classifier without fail data."
                )

        logger.info(
            "FrameClassifierDataset: %s, episodes=%d (success=%d, fail=%d), "
            "frames=%d (success=%d, fail=%d), include_state=%s, "
            "norm_stats_dir=%s",
            self.source_name,
            self._num_episodes,
            sum(1 for v in self._ep_success if v),
            sum(1 for v in self._ep_success if not v),
            len(self._frame_index),
            n_success,
            n_fail,
            self.include_state,
            norm_stats_dir,
        )

    # ------------------------------------------------------------------
    # LeRobot setup
    # ------------------------------------------------------------------

    def _setup_lerobot(self, dataset_path: str) -> None:
        from lerobot.common.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lerobot.common.datasets.utils import hf_transform_to_torch
        from PIL import Image as PILImage

        local_path = Path(dataset_path).absolute()
        self.meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        self.base = LeRobotDataset(
            local_path.name, root=local_path, download_videos=False
        )

        eps = self.base.episode_data_index
        self._ep_starts = [int(x) for x in eps["from"].tolist()]
        self._ep_ends = [int(x) for x in eps["to"].tolist()]
        self._num_episodes = len(self._ep_starts)

        def _decoding_transform(batch: dict) -> dict:
            for key in list(batch.keys()):
                vals = batch[key]
                if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
                    batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
            return hf_transform_to_torch(batch)

        self.base.hf_dataset.set_transform(_decoding_transform)

    def _scan_episode_success(self) -> list[bool]:
        """Return a per-episode success flag list.

        ``sft`` datasets are all-success by convention (no column read).
        ``rollout`` datasets read the Arrow ``is_success`` column at each
        episode's **last** frame row, matching
        :func:`examples.process.compute_returns.compute_returns_from_parquet`
        (``compute_returns.py:166`` does ``is_success_col[ep_end - 1]``).
        Reading the start row would disagree with the returns pipeline
        whenever ``is_success`` is only populated on terminal frames — which
        silently mislabels every episode as fail and poisons the entire
        classifier training signal. Fail-loud on any row-mismatch with
        returns is caught at the data-pipeline boundary, not here.
        """
        if self.dataset_type == "sft":
            return [True] * self._num_episodes

        raw_dataset = self.base.hf_dataset
        if "is_success" not in raw_dataset.column_names:
            if self.inference_mode:
                logger.warning(
                    "FrameClassifierDataset(dataset_type='rollout', "
                    "inference_mode=True) at %s lacks 'is_success' column. "
                    "Filling _ep_success with False (labels are not consumed "
                    "at inference time).",
                    self.source_name,
                )
                return [False] * self._num_episodes
            raise ValueError(
                "FrameClassifierDataset(dataset_type='rollout') requires the "
                f"LeRobot dataset at {self.source_name!r} to contain an "
                "'is_success' column. Missing."
            )
        is_success_column = raw_dataset.data.column("is_success")
        return [
            _coerce_success_flag(is_success_column[int(end) - 1].as_py())
            for end in self._ep_ends
        ]

    def _load_tasks(self, dataset_path: Path) -> dict[int, str]:
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

    # ------------------------------------------------------------------
    # Frame index
    # ------------------------------------------------------------------

    def _build_frame_index(self) -> list[tuple[int, int, int]]:
        """Return ``[(episode, frame, label)]`` entries after filtering."""
        out: list[tuple[int, int, int]] = []
        for ep in range(self._num_episodes):
            length = self._ep_ends[ep] - self._ep_starts[ep]
            if length < self.min_episode_length:
                continue
            success = bool(self._ep_success[ep])
            if success and not self.include_success:
                continue
            if not success and not self.include_fail:
                continue
            label = 0 if success else 1
            for frame in range(length):
                out.append((ep, frame, label))
        return out

    # ------------------------------------------------------------------
    # Openpi transform pipeline (state-side only; images stay raw HWC)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform(
        robot_type: str,
        model_type: str,
        action_dim: int,
        default_prompt: Optional[str],
        norm_stats_dir: Optional[str],
        asset_id: Optional[str],
    ):
        """Build the same state transform pipeline the value_dataset uses.

        Mirrors ``ValueDataset._build_transform`` so the classifier sees
        identical state statistics whether it is run at training time
        (this dataset) or at inference time inside compute_advantages.py
        (``SuccessFailClassifier.from_checkpoint`` → ``build_input_transforms``).
        """
        import openpi.models.model as _openpi_model
        import openpi.transforms as _openpi_transforms

        from rlinf.models.embodiment.openpi.policies import (
            arx_policy,
            franka_policy,
            libero_policy,
        )

        _mt_map = {
            "pi0": _openpi_model.ModelType.PI0,
            "pi05": _openpi_model.ModelType.PI05,
            "pi0_fast": _openpi_model.ModelType.PI0_FAST,
        }
        model_type_enum = _mt_map[model_type.lower()]
        robot = robot_type.lower()

        # Repack mapping mirrors ValueDataset's _REPACK_KEYS for the same
        # robots — see value_dataset.py:45-73 for the full table.
        from rlinf.data.datasets.cfg.value_dataset import (
            _REPACK_KEYS,
            _get_x2robot_mode,
        )

        x2robot_mode = _get_x2robot_mode(robot)
        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None and x2robot_mode is not None:
            repack_keys = _REPACK_KEYS["fold_towel_sm2sm"]
        if repack_keys is None:
            raise ValueError(
                f"Unknown robot type: {robot_type}. "
                f"Available: {list(_REPACK_KEYS.keys())}"
            )

        steps = [_openpi_transforms.RepackTransform(repack_keys)]
        if robot in ("libero", "libero_v2"):
            steps.append(libero_policy.LiberoInputs(model_type=model_type_enum))
        elif robot in ("franka", "franka_co_train"):
            steps.append(
                franka_policy.FrankaEEInputs(
                    action_dim=action_dim,
                    model_type=model_type_enum,
                )
            )
        elif x2robot_mode is not None:
            steps.append(
                arx_policy.ArxInputs(
                    mode=x2robot_mode,
                    action_dim=action_dim,
                    model_type=model_type_enum,
                )
            )

        steps.append(_openpi_transforms.InjectDefaultPrompt(default_prompt))

        if norm_stats_dir is not None:
            from rlinf.models.embodiment.value_model.checkpoint_utils import (
                load_norm_stats,
            )

            resolved_asset = asset_id or robot
            try:
                norm_stats = load_norm_stats(
                    Path(norm_stats_dir), asset_id=resolved_asset
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "FrameClassifierDataset was given "
                    f"norm_stats_dir={norm_stats_dir!r} asset_id={resolved_asset!r} "
                    "but no norm_stats.json was found. Either fix the path or "
                    "drop norm_stats_dir from the config to disable quantile "
                    "normalization (only safe when use_proprio=False)."
                ) from exc
            steps.append(_openpi_transforms.Normalize(norm_stats, use_quantiles=True))

        steps.append(_openpi_transforms.PadStatesAndActions(action_dim))
        return _openpi_transforms.compose(steps)

    # ------------------------------------------------------------------
    # Torch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame_index)

    @property
    def pos_weight(self) -> float:
        """Return ``n_success / n_fail`` — BCE positive-class balance weight.

        Used by the worker when ``pos_weight_mode='auto'`` to rebalance
        the loss without touching the sampler. Safe default: if only
        success frames are present (``include_fail=False`` — edge case
        where the classifier is evaluated rather than trained end-to-end),
        falls back to ``1.0``.
        """
        if self._n_fail_frames == 0:
            return 1.0
        return float(self._n_success_frames) / float(self._n_fail_frames)

    def _load_raw_sample(self, ep: int, frame: int) -> dict:
        global_idx = self._ep_starts[ep] + int(frame)
        return self.base[global_idx]

    def _load_views(
        self, raw_sample: dict
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        images: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for camera_key in self.camera_keys:
            try:
                raw = _resolve_alias(raw_sample, camera_key, _IMAGE_KEY_ALIASES)
            except KeyError:
                masks[camera_key] = False
                continue
            images[camera_key] = _to_uint8_hwc(raw)
            masks[camera_key] = True
        return images, masks

    def set_epoch(self, epoch: int) -> None:
        # The loader wrapper (_PairDataLoaderImpl) calls this; the flat
        # frame dataset has no RNG state of its own.
        del epoch

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        ep, frame, label = self._frame_index[idx]
        raw = self._load_raw_sample(ep, frame)

        images, image_masks = self._load_views(raw)

        # Inject prompt from task mapping so the downstream transform
        # pipeline has a prompt to manipulate (LiberoInputs / FrankaEEInputs
        # pass it through). The classifier itself does not consume
        # prompts — this keeps the transform behaviour identical to the
        # value dataset.
        if self._tasks and "task_index" in raw:
            ti = raw["task_index"]
            ti_int = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
            if ti_int in self._tasks:
                raw = {**raw, "prompt": self._tasks[ti_int]}

        sample: dict[str, Any] = {
            "images": images,
            "image_masks": image_masks,
            "label": int(label),
            "episode": int(ep),
            "frame_index": int(frame),
            "source_name": self.source_name,
        }

        if self.include_state:
            transformed = self._transform(raw) if self._transform is not None else raw
            state = transformed.get("state")
            if state is None:
                # Fail-loud: state was configured but the transform pipeline
                # didn't produce one. Silent defaulting would hide real
                # schema bugs (per 78bc04dd fail-loud policy).
                raise ValueError(
                    "FrameClassifierDataset(include_state=True) expected "
                    f"post-transform 'state' for episode={ep} frame={frame} in "
                    f"{self.source_name!r} but got None. Check the robot_type "
                    "repack mapping."
                )
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = np.asarray(state)
            sample["state"] = _to_float32_1d(state_np, max_dim=self.max_state_dim)
        return sample


class FrameClassifierMixtureDataset(_MixtureBase):
    """Weighted mixture of multiple :class:`FrameClassifierDataset`s.

    Thin wrapper around :class:`_MixtureBase` so multiple LeRobot sources
    can feed the classifier training loop with per-source sampling
    weights. No additional behaviour beyond what ``_MixtureBase``
    already provides; kept as a named subclass for parity with
    :class:`PairMixtureDataset` and :class:`CfgMixtureDataset`.
    """

    def __init__(
        self,
        datasets: Sequence[tuple[FrameClassifierDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        super().__init__(
            datasets=datasets,
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            seed=seed,
        )

    @property
    def pos_weight(self) -> float:
        """Return the mixture-weighted ``P(success) / P(fail)`` ratio.

        :class:`_MixtureBase._sample_step` samples a frame in two stages:
        it first picks a dataset ``i`` with probability ``p_i`` (read from
        ``_dataset_sampling_weights``, already normalised to sum-to-1) and
        then draws a frame uniformly inside that dataset. The per-sample
        expected class probabilities are therefore

            P(fail)    = sum_i p_i * (n_fail_i    / len_i)
            P(success) = sum_i p_i * (n_success_i / len_i)

        and the loss-side ``pos_weight`` that neutralises this imbalance is
        ``P(success) / P(fail)``. An earlier implementation computed
        ``sum_i p_i · n_success_i / sum_i p_i · n_fail_i`` — that formula
        silently re-weights each dataset by its *total* size (instead of
        its internal class ratio), so a large all-success dataset mixed
        with a small balanced dataset yielded a grossly inflated
        ``pos_weight``. Fixed to use per-dataset ratios. Fallback to
        ``1.0`` when the mixture has no fail samples (nothing to rebalance).
        """
        num = 0.0
        den = 0.0
        for ds, p in zip(self.datasets, self._dataset_sampling_weights):
            if p <= 0:
                continue
            length = max(1, int(len(ds)))
            n_success_i = float(getattr(ds, "_n_success_frames", 0))
            n_fail_i = float(getattr(ds, "_n_fail_frames", 0))
            num += float(p) * n_success_i / length
            den += float(p) * n_fail_i / length
        if den <= 0.0:
            return 1.0
        return num / den


def build_frame_classifier_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    world_size: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Build the training DataLoader with a ``DistributedSampler``.

    Class balance is handled at the loss via ``pos_weight`` (see
    :meth:`FrameClassifierDataset.pos_weight`), not by a
    ``WeightedRandomSampler`` — a weighted sampler would double-random
    against :class:`_MixtureBase`'s internal sampling and would not
    honour rank-aware sharding under FSDP. See the plan's §6 for the
    full rationale.
    """
    del seed  # accepted for signature parity; sampler draws fresh per-epoch

    sampler = None
    if torch.distributed.is_initialized() and world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )


__all__ = [
    "DEFAULT_CAMERA_KEYS",
    "POS_WEIGHT_MODES",
    "FrameClassifierDataset",
    "FrameClassifierMixtureDataset",
    "build_frame_classifier_loader",
]
