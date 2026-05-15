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

"""Shared utilities for CFG-style embodied data loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


def load_advantages_lookup(
    data_path: str,
    advantage_tag: str | None = None,
) -> dict[tuple[int, int], bool]:
    """Load an episode/frame advantage lookup from dataset metadata.

    Args:
        data_path: Path to a LeRobot dataset.
        advantage_tag: Optional advantage tag. When provided, loads
            ``meta/advantages_{advantage_tag}.parquet``; otherwise loads
            ``meta/advantages.parquet``.

    Returns:
        Mapping from ``(episode_index, frame_index)`` to a boolean advantage
        label.
    """
    import pandas as pd

    if advantage_tag:
        meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
    else:
        meta_path = Path(data_path) / "meta" / "advantages.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {meta_path}. "
            "Run compute_advantages.py first."
        )

    adv_df = pd.read_parquet(meta_path)
    if "advantage" not in adv_df.columns:
        raise ValueError(f"Advantage file {meta_path} missing 'advantage' column")
    return dict(
        zip(
            zip(
                adv_df["episode_index"].values.astype(int).tolist(),
                adv_df["frame_index"].values.astype(int).tolist(),
            ),
            adv_df["advantage"].values.astype(bool).tolist(),
        )
    )


def load_continuous_advantages_lookup(
    data_path: str,
    advantage_tag: str | None = None,
) -> dict[tuple[int, int], float]:
    """Load an episode/frame continuous advantage lookup from metadata."""
    import pandas as pd

    if advantage_tag:
        meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
    else:
        meta_path = Path(data_path) / "meta" / "advantages.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {meta_path}. "
            "Run compute_advantages.py first."
        )

    adv_df = pd.read_parquet(meta_path)
    required = {"episode_index", "frame_index", "advantage_continuous"}
    missing = required - set(adv_df.columns)
    if missing:
        raise ValueError(
            f"Advantage file {meta_path} missing required columns: {sorted(missing)}"
        )

    return dict(
        zip(
            zip(
                adv_df["episode_index"].values.astype(int).tolist(),
                adv_df["frame_index"].values.astype(int).tolist(),
            ),
            adv_df["advantage_continuous"].values.astype(float).tolist(),
        )
    )


def load_positive_threshold(data_path: str, advantage_tag: str) -> float:
    """Load the per-tag positive threshold used for AWBC weighting."""
    cfg_path = Path(data_path) / "meta" / "mixture_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Mixture config not found: {cfg_path}. "
            "AWBC needs tags.<advantage_tag>.positive_threshold."
        )

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    tags = cfg.get("tags") or {}
    if not isinstance(tags, dict) or advantage_tag not in tags:
        raise KeyError(
            f"Tag {advantage_tag!r} not found in {cfg_path} under 'tags'."
        )

    threshold = tags[advantage_tag].get("positive_threshold")
    if threshold is None:
        raise KeyError(
            f"Tag {advantage_tag!r} in {cfg_path} missing 'positive_threshold'."
        )
    return float(threshold)


def cast_image_features(hf_dataset: Any) -> Any:
    """Cast image columns from struct to ``datasets.Image`` for decoding."""
    from datasets import Image

    features = hf_dataset.features
    needs_cast = False
    new_features = features.copy()

    for key, feat in features.items():
        if isinstance(feat, dict) and "bytes" in feat:
            new_features[key] = Image()
            needs_cast = True

    if needs_cast:
        from lerobot.common.datasets.utils import hf_transform_to_torch

        hf_dataset = hf_dataset.cast(new_features)
        hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset


class AdvantagePreservingDataset:
    """Wrapper that restores advantage labels after OpenPI transforms."""

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None = None,
        constant_advantage: bool | None = None,
        continuous_advantages_lookup: dict[tuple[int, int], float] | None = None,
        positive_threshold: float | None = None,
        max_continuous_advantage: float | None = None,
        filter_positive_continuous: bool = False,
    ):
        self._transformed_dataset = transformed_dataset
        self._advantage_by_index = self._build_advantage_index(
            base_dataset, advantages_lookup, constant_advantage
        )
        self._advantage_continuous_by_index = self._build_continuous_advantage_index(
            base_dataset, continuous_advantages_lookup
        )
        self._positive_threshold = positive_threshold
        self._max_continuous_advantage = self._infer_max_continuous_advantage(
            base_dataset, max_continuous_advantage
        )
        self._base_dataset = (
            base_dataset
            if self._advantage_by_index is None
            or self._advantage_continuous_by_index is None
            else None
        )
        self._indices = None
        if filter_positive_continuous:
            if self._advantage_continuous_by_index is None:
                raise ValueError(
                    "filter_positive_continuous=True requires continuous advantages"
                )
            if self._positive_threshold is None:
                raise ValueError(
                    "filter_positive_continuous=True requires positive_threshold"
                )
            if self._max_continuous_advantage is None:
                raise ValueError(
                    "filter_positive_continuous=True requires max_continuous_advantage"
                )
            self._indices = [
                idx
                for idx in range(len(self._transformed_dataset))
                if self.get_advantage_weight(idx) > 0.0
            ]

    @property
    def advantage_by_index(self) -> dict[int, bool] | None:
        """Return the cached index -> advantage mapping when available."""
        return self._advantage_by_index

    @property
    def kept_ratio(self) -> float:
        """Return the fraction of transformed samples kept after filtering."""
        if self._indices is None:
            return 1.0
        total = len(self._transformed_dataset)
        return len(self._indices) / total if total > 0 else 0.0

    @staticmethod
    def _get_hf_dataset(dataset: Any) -> Any:
        """Extract the underlying HuggingFace dataset from wrapped datasets."""
        current = dataset
        while current is not None:
            if hasattr(current, "hf_dataset"):
                return current.hf_dataset
            if hasattr(current, "_dataset"):
                current = current._dataset
            else:
                return None
        return None

    def _build_advantage_index(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None,
        constant_advantage: bool | None,
    ) -> dict[int, bool] | None:
        """Build a mapping from transformed index to advantage label."""
        if constant_advantage is not None:
            return {i: bool(constant_advantage) for i in range(len(base_dataset))}

        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            logger.warning(
                "Cannot access underlying HF dataset, "
                "falling back to per-sample advantage loading (slower)."
            )
            return None

        if advantages_lookup is not None:
            ep_indices = hf_dataset["episode_index"]
            frame_indices = hf_dataset["frame_index"]
            advantage_by_index: dict[int, bool] = {}
            missing_keys: list[tuple[int, int]] = []
            for idx in range(len(hf_dataset)):
                key = (int(ep_indices[idx]), int(frame_indices[idx]))
                if key in advantages_lookup:
                    advantage_by_index[idx] = advantages_lookup[key]
                else:
                    missing_keys.append(key)
            if missing_keys:
                raise ValueError(
                    f"[AdvantagePreservingDataset] {len(missing_keys)} samples not found "
                    f"in advantages lookup (first 5: {missing_keys[:5]}). "
                    "The advantages parquet does not match this dataset. "
                    "Re-run compute_advantages.py."
                )
            return advantage_by_index

        if "advantage" in hf_dataset.column_names:
            return {
                idx: bool(value) for idx, value in enumerate(hf_dataset["advantage"])
            }

        raise ValueError(
            "[AdvantagePreservingDataset] No advantage data found: "
            "advantages_lookup is None, and 'advantage' column not in dataset. "
            "Run compute_advantages.py first."
        )

    def _build_continuous_advantage_index(
        self,
        base_dataset: Any,
        continuous_advantages_lookup: dict[tuple[int, int], float] | None,
    ) -> dict[int, float] | None:
        """Build a mapping from transformed index to continuous advantage."""
        if continuous_advantages_lookup is None:
            return None

        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            logger.warning(
                "Cannot access underlying HF dataset, "
                "falling back to per-sample continuous advantage loading (slower)."
            )
            return None

        ep_indices = hf_dataset["episode_index"]
        frame_indices = hf_dataset["frame_index"]
        advantage_by_index: dict[int, float] = {}
        missing_keys: list[tuple[int, int]] = []
        for idx in range(len(hf_dataset)):
            key = (int(ep_indices[idx]), int(frame_indices[idx]))
            if key in continuous_advantages_lookup:
                advantage_by_index[idx] = float(continuous_advantages_lookup[key])
            else:
                missing_keys.append(key)
        if missing_keys:
            raise ValueError(
                f"[AdvantagePreservingDataset] {len(missing_keys)} samples not found "
                f"in continuous advantages lookup (first 5: {missing_keys[:5]}). "
                "The advantages parquet does not match this dataset. "
                "Re-run compute_advantages.py."
            )
        return advantage_by_index

    def _infer_max_continuous_advantage(
        self,
        base_dataset: Any,
        max_continuous_advantage: float | None,
    ) -> float | None:
        """Infer the normalization maximum for continuous AWBC weights."""
        if max_continuous_advantage is not None:
            return float(max_continuous_advantage)

        if self._advantage_continuous_by_index is not None:
            if not self._advantage_continuous_by_index:
                return None
            return max(self._advantage_continuous_by_index.values())

        hf_dataset = self._get_hf_dataset(base_dataset)
        if (
            hf_dataset is not None
            and hasattr(hf_dataset, "column_names")
            and "advantage_continuous" in hf_dataset.column_names
        ):
            values = hf_dataset["advantage_continuous"]
            return max(float(value) for value in values) if len(values) > 0 else None

        return None

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else len(
            self._transformed_dataset
        )

    def _resolve_index(self, idx: int) -> int:
        return self._indices[idx] if self._indices is not None else idx

    def get_advantage(self, idx: int) -> bool:
        """Return the boolean advantage for ``idx``."""
        if self._advantage_by_index is not None:
            if idx not in self._advantage_by_index:
                raise KeyError(
                    f"[AdvantagePreservingDataset] Index {idx} not found in advantage index. "
                    f"Dataset size: {len(self._transformed_dataset)}, "
                    f"advantage index size: {len(self._advantage_by_index)}."
                )
            return self._advantage_by_index[idx]

        base_sample = self._base_dataset[idx]
        if "advantage" not in base_sample:
            raise KeyError(
                f"[AdvantagePreservingDataset] 'advantage' key not found in base_sample "
                f"at index {idx}. Run compute_advantages.py first."
            )
        advantage = base_sample["advantage"]
        if isinstance(advantage, torch.Tensor):
            advantage = bool(advantage.item())
        return bool(advantage)

    def get_advantage_continuous(self, idx: int) -> float:
        """Return the continuous advantage for ``idx``."""
        if self._advantage_continuous_by_index is not None:
            if idx not in self._advantage_continuous_by_index:
                raise KeyError(
                    f"[AdvantagePreservingDataset] Index {idx} not found in "
                    "continuous advantage index."
                )
            return float(self._advantage_continuous_by_index[idx])

        base_sample = self._base_dataset[idx]
        if "advantage_continuous" not in base_sample:
            raise KeyError(
                "[AdvantagePreservingDataset] 'advantage_continuous' key not found "
                f"in base_sample at index {idx}. Run compute_advantages.py first."
            )
        advantage = base_sample["advantage_continuous"]
        if isinstance(advantage, torch.Tensor):
            advantage = float(advantage.item())
        return float(advantage)

    def get_advantage_weight(self, idx: int) -> float:
        """Return normalized AWBC weight clipped into ``[0, 1]``."""
        if self._positive_threshold is None:
            raise ValueError("positive_threshold is required for advantage weights")
        if self._max_continuous_advantage is None:
            raise ValueError("max_continuous_advantage is required for AWBC weights")

        threshold = float(self._positive_threshold)
        denom = float(self._max_continuous_advantage) - threshold
        if denom <= 0.0:
            return 0.0

        weight = (self.get_advantage_continuous(idx) - threshold) / denom
        return min(max(weight, 0.0), 1.0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get one transformed sample and re-attach its advantage label."""
        real_idx = self._resolve_index(idx)
        sample = self._transformed_dataset[real_idx]
        sample["advantage"] = self.get_advantage(real_idx)
        if self._advantage_continuous_by_index is not None:
            sample["advantage_continuous"] = self.get_advantage_continuous(real_idx)
            sample["advantage_weight"] = self.get_advantage_weight(real_idx)
        return sample


class PositiveAdvantageOnlySubset(AdvantagePreservingDataset):
    """Subset wrapper that keeps only ``advantage=True`` samples."""

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None = None,
        constant_advantage: bool | None = None,
    ):
        super().__init__(
            base_dataset=base_dataset,
            transformed_dataset=transformed_dataset,
            advantages_lookup=advantages_lookup,
            constant_advantage=constant_advantage,
        )
        self._positive_indices = [
            idx
            for idx in range(len(self._transformed_dataset))
            if self.get_advantage(idx)
        ]

    def __len__(self) -> int:
        return len(self._positive_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transformed_dataset[self._positive_indices[idx]]


def fix_episode_data_index(dataset: Any, episodes: list[int]) -> None:
    """Fix LeRobotDataset episode indices after dataset-level filtering."""
    ep_idx_mapping = {ep: i for i, ep in enumerate(sorted(episodes))}
    max_ep_idx = max(episodes) + 1

    old_from = dataset.episode_data_index["from"]
    old_to = dataset.episode_data_index["to"]

    new_from = torch.full((max_ep_idx,), -1, dtype=old_from.dtype)
    new_to = torch.full((max_ep_idx,), -1, dtype=old_to.dtype)

    for orig_ep, new_idx in ep_idx_mapping.items():
        new_from[orig_ep] = old_from[new_idx]
        new_to[orig_ep] = old_to[new_idx]

    dataset.episode_data_index["from"] = new_from
    dataset.episode_data_index["to"] = new_to


def create_distributed_torch_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    world_size: int,
    rank: int,
    shuffle: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool = True,
) -> Any:
    """Create a PyTorch DataLoader with a distributed sampler when needed."""
    sampler = None

    if torch.distributed.is_initialized():
        if batch_size % world_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by world_size ({world_size})"
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // world_size
    else:
        local_batch_size = batch_size

    if prefetch_factor is None:
        prefetch_factor = 4 if num_workers > 0 else None
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )


class _BaseOpenPIDataLoaderImpl:
    """Shared wrapper behavior for OpenPI-backed dataloaders."""

    def __init__(self, data_config: Any, data_loader: Any):
        self._data_config = data_config
        self._data_loader = data_loader

    @property
    def sampler(self) -> Any:
        """Expose the inner sampler for compatibility with existing workers."""
        return getattr(self._data_loader, "sampler", None)

    @property
    def dataset(self) -> Any:
        """Expose the inner dataset for compatibility with existing workers."""
        return getattr(self._data_loader, "dataset", None)

    def data_config(self) -> Any:
        """Return the OpenPI data configuration."""
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        """Forward ``set_epoch`` to the wrapped sampler and dataset."""
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if self.dataset is not None and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class CFGDataLoaderImpl(_BaseOpenPIDataLoaderImpl):
    """Yield CFG training tuples, including AWBC fields when present."""

    def __iter__(self):
        from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
            Observation as CFGObservation,
        )

        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = batch["advantage"]
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, dtype=torch.bool)

            if "advantage_weight" not in batch:
                yield observation, actions, advantage
                continue

            advantage_continuous = batch["advantage_continuous"]
            if not isinstance(advantage_continuous, torch.Tensor):
                advantage_continuous = torch.tensor(
                    advantage_continuous, dtype=torch.float32
                )
            advantage_weight = batch["advantage_weight"]
            if not isinstance(advantage_weight, torch.Tensor):
                advantage_weight = torch.tensor(advantage_weight, dtype=torch.float32)

            yield observation, actions, advantage, advantage_continuous, advantage_weight


class SftPlainDataLoaderImpl(_BaseOpenPIDataLoaderImpl):
    """Yield OpenPI SFT tuples, including AWBC weights when present."""

    def __iter__(self):
        from openpi.models import model as openpi_model

        for batch in self._data_loader:
            observation = openpi_model.Observation.from_dict(batch)
            actions = batch["actions"]
            if "advantage_weight" not in batch:
                yield observation, actions
                continue

            advantage_continuous = batch["advantage_continuous"]
            if not isinstance(advantage_continuous, torch.Tensor):
                advantage_continuous = torch.tensor(
                    advantage_continuous, dtype=torch.float32
                )
            advantage_weight = batch["advantage_weight"]
            if not isinstance(advantage_weight, torch.Tensor):
                advantage_weight = torch.tensor(advantage_weight, dtype=torch.float32)

            yield observation, actions, advantage_continuous, advantage_weight
