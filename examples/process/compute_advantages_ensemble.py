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

"""
Compute advantages for CFG-RL training using a trained ensemble
BinaryValueCriticModel (ARM + ReWiND).

Per-frame advantage = ensemble-aggregated p(progress) for the pair
``(frame_t, frame_{t+k})``. The label written to the parquet is
``True`` iff ``p(progress) > positive_threshold`` (or unconditionally
``True`` for ``dataset_type == "sft"``).

Output: ``meta/advantages_{tag}.parquet`` per dataset, with columns
``[episode_index, frame_index, advantage, advantage_continuous,
   p_progress_mean, p_progress_min, p_progress_variance, member_values]``.
Terminal frames are backfilled with zero progress so the saved parquet covers
every frame expected by CFG training.
``mixture_config.yaml`` under ``meta/`` is updated with a per-tag entry.

Usage:
    python compute_advantages_ensemble.py \\
        --config-name compute_advantages_ensemble_fail150_k8_ensemble4_wco

    # Multi-GPU
    torchrun --nproc_per_node=4 compute_advantages_ensemble.py \\
        --config-name compute_advantages_ensemble_fail150_k8_ensemble4_wco
"""

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _silence_libav_logs() -> None:
    """Suppress the ``[libdav1d @ 0x..] libdav1d 0.9.2`` chatter that
    torchcodec (LeRobot's default video backend) emits every frame.

    The messages are written by libav* **directly to file-descriptor 2**
    — pyav's log level does not intercept them because torchcodec
    bypasses pyav. We splice our own ``fd=2`` through a long-lived
    ``grep -v '\\[libdav1d'`` subprocess so every write to stderr is
    filtered line-by-line; all other stderr output (our own logs,
    tracebacks, etc.) still reaches the terminal.
    """
    import atexit
    import shutil
    import subprocess
    import sys

    # pyav's log level still helps when pyav IS the backend (older lerobot
    # fallback); cheap to do alongside the fd-level filter.
    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    if not shutil.which("grep"):
        return
    # Save the original fd=2 BEFORE we redirect — we need it back at exit
    # so grep can see EOF on its stdin (otherwise grep blocks forever and
    # ``atexit`` deadlocks because our fd=2 is still a write-end of the pipe).
    saved_stderr_fd = os.dup(2)
    try:
        # grep's stdout points at the ORIGINAL stderr so filtered lines
        # keep showing. grep's stdin becomes our new fd=2.
        grep = subprocess.Popen(
            # Match any libdav1d / libav* chatter — both the `[libdav1d @
            # 0x..] libdav1d 0.9.2` form AND any continuation line that
            # contains just `libdav1d`. Cast a wider net so child workers
            # whose output formatting differs don't slip through.
            ["grep", "-v", "-E", "--line-buffered", r"libdav1d|libdav1d 0\.9"],
            stdin=subprocess.PIPE,
            stdout=saved_stderr_fd,
        )
    except Exception:
        os.close(saved_stderr_fd)
        return

    sys.stderr.flush()
    os.dup2(grep.stdin.fileno(), 2)

    def _restore_and_drain() -> None:
        # Restore fd=2 first so subsequent stderr writes (e.g. tracebacks)
        # still reach the terminal — and crucially so grep sees EOF on its
        # stdin instead of waiting on us forever.
        try:
            os.dup2(saved_stderr_fd, 2)
        except OSError:
            pass
        try:
            os.close(saved_stderr_fd)
        except OSError:
            pass
        try:
            grep.stdin.close()
        except Exception:
            pass
        try:
            grep.wait(timeout=3)
        except Exception:
            grep.kill()

    atexit.register(_restore_and_drain)


_silence_libav_logs()

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Make the rlinf package importable regardless of the cwd the user launched from.
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from rlinf.data.datasets.cfg.rewind.pair_dataset import (  # noqa: E402
    BinaryPairDataCollator,
    _LeRobotSource,
    _to_float32_1d,
)
from rlinf.models.embodiment.value_model_rewind_arm.modeling_critic import (  # noqa: E402
    BinaryValueCriticModel,
)
from rlinf.models.embodiment.value_model_rewind_arm.ensemble_modeling_critic import (  # noqa: E402
    EnsembleBinaryValueCriticModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers (mirrors compute_advantages.py; inlined to avoid
# importing that module, which has a broken `lerobot.common.datasets` top-level
# import in the openpi env that ships only `lerobot.common.datasets`).
# ---------------------------------------------------------------------------


def setup_distributed(cfg: DictConfig) -> tuple[int, int, str]:
    """Initialise torch.distributed for torchrun-launched processes."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist_cfg = cfg.get("distributed", {})
        backend = dist_cfg.get("backend", "nccl")
        timeout_seconds = dist_cfg.get("timeout", 1800)
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend, timeout=timedelta(seconds=timeout_seconds)
            )
        torch.cuda.set_device(local_rank)
        return rank, world_size, f"cuda:{local_rank}"
    return 0, 1, "cuda"


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_shard_indices(
    total_samples: int, rank: int, world_size: int
) -> tuple[int, int]:
    """Even shard with earlier ranks taking the remainder."""
    base = total_samples // world_size
    rem = total_samples % world_size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


def gather_all_advantages(
    local_df: pd.DataFrame, rank: int, world_size: int
) -> pd.DataFrame:
    if world_size == 1:
        return local_df
    all_dfs: list[Optional[list[dict[str, Any]]]] = [None] * world_size
    dist.all_gather_object(all_dfs, local_df.to_dict("records"))
    rows: list[dict[str, Any]] = []
    for shard in all_dfs:
        if shard:
            rows.extend(shard)
    merged = pd.DataFrame(rows)
    if len(merged) > 0:
        merged = merged.sort_values(["episode_index", "frame_index"]).reset_index(
            drop=True
        )
    return merged


# ---------------------------------------------------------------------------
# Inference dataset
# ---------------------------------------------------------------------------


class BinaryPairInferenceDataset(Dataset):
    """Yields one ``(frame_t, frame_{t+k})`` pair per anchor for inference.

    Differences vs :class:`PairDataset`:
        * No success-only filter — every episode contributes pairs (matches
          ``compute_advantages.py`` which scores every frame).
        * Forward direction only: ``image_t = frame_t``, ``image_tk = frame_{t+k}``,
          ``label = 0.0`` (placeholder so the existing collator works).
        * Boundary clamp identical to PairDataset: when ``t + k > T - 1`` the
          second slot is clamped to ``T - 1``.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        camera_keys: list[str],
        k: int,
        prompt: Optional[str],
        include_state: bool,
        state_max_dim: Optional[int],
        state_key: str,
        dataset_type: str,
        min_episode_length: Optional[int] = None,
    ) -> None:
        if dataset_type not in ("sft", "rollout"):
            raise ValueError(
                "BinaryPairInferenceDataset.dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}"
            )
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not camera_keys:
            raise ValueError("camera_keys must be non-empty")

        self.k = int(k)
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        self.include_state = bool(include_state)
        self.state_max_dim = state_max_dim
        self.state_key = str(state_key)
        self.dataset_type = dataset_type
        self.source_name = str(dataset_path)

        # Iterate every episode regardless of is_success; _LeRobotSource with
        # only_success=False skips the success scan and treats all episodes
        # as eligible.
        self._source = _LeRobotSource(
            dataset_path,
            only_success=False,
            dataset_type=dataset_type,
        )

        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)

        total_eps = self._source.num_episodes()
        self._eligible: list[int] = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes in {dataset_path!r} with length >= "
                f"{self._min_episode_length} (dataset has {total_eps} episodes)."
            )

        # One anchor per t in [0, T - 2]; total = sum(T_ep - 1).
        pair_positions_per_episode = np.array(
            [self._source.episode_length(ep) - 1 for ep in self._eligible],
            dtype=np.int64,
        )
        self._pair_position_ends = np.cumsum(pair_positions_per_episode)
        self._num_pair_positions = int(self._pair_position_ends[-1])

        logger.info(
            "BinaryPairInferenceDataset: source=%s, episodes=%d, k=%d, "
            "total_anchors=%d, include_state=%s, dataset_type=%s, "
            "camera_keys=%s",
            self.source_name,
            len(self._eligible),
            self.k,
            self._num_pair_positions,
            self.include_state,
            self.dataset_type,
            self.camera_keys,
        )

    def __len__(self) -> int:
        return self._num_pair_positions

    def _resolve_pair_position(self, idx: int) -> tuple[int, int, int]:
        if idx < 0 or idx >= self._num_pair_positions:
            raise IndexError(idx)
        episode_slot = int(np.searchsorted(self._pair_position_ends, idx, side="right"))
        prev_episode_end = (
            int(self._pair_position_ends[episode_slot - 1]) if episode_slot > 0 else 0
        )
        episode = int(self._eligible[episode_slot])
        t = int(idx - prev_episode_end)
        t_plus_k = min(t + self.k, self._source.episode_length(episode) - 1)
        return episode, t, t_plus_k

    def _resolve_prompt(self, episode: int, frame_idx: int) -> str:
        prompt = self._source.get_prompt(episode, frame_idx)
        if prompt:
            return prompt
        if self.prompt is None:
            raise RuntimeError(
                f"No per-episode task instruction for episode={episode} in "
                f"{self.source_name!r} and no fallback prompt was provided."
            )
        return self.prompt

    def _load_views(
        self, episode: int, frame_idx: int
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for cam in self.camera_keys:
            view = self._source.get_view(episode, frame_idx, cam)
            if view is None:
                masks[cam] = False
            else:
                views[cam] = view
                masks[cam] = True
        return views, masks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, t, t_plus_k = self._resolve_pair_position(idx)
        prompt = self._resolve_prompt(episode, t)

        views_t, mask_t = self._load_views(episode, t)
        views_tk, mask_tk = self._load_views(episode, t_plus_k)

        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": 0.0,  # placeholder; collator emits but inference ignores
            "episode": int(episode),
            "frame_idx_t": int(t),
            "frame_idx_tk": int(t_plus_k),
            "source_name": self.source_name,
        }

        if self.include_state:
            state_t = _to_float32_1d(
                self._source.get_state(episode, t, self.state_key),
                max_dim=self.state_max_dim,
            )
            state_tk = _to_float32_1d(
                self._source.get_state(episode, t_plus_k, self.state_key),
                max_dim=self.state_max_dim,
            )
            sample["state"] = state_t
            sample["state_tk"] = state_tk

        return sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_cfg(cfg: DictConfig) -> None:
    """Hard-fail on configuration mistakes — no silent fallbacks."""
    if "advantage" not in cfg:
        raise ValueError("Config missing 'advantage' section")
    if "data" not in cfg:
        raise ValueError("Config missing 'data' section")

    ckpt = cfg.advantage.get("value_checkpoint")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"value_checkpoint does not exist: {ckpt!r}")

    threshold = cfg.advantage.get("positive_threshold")
    if threshold is None:
        raise ValueError("advantage.positive_threshold is required")
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"positive_threshold must be in [0, 1] (it is a probability); "
            f"got {threshold}"
        )

    tag = cfg.advantage.get("tag")
    if not tag:
        raise ValueError("advantage.tag is required")

    k = int(cfg.data.get("k", 0))
    if k < 1:
        raise ValueError(f"data.k must be >= 1, got {k}")

    train_paths = cfg.data.get("train_data_paths")
    if not train_paths:
        raise ValueError("data.train_data_paths is empty")
    for entry in train_paths:
        ds_type = entry.get("type")
        if ds_type not in ("sft", "rollout"):
            raise ValueError(
                f"train_data_paths entry has invalid 'type'={ds_type!r}; "
                "must be 'sft' or 'rollout'"
            )
        ds_path = entry.get("dataset_path")
        if not ds_path or not Path(ds_path).exists():
            raise FileNotFoundError(f"dataset_path does not exist: {ds_path!r}")


def _move_to_device(obj: Any, device: str):
    """Recursive .to(device) for tensors nested in dicts/lists."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def _coerce_inference_model(model) -> EnsembleBinaryValueCriticModel:
    """Confirm the loaded checkpoint is an ensemble; raise otherwise."""
    if not isinstance(model, EnsembleBinaryValueCriticModel):
        raise RuntimeError(
            "Expected ensemble checkpoint, but BinaryValueCriticModel.from_checkpoint "
            f"returned {type(model).__name__}. Check that the saved config.json has "
            "ensemble_size > 1."
        )
    return model


def _build_dataloader(
    dataset: Dataset,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    collate_fn,
) -> tuple[DataLoader, int]:
    """Shard the dataset across ranks and wrap in a DataLoader."""
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Inference dataset is empty")
    start, end = get_shard_indices(total, rank, world_size)
    shard_indices = list(range(start, end))
    shard = Subset(dataset, shard_indices)
    loader = DataLoader(
        shard,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return loader, len(shard_indices)


def _records_from_predict(out, batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-sample row dicts from a single CriticOutput + batch metadata."""
    aggregated = out.predicted_values.detach().to("cpu", dtype=torch.float32)
    mean = out.prediction_mean.detach().to("cpu", dtype=torch.float32)
    minv = out.prediction_min.detach().to("cpu", dtype=torch.float32)
    var = out.prediction_variance.detach().to("cpu", dtype=torch.float32)
    members = out.member_predicted_values.detach().to("cpu", dtype=torch.float32)
    # members: [K, B]
    episodes = batch["episode"].tolist()
    frame_t = batch["frame_idx_t"].tolist()

    rows: list[dict[str, Any]] = []
    bsize = aggregated.shape[0]
    for i in range(bsize):
        rows.append(
            {
                "episode_index": int(episodes[i]),
                "frame_index": int(frame_t[i]),
                "p_progress_aggregated": float(aggregated[i].item()),
                "p_progress_mean": float(mean[i].item()),
                "p_progress_min": float(minv[i].item()),
                "p_progress_variance": float(var[i].item()),
                "member_values": [float(x) for x in members[:, i].tolist()],
            }
        )
    return rows


def _build_terminal_frame_rows(
    *,
    episode_lengths: list[int],
    member_count: int,
) -> pd.DataFrame:
    """Build default-negative rows for each episode's terminal frame."""
    rows: list[dict[str, Any]] = []
    zero_members = [0.0] * max(1, int(member_count))
    for episode_index, episode_length in enumerate(episode_lengths):
        if int(episode_length) < 1:
            continue
        rows.append(
            {
                "episode_index": int(episode_index),
                "frame_index": int(episode_length) - 1,
                "p_progress_aggregated": 0.0,
                "p_progress_mean": 0.0,
                "p_progress_min": 0.0,
                "p_progress_variance": 0.0,
                "member_values": list(zero_members),
            }
        )
    return pd.DataFrame(rows)


def _append_missing_terminal_rows(
    df: pd.DataFrame,
    *,
    episode_lengths: list[int],
    member_count: int,
) -> tuple[pd.DataFrame, int]:
    """Append any missing terminal frames with zero progress defaults."""
    terminal_rows = _build_terminal_frame_rows(
        episode_lengths=episode_lengths,
        member_count=member_count,
    )
    if terminal_rows.empty:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        return df, 0

    if len(df) == 0:
        combined = terminal_rows.sort_values(
            ["episode_index", "frame_index"]
        ).reset_index(drop=True)
        return combined, len(combined)

    existing_keys = set(
        map(
            tuple,
            df[["episode_index", "frame_index"]].astype(int).values.tolist(),
        )
    )
    missing_terminal_rows = terminal_rows[
        [
            (int(row.episode_index), int(row.frame_index)) not in existing_keys
            for row in terminal_rows.itertuples(index=False)
        ]
    ]
    combined = pd.concat([df, missing_terminal_rows], ignore_index=True)
    combined = combined.sort_values(["episode_index", "frame_index"]).reset_index(
        drop=True
    )
    return combined, len(missing_terminal_rows)


def _run_inference_for_dataset(
    *,
    model: EnsembleBinaryValueCriticModel,
    dataset_entry: DictConfig,
    cfg: DictConfig,
    rank: int,
    world_size: int,
    device: str,
) -> pd.DataFrame:
    """Run ensemble inference on one dataset; return a sorted DataFrame on rank 0."""
    dataset = BinaryPairInferenceDataset(
        dataset_path=dataset_entry.dataset_path,
        camera_keys=list(cfg.data.camera_keys),
        k=int(cfg.data.k),
        prompt=cfg.data.get("prompt", None),
        include_state=bool(cfg.data.get("include_state_in_prompt", False)),
        state_max_dim=cfg.data.get("max_state_dim", None),
        state_key=str(cfg.data.get("state_key", "state")),
        dataset_type=dataset_entry.type,
    )

    collator = BinaryPairDataCollator(
        processor=model.members[0].processor,
        max_length=int(getattr(model.config, "max_token_len", 200)),
        train=False,
    )

    loader, shard_size = _build_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=int(cfg.advantage.batch_size),
        num_workers=int(cfg.advantage.num_dataloader_workers_per_gpu),
        prefetch_factor=int(cfg.advantage.prefetch_factor),
        collate_fn=collator,
    )

    if rank == 0:
        logger.info(
            "Dataset %s: total_anchors=%d, rank0 shard_size=%d, batch_size=%d",
            dataset_entry.dataset_path,
            len(dataset),
            shard_size,
            int(cfg.advantage.batch_size),
        )

    local_rows: list[dict[str, Any]] = []
    pbar = tqdm(
        loader,
        desc=f"[rank{rank}] {Path(dataset_entry.dataset_path).name}",
        disable=(rank != 0),
        total=len(loader),
    )
    for batch in pbar:
        observation = _move_to_device(batch["observation"], device)
        with torch.inference_mode():
            out = model.predict(observation)
        local_rows.extend(_records_from_predict(out, batch))

    local_df = pd.DataFrame(local_rows)
    if world_size > 1:
        dist.barrier()
        df = gather_all_advantages(local_df, rank, world_size)
    else:
        df = local_df

    if rank == 0:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        episode_lengths = [
            dataset._source.episode_length(ep)
            for ep in range(dataset._source.num_episodes())
        ]
        df, num_appended = _append_missing_terminal_rows(
            df,
            episode_lengths=episode_lengths,
            member_count=int(model.config.ensemble_size),
        )
        if num_appended > 0:
            logger.info(
                "Appended %d terminal frames with default-negative scores for %s",
                num_appended,
                dataset_entry.dataset_path,
            )
    return df


def _finalise_dataframe(
    df: pd.DataFrame,
    *,
    dataset_type: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply threshold, force-True for sft, and project to the final columns."""
    if df.empty:
        raise RuntimeError(
            "Empty DataFrame after gather — no predictions were produced for this dataset"
        )
    df = df.copy()
    df["advantage_continuous"] = df["p_progress_aggregated"]
    if dataset_type.lower() == "sft":
        df["advantage"] = True
    else:
        df["advantage"] = df["advantage_continuous"] > float(threshold)
    return df[
        [
            "episode_index",
            "frame_index",
            "advantage",
            "advantage_continuous",
            "p_progress_mean",
            "p_progress_min",
            "p_progress_variance",
            "member_values",
        ]
    ]


def _save_advantages_parquet(df: pd.DataFrame, dataset_path: str, tag: str) -> Path:
    meta_dir = Path(dataset_path) / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Dataset meta dir does not exist: {meta_dir}")
    out_path = meta_dir / f"advantages_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _update_mixture_config(
    *,
    dataset_path: str,
    tag: str,
    positive_threshold: float,
    inference_mode: str,
    ensemble_size: int,
    total_samples: int,
    num_positive: int,
) -> Path:
    """Merge a per-tag entry into ``meta/mixture_config.yaml``.

    Preserves any existing top-level keys and any other tags already
    recorded under ``tags:``.
    """
    meta_dir = Path(dataset_path) / "meta"
    cfg_path = meta_dir / "mixture_config.yaml"
    existing: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"mixture_config.yaml at {cfg_path} is not a mapping; refusing to overwrite"
                )
            existing = loaded
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(
            f"mixture_config.yaml 'tags' field at {cfg_path} is not a mapping"
        )
    tags[str(tag)] = {
        "positive_threshold": float(positive_threshold),
        "inference_mode": str(inference_mode),
        "ensemble_size": int(ensemble_size),
        "total_samples": int(total_samples),
        "num_positive": int(num_positive),
    }
    existing["tags"] = tags
    with open(cfg_path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return cfg_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_advantages_ensemble_fail150_k8_ensemble4_wco",
)
def main(cfg: DictConfig) -> None:
    rank, world_size, device = setup_distributed(cfg)

    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if rank == 0:
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    _validate_cfg(cfg)

    inference_mode = str(cfg.advantage.model.get("inference_mode", "wco"))
    precision = cfg.advantage.model.get("precision", None)
    threshold = float(cfg.advantage.positive_threshold)
    tag = str(cfg.advantage.tag)

    if rank == 0:
        logger.info(
            "Loading ensemble checkpoint from %s", cfg.advantage.value_checkpoint
        )
    raw_model = BinaryValueCriticModel.from_checkpoint(
        cfg.advantage.value_checkpoint,
        device=device,
        env_type=str(cfg.data.robot_type),
        model_type=str(cfg.data.model_type),
        inference_mode=inference_mode,
        precision=precision,
    )
    model = _coerce_inference_model(raw_model)

    if rank == 0:
        logger.info(
            "Ensemble loaded: ensemble_size=%d, inference_mode=%s, "
            "max_token_len=%d, include_state_in_prompt=%s",
            int(model.config.ensemble_size),
            str(model.config.inference_mode),
            int(getattr(model.config, "max_token_len", 200)),
            bool(getattr(model.config, "include_state_in_prompt", False)),
        )

    try:
        for ds_idx, entry in enumerate(cfg.data.train_data_paths):
            if rank == 0:
                logger.info(
                    "[%d/%d] Inference on %s (type=%s)",
                    ds_idx + 1,
                    len(cfg.data.train_data_paths),
                    entry.dataset_path,
                    entry.type,
                )
            df = _run_inference_for_dataset(
                model=model,
                dataset_entry=entry,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank != 0:
                if world_size > 1:
                    dist.barrier()
                continue

            final_df = _finalise_dataframe(
                df, dataset_type=str(entry.type), threshold=threshold
            )
            out_path = _save_advantages_parquet(final_df, entry.dataset_path, tag)
            num_positive = int(final_df["advantage"].sum())
            total_samples = int(len(final_df))
            mix_path = _update_mixture_config(
                dataset_path=entry.dataset_path,
                tag=tag,
                positive_threshold=threshold,
                inference_mode=str(model.config.inference_mode),
                ensemble_size=int(model.config.ensemble_size),
                total_samples=total_samples,
                num_positive=num_positive,
            )
            logger.info(
                "Wrote %s (rows=%d, positive=%d/%d, p_mean_avg=%.4f). Updated %s",
                out_path,
                total_samples,
                num_positive,
                total_samples,
                float(final_df["p_progress_mean"].mean()),
                mix_path,
            )

            if world_size > 1:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
