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

"""Inject classifier ``p_fail`` / ``logit_fail`` into an existing ensemble
advantages parquet, optionally applying classifier correction immediately.

Separate from ``recompute_advantages_ensemble_with_classifier.py`` so the
CPU-only threshold sweep stays cheap once ``p_fail`` is injected. Typical
workflow for a legacy parquet without classifier columns::

    # Step 1 (GPU): inject p_fail / logit_fail, upgrade legacy schema
    python inject_classifier_into_advantages.py \\
        --dataset_paths /p1 --dataset_types rollout \\
        --classifier_checkpoint /path/to/actor \\
        --source_tag old --new_tag old_clf1k \\
        --camera_keys image wrist_image \\
        --model_type pi05 --robot_type libero

    # Step 2 (CPU): sweep λ / fail_threshold on top of the injected parquet
    python recompute_advantages_ensemble_with_classifier.py \\
        --dataset_paths /p1 --source_tag old_clf1k --new_tag old_clf1k_lam1_t04 \\
        --classifier_lambda 1.0 --classifier_fail_threshold 0.4

Guarantees (all fail-loud):
    * Source parquet must carry ``member_values`` with uniform length > 1 and
      (if metadata has ``ensemble_size``) a matching count.
    * A ``has_classifier_correction=True`` source is rejected by default
      (use ``--override_corrected_source`` if intentional).
    * Legacy parquet missing ``ensemble_signed_score`` is upgraded from
      ``advantage_continuous`` only when that column stays in ``[-1, 1]``.
    * Per-frame merge is ``validate='one_to_one'``; any missing row raises.
    * If ``--classifier_lambda`` / ``--classifier_fail_threshold`` are provided,
      classifier correction is applied using the same formula as
      ``recompute_advantages_ensemble_with_classifier.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _silence_libav_logs() -> None:
    """Suppress libdav1d / libav stderr chatter (LeRobot video backend)."""
    import atexit
    import shutil
    import subprocess
    import sys as _sys

    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    if not shutil.which("grep"):
        return
    saved_stderr_fd = os.dup(2)
    try:
        grep = subprocess.Popen(
            ["grep", "-v", "-E", "--line-buffered", r"libdav1d|libdav1d 0\.9"],
            stdin=subprocess.PIPE,
            stdout=saved_stderr_fd,
        )
    except Exception:
        os.close(saved_stderr_fd)
        return
    _sys.stderr.flush()
    os.dup2(grep.stdin.fileno(), 2)

    def _restore() -> None:
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

    atexit.register(_restore)


_silence_libav_logs()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from rlinf.data.datasets.cfg.success_fail_dataset import (  # noqa: E402
    FrameClassifierDataset,
)
from rlinf.models.embodiment.success_fail_classifier import (  # noqa: E402
    FrameClassifierDataCollator,
    SuccessFailClassifier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (inlined instead of importing from compute_advantages_ensemble.py to
# avoid the fragile cross-script import that would pull in its side-effect-
# heavy top-level module state).
# ---------------------------------------------------------------------------


def _first_non_none(*values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        return v
    return None


def _none_if_empty_like(value: Optional[str]) -> Optional[str]:
    """Treat common CLI sentinels as an omitted optional string."""
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"none", "null"}:
        return None
    return stripped


def _setup_distributed(timeout_seconds: int = 3600) -> tuple[int, int, str]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", timeout=timedelta(seconds=timeout_seconds)
            )
        torch.cuda.set_device(local_rank)
        return rank, world_size, f"cuda:{local_rank}"
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _get_shard_indices(total: int, rank: int, world_size: int) -> tuple[int, int]:
    base = total // world_size
    rem = total % world_size
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (rank - rem) * base
        end = start + base
    return start, end


def _gather_dfs(local_df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
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


def _move_to_device(obj: Any, device: str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# Mixture-config I/O (shared pattern with recompute_advantages_ensemble_…)
# ---------------------------------------------------------------------------


def _mixture_config_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "mixture_config.yaml"


def _read_mixture_config(dataset_path: Path) -> dict[str, Any]:
    p = _mixture_config_path(dataset_path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"mixture_config.yaml at {p} is not a mapping; refusing to read"
        )
    return loaded


def _write_mixture_config_tag(
    dataset_path: Path, tag: str, new_entry: dict[str, Any]
) -> Path:
    p = _mixture_config_path(dataset_path)
    existing = _read_mixture_config(dataset_path)
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(f"mixture_config.yaml at {p} has non-mapping 'tags'")
    tags[str(tag)] = new_entry
    existing["tags"] = tags
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# Source validation + schema upgrade
# ---------------------------------------------------------------------------


def _validate_member_values_column(df: pd.DataFrame, source_meta: dict) -> int:
    """Same four-step ensemble gate used in recompute (Plan Part D/E)."""
    if "member_values" not in df.columns:
        raise ValueError(
            "Source parquet lacks 'member_values' column — this is required "
            "to confirm the source is an ensemble-produced parquet. Refusing "
            "to inject classifier scores."
        )
    lengths = df["member_values"].apply(len)
    if lengths.nunique() != 1:
        raise ValueError(
            f"Source parquet has non-uniform member_values lengths "
            f"({sorted(set(lengths.tolist()))[:5]}...); ensemble schema broken"
        )
    length = int(lengths.iloc[0])
    if length <= 1:
        raise ValueError(
            f"member_values length is {length}; ensemble must have >1 member"
        )
    meta_size = source_meta.get("ensemble_size")
    if meta_size is not None and int(meta_size) != length:
        raise ValueError(
            f"source tag ensemble_size={meta_size} does not match parquet "
            f"member_values length={length}"
        )
    return length


def _upgrade_legacy_schema(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Fill ``ensemble_signed_score`` from ``advantage_continuous`` for
    pre-rename parquets, but only when the column stays in ``[-1, 1]``.

    Out-of-range values are a strong hint the parquet is already
    classifier-corrected (and the metadata flag was lost); refuse to upgrade.
    """
    if "ensemble_signed_score" in df.columns:
        return df
    if "advantage_continuous" not in df.columns:
        raise ValueError(
            f"{source_path} lacks both 'ensemble_signed_score' and "
            "'advantage_continuous' — cannot reconstruct raw ensemble signal"
        )
    ac = df["advantage_continuous"].astype(float)
    eps = 1e-3
    lo, hi = float(ac.min()), float(ac.max())
    if lo < -1.0 - eps or hi > 1.0 + eps:
        raise ValueError(
            f"{source_path} advantage_continuous has range [{lo:.4f}, "
            f"{hi:.4f}] outside [-1, 1]; refusing to upgrade to "
            "ensemble_signed_score. Parquet looks already classifier-corrected "
            "— check mixture_config tag metadata."
        )
    df = df.copy()
    df["ensemble_signed_score"] = ac
    logger.info(
        "Legacy schema upgrade: ensemble_signed_score ← advantage_continuous for %s",
        source_path,
    )
    return df


# ---------------------------------------------------------------------------
# Classifier loading + per-dataset kwargs (first-non-none everywhere)
# ---------------------------------------------------------------------------


def _load_classifier(args: argparse.Namespace, device: str) -> SuccessFailClassifier:
    classifier = SuccessFailClassifier.from_checkpoint(
        args.classifier_checkpoint,
        device=device,
        env_type=str(args.robot_type),
        model_type=str(args.model_type),
        default_prompt=args.default_prompt,
        image_size=args.image_size,
        precision=args.precision,
        max_state_dim=args.max_state_dim,
        action_dim=args.action_dim,
    )
    camera_keys = tuple(args.camera_keys)
    if tuple(classifier.processor.image_keys) != camera_keys:
        logger.info(
            "Overriding classifier.processor.image_keys %s → %s",
            classifier.processor.image_keys,
            camera_keys,
        )
        classifier.processor.image_keys = camera_keys
    return classifier


def _classifier_dataset(
    *, dataset_path: str, dataset_type: str, args: argparse.Namespace, classifier
) -> FrameClassifierDataset:
    return FrameClassifierDataset(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        camera_keys=tuple(args.camera_keys),
        include_state=bool(getattr(classifier.config, "use_proprio", False)),
        state_key=str(args.state_key),
        max_state_dim=int(args.max_state_dim),
        robot_type=str(args.robot_type),
        model_type=str(args.model_type),
        action_dim=int(args.action_dim),
        default_prompt=args.default_prompt,
        norm_stats_dir=_none_if_empty_like(args.norm_stats_dir),
        asset_id=_none_if_empty_like(args.asset_id),
        include_success=True,
        include_fail=True,
        min_episode_length=int(args.min_episode_length),
        inference_mode=True,
    )


def _run_classifier_inference(
    *,
    classifier,
    ds: FrameClassifierDataset,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: str,
) -> pd.DataFrame:
    """Shard + forward the classifier over every frame of ``ds``.

    Returns a DataFrame on rank 0 (empty on other ranks) with columns
    ``[episode_index, frame_index, p_fail, logit_fail]``.
    """
    total = len(ds)
    if total == 0:
        raise RuntimeError(f"FrameClassifierDataset empty for {ds.source_name!r}")
    start, end = _get_shard_indices(total, rank, world_size)
    shard = Subset(ds, list(range(start, end)))

    loader = DataLoader(
        shard,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        shuffle=False,
        collate_fn=FrameClassifierDataCollator(
            processor=classifier.processor,
            use_proprio=bool(getattr(classifier.config, "use_proprio", False)),
        ),
    )

    if rank == 0:
        logger.info(
            "Classifier inference: total_frames=%d, rank0 shard=%d, batch_size=%d",
            total, end - start, int(args.batch_size),
        )

    rows: list[dict[str, Any]] = []
    pbar = tqdm(
        loader,
        desc=f"[rank{rank}] classifier {Path(ds.source_name).name}",
        disable=(rank != 0),
        total=len(loader),
    )
    for batch in pbar:
        observation = _move_to_device(batch["observation"], device)
        with torch.inference_mode():
            out = classifier.predict(observation)
        logits = out.logits.squeeze(-1).detach().to("cpu", dtype=torch.float32)
        probs = torch.sigmoid(logits)
        episodes = batch["episode"].tolist()
        frame_idx = batch["frame_index"].tolist()
        for i in range(int(logits.shape[0])):
            rows.append({
                "episode_index": int(episodes[i]),
                "frame_index": int(frame_idx[i]),
                "p_fail": float(probs[i].item()),
                "logit_fail": float(logits[i].item()),
            })
    local_df = pd.DataFrame(rows)

    if world_size > 1:
        dist.barrier()
    df = _gather_dfs(local_df, rank, world_size)
    if rank == 0 and len(df) > 0:
        dup = df.duplicated(subset=["episode_index", "frame_index"], keep=False)
        if dup.any():
            raise RuntimeError(
                f"classifier produced {int(dup.sum())} duplicate "
                "(episode, frame) keys after gather — shard logic is broken"
            )
    return df


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------


_INJECT_OUTPUT_PREFERRED_COLS = [
    "episode_index",
    "frame_index",
    "advantage",
    "advantage_continuous",
    "ensemble_signed_score",
    "p_progress_mean",
    "p_progress_min",
    "p_progress_variance",
    "member_values",
    "expected_stride_normalized",
    "entropy_aggregated",
    "entropy_member_mean",
    "entropy_member_variance",
    "p_fail",
    "logit_fail",
]


def process_one_dataset(
    *,
    dataset_path: Path,
    dataset_type: str,
    classifier,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: str,
) -> Optional[tuple[Path, Path]]:
    """Inject ``p_fail`` / ``logit_fail`` into one dataset's advantages parquet.

    All ranks run classifier inference; only rank 0 writes files. Returns
    ``None`` on non-zero ranks, ``(parquet_path, mixture_config_path)`` on rank 0.
    """
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    source_path = dataset_path / "meta" / f"advantages_{args.source_tag}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source parquet not found: {source_path}. Pass the correct --source_tag."
        )

    # --- Source metadata / schema checks (fail-loud) ---
    mix = _read_mixture_config(dataset_path)
    tags = mix.get("tags") or {}
    source_meta = tags.get(args.source_tag, {}) if isinstance(tags, dict) else {}
    if not isinstance(source_meta, dict):
        raise RuntimeError(
            f"mixture_config tag {args.source_tag!r} is not a mapping"
        )

    if source_meta.get("has_classifier_correction") and not args.override_corrected_source:
        raise RuntimeError(
            f"source tag {args.source_tag!r} is already classifier-corrected "
            "(has_classifier_correction=True in metadata). Pass "
            "--override_corrected_source to re-inject intentionally."
        )

    df = pd.read_parquet(source_path)
    _validate_member_values_column(df, source_meta)

    # Duplicate-key sanity (before any merge).
    if df.duplicated(subset=["episode_index", "frame_index"]).any():
        raise ValueError(
            f"{source_path} has duplicated (episode_index, frame_index) rows"
        )

    if "p_fail" in df.columns and not args.overwrite_existing_p_fail:
        raise ValueError(
            f"{source_path} already has a 'p_fail' column. Pass "
            "--overwrite_existing_p_fail to replace it."
        )

    df = _upgrade_legacy_schema(df, source_path)

    # --- Classifier inference (all ranks participate) ---
    ds = _classifier_dataset(
        dataset_path=str(dataset_path),
        dataset_type=dataset_type,
        args=args,
        classifier=classifier,
    )
    clf_df = _run_classifier_inference(
        classifier=classifier, ds=ds, args=args, rank=rank, world_size=world_size,
        device=device,
    )

    if rank != 0:
        return None

    if "p_fail" in df.columns:
        df = df.drop(columns=["p_fail", "logit_fail"], errors="ignore")

    merged = df.merge(
        clf_df[["episode_index", "frame_index", "p_fail", "logit_fail"]],
        on=["episode_index", "frame_index"],
        how="left",
        validate="one_to_one",
    )
    missing = merged[["p_fail", "logit_fail"]].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(
            f"classifier missed {int(missing)} rows for {dataset_path} after "
            "left-join; shard coverage or key alignment broken"
        )

    # --- Metadata: copy + enforce required fields + record classifier provenance ---
    cli_overrides: dict[str, Any] = {
        "dataset_type": dataset_type,
        "positive_threshold": args.positive_threshold,
        "inference_mode": args.inference_mode,
        "num_bins": args.num_bins,
    }
    new_entry: dict[str, Any] = dict(source_meta)
    for key, cli_val in cli_overrides.items():
        resolved = _first_non_none(cli_val, new_entry.get(key))
        if resolved is None:
            raise ValueError(
                f"source tag {args.source_tag!r} missing {key!r} and "
                f"--{key} not provided on the CLI"
            )
        new_entry[key] = resolved

    positive_threshold = float(new_entry["positive_threshold"])
    apply_classifier_correction = args.classifier_lambda is not None
    if apply_classifier_correction:
        hinge = (merged["p_fail"] - float(args.classifier_fail_threshold)).clip(
            lower=0.0
        )
        merged["advantage_continuous"] = (
            merged["ensemble_signed_score"] - float(args.classifier_lambda) * hinge
        )
        if str(dataset_type).lower() == "sft":
            merged["advantage"] = True
        else:
            merged["advantage"] = merged["advantage_continuous"] > positive_threshold

    # Column ordering: keep all existing columns but surface classifier pair at the end.
    preferred = [c for c in _INJECT_OUTPUT_PREFERRED_COLS if c in merged.columns]
    extras = [c for c in merged.columns if c not in preferred]
    merged = merged[preferred + extras]

    out_path = dataset_path / "meta" / f"advantages_{args.new_tag}.parquet"
    merged.to_parquet(out_path, index=False)

    # ensemble_size: recover from member_values if the source lacked it.
    new_entry["ensemble_size"] = int(
        _first_non_none(
            args.ensemble_size,
            new_entry.get("ensemble_size"),
            int(len(merged.iloc[0]["member_values"])),
        )
    )
    new_entry["total_samples"] = int(len(merged))
    new_entry["num_positive"] = int(merged["advantage"].sum())
    new_entry.update({
        "has_classifier_correction": bool(apply_classifier_correction),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "classifier_model_type": "success_fail_classifier",
        "classifier_aggregation": "sigmoid_of_single_logit",
        "derived_from_tag": args.source_tag,
    })
    if apply_classifier_correction:
        new_entry.update({
            "classifier_lambda": float(args.classifier_lambda),
            "classifier_fail_threshold": float(args.classifier_fail_threshold),
        })

    mix_path = _write_mixture_config_tag(dataset_path, args.new_tag, new_entry)
    logger.info(
        "Injected %s: rows=%d, corrected=%s, p_fail stats mean=%.3f std=%.3f "
        "min=%.3f max=%.3f",
        dataset_path.name,
        len(merged),
        apply_classifier_correction,
        float(merged["p_fail"].mean()),
        float(merged["p_fail"].std()),
        float(merged["p_fail"].min()),
        float(merged["p_fail"].max()),
    )
    return out_path, mix_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inject classifier p_fail / logit_fail into an existing ensemble "
            "advantages parquet. Optionally applies classifier correction when "
            "--classifier_lambda and --classifier_fail_threshold are provided."
        )
    )
    p.add_argument("--dataset_paths", type=Path, nargs="+", required=True)
    p.add_argument(
        "--dataset_types",
        nargs="+",
        choices=("sft", "rollout"),
        required=True,
        help="One per --dataset_paths. Determines advantage bool label semantics.",
    )
    p.add_argument("--classifier_checkpoint", required=True)
    p.add_argument("--source_tag", required=True)
    p.add_argument("--new_tag", required=True)

    # Data-side config (mirrors compute_advantages_ensemble.yaml data.*).
    p.add_argument("--camera_keys", nargs="+", default=["image", "wrist_image"])
    p.add_argument("--state_key", default="state")
    p.add_argument("--model_type", default="pi05")
    p.add_argument("--robot_type", default="libero")
    p.add_argument(
        "--norm_stats_dir",
        default=os.environ.get("LIBERO_NORM_STATS_DIR"),
        help=(
            "Norm stats directory for the openpi input transform. Defaults to "
            "$LIBERO_NORM_STATS_DIR."
        ),
    )
    p.add_argument("--asset_id", default=None)
    p.add_argument("--default_prompt", default=None)
    p.add_argument("--max_state_dim", type=int, default=32)
    p.add_argument("--action_dim", type=int, default=32)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--min_episode_length", type=int, default=1)

    # Metadata carry-over knobs (required only when the source tag is missing them).
    p.add_argument("--positive_threshold", type=float, default=None)
    p.add_argument("--inference_mode", default=None)
    p.add_argument("--num_bins", type=int, default=None)
    p.add_argument("--ensemble_size", type=int, default=None)
    p.add_argument(
        "--classifier_lambda",
        type=float,
        default=None,
        help=(
            "Optional lambda weight for immediate classifier correction. Must "
            "be provided together with --classifier_fail_threshold."
        ),
    )
    p.add_argument(
        "--classifier_fail_threshold",
        type=float,
        default=None,
        help=(
            "Optional p_fail hinge knee in [0, 1] for immediate classifier "
            "correction. Must be provided together with --classifier_lambda."
        ),
    )

    # Runtime knobs.
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--device", default="cuda")

    # Narrow override switches (see plan risk table).
    p.add_argument(
        "--overwrite_existing_p_fail",
        action="store_true",
        help="Allow overwriting an existing p_fail column in the source parquet.",
    )
    p.add_argument(
        "--override_corrected_source",
        action="store_true",
        help=(
            "Allow re-injecting over a source tag whose "
            "has_classifier_correction=True. Use with caution; normally you "
            "should inject onto the uncorrected parent tag instead."
        ),
    )

    args = p.parse_args(argv)
    args.norm_stats_dir = _none_if_empty_like(args.norm_stats_dir)
    args.asset_id = _none_if_empty_like(args.asset_id)

    if args.new_tag == args.source_tag:
        p.error("--new_tag must differ from --source_tag")
    if len(args.dataset_types) != len(args.dataset_paths):
        p.error(
            f"--dataset_types length {len(args.dataset_types)} does not match "
            f"--dataset_paths length {len(args.dataset_paths)}"
        )
    if args.batch_size <= 0:
        p.error(f"--batch_size must be > 0 (got {args.batch_size})")
    if args.num_workers < 0:
        p.error(f"--num_workers must be >= 0 (got {args.num_workers})")
    if args.prefetch_factor < 1:
        p.error(f"--prefetch_factor must be >= 1 (got {args.prefetch_factor})")
    if args.positive_threshold is not None and not (
        -1.0 <= args.positive_threshold <= 1.0
    ):
        p.error(
            "--positive_threshold must be in [-1, 1] "
            f"(got {args.positive_threshold})"
        )
    if (args.classifier_lambda is None) != (
        args.classifier_fail_threshold is None
    ):
        p.error(
            "--classifier_lambda and --classifier_fail_threshold must be "
            "provided together"
        )
    if args.classifier_lambda is not None and args.classifier_lambda < 0.0:
        p.error(
            f"--classifier_lambda must be >= 0 (got {args.classifier_lambda})"
        )
    if args.classifier_fail_threshold is not None and not (
        0.0 <= args.classifier_fail_threshold <= 1.0
    ):
        p.error(
            "--classifier_fail_threshold must be in [0, 1] "
            f"(got {args.classifier_fail_threshold})"
        )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    rank, world_size, device = _setup_distributed()
    try:
        classifier = _load_classifier(args, device)
        if rank == 0:
            logger.info(
                "Classifier loaded: use_proprio=%s, max_state_dim=%s",
                getattr(classifier.config, "use_proprio", None),
                getattr(classifier.config, "max_state_dim", None),
            )
        for ds_path, ds_type in zip(args.dataset_paths, args.dataset_types):
            if rank == 0:
                logger.info("Injecting classifier scores into %s (%s)", ds_path, ds_type)
            result = process_one_dataset(
                dataset_path=ds_path,
                dataset_type=ds_type,
                classifier=classifier,
                args=args,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if rank == 0 and result is not None:
                out_path, mix_path = result
                logger.info("  wrote %s; updated %s", out_path, mix_path)
            if world_size > 1:
                dist.barrier()
        if rank == 0:
            logger.info("Done.")
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
