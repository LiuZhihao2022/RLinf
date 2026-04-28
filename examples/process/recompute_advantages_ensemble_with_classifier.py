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

"""Recompute ensemble ``advantage_continuous`` / ``advantage`` with new
classifier-correction parameters — pure CPU, no GPU inference.

Reads an existing ``meta/advantages_{source_tag}.parquet`` that already
carries ``ensemble_signed_score`` and ``p_fail`` columns (populated by
``compute_advantages_ensemble.py`` with classifier=on, or by
``inject_classifier_into_advantages.py``), applies the hinge formula

    advantage_continuous = ensemble_signed_score
                          - λ · max(p_fail − fail_threshold, 0)

and writes a new ``meta/advantages_{new_tag}.parquet`` plus a merged
``tags[new_tag]`` entry into ``meta/mixture_config.yaml``.

Typical usage::

    python recompute_advantages_ensemble_with_classifier.py \\
        --dataset_paths /p1 /p2 \\
        --source_tag  libero_task0_..._wco_step7000_clf1k \\
        --new_tag     libero_task0_..._wco_step7000_clf1k_lam2_t03 \\
        --classifier_lambda 2.0 \\
        --classifier_fail_threshold 0.3

Defaults for ``positive_threshold`` / ``dataset_type`` / ``classifier_checkpoint``
are read from the source tag's metadata; CLI flags override. Each of these
**must** be resolvable — otherwise the script raises so downstream parquets
never carry semantically-wrong metadata.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# First-non-none helper (replicated inline — kept simple, avoids a fragile
# cross-script import from compute_advantages_ensemble.py).
# ---------------------------------------------------------------------------


def _first_non_none(*values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v == "":
            continue
        return v
    return None


# ---------------------------------------------------------------------------
# Mixture config I/O
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
    """Merge ``tags[tag] = new_entry`` into ``meta/mixture_config.yaml``.

    Top-level keys are treated as read-only: existing ``advantage_tag`` /
    ``datasets`` / etc. are preserved; other tags under ``tags:`` are
    preserved too. Only ``tags[tag]`` is overwritten.
    """
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
# Source validation
# ---------------------------------------------------------------------------


def _validate_member_values_column(df: pd.DataFrame, source_meta: dict) -> int:
    """Enforce ensemble-schema gate; return the inferred ``ensemble_size``.

    Four checks (see plan Part D):
        1. column exists
        2. per-row length is uniform
        3. length > 1 (a real ensemble, not a degenerate single-model dump)
        4. if ``source_meta`` carries ``ensemble_size``, it matches the length
    """
    if "member_values" not in df.columns:
        raise ValueError(
            "Source parquet lacks 'member_values' column — this is required "
            "to confirm the source is an ensemble-produced parquet. Refusing "
            "to operate on it."
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


def _load_source_parquet(
    dataset_path: Path, source_tag: str
) -> tuple[pd.DataFrame, Path]:
    meta_dir = dataset_path / "meta"
    source_path = meta_dir / f"advantages_{source_tag}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source parquet not found: {source_path}. Did you pass the "
            "correct --source_tag?"
        )
    return pd.read_parquet(source_path), source_path


def _require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns {missing}; present columns: "
            f"{sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Core recompute
# ---------------------------------------------------------------------------


_OUTPUT_COLS = [
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


def _recompute_advantages(
    df: pd.DataFrame,
    *,
    dataset_type: str,
    positive_threshold: float,
    classifier_lambda: float,
    classifier_fail_threshold: float,
) -> pd.DataFrame:
    """Apply the hinge + threshold on an existing DataFrame."""
    out = df.copy()
    hinge = (out["p_fail"] - float(classifier_fail_threshold)).clip(lower=0.0)
    out["advantage_continuous"] = (
        out["ensemble_signed_score"] - float(classifier_lambda) * hinge
    )
    if dataset_type.lower() == "sft":
        out["advantage"] = True
    else:
        out["advantage"] = out["advantage_continuous"] > float(positive_threshold)
    # Keep only canonical column set, but tolerate missing optional ones
    # (e.g. legacy parquet without entropy columns).
    present = [c for c in _OUTPUT_COLS if c in out.columns]
    return out[present]


def process_one_dataset(
    *,
    dataset_path: Path,
    source_tag: str,
    new_tag: str,
    classifier_lambda: float,
    classifier_fail_threshold: float,
    positive_threshold_cli: Optional[float],
    dataset_type_cli: Optional[str],
    classifier_checkpoint_cli: Optional[str],
) -> tuple[Path, Path]:
    """Recompute one dataset's ``advantages_{new_tag}.parquet`` + metadata.

    Returns ``(parquet_path, mixture_config_path)``.
    """
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    mix = _read_mixture_config(dataset_path)
    tags = mix.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(
            f"{_mixture_config_path(dataset_path)} has non-mapping 'tags'"
        )
    source_meta = tags.get(source_tag) or {}
    if not isinstance(source_meta, dict):
        raise RuntimeError(
            f"mixture_config tag {source_tag!r} is not a mapping"
        )

    df, src_path = _load_source_parquet(dataset_path, source_tag)
    _require_columns(
        df,
        ["episode_index", "frame_index", "ensemble_signed_score", "p_fail"],
        context=f"{src_path}",
    )
    # Fail-loud: duplicated keys would silently halve rows under recompute.
    if df.duplicated(subset=["episode_index", "frame_index"]).any():
        raise ValueError(
            f"{src_path} has duplicated (episode_index, frame_index) rows"
        )
    _validate_member_values_column(df, source_meta)

    dataset_type = _first_non_none(dataset_type_cli, source_meta.get("dataset_type"))
    if dataset_type is None:
        raise ValueError(
            f"dataset_type unresolved for tag {source_tag!r}: not in metadata "
            "and --dataset_types not provided"
        )
    dataset_type = str(dataset_type)
    if dataset_type not in ("sft", "rollout"):
        raise ValueError(
            f"dataset_type must be 'sft' or 'rollout', got {dataset_type!r}"
        )

    positive_threshold = _first_non_none(
        positive_threshold_cli, source_meta.get("positive_threshold")
    )
    if positive_threshold is None:
        raise ValueError(
            f"positive_threshold unresolved for tag {source_tag!r}: not in "
            "metadata and --positive_threshold not provided"
        )
    positive_threshold = float(positive_threshold)

    classifier_checkpoint = _first_non_none(
        classifier_checkpoint_cli, source_meta.get("classifier_checkpoint")
    )
    if classifier_checkpoint is None:
        raise ValueError(
            f"classifier_checkpoint unresolved for tag {source_tag!r}: not in "
            "metadata and --classifier_checkpoint not provided"
        )

    new_df = _recompute_advantages(
        df,
        dataset_type=dataset_type,
        positive_threshold=positive_threshold,
        classifier_lambda=classifier_lambda,
        classifier_fail_threshold=classifier_fail_threshold,
    )

    # Write parquet
    out_path = dataset_path / "meta" / f"advantages_{new_tag}.parquet"
    new_df.to_parquet(out_path, index=False)

    # Build new tag metadata — start from source, override classifier params
    new_entry = dict(source_meta)
    new_entry.update({
        "positive_threshold": positive_threshold,
        "total_samples": int(len(new_df)),
        "num_positive": int(new_df["advantage"].sum()),
        "dataset_type": dataset_type,
        "has_classifier_correction": True,
        "classifier_lambda": float(classifier_lambda),
        "classifier_fail_threshold": float(classifier_fail_threshold),
        "classifier_model_type": source_meta.get(
            "classifier_model_type", "success_fail_classifier"
        ),
        "classifier_aggregation": source_meta.get(
            "classifier_aggregation", "sigmoid_of_single_logit"
        ),
        "classifier_checkpoint": str(classifier_checkpoint),
        "derived_from_tag": source_tag,
    })
    mix_path = _write_mixture_config_tag(dataset_path, new_tag, new_entry)

    logger.info(
        "  %s: rows=%d, positive=%d/%d, advantage_continuous range=[%.4f, %.4f]",
        dataset_path.name,
        len(new_df),
        int(new_df["advantage"].sum()),
        len(new_df),
        float(new_df["advantage_continuous"].min()),
        float(new_df["advantage_continuous"].max()),
    )
    return out_path, mix_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute advantage_continuous / advantage using new classifier "
            "correction parameters. Pure CPU; reads ensemble_signed_score and "
            "p_fail from an existing advantages parquet."
        )
    )
    parser.add_argument(
        "--dataset_paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more LeRobot dataset roots (each containing a meta/ dir).",
    )
    parser.add_argument(
        "--source_tag",
        required=True,
        help="Tag of the existing advantages parquet to read as the baseline.",
    )
    parser.add_argument(
        "--new_tag",
        required=True,
        help="Output tag; written to meta/advantages_{new_tag}.parquet.",
    )
    parser.add_argument(
        "--classifier_lambda",
        type=float,
        required=True,
        help="λ weight on the classifier hinge penalty.",
    )
    parser.add_argument(
        "--classifier_fail_threshold",
        type=float,
        required=True,
        help="p_fail hinge knee in [0, 1].",
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=None,
        help=(
            "Signed-score threshold in [-1, 1]. If omitted, read from the "
            "source tag's mixture_config metadata."
        ),
    )
    parser.add_argument(
        "--dataset_types",
        nargs="+",
        default=None,
        choices=("sft", "rollout"),
        help=(
            "One entry per --dataset_paths, overriding mixture_config. If "
            "omitted, dataset_type is read from tag metadata."
        ),
    )
    parser.add_argument(
        "--classifier_checkpoint",
        default=None,
        help=(
            "Override the classifier_checkpoint field written into the new "
            "tag's metadata. Required when the source tag does not carry it."
        ),
    )
    args = parser.parse_args(argv)

    if args.new_tag == args.source_tag:
        parser.error("--new_tag must differ from --source_tag")
    if args.dataset_types is not None and len(args.dataset_types) != len(
        args.dataset_paths
    ):
        parser.error(
            f"--dataset_types length {len(args.dataset_types)} does not match "
            f"--dataset_paths length {len(args.dataset_paths)}"
        )
    if args.classifier_lambda < 0.0:
        parser.error(f"--classifier_lambda must be >= 0 (got {args.classifier_lambda})")
    if not (0.0 <= args.classifier_fail_threshold <= 1.0):
        parser.error(
            "--classifier_fail_threshold must be in [0, 1] "
            f"(got {args.classifier_fail_threshold})"
        )
    if args.positive_threshold is not None and not (
        -1.0 <= args.positive_threshold <= 1.0
    ):
        parser.error(
            "--positive_threshold must be in [-1, 1] "
            f"(got {args.positive_threshold})"
        )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    logger.info(
        "Recompute: source_tag=%s → new_tag=%s, λ=%s, fail_threshold=%s",
        args.source_tag,
        args.new_tag,
        args.classifier_lambda,
        args.classifier_fail_threshold,
    )
    for i, ds_path in enumerate(args.dataset_paths):
        ds_type_cli = (
            args.dataset_types[i] if args.dataset_types is not None else None
        )
        out_path, mix_path = process_one_dataset(
            dataset_path=ds_path,
            source_tag=args.source_tag,
            new_tag=args.new_tag,
            classifier_lambda=args.classifier_lambda,
            classifier_fail_threshold=args.classifier_fail_threshold,
            positive_threshold_cli=args.positive_threshold,
            dataset_type_cli=ds_type_cli,
            classifier_checkpoint_cli=args.classifier_checkpoint,
        )
        logger.info("  wrote %s; updated %s", out_path, mix_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
