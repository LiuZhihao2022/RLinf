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

"""Frame-level success/fail classifier factory.

Exposes ``get_model(cfg, torch_dtype=None)`` with the same signature as
:func:`rlinf.models.embodiment.value_model_rewind_arm.get_model` so the
router in ``rlinf/models/__init__.py`` can dispatch on ``model_type``
uniformly.

The classifier's config is rehydrated via
``defaults ← checkpoint config.json ← Hydra cfg``, matching the
:class:`BinaryValueConfig` plumbing in ``value_model_rewind_arm/__init__.py``
so offline tooling that touches one model's config path touches the other
the same way.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Optional, Union

import safetensors.torch
import torch
from omegaconf import DictConfig

from .configuration import SuccessFailClassifierConfig
from .data_collator import FrameClassifierDataCollator
from .modeling_classifier import CriticOutput, SuccessFailClassifier
from .processing import SuccessFailClassifierImageProcessor

logger = logging.getLogger(__name__)


_SUCCESS_FAIL_CONFIG_DEFAULTS: dict[str, Any] = {
    "vision_repo_id": "",
    "vision_revision": None,
    "image_size": 224,
    "use_proprio": True,
    "proprio_dim": 32,
    "proprio_hidden": 128,
    "hidden_dim": 256,
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "freeze_vision": False,
    "use_gradient_checkpointing": False,
    "max_state_dim": 32,
}

assert all(
    not isinstance(v, (dict, DictConfig))
    for v in _SUCCESS_FAIL_CONFIG_DEFAULTS.values()
), (
    "_SUCCESS_FAIL_CONFIG_DEFAULTS must stay flat — build_success_fail_classifier_config "
    "merges only top-level fields."
)


def _is_override(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def _candidate_checkpoint_dirs(
    model_path: Union[str, os.PathLike, None],
) -> list[str]:
    """Return candidate directories that may hold a ``config.json``.

    Accepts a step dir, an ``actor/`` subdir, a model_state_dict subdir, or
    a direct file path (e.g. ``.../full_weights.pt``). Walks up to two
    parents and probes for ``actor/`` subdirs. Identical behaviour to the
    same-named helper in ``value_model_rewind_arm``.
    """
    if model_path is None:
        return []
    resolved = os.path.abspath(os.fspath(model_path))
    start = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)

    candidates: list[str] = []
    current = start
    for _ in range(3):
        if not current or not os.path.isdir(current):
            break
        if current not in candidates:
            candidates.append(current)
        actor_subdir = os.path.join(current, "actor")
        if os.path.isdir(actor_subdir) and actor_subdir not in candidates:
            candidates.append(actor_subdir)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return candidates


def load_success_fail_classifier_checkpoint_config(
    model_path: Union[str, os.PathLike, None],
) -> Optional[SuccessFailClassifierConfig]:
    """Load ``SuccessFailClassifierConfig`` saved alongside a checkpoint."""
    for checkpoint_dir in _candidate_checkpoint_dirs(model_path):
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            continue
        try:
            config = SuccessFailClassifierConfig.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load SuccessFailClassifierConfig from %s: %s",
                checkpoint_dir,
                exc,
            )
            continue
        logger.info(
            "Loaded SuccessFailClassifierConfig metadata from %s",
            checkpoint_dir,
        )
        return config
    return None


def _resolve_precision_dtype(precision) -> str:
    if precision in ("bf16", "bf16-mixed", "bfloat16"):
        return "bfloat16"
    if precision in ("fp32", "32", "32-true", "float32"):
        return "float32"
    if precision in ("fp16", "16", "16-mixed", "float16"):
        return "float16"
    # Default to fp32 for classifier (small model, fp32 master is safer
    # under FSDP mixed precision — mirrors _ensure_binary_value_precision_cfg).
    return "float32"


def build_success_fail_classifier_config(
    cfg: DictConfig,
    checkpoint_config: Optional[SuccessFailClassifierConfig] = None,
) -> SuccessFailClassifierConfig:
    """Merge defaults ← checkpoint config ← Hydra cfg.

    ``None`` / empty-string values on either source are treated as
    "not provided" so partially-specified Hydra configs do not erase
    valid checkpoint metadata. ``cfg`` wins on ties.
    """
    if checkpoint_config is None:
        checkpoint_config = load_success_fail_classifier_checkpoint_config(
            getattr(cfg, "model_path", None)
        )

    merged: dict[str, Any] = dict(_SUCCESS_FAIL_CONFIG_DEFAULTS)
    for source_label, source in (
        ("checkpoint_config", checkpoint_config),
        ("cfg", cfg),
    ):
        if source is None:
            continue
        for field in _SUCCESS_FAIL_CONFIG_DEFAULTS:
            value = getattr(source, field, None)
            if not _is_override(value):
                continue
            if isinstance(value, (dict, DictConfig)):
                raise TypeError(
                    f"{source_label}.{field} is {type(value).__name__}; "
                    "build_success_fail_classifier_config merges non-recursively. "
                    "Pass the flat model sub-config (e.g. cfg.actor.model)."
                )
            merged[field] = value

    precision = getattr(cfg, "precision", None)
    if precision is None and checkpoint_config is not None:
        precision = getattr(checkpoint_config, "precision", None) or getattr(
            checkpoint_config, "dtype", None
        )
    merged["dtype"] = _resolve_precision_dtype(precision)

    return SuccessFailClassifierConfig(**merged)


def save_success_fail_classifier_checkpoint_assets(
    save_path: str,
    cfg: Union[DictConfig, SuccessFailClassifierConfig],
    processor: Optional[SuccessFailClassifierImageProcessor] = None,
) -> None:
    """Persist config + image processor next to checkpoint weights."""
    os.makedirs(save_path, exist_ok=True)
    model_config = (
        cfg
        if isinstance(cfg, SuccessFailClassifierConfig)
        else build_success_fail_classifier_config(cfg)
    )
    model_config.to_json_file(
        os.path.join(save_path, "config.json"), use_diff=False
    )
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(save_path)
        logger.info(
            "Saved SuccessFailClassifier processor assets to %s", save_path
        )
    logger.info(
        "Saved SuccessFailClassifier checkpoint metadata to %s", save_path
    )


def _strip_model_prefix(state_dict: dict, model: torch.nn.Module) -> dict:
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if len(model_keys & ckpt_keys) == 0:
        stripped = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        if len(set(stripped.keys()) & model_keys) > 0:
            logger.info("Stripped 'model.' prefix from checkpoint keys")
            return stripped
    return state_dict


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    if path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=False)
    if os.path.isdir(path):
        weight_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_paths:
            weight_paths = sorted(glob.glob(os.path.join(path, "*.pt")))
        sd: dict = {}
        for wp in weight_paths:
            if wp.endswith(".safetensors"):
                sd.update(safetensors.torch.load_file(wp, device="cpu"))
            else:
                sd.update(torch.load(wp, map_location="cpu", weights_only=False))
        return sd
    return {}


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    if model_path is None:
        return None
    candidates = (
        os.path.join(model_path, "model_state_dict", "full_weights.pt"),
        os.path.join(model_path, "actor", "model_state_dict", "full_weights.pt"),
    )
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return model_path


def _load_checkpoint_into_model(
    model: torch.nn.Module, state_dict: dict, model_path: str
) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded fine-tuned checkpoint from %s (missing=%d, unexpected=%d)",
        model_path,
        len(missing),
        len(unexpected),
    )


def get_model(
    cfg: DictConfig,
    torch_dtype=None,
) -> SuccessFailClassifier:
    """Build a :class:`SuccessFailClassifier` from a Hydra model config.

    Signature matches
    :func:`rlinf.models.embodiment.value_model_rewind_arm.get_model` so the
    central router in ``rlinf/models/__init__.py`` can dispatch via
    ``from rlinf.models.embodiment.success_fail_classifier import get_model``.

    Expected ``cfg`` keys (missing fields fall back to
    ``_SUCCESS_FAIL_CONFIG_DEFAULTS``): ``vision_repo_id``, ``use_proprio``,
    ``proprio_dim``, ``hidden_dim``, ``dropout``, ``label_smoothing``,
    ``freeze_vision``, ``precision`` / ``dtype``, ``image_size``,
    ``max_state_dim``, ``use_gradient_checkpointing``. Optional
    ``model_path`` points at a saved checkpoint; weights are loaded when
    present.
    """
    del torch_dtype  # accepted for interface parity

    checkpoint_config = load_success_fail_classifier_checkpoint_config(
        getattr(cfg, "model_path", None)
    )
    config = build_success_fail_classifier_config(
        cfg, checkpoint_config=checkpoint_config
    )

    model_path = _resolve_model_path(getattr(cfg, "model_path", None))
    state_dict: dict = {}
    if model_path and os.path.exists(model_path):
        state_dict = _load_state_dict(model_path)

    model = SuccessFailClassifier(config)
    logger.info("Created SuccessFailClassifier (DINOv2-based)")

    if state_dict:
        model_state_dict = _strip_model_prefix(state_dict, model)
        _load_checkpoint_into_model(model, model_state_dict, model_path)
    else:
        logger.info(
            "No model_path provided; using from_pretrained() backbone weights "
            "and freshly initialised proprio/head."
        )
    return model


__all__ = [
    "CriticOutput",
    "FrameClassifierDataCollator",
    "SuccessFailClassifier",
    "SuccessFailClassifierConfig",
    "SuccessFailClassifierImageProcessor",
    "_SUCCESS_FAIL_CONFIG_DEFAULTS",
    "_candidate_checkpoint_dirs",
    "build_success_fail_classifier_config",
    "get_model",
    "load_success_fail_classifier_checkpoint_config",
    "save_success_fail_classifier_checkpoint_assets",
]
