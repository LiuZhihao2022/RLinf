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

"""Pistar06 evorl value model factory.

Provides ``get_model(cfg)`` matching the signature of
``value_model.get_value_model`` so the router in ``rlinf/models/__init__.py``
can dispatch to either variant uniformly. The processor-agnostic utilities
(``checkpoint_utils``, ``data_collator``) are reused from ``value_model/``;
only the model/config live here, since those are what genuinely differ
between the two variants.
"""

import glob
import logging
import os

import safetensors.torch
import torch
from omegaconf import DictConfig

from .configuration import Pistar06EvorlConfig
from .modeling_critic import CriticOutput, Pistar06ValueCriticModel

logger = logging.getLogger(__name__)


def _strip_model_prefix(state_dict: dict, model) -> dict:
    """Strip ``model.`` prefix from checkpoint keys if needed.

    Standalone copy of the same helper in value_model/__init__.py — kept here
    so the new module has zero dependency on its sibling.
    """
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
    """Load a state dict from .safetensors / .pt / .pth file or directory."""
    if path.endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    elif path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=False)
    elif os.path.isdir(path):
        weight_paths = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not weight_paths:
            weight_paths = sorted(glob.glob(os.path.join(path, "*.pt")))
        sd = {}
        for wp in weight_paths:
            if wp.endswith(".safetensors"):
                sd.update(safetensors.torch.load_file(wp, device="cpu"))
            else:
                sd.update(torch.load(wp, map_location="cpu", weights_only=False))
        return sd
    return {}


def get_model(cfg: DictConfig, torch_dtype=None) -> Pistar06ValueCriticModel:
    """Build a ``Pistar06ValueCriticModel``.

    Signature matches ``rlinf.models.embodiment.value_model.get_value_model``
    so the router in ``rlinf/models/__init__.py`` can dispatch via
    ``from rlinf.models.embodiment.value_model_evorl import get_model``.

    Args:
        cfg: Hydra model config. Expected keys:
            - vision_repo_id, language_repo_id (local paths or HF repo ids)
            - num_bins, v_min, v_max, fusion_hidden_dim, dropout
            - precision (one of bf16/fp32/fp16)
            - freeze_vision_encoder, freeze_language_model
            - use_gradient_checkpointing
            - action_dim, action_horizon, max_token_len (interface compat)
            - model_path: optional fine-tuned checkpoint path
        torch_dtype: unused, kept for interface parity with get_value_model.

    Returns:
        ``Pistar06ValueCriticModel`` instance.
    """
    precision = getattr(cfg, "precision", "bf16")
    if precision in ("bf16", "bf16-mixed"):
        dtype = "bfloat16"
    elif precision in ("fp32", "32", "32-true"):
        dtype = "float32"
    elif precision in ("fp16", "16", "16-mixed"):
        dtype = "float16"
    else:
        dtype = "bfloat16"

    config = Pistar06EvorlConfig(
        vision_repo_id=getattr(cfg, "vision_repo_id", "") or "",
        language_repo_id=getattr(cfg, "language_repo_id", "") or "",
        fusion_hidden_dim=getattr(cfg, "fusion_hidden_dim", 512),
        num_bins=getattr(cfg, "num_bins", 201),
        bin_min=getattr(cfg, "v_min", -1.0),
        bin_max=getattr(cfg, "v_max", 0.0),
        dropout=getattr(cfg, "dropout", 0.1),
        dtype=dtype,
        freeze_vision_encoder=getattr(cfg, "freeze_vision_encoder", False),
        freeze_language_model=getattr(cfg, "freeze_language_model", False),
        use_gradient_checkpointing=getattr(cfg, "use_gradient_checkpointing", False),
        action_dim=getattr(cfg, "action_dim", 32),
        action_horizon=getattr(cfg, "action_horizon", 50),
        max_token_len=getattr(cfg, "max_token_len", 200),
        include_state_in_prompt=getattr(cfg, "include_state_in_prompt", True),
        max_state_dim=getattr(cfg, "max_state_dim", 32),
        state_discretization_bins=getattr(cfg, "state_discretization_bins", 256),
    )

    model = Pistar06ValueCriticModel(config)
    logger.info("Created Pistar06ValueCriticModel (evorl)")

    # Optional: resume from a fine-tuned checkpoint.
    model_path = getattr(cfg, "model_path", None)
    if model_path is not None:
        full_weights_path = os.path.join(
            model_path, "model_state_dict", "full_weights.pt"
        )
        actor_full_weights_path = os.path.join(
            model_path, "actor", "model_state_dict", "full_weights.pt"
        )
        if os.path.exists(full_weights_path):
            model_path = full_weights_path
        elif os.path.exists(actor_full_weights_path):
            model_path = actor_full_weights_path

    if model_path and os.path.exists(model_path):
        state_dict = _load_state_dict(model_path)
        if state_dict:
            state_dict = _strip_model_prefix(state_dict, model)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded fine-tuned checkpoint from %s (missing=%d, unexpected=%d)",
                model_path,
                len(missing),
                len(unexpected),
            )
    else:
        logger.info("No model_path provided; using from_pretrained() backbone weights.")

    return model


__all__ = [
    "get_model",
    "Pistar06EvorlConfig",
    "Pistar06ValueCriticModel",
    "CriticOutput",
]
