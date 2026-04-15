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

"""Pistar06 distributional value model.

Faithful port of the Pistar06Model class from
``/mnt/project_rlinf/liuzhihao/Evo-RL/src/lerobot/values/pistar06/modeling_pistar06.py``.
The lerobot-side wrappers (Pistar06Policy, training-target builders, processor
glue) are intentionally dropped — they are replaced by the RLinf-native adapter
in ``modeling_critic.py`` and the standalone processor in ``processing.py``.

Architecture:
    SigLIP vision encoder (default 384x384) + Gemma3-270m language backbone
    -> per-camera image projector -> per-token language projector
    -> mean-pool across cameras + language pool
    -> concat(image_pooled, language_pool) -> LayerNorm -> 2-layer MLP
    -> categorical value distribution over ``num_bins`` atoms.

The model expects images already in the [0, 1] float range (or uint8); it
applies its own SigLIP-style mean/std normalization internally.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

if TYPE_CHECKING:
    from .configuration import Pistar06EvorlConfig

try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
    )

    _transformers_available = True
except ImportError:
    AutoConfig = None
    AutoImageProcessor = None
    AutoModel = None
    AutoModelForCausalLM = None
    _transformers_available = False


# ---------------------------------------------------------------------------
# Categorical value-distribution helpers (faithful port)
# ---------------------------------------------------------------------------


def build_bin_centers(
    num_bins: int,
    bin_min: float,
    bin_max: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Evenly-spaced bin centers used as atoms of the value distribution."""
    return torch.linspace(
        bin_min, bin_max, num_bins, dtype=torch.float32, device=device
    )


def project_values_to_bins(
    values: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Project scalar value targets to a soft distribution over bin centers.

    Linear interpolation between adjacent bins. Values outside the bin range
    are clipped.
    """
    if values.ndim != 1:
        raise ValueError(f"'values' must be rank-1, got shape={tuple(values.shape)}.")
    if bin_centers.ndim != 1:
        raise ValueError(
            f"'bin_centers' must be rank-1, got shape={tuple(bin_centers.shape)}."
        )
    if bin_centers.shape[0] < 2:
        raise ValueError("At least 2 bins are required.")

    values = values.clamp(min=bin_centers[0], max=bin_centers[-1])
    step = bin_centers[1] - bin_centers[0]
    scaled = (values - bin_centers[0]) / step
    low = torch.floor(scaled).long()
    high = torch.clamp(low + 1, max=bin_centers.shape[0] - 1)
    high_weight = (scaled - low.float()).clamp(0.0, 1.0)
    low_weight = 1.0 - high_weight

    target = torch.zeros(
        values.shape[0], bin_centers.shape[0], device=values.device, dtype=torch.float32
    )
    target.scatter_add_(1, low.unsqueeze(1), low_weight.unsqueeze(1))
    target.scatter_add_(1, high.unsqueeze(1), high_weight.unsqueeze(1))
    return target


def expected_value_from_logits(
    logits: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Expected scalar value from categorical logits over bins."""
    probs = functional.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)


# ---------------------------------------------------------------------------
# Module-level helpers (ported verbatim from the source, lerobot-free)
# ---------------------------------------------------------------------------


def _resolve_load_dtype(dtype_name: str) -> torch.dtype:
    requested_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    if requested_dtype == torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return requested_dtype


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = True


def _extract_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            f"Cannot infer hidden size for model type {type(model)}: missing `.config`."
        )
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError(f"Cannot infer hidden size for model config type {type(config)}.")


def _extract_vision_feature_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            f"Cannot infer vision feature size for model type {type(model)}: "
            "missing `.config`."
        )
    if hasattr(config, "projection_dim"):
        return int(config.projection_dim)
    if hasattr(config, "vision_config") and hasattr(
        config.vision_config, "projection_dim"
    ):
        return int(config.vision_config.projection_dim)
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(
        config.vision_config, "hidden_size"
    ):
        return int(config.vision_config.hidden_size)
    raise ValueError(
        f"Cannot infer vision feature size for model config type {type(config)}."
    )


def _validate_loading_info(
    repo_id: str, model_label: str, loading_info: dict[str, list] | None
) -> None:
    if loading_info is None:
        return
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    if not missing and not unexpected and not mismatched:
        return
    raise RuntimeError(
        f"Pretrained weights for {model_label} from '{repo_id}' did not load "
        f"cleanly: missing={len(missing)} unexpected={len(unexpected)} "
        f"mismatched={len(mismatched)}. "
        "This usually indicates a model class/checkpoint mismatch."
    )


def _resolve_image_size(image_processor: Any) -> tuple[int, int]:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return 384, 384


def _resolve_norm_stats(
    image_processor: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean_raw = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    std_raw = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
    if len(mean_raw) != 3 or len(std_raw) != 3:
        raise ValueError(
            f"Expected RGB normalization stats of len=3, got mean={mean_raw} "
            f"std={std_raw}."
        )
    mean = (float(mean_raw[0]), float(mean_raw[1]), float(mean_raw[2]))
    std = (float(std_raw[0]), float(std_raw[1]), float(std_raw[2]))
    if any(v <= 0 for v in std):
        raise ValueError(f"Invalid image std values: {std}.")
    return mean, std


def _load_language_model(
    repo_id: str,
    revision: str | None,
    dtype: torch.dtype,
) -> nn.Module:
    if AutoConfig is None or AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError(
            "transformers is not installed. Please install the embodied stack."
        )

    model_config = AutoConfig.from_pretrained(repo_id, revision=revision)
    architectures = getattr(model_config, "architectures", None) or []
    prefer_causal_lm = any(
        isinstance(arch, str) and arch.endswith("ForCausalLM") for arch in architectures
    )

    if prefer_causal_lm:
        lm_with_head, loading_info = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=dtype,
            output_loading_info=True,
        )
        _validate_loading_info(repo_id, "language_model(causal_lm)", loading_info)
        if not hasattr(lm_with_head, "model"):
            raise RuntimeError(
                f"AutoModelForCausalLM loaded from '{repo_id}' does not expose "
                "`.model` text backbone."
            )
        return lm_with_head.model

    language_model, loading_info = AutoModel.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=dtype,
        output_loading_info=True,
    )
    _validate_loading_info(repo_id, "language_model(auto_model)", loading_info)
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"AutoModel loaded from '{repo_id}' returned unexpected type: "
            f"{type(language_model)}."
        )
    return language_model


# ---------------------------------------------------------------------------
# Pistar06Model (faithful port from the source)
# ---------------------------------------------------------------------------


class Pistar06Model(nn.Module):
    """SigLIP + Gemma3-270m -> concat + MLP -> categorical value head.

    Faithful port from the Evo-RL source. Accepts ``[0, 1]`` float (or uint8)
    images and applies SigLIP-style ``(x - mean) / std`` normalization
    internally, then resizes to the vision encoder's native resolution if
    needed and runs the multimodal forward.
    """

    def __init__(self, cfg: "Pistar06EvorlConfig"):
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError(
                "transformers is not installed. Please install the embodied stack."
            )

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            torch_dtype=self.model_dtype,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            use_fast=True,
        )
        image_height, image_width = _resolve_image_size(image_processor)
        image_mean, image_std = _resolve_norm_stats(image_processor)
        self.image_resolution = (image_height, image_width)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        vision_feature_size = _extract_vision_feature_size(self.vision_encoder)
        language_hidden_size = _extract_hidden_size(self.language_model)

        self.image_projector = nn.Sequential(
            nn.Linear(vision_feature_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(language_hidden_size, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_norm = nn.LayerNorm(cfg.fusion_hidden_dim * 2)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_dim * 2, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_bins),
        )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)

        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)

    # -----------------------------------------------------------------------
    # Forward pieces
    # -----------------------------------------------------------------------

    def _encode_images(self, flat_images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)

        vision_outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        if (
            hasattr(vision_outputs, "pooler_output")
            and vision_outputs.pooler_output is not None
        ):
            return vision_outputs.pooler_output
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError(
            "Unsupported vision encoder output. Expected pooler_output or "
            "last_hidden_state."
        )

    def _encode_language(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError(
                "Language model output does not contain `last_hidden_state`."
            )

        token_mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        return (hidden * token_mask).sum(dim=1) / denom

    def _preprocess_images(
        self, images: Tensor, image_attention_mask: Tensor
    ) -> Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"'images' must have shape [B,N,C,H,W], got {tuple(images.shape)}."
            )
        if image_attention_mask.ndim != 2:
            raise ValueError(
                "'image_attention_mask' must have shape [B,N], got "
                f"{tuple(image_attention_mask.shape)}."
            )

        bsize, num_cameras = images.shape[:2]
        if (
            image_attention_mask.shape[0] != bsize
            or image_attention_mask.shape[1] != num_cameras
        ):
            raise ValueError(
                "Batch shape mismatch between images and image_attention_mask."
            )

        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32) / 255.0
        else:
            images = images.to(dtype=torch.float32)
            if bool(torch.max(images) > 1.0) or bool(torch.min(images) < 0.0):
                images = (images / 255.0).clamp(0.0, 1.0)

        flat_images = images.reshape(bsize * num_cameras, *images.shape[2:])
        if flat_images.shape[-2:] != self.image_resolution:
            flat_images = functional.interpolate(
                flat_images,
                size=self.image_resolution,
                mode="bilinear",
                align_corners=False,
            )

        mean = self.image_mean.to(
            device=flat_images.device, dtype=flat_images.dtype
        ).view(1, 3, 1, 1)
        std = self.image_std.to(
            device=flat_images.device, dtype=flat_images.dtype
        ).view(1, 3, 1, 1)
        flat_images = (flat_images - mean) / std
        flat_images = flat_images.reshape(
            bsize,
            num_cameras,
            flat_images.shape[1],
            flat_images.shape[2],
            flat_images.shape[3],
        )

        camera_mask = image_attention_mask.to(
            device=flat_images.device, dtype=flat_images.dtype
        ).view(bsize, num_cameras, 1, 1, 1)
        flat_images = flat_images * camera_mask
        return flat_images

    def _compute_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        """Compute the LayerNorm'd joint multimodal feature.

        This is the analog of ``ValueCriticModel``'s ``cls_hidden`` — the
        feature vector that gets fed into the value head. Splitting it out
        from ``forward`` lets ``Pistar06ValueCriticModel`` expose it via
        ``CriticOutput.hidden_states`` from both the training-side ``forward``
        and the inference-side ``predict`` paths.

        Returns:
            Tensor of shape ``[B, fusion_hidden_dim * 2]``.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"'input_ids' must have shape [B, T], got {tuple(input_ids.shape)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "'attention_mask' must have shape [B, T], got "
                f"{tuple(attention_mask.shape)}."
            )
        if images.ndim != 5:
            raise ValueError(
                f"'images' must have shape [B, N, C, H, W], got {tuple(images.shape)}."
            )
        if image_attention_mask.ndim != 2:
            raise ValueError(
                "'image_attention_mask' must have shape [B, N], got "
                f"{tuple(image_attention_mask.shape)}."
            )

        bsize = input_ids.shape[0]
        if attention_mask.shape[0] != bsize:
            raise ValueError(
                "Batch size mismatch between input_ids and attention_mask."
            )
        if images.shape[0] != bsize or image_attention_mask.shape[0] != bsize:
            raise ValueError("Batch size mismatch between language and image inputs.")
        if images.shape[1] == 0:
            raise ValueError("At least one camera is required for Pistar06Model.")

        image_attention_mask = image_attention_mask.to(
            dtype=torch.bool, device=images.device
        )
        if not torch.all(image_attention_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid camera input.")
        language_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if not torch.all(language_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token.")

        processed_images = self._preprocess_images(images, image_attention_mask)
        num_cameras = processed_images.shape[1]
        flat_images = processed_images.reshape(
            bsize * num_cameras, *processed_images.shape[2:]
        )
        flat_images = flat_images.to(dtype=self.model_dtype)
        image_context = (
            torch.no_grad() if self.cfg.freeze_vision_encoder else nullcontext()
        )
        with image_context:
            image_features = self._encode_images(flat_images)

        language_context = (
            torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        )
        with language_context:
            language_features = self._encode_language(
                input_ids=input_ids, attention_mask=language_mask.long()
            )

        feature_dtype = torch.float32
        image_features = image_features.to(dtype=feature_dtype)
        language_features = language_features.to(dtype=feature_dtype)
        image_tokens = self.image_projector(image_features).view(bsize, num_cameras, -1)
        camera_token_mask = image_attention_mask.unsqueeze(-1).to(
            dtype=image_tokens.dtype
        )
        image_tokens = image_tokens * camera_token_mask

        camera_denominator = (
            image_attention_mask.sum(dim=1, keepdim=True)
            .to(dtype=image_tokens.dtype)
            .clamp_min(1.0)
        )
        image_pooled = image_tokens.sum(dim=1) / camera_denominator
        language_token = self.language_projector(language_features)

        joint_features = torch.cat([image_pooled, language_token], dim=-1)
        return self.final_norm(joint_features)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        """Compute value-distribution logits.

        Thin wrapper over ``_compute_features`` + ``value_head`` so callers
        that want the intermediate hidden state can call ``_compute_features``
        directly without re-running the encoders.
        """
        features = self._compute_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )
        return self.value_head(features)
