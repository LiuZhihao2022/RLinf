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

"""Frame-level success/fail classifier configuration.

Independent binary classifier built on top of a pretrained DINOv2-s visual
backbone plus an optional proprioceptive MLP. Used at Step 3
(compute_advantages) inference time to correct the original value with
``V_final = V_orig - lambda * logit``.

Training supervision is **episode-level** ``is_success`` broadcast to every
frame of the episode — not true frame-level labels. The classifier tolerates
this label noise via ``label_smoothing`` and relies on the statistical
assumption that fail episodes' typical failure poses also occur at the
transient bad frames inside success episodes.
"""

from typing import Optional

from transformers import PretrainedConfig


class SuccessFailClassifierConfig(PretrainedConfig):
    """Configuration for :class:`SuccessFailClassifier`.

    The vision encoder is a DINOv2-s (default ``facebook/dinov2-small``,
    output dim 384). An optional proprio MLP branch fuses robot state when
    ``use_proprio=True``. A small MLP head produces one raw logit per frame
    (``> 0`` means ``fail-like``, ``< 0`` means ``success-like``).
    """

    model_type = "success_fail_classifier"

    def __init__(
        self,
        # Vision backbone
        vision_repo_id: str = "",
        vision_revision: Optional[str] = None,
        image_size: int = 224,
        # Proprio branch
        use_proprio: bool = True,
        proprio_dim: int = 32,
        proprio_hidden: int = 128,
        # Head
        hidden_dim: int = 256,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        # Runtime
        dtype: str = "float32",
        precision: Optional[str] = None,
        freeze_vision: bool = False,
        use_gradient_checkpointing: bool = False,
        # openpi transform compat (mirrors value_evorl so the inference-time
        # build_input_transforms path works unchanged)
        max_state_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_repo_id = vision_repo_id
        self.vision_revision = vision_revision
        self.image_size = int(image_size)

        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim)
        self.proprio_hidden = int(proprio_hidden)

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.label_smoothing = float(label_smoothing)

        self.dtype = precision if precision is not None else dtype
        self.precision = self.dtype
        self.freeze_vision = bool(freeze_vision)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.max_state_dim = int(max_state_dim)

        self._validate()

    def _validate(self) -> None:
        if not self.vision_repo_id:
            raise ValueError(
                "SuccessFailClassifierConfig.vision_repo_id must be a non-empty "
                "path or HF repo id (e.g. 'facebook/dinov2-small')"
            )
        if self.image_size <= 0 or self.image_size % 14 != 0:
            raise ValueError(
                "image_size must be > 0 and a multiple of 14 (DINOv2 patch size); "
                f"got {self.image_size}"
            )
        if self.use_proprio and self.proprio_dim <= 0:
            raise ValueError(
                f"proprio_dim must be > 0 when use_proprio=True, got {self.proprio_dim}"
            )
        if self.proprio_hidden <= 0:
            raise ValueError(f"proprio_hidden must be > 0, got {self.proprio_hidden}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}"
            )
        if self.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError(
                f"dtype must be one of bfloat16/float32/float16, got {self.dtype}"
            )
        if self.max_state_dim <= 0:
            raise ValueError("max_state_dim must be > 0")

    def to_diff_dict(self) -> dict:
        """Return the full config dict.

        Mirrors :class:`BinaryValueConfig.to_diff_dict`: we cannot build an
        empty default config (``vision_repo_id`` is required), so the diff
        fallback in :meth:`PretrainedConfig.to_diff_dict` does not apply.
        """
        return self.to_dict()
