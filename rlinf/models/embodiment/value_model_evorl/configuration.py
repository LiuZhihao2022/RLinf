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

"""Pistar06 evorl value model configuration.

Standalone PretrainedConfig — does NOT inherit from anything in
``rlinf.models.embodiment.value_model.configuration``.
"""

from typing import Optional

from transformers import PretrainedConfig


class Pistar06EvorlConfig(PretrainedConfig):
    """Configuration for the Pistar06ValueCriticModel (evorl variant).

    The Pistar06 stack uses a SigLIP vision encoder
    (``google/siglip-so400m-patch14-384``, native 384x384) and a Gemma3-270m
    language backbone, fused via concat + 2-layer MLP. This config holds the
    fields needed by ``Pistar06Model`` and ``Pistar06ValueCriticModel``.
    """

    model_type = "pistar06_evorl"

    def __init__(
        self,
        # Backbones
        vision_repo_id: str = "",
        language_repo_id: str = "",
        vision_revision: Optional[str] = None,
        language_revision: Optional[str] = None,
        # Fusion + value head
        fusion_hidden_dim: int = 512,
        num_bins: int = 201,
        bin_min: float = -1.0,
        bin_max: float = 0.0,
        dropout: float = 0.1,
        # Runtime
        dtype: str = "bfloat16",
        precision: Optional[str] = None,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = False,
        use_gradient_checkpointing: bool = False,
        # Interface compat fields (consumed by ValueDataset / Pistar06ValueProcessor)
        action_dim: int = 32,
        action_horizon: int = 50,
        max_token_len: int = 200,
        # State-aware prompt (aligns with external Evo-RL reference).
        include_state_in_prompt: bool = True,
        max_state_dim: int = 32,
        state_discretization_bins: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_repo_id = vision_repo_id
        self.language_repo_id = language_repo_id
        self.vision_revision = vision_revision
        self.language_revision = language_revision

        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.dropout = dropout

        self.dtype = precision if precision is not None else dtype
        self.precision = self.dtype
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_language_model = freeze_language_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.max_token_len = max_token_len

        self.include_state_in_prompt = include_state_in_prompt
        self.max_state_dim = max_state_dim
        self.state_discretization_bins = state_discretization_bins

        self._validate()

    def _validate(self) -> None:
        if not self.vision_repo_id:
            raise ValueError(
                "Pistar06EvorlConfig.vision_repo_id must be a non-empty path or "
                "HF repo id"
            )
        if not self.language_repo_id:
            raise ValueError(
                "Pistar06EvorlConfig.language_repo_id must be a non-empty path or "
                "HF repo id"
            )
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be > 0")
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if self.bin_min >= self.bin_max:
            raise ValueError(
                f"bin_min must be < bin_max, got {self.bin_min} and {self.bin_max}"
            )
        if self.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError(
                f"dtype must be one of bfloat16/float32/float16, got {self.dtype}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.max_token_len <= 0:
            raise ValueError("max_token_len must be > 0")
        if self.max_state_dim <= 0:
            raise ValueError("max_state_dim must be > 0")
        if self.state_discretization_bins < 2:
            raise ValueError("state_discretization_bins must be >= 2")
