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

"""Advantage-weighted flow-matching loss helpers for OpenPI SFT."""

import torch


def compute_advantage_weighted_loss(
    per_sample_loss: torch.Tensor,
    advantage_weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ``sum(loss_i * weight_i) / sum(weight_i)``."""
    weights = advantage_weight.to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    ).reshape_as(per_sample_loss)
    weight_sum = weights.sum()
    weighted_sum = (per_sample_loss * weights).sum()
    return weighted_sum / weight_sum.clamp_min(eps)
