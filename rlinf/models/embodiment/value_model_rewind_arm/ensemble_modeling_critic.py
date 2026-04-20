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

"""Ensemble wrapper for the ARM + ReWiND binary value critic.

This module owns every ensemble-only concept in the stack:

* :class:`EnsembleCriticOutput` — extends :class:`CriticOutput` with
  member-wise and aggregate prediction statistics that only make sense
  under a deep ensemble.
* :class:`EnsembleBinaryValueCriticModel` — the
  :class:`BinaryValueCriticModel` members are the ones that actually
  produce logits; this class is purely responsible for cloning /
  re-seeding members, per-member training orchestration (via
  ``forward(..., member_idx=int)``), and aggregating member predictions
  into a single inference output under the configured
  ``inference_mode``.

The single-model path in :mod:`modeling_critic` intentionally knows
nothing about ensembles — no ``member_predicted_values`` fake-stats
padding, no ``hasattr(model, "members")`` duck typing. That separation
is the whole reason this module exists.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .configuration import BinaryValueConfig
from .modeling_critic import BinaryValueCriticModel, CriticOutput
from .modeling_rewind_arm import CLASS_PROGRESS, CLASS_REGRESS

logger = logging.getLogger(__name__)


@dataclass
class EnsembleCriticOutput(CriticOutput):
    """CriticOutput extended with ensemble aggregate stats.

    All four fields are only populated by
    :meth:`EnsembleBinaryValueCriticModel.predict` (the inference path).
    The per-member training path returns a plain :class:`CriticOutput`
    with these fields absent — there's no cross-member aggregate to
    report when each member is trained on its own independent random
    micro-batch.
    """

    member_predicted_values: Optional[torch.FloatTensor] = None
    prediction_mean: Optional[torch.FloatTensor] = None
    prediction_min: Optional[torch.FloatTensor] = None
    prediction_variance: Optional[torch.FloatTensor] = None


def clone_ensemble_members(
    base_member: nn.Module,
    ensemble_size: int,
) -> list[nn.Module]:
    """Clone a base member ``ensemble_size`` times."""
    if ensemble_size < 1:
        raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")
    return [base_member] + [
        copy.deepcopy(base_member) for _ in range(ensemble_size - 1)
    ]


def _reinitialize_module_parameters(module: nn.Module, seed: int) -> None:
    """Reset all resettable submodules under ``module`` with a fixed seed."""
    cuda_devices = sorted(
        {
            int(parameter.device.index)
            for parameter in module.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        for submodule in module.modules():
            if hasattr(submodule, "reset_parameters"):
                submodule.reset_parameters()


def reinitialize_member_value_heads(
    members: list[nn.Module],
    head_seed_base: int,
) -> None:
    """Reinitialize each member's value head with a distinct seed."""
    for member_idx, member in enumerate(members):
        value_head = getattr(getattr(member, "model", None), "value_head", None)
        if value_head is None:
            raise AttributeError(
                f"Ensemble member {member_idx} does not expose model.value_head"
            )
        _reinitialize_module_parameters(value_head, int(head_seed_base) + member_idx)


def build_ensemble_members(
    base_member: nn.Module,
    ensemble_size: int,
    head_seed_base: int,
) -> list[nn.Module]:
    """Clone a base member and reinitialize only the value heads."""
    members = clone_ensemble_members(base_member, ensemble_size)
    reinitialize_member_value_heads(members, head_seed_base)
    return members


class EnsembleBinaryValueCriticModel(nn.Module):
    """Deep-ensemble wrapper for :class:`BinaryValueCriticModel`."""

    def __init__(
        self,
        config: BinaryValueConfig,
        members: list[BinaryValueCriticModel],
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError(
                "EnsembleBinaryValueCriticModel requires at least one member"
            )

        self.config = config
        self.members = nn.ModuleList(members)
        self.gradient_checkpointing_enabled = False

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return self.members[0]._no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return self.members[0]._no_split_names

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True
        for member in self.members:
            member.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        for member in self.members:
            member.gradient_checkpointing_disable()

    def attach_runtime_assets(self, processor, input_transform, device) -> None:
        """Attach inference-time runtime assets to self AND every member.

        Overrides :meth:`BinaryValueCriticModel.attach_runtime_assets`:
        the ensemble's own ``infer`` / ``infer_batch`` reads
        ``self._input_transform`` directly, while the per-member
        ``_prepare_observation*`` delegations in this class call into
        ``members[0]``. Both surfaces must see the same assets, so we
        push them down to every member rather than duck-typing from the
        outside.
        """
        self.processor = processor
        self._input_transform = input_transform
        self._device = device
        for member in self.members:
            member.attach_runtime_assets(processor, input_transform, device)

    @staticmethod
    def _binary_logits_from_progress_probability(progress_prob: Tensor) -> Tensor:
        """Build a 2-logit representation from a binary progress probability."""
        clamped_progress = progress_prob.float().clamp(min=1e-6, max=1.0 - 1e-6)
        progress_margin = torch.logit(clamped_progress)
        return torch.stack(
            (-0.5 * progress_margin, 0.5 * progress_margin),
            dim=-1,
        ).to(dtype=progress_prob.dtype)

    @staticmethod
    def _gather_member_batch_values(
        member_tensor: Tensor, member_indices: Tensor
    ) -> Tensor:
        """Gather one member prediction per batch item from ``[E, B, ...]`` tensors."""
        batch_indices = torch.arange(
            member_tensor.shape[1], device=member_tensor.device
        )
        return member_tensor[member_indices, batch_indices]

    def _aggregate_member_predictions(
        self,
        member_logits: Tensor,
        member_probs: Tensor,
        member_predicted_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        prediction_mean = member_predicted_values.mean(dim=0)
        prediction_min, worst_member_indices = member_predicted_values.min(dim=0)
        prediction_variance = member_predicted_values.var(dim=0, unbiased=False)

        if self.config.inference_mode == "mo":
            aggregated_probs = member_probs.mean(dim=0)
            aggregated = aggregated_probs[:, CLASS_PROGRESS]
            aggregated_logits = self._binary_logits_from_progress_probability(
                aggregated
            )
        elif self.config.inference_mode == "wco":
            aggregated_logits = self._gather_member_batch_values(
                member_logits,
                worst_member_indices,
            )
            aggregated_probs = self._gather_member_batch_values(
                member_probs,
                worst_member_indices,
            )
            aggregated = aggregated_probs[:, CLASS_PROGRESS]
        elif self.config.inference_mode == "uwo":
            member_margins = (
                member_logits[..., CLASS_PROGRESS] - member_logits[..., CLASS_REGRESS]
            )
            aggregated_margin = member_margins.mean(dim=0) - (
                self.config.uwo_lambda * member_margins.var(dim=0, unbiased=False)
            )
            aggregated_logits = torch.stack(
                (-0.5 * aggregated_margin, 0.5 * aggregated_margin),
                dim=-1,
            )
            aggregated_probs = torch.softmax(aggregated_logits, dim=-1)
            aggregated = aggregated_probs[:, CLASS_PROGRESS]
        else:
            raise ValueError(
                f"Unsupported inference_mode: {self.config.inference_mode}"
            )

        return (
            aggregated,
            aggregated_logits,
            aggregated_probs,
            prediction_mean,
            prediction_min,
            prediction_variance,
        )

    def forward(
        self,
        observation,
        labels=None,
        *,
        member_idx: Optional[int] = None,
        **kwargs,
    ) -> CriticOutput:
        """Training / inference dispatch.

        * ``labels is None`` → inference; delegates to :meth:`predict`,
          which returns a full :class:`EnsembleCriticOutput` with member
          and aggregate stats.
        * ``labels is not None`` → training; ``member_idx`` **must** be an
          int. The caller (the FSDP SFT worker) drives an outer loop
          over members and feeds each member its OWN fresh micro batch
          from the dataloader, so each member's training trajectory
          sees independent random data — the bagging-style randomness
          that makes ensemble prediction variance a meaningful epistemic
          uncertainty signal. There is intentionally no
          "split one batch across members" path: it wouldn't give
          independent random data (members would see disjoint slices of
          the same micro batch) and it isn't exercised by any caller in
          this repo.
        """
        if labels is None:
            return self.predict(observation)

        if member_idx is None:
            raise ValueError(
                "EnsembleBinaryValueCriticModel.forward requires member_idx "
                "to be an int during training. The ensemble worker must drive "
                "a per-member outer loop and feed each member its own micro "
                "batch; there is no parallel batch-slicing path."
            )

        member = self.members[int(member_idx)]
        member_output = member(observation=observation, labels=labels, **kwargs)
        if member_output.loss is None:
            raise RuntimeError(
                f"Ensemble member {int(member_idx)} returned no loss during training"
            )
        return member_output

    @torch.no_grad()
    def predict(self, observation) -> EnsembleCriticOutput:
        member_outputs = [member.predict(observation) for member in self.members]
        member_logits = torch.stack([output.logits for output in member_outputs], dim=0)
        member_probs = torch.stack([output.probs for output in member_outputs], dim=0)
        member_predicted_values = torch.stack(
            [output.predicted_values for output in member_outputs],
            dim=0,
        )
        (
            aggregated,
            aggregated_logits,
            aggregated_probs,
            prediction_mean,
            prediction_min,
            prediction_variance,
        ) = self._aggregate_member_predictions(
            member_logits,
            member_probs,
            member_predicted_values,
        )

        return EnsembleCriticOutput(
            predicted_values=aggregated,
            logits=aggregated_logits,
            probs=aggregated_probs,
            atoms=None,
            hidden_states=None,
            member_predicted_values=member_predicted_values,
            prediction_mean=prediction_mean,
            prediction_min=prediction_min,
            prediction_variance=prediction_variance,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        return self.predict(observation).predicted_values

    @staticmethod
    def _prepare_observation_cpu(inputs: dict, processor) -> dict:
        return BinaryValueCriticModel._prepare_observation_cpu(inputs, processor)

    def _prepare_observation(self, inputs: dict) -> dict:
        return self.members[0]._prepare_observation(inputs)

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        return self.members[0]._prepare_observation_batch(inputs_list)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        import numpy as np

        inputs = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in obs.items()
        }
        inputs = self._input_transform(inputs)
        observation = self._prepare_observation(inputs)
        result = self.predict(observation)

        return {
            "value": float(result.predicted_values[0].item()),
            "member_values": result.member_predicted_values[:, 0].tolist(),
            "value_mean": float(result.prediction_mean[0].item()),
            "value_min": float(result.prediction_min[0].item()),
            "value_variance": float(result.prediction_variance[0].item()),
            "state": obs.get("state", np.array([])),
        }

    @torch.no_grad()
    def infer_batch(
        self,
        obs_list: list[dict],
        *,
        batch_size: int = 64,
        pretransformed: bool = False,
        already_cpu_prepared: bool = False,
    ) -> list[dict]:
        import numpy as np

        if not obs_list:
            return []

        device = getattr(self, "_device", "cuda")
        all_outputs = []

        for batch_start in range(0, len(obs_list), batch_size):
            batch_end = min(batch_start + batch_size, len(obs_list))
            batch_obs = obs_list[batch_start:batch_end]

            if already_cpu_prepared:
                first = batch_obs[0]
                if isinstance(first.get("images"), dict):
                    batched_images = {
                        key: torch.cat(
                            [obs["images"][key] for obs in batch_obs], dim=0
                        ).to(device)
                        for key in first["images"]
                    }
                    batched_masks = {
                        key: torch.cat(
                            [obs["image_masks"][key] for obs in batch_obs], dim=0
                        ).to(device)
                        for key in first["image_masks"]
                    }
                else:
                    batched_images = torch.cat(
                        [obs["images"] for obs in batch_obs], dim=0
                    ).to(device)
                    batched_masks = torch.cat(
                        [obs["image_masks"] for obs in batch_obs], dim=0
                    ).to(device)

                observation = {
                    "images": batched_images,
                    "image_masks": batched_masks,
                    "tokenized_prompt": torch.cat(
                        [obs["tokenized_prompt"] for obs in batch_obs], dim=0
                    ).to(device),
                    "tokenized_prompt_mask": torch.cat(
                        [obs["tokenized_prompt_mask"] for obs in batch_obs], dim=0
                    ).to(device),
                }
            else:
                inputs_list = []
                for obs in batch_obs:
                    inputs = {
                        key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in obs.items()
                    }
                    if not pretransformed:
                        inputs = self._input_transform(inputs)
                    inputs_list.append(inputs)

                observation = self._prepare_observation_batch(inputs_list)

            result = self.predict(observation)
            values = result.predicted_values.cpu()
            member_values = result.member_predicted_values.cpu()
            value_mean = result.prediction_mean.cpu()
            value_min = result.prediction_min.cpu()
            value_variance = result.prediction_variance.cpu()

            for idx in range(len(batch_obs)):
                all_outputs.append(
                    {
                        "value": float(values[idx].item()),
                        "member_values": member_values[:, idx].tolist(),
                        "value_mean": float(value_mean[idx].item()),
                        "value_min": float(value_min[idx].item()),
                        "value_variance": float(value_variance[idx].item()),
                    }
                )

        return all_outputs

    @classmethod
    def from_checkpoint(cls, *args, **kwargs):
        from .modeling_critic import BinaryValueCriticModel

        return BinaryValueCriticModel.from_checkpoint(*args, **kwargs)


__all__ = [
    "EnsembleBinaryValueCriticModel",
    "EnsembleCriticOutput",
    "build_ensemble_members",
    "clone_ensemble_members",
    "reinitialize_member_value_heads",
]
