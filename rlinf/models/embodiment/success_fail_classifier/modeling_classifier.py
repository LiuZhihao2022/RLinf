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

"""Frame-level success/fail classifier — RLinf-facing entry point.

Independent binary classifier consumed at Step 3 (compute_advantages). The
observation contract is a stripped-down analogue of
:class:`BinaryValueCriticModel.forward`:

    * ``images: dict[cam_name, Tensor[B, 3, H, W]]`` — single-frame (no
      pair axis), ImageNet-normalized by
      :class:`SuccessFailClassifierImageProcessor`.
    * ``image_masks: dict[cam_name, Tensor[B] bool]`` — whether each sample
      actually carried this camera view.
    * ``state: Tensor[B, proprio_dim]`` — optional; present iff
      ``config.use_proprio``.

Forward returns a :class:`CriticOutput` whose field layout is duck-type
compatible with :class:`BinaryValueCriticModel.CriticOutput` (same names,
same dtypes) so any downstream worker that treats the result polymorphically
keeps working. Key differences from the binary rewind_arm variant:

    * Output ``logits`` is ``[B, 1]`` (one raw logit per sample);
      ``probs`` and ``predicted_values`` are both ``sigmoid(logit)`` of
      shape ``[B]``. ``probs > 0.5`` ↔ model predicts fail.
    * Labels are ``float32`` in ``{0, 1}`` (BCE target) rather than long
      bin indices (CE target). Label smoothing is applied inline:
      ``y_smooth = y * (1 - 2ε) + ε``.
    * ``pos_weight`` may be passed at forward time to rebalance the
      positive (= fail) class; driven by
      :class:`FSDPFrameClassifierSftWorker` via the ``pos_weight_mode``
      config knob.

Inference surface:

    * :meth:`predict` — no-grad single-batch forward (no loss).
    * :meth:`predict_logit_batch` — new API that returns the raw logit
      array ``np.ndarray[N]`` for the Phase 1.5 advantage-correction path
      in ``compute_advantages.py``.
    * :meth:`infer` / :meth:`infer_batch` — parallel to
      :meth:`BinaryValueCriticModel.infer_batch`; returns ``{"value":
      float}`` where ``value = sigmoid(logit) ∈ [0, 1]``. Kept for parity
      with generic value-model tooling.
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .configuration import SuccessFailClassifierConfig

logger = logging.getLogger(__name__)


def _load_image_processor_from_checkpoint(
    checkpoint_dir: str, default_image_size: int
):
    """Find the saved ``preprocessor_config.json`` across candidate dirs.

    The processor carries ``image_keys``, which is dataset-specific (e.g.
    ``["image", "wrist_image"]`` for libero vs the default
    ``["base_0_rgb", ...]``). If training used non-default keys and we
    fall back to the defaults, :meth:`SuccessFailClassifierImageProcessor.
    process_images` silently zero-fills every camera with ``mask=False``
    and the model sees a black image. So: walk the same candidate list
    used for config/weights (:func:`_candidate_checkpoint_dirs`) before
    giving up — accept step dir, actor subdir, inner subdir, or a file
    path.
    """
    from . import _candidate_checkpoint_dirs
    from .processing import SuccessFailClassifierImageProcessor

    for candidate in _candidate_checkpoint_dirs(checkpoint_dir):
        preproc_path = pathlib.Path(candidate) / "preprocessor_config.json"
        if not preproc_path.exists():
            continue
        try:
            processor = SuccessFailClassifierImageProcessor.from_pretrained(
                candidate,
                local_files_only=True,
            )
            logger.info(
                "  Loaded success_fail_classifier image processor from %s",
                candidate,
            )
            return processor
        except (OSError, ValueError) as exc:
            logger.warning(
                "  Failed to load processor from %s: %s", candidate, exc
            )
    logger.info(
        "  No image processor config found under any candidate directory; "
        "falling back to defaults (image_size=%d, "
        "image_keys=DEFAULT_IMAGE_KEYS). If training used custom camera "
        "keys this will silently zero the visual input — double-check that "
        "the checkpoint dir contains preprocessor_config.json.",
        default_image_size,
    )
    return SuccessFailClassifierImageProcessor(image_size=default_image_size)


@dataclass
class CriticOutput(ModelOutput):
    """Output dataclass for the frame-level success/fail classifier.

    Field names match :class:`rlinf.models.embodiment.value_model_rewind_arm.\
modeling_critic.CriticOutput` so worker code stays duck-type compatible.
    Classifier-specific shapes:

        * ``logits`` — ``[B, 1]`` raw logit (``> 0`` = fail-like).
        * ``probs`` — ``[B]`` softmax / sigmoid probability of ``fail``.
        * ``predicted_values`` — ``[B]`` sigmoid probability (= probs).
          Kept as the "headline" scalar for downstream tooling.
        * ``atoms`` — ``None`` (retained for slot parity with value_model).
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    expert_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    cat_acc_best: Optional[torch.FloatTensor] = None
    cat_acc_neighbor: Optional[torch.FloatTensor] = None
    mae: Optional[torch.FloatTensor] = None


class SuccessFailClassifier(nn.Module):
    """DINOv2-s visual backbone + optional proprio MLP + binary MLP head."""

    def __init__(self, config: SuccessFailClassifierConfig):
        super().__init__()
        self.config = config
        self.use_proprio = bool(config.use_proprio)
        self.label_smoothing = float(config.label_smoothing)
        self.gradient_checkpointing_enabled = False

        self._build_visual()
        self._build_proprio()
        self._build_head()

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def _build_visual(self) -> None:
        """Instantiate the DINOv2 visual backbone via HuggingFace transformers."""
        from transformers import AutoModel

        self.visual = AutoModel.from_pretrained(
            self.config.vision_repo_id,
            revision=self.config.vision_revision,
        )
        self._visual_feat_dim = int(self.visual.config.hidden_size)

        if self.config.freeze_vision:
            for p in self.visual.parameters():
                p.requires_grad = False
            self.visual.eval()
            logger.info(
                "SuccessFailClassifier: vision encoder frozen (%d params)",
                sum(p.numel() for p in self.visual.parameters()),
            )

    def _build_proprio(self) -> None:
        if not self.use_proprio:
            self.proprio_enc = None
            return
        d_in = int(self.config.proprio_dim)
        d_hidden = int(self.config.proprio_hidden)
        self.proprio_enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def _build_head(self) -> None:
        d_in = self._visual_feat_dim
        if self.use_proprio:
            d_in += int(self.config.proprio_hidden)
        d_hidden = int(self.config.hidden_dim)
        p = float(self.config.dropout)
        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(d_hidden, 1),
        )

    @property
    def _no_split_modules(self) -> list[str]:
        return ["Dinov2Embeddings", "LayerNorm"]

    @property
    def _no_split_names(self) -> list[str]:
        return ["proprio_enc", "head"]

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True
        if hasattr(self.visual, "gradient_checkpointing_enable"):
            self.visual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        logger.info("Enabled gradient checkpointing for SuccessFailClassifier")

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        if hasattr(self.visual, "gradient_checkpointing_disable"):
            self.visual.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for SuccessFailClassifier")

    def attach_runtime_assets(
        self,
        processor,
        input_transform,
        device,
    ) -> None:
        """Wire up the inference-time image processor, input transform, and device.

        Called from :meth:`from_checkpoint`. Mirrors the rewind_arm hook so
        ``infer`` / ``infer_batch`` / ``predict_logit_batch`` can run
        end-to-end on raw LeRobot observations without the caller managing
        processor state.
        """
        self.processor = processor
        self._input_transform = input_transform
        self._device = device

    # ------------------------------------------------------------------
    # Vision encoder helpers
    # ------------------------------------------------------------------

    def _encode_image(self, pixel_values: Tensor) -> Tensor:
        """Run DINOv2 on a ``[B, 3, H, W]`` batch → ``[B, hidden_size]``."""
        outputs = self.visual(pixel_values=pixel_values)
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        # Fall back to CLS token from last_hidden_state (DINOv2 family has
        # this field for all variants; pooler is optional in some HF builds).
        return outputs.last_hidden_state[:, 0]

    def _pool_across_cameras(
        self,
        images: dict[str, Tensor],
        image_masks: dict[str, Tensor],
    ) -> Tensor:
        """Masked mean-pool vision features across cameras.

        Each camera contributes a ``[B, hidden_size]`` feature tensor; the
        pool is a per-sample weighted mean using the camera masks. Samples
        whose masks are all ``False`` (no camera active) get an all-zero
        pooled feature with the denominator clamped to ``1.0`` to avoid
        NaN — mirroring the safe-divide in
        :meth:`Pistar06Model._pool_vision_features`.
        """
        sorted_cams = sorted(images.keys())
        if not sorted_cams:
            raise ValueError("observation['images'] is empty")
        template = images[sorted_cams[0]]
        bsize = template.shape[0]
        device = template.device
        dtype = template.dtype

        feats: list[Tensor] = []
        weights: list[Tensor] = []
        for cam in sorted_cams:
            img = images[cam]
            feat = self._encode_image(img)

            mask = image_masks.get(cam)
            if mask is None:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            if mask.dim() == 0:
                mask = mask.view(1).expand(bsize)
            elif mask.dim() == 2:
                # Treat first-dim as batch, collapse by any-true along the rest.
                mask = mask.any(dim=-1)
            mask_f = mask.to(dtype=dtype).unsqueeze(-1)  # [B, 1]
            feats.append(feat * mask_f)
            weights.append(mask_f)

        total = torch.stack(feats, dim=0).sum(dim=0)  # [B, hidden_size]
        total_w = torch.stack(weights, dim=0).sum(dim=0).clamp(min=1.0)
        return total / total_w

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        pos_weight: Optional[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """BCE-with-logits + label smoothing on ``{0, 1}`` frame labels.

        Args:
            logits: ``[B]`` raw logits.
            labels: ``[B]`` float32 in ``{0, 1}`` (0 = success ep, 1 = fail ep).
            pos_weight: Optional scalar tensor to rebalance the positive
                (= fail) class. Passed straight through to
                :func:`F.binary_cross_entropy_with_logits`.

        Returns:
            ``(loss, metrics)`` where ``metrics`` has the same keys the
            rewind_arm CriticOutput populates (``acc_best``,
            ``acc_neighbor``, ``mae``) for logging parity.
        """
        if labels.dim() != 1:
            raise ValueError(f"labels must be rank-1, got {tuple(labels.shape)}")
        if logits.dim() != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits ({tuple(logits.shape)}) must be rank-1 and match labels "
                f"({tuple(labels.shape)})"
            )

        y = labels.to(dtype=logits.dtype)
        eps = float(self.label_smoothing)
        y_smooth = y * (1.0 - 2.0 * eps) + eps

        loss = F.binary_cross_entropy_with_logits(
            logits, y_smooth, pos_weight=pos_weight, reduction="mean"
        )

        with torch.no_grad():
            preds = (logits >= 0.0).to(dtype=torch.long)
            targets_long = y.round().to(dtype=torch.long)
            acc_best = (preds == targets_long).to(dtype=torch.float32).mean()

        metrics = {
            "acc_best": acc_best,
            # acc_neighbor / mae have no meaning for single-logit BCE; keep
            # as zero-valued scalars so CriticOutput slots stay populated
            # and aggregation code downstream does not crash on ``None``.
            "acc_neighbor": torch.zeros((), device=logits.device),
            "mae": torch.zeros((), device=logits.device),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def _forward_features(self, observation: dict) -> Tensor:
        """Compute fused ``[B, head_in]`` features from an observation dict."""
        images = observation["images"]
        image_masks = observation.get("image_masks", {})
        vision_feat = self._pool_across_cameras(images, image_masks)

        if self.use_proprio:
            state = observation.get("state")
            if state is None:
                raise ValueError(
                    "SuccessFailClassifier(use_proprio=True) requires "
                    "observation['state']; got None."
                )
            proprio_feat = self.proprio_enc(state.to(dtype=vision_feat.dtype))
            return torch.cat([vision_feat, proprio_feat], dim=-1)
        return vision_feat

    def forward(
        self,
        observation: dict,
        labels: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> CriticOutput:
        """Forward pass.

        Args:
            observation: See module docstring for the expected schema.
            labels: ``[B]`` float in ``{0, 1}`` — success / fail episode
                label broadcast to every frame of that episode.
            pos_weight: Optional scalar tensor for BCE rebalancing.

        Returns:
            A fully populated :class:`CriticOutput`. When ``labels`` is
            ``None`` only inference-relevant fields are populated.
        """
        del kwargs  # forward-compat signature parity with other value models

        fused = self._forward_features(observation)
        raw_logits = self.head(fused).squeeze(-1)  # [B]
        probs = torch.sigmoid(raw_logits)

        loss: Optional[Tensor] = None
        cat_metrics: Optional[dict[str, Tensor]] = None
        if labels is not None:
            loss, cat_metrics = self._compute_loss(raw_logits, labels, pos_weight)

        return CriticOutput(
            loss=loss,
            predicted_values=probs,
            logits=raw_logits.unsqueeze(-1),  # [B, 1]
            probs=probs,
            atoms=None,
            expert_loss=loss,
            hidden_states=fused,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
        )

    @torch.no_grad()
    def predict(self, observation: dict) -> CriticOutput:
        """No-grad forward — populates only inference-relevant fields."""
        fused = self._forward_features(observation)
        raw_logits = self.head(fused).squeeze(-1)
        probs = torch.sigmoid(raw_logits)
        return CriticOutput(
            predicted_values=probs,
            logits=raw_logits.unsqueeze(-1),
            probs=probs,
            atoms=None,
            hidden_states=fused,
        )

    @torch.no_grad()
    def predict_value(self, observation: dict) -> Tensor:
        """Return ``P(fail) ∈ [0, 1]`` per sample (shape ``[B]``)."""
        return self.predict(observation).predicted_values

    # ------------------------------------------------------------------
    # Observation prep (CPU-only, safe for DataLoader workers)
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_observation_cpu(inputs: dict, processor) -> dict:
        """CPU-only observation prep — safe for DataLoader workers.

        Mirrors :meth:`BinaryValueCriticModel._prepare_observation_cpu` but
        without the tokenization branch (this classifier has no prompt
        input). The caller is responsible for having already applied the
        openpi input_transform pipeline — this helper only handles the
        image / state → processor-ready tensor conversion.
        """
        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key, value in inputs.items():
                if "image" in key.lower() and isinstance(
                    value, (np.ndarray, torch.Tensor)
                ):
                    normalized = key
                    for prefix in (
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ):
                        normalized = normalized.replace(prefix, "")
                    images_dict[normalized] = value

        images_bhwc: dict[str, torch.Tensor] = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)  # CHW -> BHWC
                else:
                    img = img.unsqueeze(0)  # HWC -> BHWC
            elif img.dim() == 4 and img.shape[1] == 3:
                img = img.permute(0, 2, 3, 1)  # BCHW -> BHWC
            images_bhwc[cam_name] = img

        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch: dict[str, torch.Tensor] = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor(
                        [mask], dtype=torch.bool
                    )
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor(
                        [True], dtype=torch.bool
                    )
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        processed = processor(
            images=images_bhwc,
            image_masks=image_masks_batch,
        )

        obs: dict[str, Any] = {
            "images": processed["pixel_values"],
            "image_masks": processed["image_masks"],
        }

        state_val = inputs.get("state")
        if state_val is not None:
            if isinstance(state_val, torch.Tensor):
                state_t = state_val.detach().float()
            else:
                state_t = torch.from_numpy(
                    np.asarray(state_val, dtype=np.float32)
                )
            state_t = state_t.reshape(-1).unsqueeze(0)
            obs["state"] = state_t
        return obs

    def _prepare_observation(self, inputs: dict) -> dict:
        """GPU-ready observation for a single raw input dict."""
        processor = getattr(self, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Processor not attached. Use from_checkpoint() or "
                "attach_runtime_assets() first."
            )
        device = getattr(self, "_device", "cuda")
        obs = self._prepare_observation_cpu(inputs, processor)
        obs["images"] = {k: v.to(device) for k, v in obs["images"].items()}
        obs["image_masks"] = {k: v.to(device) for k, v in obs["image_masks"].items()}
        if "state" in obs:
            obs["state"] = obs["state"].to(device)
        return obs

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        """Concat a list of already-CPU-prepped observations into a batch."""
        all_images: list[dict[str, torch.Tensor]] = []
        all_image_masks: list[dict[str, torch.Tensor]] = []
        all_states: list[Optional[torch.Tensor]] = []

        for inputs in inputs_list:
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_states.append(single_obs.get("state"))

        batched_images = {
            k: torch.cat([img[k] for img in all_images], dim=0)
            for k in all_images[0]
        }
        batched_masks = {
            k: torch.cat([m[k] for m in all_image_masks], dim=0)
            for k in all_image_masks[0]
        }
        batch: dict[str, Any] = {
            "images": batched_images,
            "image_masks": batched_masks,
        }
        if all(s is not None for s in all_states):
            batch["state"] = torch.cat(all_states, dim=0)
        return batch

    # ------------------------------------------------------------------
    # Inference-facing APIs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Infer from a single raw observation; returns ``{"value": float}``.

        ``value = sigmoid(logit) ∈ [0, 1]`` — the probability the frame is
        a fail-like frame. Parallel to ``BinaryValueCriticModel.infer``
        (which also returns ``{"value": float}``) so generic value-model
        tooling treats this model identically.
        """
        inputs = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()
        }
        inputs = self._input_transform(inputs)
        observation = self._prepare_observation(inputs)
        result = self.predict(observation)
        return {
            "value": float(result.predicted_values[0].item()),
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
        """Batched :meth:`infer`; returns one ``{"value": float}`` per input.

        ``pretransformed`` and ``already_cpu_prepared`` follow the same
        contract as :meth:`BinaryValueCriticModel.infer_batch`. Use
        :meth:`predict_logit_batch` when you need raw logits instead of
        the ``sigmoid(logit)`` value.
        """
        if not obs_list:
            return []
        raw_logits = self._predict_logits_impl(
            obs_list,
            batch_size=batch_size,
            pretransformed=pretransformed,
            already_cpu_prepared=already_cpu_prepared,
        )
        probs = 1.0 / (1.0 + np.exp(-raw_logits))
        return [{"value": float(p)} for p in probs]

    @torch.no_grad()
    def predict_logit_batch(
        self,
        obs_list: list[dict],
        *,
        batch_size: int = 64,
        pretransformed: bool = False,
        already_cpu_prepared: bool = False,
    ) -> np.ndarray:
        """Return raw logits for a list of raw observations.

        New API (not on :class:`BinaryValueCriticModel`). Used by the
        Step 3 advantage-correction path in ``compute_advantages.py`` to
        apply ``V_final = V_orig - lambda · logit`` without re-deriving
        the logit from ``sigmoid(value)``.

        Returns:
            ``np.ndarray[len(obs_list)]`` of float32 logits, in the same
            order as ``obs_list``.
        """
        if not obs_list:
            return np.zeros(0, dtype=np.float32)
        return self._predict_logits_impl(
            obs_list,
            batch_size=batch_size,
            pretransformed=pretransformed,
            already_cpu_prepared=already_cpu_prepared,
        )

    def _predict_logits_impl(
        self,
        obs_list: list[dict],
        *,
        batch_size: int,
        pretransformed: bool,
        already_cpu_prepared: bool,
    ) -> np.ndarray:
        device = getattr(self, "_device", "cuda")
        all_logits: list[np.ndarray] = []

        for batch_start in range(0, len(obs_list), batch_size):
            batch_end = min(batch_start + batch_size, len(obs_list))
            batch_obs = obs_list[batch_start:batch_end]

            if already_cpu_prepared:
                first = batch_obs[0]
                batched_images = {
                    k: torch.cat([o["images"][k] for o in batch_obs], dim=0).to(
                        device
                    )
                    for k in first["images"]
                }
                batched_masks = {
                    k: torch.cat([o["image_masks"][k] for o in batch_obs], dim=0).to(
                        device
                    )
                    for k in first["image_masks"]
                }
                observation: dict[str, Any] = {
                    "images": batched_images,
                    "image_masks": batched_masks,
                }
                if "state" in first:
                    observation["state"] = torch.cat(
                        [o["state"] for o in batch_obs], dim=0
                    ).to(device)
            else:
                inputs_list = []
                for obs in batch_obs:
                    inputs = {
                        k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in obs.items()
                    }
                    if not pretransformed:
                        inputs = self._input_transform(inputs)
                    inputs_list.append(inputs)
                observation = self._prepare_observation_batch(inputs_list)

            result = self.predict(observation)
            # result.logits is [B, 1]; flatten to [B] numpy.
            all_logits.append(
                result.logits.squeeze(-1).detach().float().cpu().numpy()
            )

        return np.concatenate(all_logits, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # from_checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir,
        *,
        device: str = "cuda",
        env_type: str = "libero",
        model_type: str = "pi05",
        default_prompt: Optional[str] = None,
        norm_stats: Optional[dict] = None,
        vision_repo_id: Optional[str] = None,
        use_proprio: Optional[bool] = None,
        proprio_dim: Optional[int] = None,
        image_size: Optional[int] = None,
        label_smoothing: Optional[float] = None,
        precision: Optional[str] = None,
        max_state_dim: Optional[int] = None,
        action_dim: int = 32,
        **kwargs,
    ) -> "SuccessFailClassifier":
        """Build a classifier from a saved FSDP checkpoint, ready for inference.

        Mirrors :meth:`BinaryValueCriticModel.from_checkpoint`. The loaded
        weights come from ``<checkpoint>/actor/model_state_dict/full_weights.pt``
        (or a bare directory layout). The classifier-side config
        (``config.json``) is read and merged with Hydra-level overrides so
        the inference-time processor + input_transform match training.
        """
        del kwargs  # accepted for signature forward-compat with value models

        from omegaconf import OmegaConf

        from rlinf.models.embodiment.value_model.checkpoint_utils import (
            build_input_transforms,
            load_norm_stats,
        )

        from . import _candidate_checkpoint_dirs, get_model
        from .processing import SuccessFailClassifierImageProcessor

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        logger.info(
            "Loading SuccessFailClassifier from %s", checkpoint_dir
        )

        cfg_dict: dict[str, Any] = {"model_path": str(checkpoint_dir)}
        optional_overrides = {
            "vision_repo_id": vision_repo_id,
            "use_proprio": use_proprio,
            "proprio_dim": proprio_dim,
            "image_size": image_size,
            "label_smoothing": label_smoothing,
            "precision": precision,
            "max_state_dim": max_state_dim,
        }
        for key, value in optional_overrides.items():
            if value is not None:
                cfg_dict[key] = value
        cfg = OmegaConf.create(cfg_dict)
        model = get_model(cfg)
        if model is None:
            raise RuntimeError(
                f"get_model() returned None for {checkpoint_dir}; "
                "check the checkpoint config.json and vision_repo_id."
            )

        image_size_value = int(getattr(model.config, "image_size", 224))
        image_processor = _load_image_processor_from_checkpoint(
            str(checkpoint_dir), default_image_size=image_size_value
        )

        if norm_stats is None:
            try:
                norm_stats = load_norm_stats(checkpoint_dir, env_type)
                logger.info(f"  Loaded norm stats with asset_id={env_type}")
            except FileNotFoundError:
                logger.warning(
                    "  Could not find norm stats in %s — proceeding without "
                    "normalization",
                    checkpoint_dir,
                )
        if norm_stats and "return" in norm_stats:
            norm_stats = {k: v for k, v in norm_stats.items() if k != "return"}

        use_quantile_norm = model_type.lower() != "pi0"
        transforms = build_input_transforms(
            env_type=env_type,
            model_type=model_type,
            action_dim=action_dim,
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
        )
        from openpi.transforms import compose

        input_transform = compose(transforms)
        model.attach_runtime_assets(
            processor=image_processor,
            input_transform=input_transform,
            device=device,
        )

        model = model.to(device)
        model.eval()

        logger.info("SuccessFailClassifier.from_checkpoint ready for inference")
        return model


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = [
    "CriticOutput",
    "SuccessFailClassifier",
]
