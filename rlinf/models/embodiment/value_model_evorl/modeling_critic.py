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

"""Pistar06 evorl value critic model — RLinf-facing entry point.

Standalone module — does NOT import anything from
``rlinf.models.embodiment.value_model``. The class ``Pistar06ValueCriticModel``
exposes the same public API as ``ValueCriticModel`` so the FSDP value-SFT
worker and ``compute_advantages.py`` can use both interchangeably via simple
``model_type`` dispatch.

Public API (mirrors value_model.modeling_critic.ValueCriticModel):
    - forward(observation, target_values=None) -> CriticOutput
    - predict(observation) -> CriticOutput
    - predict_value(observation) -> Tensor
    - predict_distribution(observation) -> tuple[Tensor, Tensor, Tensor]
    - infer(obs) -> dict
    - infer_batch(obs_list, **kwargs) -> list[dict]
    - from_checkpoint(checkpoint_dir, **kwargs) -> Pistar06ValueCriticModel
    - gradient_checkpointing_enable() / .disable()
    - _no_split_modules / _no_split_names properties
    - _prepare_observation_cpu(inputs, processor) staticmethod
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .configuration import Pistar06EvorlConfig
from .modeling_pistar06 import Pistar06Model, build_bin_centers

logger = logging.getLogger(__name__)


@dataclass
class CriticOutput(ModelOutput):
    """Output for the Pistar06 critic model.

    Standalone copy — fields mirror ``value_model.modeling_critic.CriticOutput``
    so worker code (``FSDPValueSftWorker``) and ``compute_advantages.py`` can
    consume either model's output via duck typing.
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


class Pistar06ValueCriticModel(nn.Module):
    """Pistar06 value model — fully independent of value_model/.

    Wraps the faithful ``Pistar06Model`` port (SigLIP + Gemma3 + concat MLP)
    with the RLinf-side observation/forward contract:

        - ``forward(observation, target_values)`` returns a ``CriticOutput``
        - ``observation`` is a dict in the same format produced by
          ``ValueDataCollator`` / ``ValueDataset``: it has
          ``images: dict[cam_name, Tensor[B,3,384,384]]`` (in [0,1] range),
          ``image_masks: dict[cam_name, Tensor[B]]``, ``tokenized_prompt``,
          ``tokenized_prompt_mask``.

    The class layout intentionally mirrors ``ValueCriticModel`` so the FSDP
    SFT worker and the offline advantage pipeline can dispatch between
    variants without any other changes.
    """

    def __init__(self, config: Pistar06EvorlConfig):
        super().__init__()
        self.config = config
        self.model = Pistar06Model(config)

        self.num_bins = config.num_bins
        self.v_min = config.bin_min
        self.v_max = config.bin_max
        self.delta_z = (config.bin_max - config.bin_min) / (config.num_bins - 1)
        self.register_buffer(
            "bin_centers",
            build_bin_centers(config.num_bins, config.bin_min, config.bin_max),
            persistent=False,
        )

        self.gradient_checkpointing_enabled = False

        # FSDP wrap-name tagging (mirrors value_model/modeling_critic.py:160-162)
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "SiglipVisionEmbeddings",
            "Gemma3RotaryEmbedding",
            "LayerNorm",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return ["image_projector", "language_projector", "value_head"]

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        for submod in (self.model.vision_encoder, self.model.language_model):
            if hasattr(submod, "gradient_checkpointing_enable"):
                submod.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        logger.info("Enabled gradient checkpointing for Pistar06ValueCriticModel")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        for submod in (self.model.vision_encoder, self.model.language_model):
            if hasattr(submod, "gradient_checkpointing_disable"):
                submod.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for Pistar06ValueCriticModel")

    # ------------------------------------------------------------------
    # Observation -> tensor adapter
    # ------------------------------------------------------------------

    def _stack_observation(
        self, observation: dict
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert worker/collator observation dict to Pistar06Model forward args.

        Returns: (input_ids[B,T], attention_mask[B,T], images[B,N,3,H,W],
                  image_attention_mask[B,N]).
        """
        images_dict = observation["images"]
        image_masks_dict = observation.get("image_masks", {})
        input_ids = observation["tokenized_prompt"]
        attention_mask = observation["tokenized_prompt_mask"]

        sorted_cams = sorted(images_dict.keys())
        img_list = [images_dict[k] for k in sorted_cams]
        if not img_list:
            raise ValueError("observation['images'] is empty")
        bsize = img_list[0].shape[0]
        device = img_list[0].device

        mask_list = [
            image_masks_dict.get(k, torch.ones(bsize, dtype=torch.bool, device=device))
            for k in sorted_cams
        ]

        images_stacked = torch.stack(img_list, dim=1)  # [B, N, 3, H, W]
        image_mask_stacked = torch.stack(mask_list, dim=1).to(torch.bool)  # [B, N]
        return (
            input_ids.long(),
            attention_mask.long(),
            images_stacked,
            image_mask_stacked,
        )

    # ------------------------------------------------------------------
    # Categorical loss helper (independent copy)
    # ------------------------------------------------------------------

    def _compute_categorical_loss(self, logits, target_values):
        """Categorical loss via Dirac delta projection onto bins.

        Standalone copy — same logic and identical signature as
        ``ValueCriticModel._compute_categorical_loss``.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        target_values = target_values.clamp(self.v_min, self.v_max)
        b = (target_values - self.v_min) / self.delta_z
        l = b.floor().long().clamp(0, self.num_bins - 1)  # noqa: E741
        u = b.ceil().long().clamp(0, self.num_bins - 1)

        d_to_l, d_to_u = b - l.float(), u.float() - b
        same_bin = l == u
        d_to_l = torch.where(same_bin, torch.zeros_like(d_to_l), d_to_l)
        d_to_u = torch.where(same_bin, torch.ones_like(d_to_u), d_to_u)

        batch_size = target_values.shape[0]
        target_probs = torch.zeros(
            batch_size, self.num_bins, device=target_values.device
        )
        batch_idx = torch.arange(batch_size, device=target_values.device)
        target_probs[batch_idx, l] += d_to_u
        target_probs[batch_idx, u] += d_to_l

        loss = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)

        pred_bin = logits.argmax(dim=-1)
        best_target_bin = torch.where(d_to_u >= d_to_l, l, u)
        acc_best = (pred_bin == best_target_bin).float().mean()
        acc_neighbor = ((pred_bin == l) | (pred_bin == u)).float().mean()

        dist_to_l = (pred_bin - l).abs()
        dist_to_u = (pred_bin - u).abs()
        min_dist = torch.min(dist_to_l, dist_to_u).float()
        mae = (min_dist * self.delta_z).mean()

        metrics = {
            "acc_best": acc_best,
            "acc_neighbor": acc_neighbor,
            "mae": mae,
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, observation, target_values=None, **kwargs) -> CriticOutput:
        """Forward pass.

        Mirrors ``ValueCriticModel.forward``: stacks the observation, runs
        the multimodal forward to get hidden state + logits, computes the
        categorical loss + metrics if ``target_values`` is provided, and
        returns a fully populated ``CriticOutput`` (including
        ``hidden_states``).
        """
        input_ids, attention_mask, images, image_mask = self._stack_observation(
            observation
        )

        # Pistar06ValueImageProcessor outputs [0, 1] BCHW; Pistar06Model
        # applies SigLIP-style mean/std normalization internally.
        # Splitting features and logits lets us expose hidden_states.
        hidden_states = self.model._compute_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
        )  # [B, fusion_hidden_dim * 2]
        logits = self.model.value_head(hidden_states)  # [B, num_bins]

        probs = F.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(device=logits.device)
        predicted_values = (probs * bin_centers).sum(dim=-1)

        expert_loss = None
        cat_metrics = None
        if target_values is not None:
            expert_loss, cat_metrics = self._compute_categorical_loss(
                logits, target_values
            )

        expert_loss_mean = expert_loss.mean() if expert_loss is not None else None

        return CriticOutput(
            loss=expert_loss_mean,
            predicted_values=predicted_values,
            logits=logits,
            probs=probs,
            atoms=bin_centers,
            expert_loss=expert_loss_mean,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
        )

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference forward — standalone, mirrors ValueCriticModel.predict.

        Does NOT call ``self.forward``: matches the structural pattern of
        ``ValueCriticModel.predict``, which is its own ``@torch.no_grad()``
        path that skips the loss / metric computation entirely. Returns a
        ``CriticOutput`` with only the inference-relevant fields populated
        (``predicted_values``, ``logits``, ``probs``, ``atoms``,
        ``hidden_states``); loss and metric fields stay at their dataclass
        defaults of ``None``.
        """
        input_ids, attention_mask, images, image_mask = self._stack_observation(
            observation
        )

        hidden_states = self.model._compute_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
        )
        logits = self.model.value_head(hidden_states)

        probs = F.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(device=logits.device)
        predicted_values = (probs * bin_centers).sum(dim=-1)

        return CriticOutput(
            predicted_values=predicted_values,
            logits=logits,
            probs=probs,
            atoms=bin_centers,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Predict scalar values. Mirrors ValueCriticModel.predict_value."""
        return self.predict(observation).predicted_values

    @torch.no_grad()
    def predict_distribution(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        """Predict (values, probs, atoms). Mirrors ValueCriticModel."""
        out = self.predict(observation)
        return out.predicted_values, out.probs, out.atoms

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
        num_return_bins: int = 201,
        return_min: float = -1.0,
        return_max: float = 0.0,
        tokenizer_path: Optional[str] = None,
        vision_repo_id: Optional[str] = None,
        language_repo_id: Optional[str] = None,
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        # State-in-prompt and interface compat — must match training config.
        include_state_in_prompt: bool = True,
        max_state_dim: int = 32,
        state_discretization_bins: int = 256,
        max_token_len: int = 200,
        action_dim: int = 32,
        action_horizon: int = 50,
        **kwargs,
    ) -> "Pistar06ValueCriticModel":
        """Build a Pistar06ValueCriticModel from a checkpoint, ready for inference.

        Mirrors ValueCriticModel.from_checkpoint so compute_advantages.py can
        dispatch between the two model variants without other changes.
        """
        import pathlib

        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        from rlinf.models.embodiment.value_model.checkpoint_utils import (
            build_input_transforms,
            has_tokenizer_files,
            load_norm_stats,
        )

        from . import get_model
        from .processing import Pistar06ValueProcessor

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        logger.info(f"Loading Pistar06 evorl value model from {checkpoint_dir}")

        cfg = OmegaConf.create(
            {
                "model_path": str(checkpoint_dir),
                "vision_repo_id": vision_repo_id,
                "language_repo_id": language_repo_id,
                "num_bins": num_return_bins,
                "v_min": return_min,
                "v_max": return_max,
                "fusion_hidden_dim": fusion_hidden_dim,
                "dropout": dropout,
                "include_state_in_prompt": include_state_in_prompt,
                "max_state_dim": max_state_dim,
                "state_discretization_bins": state_discretization_bins,
                "max_token_len": max_token_len,
                "action_dim": action_dim,
                "action_horizon": action_horizon,
            }
        )
        model = get_model(cfg)

        # Tokenizer resolution
        if tokenizer_path:
            logger.info("  Using explicit tokenizer_path: %s", tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, add_bos_token=True, local_files_only=True
            )
        elif has_tokenizer_files(checkpoint_dir):
            logger.info("  Found tokenizer files in checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=True, local_files_only=True
            )
        else:
            raise ValueError(
                f"No tokenizer found. Set tokenizer_path or ensure checkpoint "
                f"contains tokenizer files. checkpoint_dir={checkpoint_dir}"
            )
        # Read state-in-prompt fields off the just-constructed model.config so
        # inference-time prompt construction matches what the model was trained
        # on. Backward-compat defaults kick in when the checkpoint predates
        # these fields.
        processor = Pistar06ValueProcessor(
            tokenizer=tokenizer,
            max_token_len=getattr(model.config, "max_token_len", 200),
            include_state_in_prompt=getattr(
                model.config, "include_state_in_prompt", True
            ),
            max_state_dim=getattr(model.config, "max_state_dim", 32),
            state_discretization_bins=getattr(
                model.config, "state_discretization_bins", 256
            ),
        )

        model.processor = processor
        model = model.to(device)
        model.eval()

        # Norm stats
        if norm_stats is None:
            try:
                norm_stats = load_norm_stats(checkpoint_dir, env_type)
                logger.info(f"Loaded norm stats with asset_id={env_type}")
            except FileNotFoundError:
                logger.warning(
                    f"Could not find norm stats in {checkpoint_dir}, "
                    "proceeding without normalization"
                )

        if norm_stats and "return" in norm_stats:
            norm_stats = {k: v for k, v in norm_stats.items() if k != "return"}

        use_quantile_norm = model_type.lower() != "pi0"

        transforms = build_input_transforms(
            env_type=env_type,
            model_type=model_type,
            action_dim=getattr(model.config, "action_dim", 32),
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
        )

        from openpi.transforms import compose

        model._input_transform = compose(transforms)
        model._device = device

        logger.info("Pistar06ValueCriticModel.from_checkpoint ready for inference")
        return model

    @staticmethod
    def _prepare_observation_cpu(inputs: dict, processor) -> dict:
        """CPU-only observation preparation (safe for DataLoader workers).

        Standalone copy of ValueCriticModel._prepare_observation_cpu adapted
        to use Pistar06ValueProcessor (which outputs [0, 1] BCHW 384x384
        images instead of [-1, 1] 224x224).
        """
        import numpy as np

        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key in inputs:
                if "image" in key.lower() and isinstance(
                    inputs[key], (np.ndarray, torch.Tensor)
                ):
                    img_key = key
                    for prefix in [
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ]:
                        img_key = img_key.replace(prefix, "")
                    images_dict[img_key] = inputs[key]

        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        images_bhwc = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)  # CHW -> BHWC
                else:
                    img = img.unsqueeze(0)  # HWC -> BHWC
            elif img.dim() == 4:
                if img.shape[1] == 3:
                    img = img.permute(0, 2, 3, 1)  # BCHW -> BHWC
            images_bhwc[cam_name] = img

        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor([mask], dtype=torch.bool)
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

        # Prompt + optional state → tokens. Delegating to
        # ``processor._build_prefix_text`` keeps inference parity with the
        # training-time ``_tokenize_single`` path.
        state_val = inputs.get("state")
        prefix_text = processor._build_prefix_text(prompt, state_val)
        tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=True)
        seq_len = len(tokens)
        max_length = processor.max_token_len
        if seq_len < max_length:
            padding_len = max_length - seq_len
            tok_mask = [True] * seq_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
        else:
            tokens = tokens[:max_length]
            tok_mask = [True] * max_length

        return {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": torch.tensor([tokens], dtype=torch.long),
            "tokenized_prompt_mask": torch.tensor([tok_mask], dtype=torch.bool),
        }

    def _prepare_observation(self, inputs: dict) -> dict:
        """Prepare observation dict on the model's device for direct forward.

        Independent copy of ValueCriticModel._prepare_observation. Uses
        ``self.processor`` and ``self._device`` (both set by from_checkpoint).
        """
        import numpy as np

        processor = getattr(self, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Model processor not attached. Use from_checkpoint() or "
                "attach manually."
            )

        device = getattr(self, "_device", "cuda")

        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key in inputs:
                if "image" in key.lower() and isinstance(
                    inputs[key], (np.ndarray, torch.Tensor)
                ):
                    img_key = key
                    for prefix in [
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ]:
                        img_key = img_key.replace(prefix, "")
                    images_dict[img_key] = inputs[key]

        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        images_bhwc = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)
                else:
                    img = img.unsqueeze(0)
            elif img.dim() == 4:
                if img.shape[1] == 3:
                    img = img.permute(0, 2, 3, 1)
            images_bhwc[cam_name] = img

        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor([mask], dtype=torch.bool)
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

        # Prompt + optional state → tokens. Delegating to
        # ``processor._build_prefix_text`` keeps inference parity with the
        # training-time ``_tokenize_single`` path.
        state_val = inputs.get("state")
        prefix_text = processor._build_prefix_text(prompt, state_val)
        tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=True)
        seq_len = len(tokens)
        max_length = processor.max_token_len
        if seq_len < max_length:
            padding_len = max_length - seq_len
            mask = [True] * seq_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
        else:
            tokens = tokens[:max_length]
            mask = [True] * max_length

        pixel_values = processed_img["pixel_values"]
        image_masks = processed_img["image_masks"]

        if isinstance(pixel_values, dict):
            images_on_device = {k: v.to(device) for k, v in pixel_values.items()}
        else:
            images_on_device = pixel_values.to(device)

        if isinstance(image_masks, dict):
            masks_on_device = {k: v.to(device) for k, v in image_masks.items()}
        else:
            masks_on_device = image_masks.to(device)

        return {
            "images": images_on_device,
            "image_masks": masks_on_device,
            "tokenized_prompt": torch.tensor([tokens], dtype=torch.long, device=device),
            "tokenized_prompt_mask": torch.tensor(
                [mask], dtype=torch.bool, device=device
            ),
        }

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        """Prepare batched observation dict from a list of inputs."""
        all_images = []
        all_image_masks = []
        all_tokens = []
        all_masks = []

        for inputs in inputs_list:
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_tokens.append(single_obs["tokenized_prompt"])
            all_masks.append(single_obs["tokenized_prompt_mask"])

        if isinstance(all_images[0], dict):
            batched_images = {
                k: torch.cat([img[k] for img in all_images], dim=0)
                for k in all_images[0]
            }
            batched_masks = {
                k: torch.cat([m[k] for m in all_image_masks], dim=0)
                for k in all_image_masks[0]
            }
        else:
            batched_images = torch.cat(all_images, dim=0)
            batched_masks = torch.cat(all_image_masks, dim=0)

        return {
            "images": batched_images,
            "image_masks": batched_masks,
            "tokenized_prompt": torch.cat(all_tokens, dim=0),
            "tokenized_prompt_mask": torch.cat(all_masks, dim=0),
        }

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Infer value from a single raw observation.

        Returns ``{"value": float, "state": np.ndarray}``.
        """
        import numpy as np

        inputs = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()
        }
        inputs = self._input_transform(inputs)
        observation = self._prepare_observation(inputs)

        values = self.predict_value(observation)
        return {
            "value": float(values[0].item()),
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
        """Batched inference. Returns one ``{"value": float}`` per input.

        Independent copy of ``ValueCriticModel.infer_batch``. Calls
        ``self.predict_value(observation)`` which routes through our own
        forward → ``Pistar06Model.forward`` path.
        """
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
                        k: torch.cat([obs["images"][k] for obs in batch_obs], dim=0).to(
                            device
                        )
                        for k in first["images"]
                    }
                    batched_masks = {
                        k: torch.cat(
                            [obs["image_masks"][k] for obs in batch_obs], dim=0
                        ).to(device)
                        for k in first["image_masks"]
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
                        k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in obs.items()
                    }
                    if not pretransformed:
                        inputs = self._input_transform(inputs)
                    inputs_list.append(inputs)

                observation = self._prepare_observation_batch(inputs_list)

            values = self.predict_value(observation).cpu()

            for i in range(len(batch_obs)):
                all_outputs.append({"value": float(values[i])})

        return all_outputs


__all__ = [
    "CriticOutput",
    "Pistar06ValueCriticModel",
]
