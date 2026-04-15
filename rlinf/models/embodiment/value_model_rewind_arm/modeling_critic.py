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

"""ARM + ReWiND binary value critic — RLinf-facing entry point.

Parallel to ``rlinf.models.embodiment.value_model_evorl.modeling_critic``:
the observation/forward contract is byte-for-byte compatible (same keys in
the observation dict, same ``images: dict[cam_name, Tensor[B,3,H,W]]``
layout, same CriticOutput dataclass), so the FSDP value-SFT worker and the
offline advantage pipeline can dispatch on ``model_type`` alone.

The only real differences from the evorl variant:
    * ``forward(observation, labels)`` replaces ``(observation, target_values)``
      — labels are in ``{-1, +1}``.
    * :meth:`_compute_binary_loss` replaces ``_compute_categorical_loss``.
    * ``CriticOutput.predicted_values`` holds the sigmoid probability instead
      of the expected value over bins, and ``CriticOutput.atoms`` is ``None``.

Public API (mirrors value_model.modeling_critic.ValueCriticModel):
    - forward(observation, labels=None) -> CriticOutput
    - predict(observation) -> CriticOutput
    - predict_value(observation) -> Tensor   (sigmoid probability)
    - infer(obs) -> dict
    - infer_batch(obs_list, **kwargs) -> list[dict]
    - from_checkpoint(checkpoint_dir, **kwargs) -> BinaryValueCriticModel
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

from .configuration import BinaryValueConfig
from .modeling_rewind_arm import CLASS_PROGRESS, NUM_CLASSES, RewindArmBackbone

logger = logging.getLogger(__name__)


@dataclass
class CriticOutput(ModelOutput):
    """Output for the ARM + ReWiND binary critic.

    Field list deliberately matches :class:`~rlinf.models.embodiment.value_model.\
modeling_critic.CriticOutput` so worker code stays duck-type-compatible.
    For the binary variant:

        * ``logits`` is ``[B]`` (a single logit per pair).
        * ``probs`` is ``sigmoid(logits)`` — shape ``[B]``.
        * ``predicted_values`` equals ``probs`` (same scalar-per-sample).
        * ``atoms`` is always ``None``.
        * ``cat_acc_best`` carries binary accuracy for parity with evorl
          logging (the other cat_* fields stay ``None``).
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


class BinaryValueCriticModel(nn.Module):
    """ARM + ReWiND binary value model — parallel to Pistar06ValueCriticModel.

    Wraps the :class:`RewindArmBackbone` (SigLIP + Gemma3 + per-frame
    concat MLP) with the RLinf-side observation/forward contract:

        - ``forward(observation, labels)`` returns a :class:`CriticOutput`.
        - ``observation`` is a dict in the same format produced by
          :class:`~rlinf.models.embodiment.value_model.data_collator.\
ValueDataCollator`: ``images: dict[cam_name, Tensor[B,3,H,W]]`` in [0, 1],
          ``image_masks: dict[cam_name, Tensor[B]]``, ``tokenized_prompt``,
          ``tokenized_prompt_mask``. The ``cam_name`` entries are
          ``("frame_t_rgb", "frame_tk_rgb")`` — one slot per pair-frame, in
          that order.

    The class layout mirrors ``Pistar06ValueCriticModel`` so FSDP wrap names
    and the offline pipeline keep working unchanged.
    """

    def __init__(self, config: BinaryValueConfig):
        super().__init__()
        self.config = config
        self.model = RewindArmBackbone(config)
        self.label_smoothing = float(config.label_smoothing)
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
        logger.info("Enabled gradient checkpointing for BinaryValueCriticModel")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        for submod in (self.model.vision_encoder, self.model.language_model):
            if hasattr(submod, "gradient_checkpointing_disable"):
                submod.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for BinaryValueCriticModel")

    # ------------------------------------------------------------------
    # Observation -> tensor adapter
    # ------------------------------------------------------------------

    def _stack_observation(
        self, observation: dict
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert collator observation dict to :class:`RewindArmBackbone` args.

        Expects each ``observation["images"][cam_name]`` to be shape
        ``[B, num_frames, 3, H, W]`` — the camera axis is the dict key,
        and the frame axis is stacked within the tensor by
        :class:`BinaryPairDataCollator`.

        Returns:
            ``(input_ids[B,T], attention_mask[B,T],
              images[B, num_cameras, num_frames, 3, H, W],
              image_attention_mask[B, num_cameras, num_frames])``.
        """
        images_dict = observation["images"]
        image_masks_dict = observation.get("image_masks", {})
        input_ids = observation["tokenized_prompt"]
        attention_mask = observation["tokenized_prompt_mask"]

        sorted_cams = sorted(images_dict.keys())
        img_list = [images_dict[k] for k in sorted_cams]
        if not img_list:
            raise ValueError("observation['images'] is empty")
        template = img_list[0]
        if template.ndim != 5:
            raise ValueError(
                "observation['images'][cam] must have shape "
                f"[B, num_frames, C, H, W]; got {tuple(template.shape)}"
            )
        bsize, num_frames = template.shape[0], template.shape[1]
        device = template.device

        # Camera axis is the dict key — stack along a new dim=1.
        images_stacked = torch.stack(img_list, dim=1)
        # images_stacked: [B, num_cameras, num_frames, C, H, W]

        mask_list = [
            image_masks_dict.get(
                k,
                torch.ones(bsize, num_frames, dtype=torch.bool, device=device),
            )
            for k in sorted_cams
        ]
        # Masks can be either [B, num_frames] (per-camera) or [B] (treated
        # as a per-camera scalar broadcast across frames).
        normalised_masks: list[Tensor] = []
        for m in mask_list:
            if m.ndim == 1:
                m = m.unsqueeze(-1).expand(-1, num_frames)
            normalised_masks.append(m.to(torch.bool))
        image_mask_stacked = torch.stack(normalised_masks, dim=1)
        # image_mask_stacked: [B, num_cameras, num_frames]
        return (
            input_ids.long(),
            attention_mask.long(),
            images_stacked,
            image_mask_stacked,
        )

    # ------------------------------------------------------------------
    # Binary loss helper (replaces the evorl categorical loss)
    # ------------------------------------------------------------------

    def _compute_binary_loss(self, logits, labels):
        """2-way cross-entropy with optional label smoothing.

        Mirrors the evorl ``_compute_categorical_loss`` return shape so the
        rest of the critic (forward, predict, CriticOutput) slots in
        unchanged.

        Args:
            logits: Shape ``[B, 2]`` — ``[regress_logit, progress_logit]``.
            labels: Shape ``[B]`` with values in ``{-1, +1}``. ``-1`` → regress,
                ``+1`` → progress.

        Returns:
            Tuple of (per-sample loss of shape ``[B]``, metrics dict).
            Metric keys match the evorl slot names so
            ``CriticOutput.cat_acc_best`` can carry classification accuracy;
            the other cat_* fields are placeholders.
        """
        if logits.ndim != 2 or logits.shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"logits must have shape [B, {NUM_CLASSES}], got {tuple(logits.shape)}"
            )
        # {-1, +1} → {0, 1} int64 class indices.
        targets = ((labels.to(dtype=torch.float32) + 1.0) / 2.0).long()
        loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        pred_class = logits.argmax(dim=-1)
        acc_best = (pred_class == targets).to(dtype=torch.float32).mean()

        metrics = {
            "acc_best": acc_best,
            "acc_neighbor": torch.zeros((), device=logits.device),
            "mae": torch.zeros((), device=logits.device),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, observation, labels=None, **kwargs) -> CriticOutput:
        """Forward pass — parallel to Pistar06ValueCriticModel.forward.

        Stacks the observation, runs the multimodal backbone, squeezes the
        single-logit head, and — if ``labels`` are provided — computes BCE +
        accuracy via :meth:`_compute_binary_loss`. Returns a fully populated
        :class:`CriticOutput`.
        """
        input_ids, attention_mask, images, image_mask = self._stack_observation(
            observation
        )

        # Pistar06ValueImageProcessor emits [0, 1] BCHW images; the
        # backbone applies SigLIP-style mean/std normalization internally.
        # Splitting features and logits keeps `hidden_states` exposed.
        hidden_states = self.model._compute_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
        )  # [B, fusion_hidden_dim * (num_frames_per_pair + 1)]
        logits = self.model.value_head(hidden_states)  # [B, 2]

        probs = F.softmax(logits, dim=-1)  # [B, 2] — [p_regress, p_progress]
        predicted_values = probs[:, CLASS_PROGRESS]  # [B] — P(progress)

        expert_loss = None
        cat_metrics = None
        if labels is not None:
            expert_loss, cat_metrics = self._compute_binary_loss(logits, labels)

        expert_loss_mean = expert_loss.mean() if expert_loss is not None else None

        return CriticOutput(
            loss=expert_loss_mean,
            predicted_values=predicted_values,
            logits=logits,
            probs=probs,
            atoms=None,
            expert_loss=expert_loss_mean,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
        )

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference forward — parallel to Pistar06ValueCriticModel.predict.

        Separate from :meth:`forward` so the inference path stays cheap and
        keeps the evorl structural pattern. The returned ``CriticOutput``
        populates only inference-relevant fields; loss and metric fields
        stay at their dataclass defaults of ``None``.
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
        logits = self.model.value_head(hidden_states)  # [B, 2]
        probs = F.softmax(logits, dim=-1)  # [B, 2]

        return CriticOutput(
            predicted_values=probs[:, CLASS_PROGRESS],
            logits=logits,
            probs=probs,
            atoms=None,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Return P(progress) per sample (shape ``[B]``)."""
        return self.predict(observation).predicted_values

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
        label_smoothing: float = 0.05,
        num_frames_per_pair: int = 2,
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
        **kwargs,
    ) -> "BinaryValueCriticModel":
        """Build a BinaryValueCriticModel from a checkpoint, ready for inference.

        Mirrors ``Pistar06ValueCriticModel.from_checkpoint`` so the offline
        pipeline (compute_advantages, etc.) dispatches on ``model_type`` and
        loads either variant via the same call shape.
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
        logger.info(f"Loading ARM+ReWiND binary value model from {checkpoint_dir}")

        cfg = OmegaConf.create(
            {
                "model_path": str(checkpoint_dir),
                "vision_repo_id": vision_repo_id,
                "language_repo_id": language_repo_id,
                "label_smoothing": label_smoothing,
                "num_frames_per_pair": num_frames_per_pair,
                "fusion_hidden_dim": fusion_hidden_dim,
                "dropout": dropout,
                "include_state_in_prompt": include_state_in_prompt,
                "max_state_dim": max_state_dim,
                "state_discretization_bins": state_discretization_bins,
                "max_token_len": max_token_len,
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
            # Binary model has no action branch — pass a default so the
            # shared openpi transform (PadStatesAndActions) still works.
            action_dim=kwargs.get("action_dim", 32),
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
        )

        from openpi.transforms import compose

        model._input_transform = compose(transforms)
        model._device = device

        logger.info("BinaryValueCriticModel.from_checkpoint ready for inference")
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
        forward → ``RewindArmBackbone.forward`` path.
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
    "BinaryValueCriticModel",
    "CriticOutput",
]
