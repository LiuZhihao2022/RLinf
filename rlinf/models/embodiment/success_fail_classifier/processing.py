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

"""Image processor for the frame-level success/fail classifier.

The backbone is a DINOv2-s ViT pretrained with ImageNet statistics, so this
processor resizes each camera view to a square ``image_size`` (a multiple of
14, default 224) and applies ImageNet mean/std normalization. Output is a
dict ``{cam_name: Tensor[B, 3, H, W]}`` in the float32 normalized range,
ready for direct DINOv2 consumption.

Distinct from :class:`Pistar06ValueImageProcessor` which keeps [0, 1] floats
and defers normalization to the backbone — DINOv2 has no built-in
normalization, so we apply it here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.utils import TensorType

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default camera views align with the evorl / rewind_arm convention so any
# LeRobot dataset repacked for openpi plugs in without renaming.
DEFAULT_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def _resize_square(image_bhwc: torch.Tensor, target: int) -> torch.Tensor:
    """Resize a BHWC float image (values in [0, 1]) to ``target × target``.

    Aspect-ratio-preserving resize with zero padding — keeps the classifier
    robust to non-square camera resolutions (e.g. 256x256 libero vs native
    franka aspect). Mirrors :func:`resize_with_pad` in the Pistar06
    processor but simpler (no uint8 path, target is always square).
    """
    img_bchw = image_bhwc.permute(0, 3, 1, 2)  # BHWC -> BCHW
    _, _, cur_h, cur_w = img_bchw.shape
    ratio = max(cur_w / target, cur_h / target)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))
    resized = F.interpolate(
        img_bchw,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    pad_h0, rem_h = divmod(target - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(target - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    padded = F.pad(
        resized,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=0.0,
    )
    return padded  # BCHW, [0, 1]


class SuccessFailClassifierImageProcessor(ImageProcessingMixin):
    """Resize + ImageNet-normalize multi-camera input for DINOv2.

    Output images are BCHW float32 tensors already mean-subtracted and
    std-divided. The downstream model simply calls ``dinov2(image)`` without
    further normalization.
    """

    model_input_names: ClassVar[list[str]] = ["pixel_values", "image_masks"]

    def __init__(
        self,
        image_size: int = 224,
        do_resize: bool = True,
        image_keys: Sequence[str] = DEFAULT_IMAGE_KEYS,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if int(image_size) <= 0 or int(image_size) % 14 != 0:
            raise ValueError(
                "image_size must be a positive multiple of 14 (DINOv2 patch size); "
                f"got {image_size}"
            )
        self.image_size = int(image_size)
        self.do_resize = bool(do_resize)
        self.image_keys = tuple(image_keys)
        self.mean = tuple(float(x) for x in mean)
        self.std = tuple(float(x) for x in std)
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean / std must be length-3 tuples (RGB channels)")

    def _to_bhwc_float01(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize raw image to BHWC float in [0, 1].

        Accepts HWC / CHW / BHWC / BCHW layouts and uint8 / float. Mirrors
        the permissive conversion path in ``BinaryValueCriticModel._prepare
        _observation_cpu`` so the same raw LeRobot frames can be fed into
        either pipeline without pre-conversion.
        """
        if image.dim() == 3:
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.permute(1, 2, 0)  # CHW -> HWC
            image = image.unsqueeze(0)  # -> BHWC
        elif image.dim() == 4:
            if image.shape[1] == 3 and image.shape[-1] != 3:
                image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC
        else:
            raise ValueError(f"Expected 3D or 4D image tensor, got {image.dim()}D")

        image = image.float()
        if image.max() > 1.5:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    def _normalize_bchw(self, image_bchw: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=image_bchw.dtype, device=image_bchw.device)
        std = torch.tensor(self.std, dtype=image_bchw.dtype, device=image_bchw.device)
        return (image_bchw - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    def process_images(
        self,
        images_dict: dict[str, torch.Tensor],
        image_masks_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Process a multi-camera batch.

        Missing camera keys get zero-filled placeholders with mask ``False``
        so downstream code sees a rectangular camera dict. Present cameras
        are resized → normalized → returned as BCHW float tensors.
        """
        out_images: dict[str, torch.Tensor] = {}
        out_masks: dict[str, torch.Tensor] = {}

        batch_size: Optional[int] = None
        template_device: Optional[torch.device] = None
        for key in images_dict:
            if images_dict[key] is not None:
                batch_size = images_dict[key].shape[0]
                template_device = images_dict[key].device
                break

        for key in self.image_keys:
            image = images_dict.get(key)

            if image is None:
                if batch_size is not None:
                    placeholder = torch.zeros(
                        batch_size,
                        3,
                        self.image_size,
                        self.image_size,
                        device=template_device,
                    )
                    out_images[key] = self._normalize_bchw(placeholder)
                    out_masks[key] = torch.zeros(
                        batch_size, dtype=torch.bool, device=template_device
                    )
                continue

            image_bhwc = self._to_bhwc_float01(image)
            if self.do_resize and image_bhwc.shape[1:3] != (
                self.image_size,
                self.image_size,
            ):
                image_bchw = _resize_square(image_bhwc, self.image_size)
            else:
                image_bchw = image_bhwc.permute(0, 3, 1, 2)

            out_images[key] = self._normalize_bchw(image_bchw)

            bsize = image_bchw.shape[0]
            if image_masks_dict is not None and key in image_masks_dict:
                mask = image_masks_dict[key]
                if isinstance(mask, (bool, np.bool_)):
                    mask = torch.tensor([mask] * bsize, dtype=torch.bool)
                out_masks[key] = mask.to(torch.bool)
            else:
                out_masks[key] = torch.ones(
                    bsize, dtype=torch.bool, device=image_bchw.device
                )

        return out_images, out_masks

    def __call__(
        self,
        images: dict[str, torch.Tensor],
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        del return_tensors, kwargs  # reserved for HF API parity
        pixel_values, image_masks_out = self.process_images(images, image_masks)
        return {"pixel_values": pixel_values, "image_masks": image_masks_out}


__all__ = [
    "SuccessFailClassifierImageProcessor",
    "DEFAULT_IMAGE_KEYS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
