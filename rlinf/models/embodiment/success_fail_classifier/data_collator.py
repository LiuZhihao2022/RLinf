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

"""Collator for single-frame success/fail classifier training.

Parallel to :class:`BinaryPairDataCollator` but without the pair axis —
every sample is one frame, so the output is ``images: dict[cam, Tensor[B,
3, H, W]]`` (no ``num_frames`` dim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from .processing import SuccessFailClassifierImageProcessor


@dataclass
class FrameClassifierDataCollator:
    """Collator for :class:`FrameClassifierDataset` outputs.

    Expects each sample dict to contain:

    - ``images``: ``dict[cam_name, np.ndarray]`` of HWC uint8 raw frames.
      Missing cameras are permitted.
    - ``image_masks``: ``dict[cam_name, bool]`` mask values.
    - ``state``: ``np.ndarray`` (shape ``[D]``) when the dataset threads
      proprioceptive state; may be absent when ``use_proprio=False``.
    - ``label``: ``int`` in ``{0, 1}`` (0 = success ep, 1 = fail ep).
    - ``episode``, ``frame_index``, ``source_name``: bookkeeping fields,
      threaded for logging / debugging.

    Output batch dict:

    ``{"observation": {"images": ..., "image_masks": ..., "state": ...},``
    ``  "labels": Tensor[B] float32, "episode": ..., "frame_index": ...}``
    """

    processor: SuccessFailClassifierImageProcessor
    use_proprio: bool = True

    def _collect_per_camera(
        self,
        examples: list[dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Gather per-camera image tensors and masks for a batch.

        Handles missing cameras per-sample by replacing with zero
        placeholders at the raw HWC-uint8 stage; the processor then
        normalizes those to zero-mean tensors with the mask recording
        which samples lacked the view. Same logic as
        :class:`BinaryPairDataCollator._collect_per_camera` but without
        the pair axis.
        """
        camera_keys: set[str] = set()
        for ex in examples:
            camera_keys.update(ex["images"].keys())
        if not camera_keys:
            raise ValueError(
                "FrameClassifierDataCollator received a batch with no camera views."
            )

        bsize = len(examples)
        images_out: dict[str, torch.Tensor] = {}
        masks_out: dict[str, torch.Tensor] = {}
        for cam in sorted(camera_keys):
            frames: list[Optional[np.ndarray]] = []
            mask_vec: list[bool] = []
            shapes: list[tuple[int, ...]] = []
            for ex in examples:
                v = ex["images"].get(cam)
                if v is None:
                    frames.append(None)
                    mask_vec.append(False)
                else:
                    frames.append(v)
                    shapes.append(tuple(int(d) for d in v.shape))
                    mask_vec.append(bool(ex.get("image_masks", {}).get(cam, True)))

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 1:
                raise ValueError(
                    "FrameClassifierDataCollator saw incompatible raw image "
                    f"shapes for camera={cam!r}: {unique_shapes}. All samples "
                    "in a batch must share the same raw (H, W) per camera."
                )
            if unique_shapes:
                h, w = unique_shapes[0][:2]
            else:
                h, w = 1, 1

            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            stacked = torch.from_numpy(
                np.stack([f if f is not None else placeholder for f in frames])
            )
            images_out[cam] = stacked
            masks_out[cam] = torch.tensor(mask_vec, dtype=torch.bool)

        for cam, t in images_out.items():
            if t.shape[0] != bsize:
                raise RuntimeError(
                    f"Unexpected batch shape for cam={cam}: {tuple(t.shape)}"
                )
        return images_out, masks_out

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("FrameClassifierDataCollator received an empty batch")

        raw_images, raw_masks = self._collect_per_camera(examples)
        processed = self.processor(
            images=raw_images,
            image_masks=raw_masks,
        )

        observation: dict[str, Any] = {
            "images": processed["pixel_values"],
            "image_masks": processed["image_masks"],
        }

        if self.use_proprio:
            states_list = [ex.get("state") for ex in examples]
            missing = [i for i, s in enumerate(states_list) if s is None]
            if missing:
                raise ValueError(
                    "FrameClassifierDataCollator(use_proprio=True) received "
                    f"samples missing 'state': indices {missing[:5]}"
                )
            state_np = np.stack(
                [np.asarray(s, dtype=np.float32).reshape(-1) for s in states_list]
            )
            observation["state"] = torch.from_numpy(state_np)

        labels = torch.tensor(
            [int(ex["label"]) for ex in examples], dtype=torch.float32
        )
        batch: dict[str, Any] = {
            "observation": observation,
            "labels": labels,
        }
        if "episode" in examples[0]:
            batch["episode"] = torch.tensor(
                [int(ex.get("episode", -1)) for ex in examples], dtype=torch.long
            )
        if "frame_index" in examples[0]:
            batch["frame_index"] = torch.tensor(
                [int(ex.get("frame_index", -1)) for ex in examples], dtype=torch.long
            )
        return batch


__all__ = ["FrameClassifierDataCollator"]
