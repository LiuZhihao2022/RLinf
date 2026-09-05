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
#
# Mirrored from openpi/src/openpi/policies/robodojo_policy.py so the RLinf-side
# CFGRL / value pipelines can import it without depending on that checkout.
# Keep the two copies in sync.
"""Policy-side transforms for RoboDojo's dual-arm ARX X5 contract.

RoboDojo represents both state and action as 14 values in this order:

    left arm joints (6), left gripper (1),
    right arm joints (6), right gripper (1)

This module only adapts field names, image layout, and model padding. Delta
joint actions and normalization belong to the training configuration pipeline,
not to this robot-interface transform.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms

STATE_DIM = 14


def make_robodojo_arx_x5_example() -> dict:
    """Create one correctly shaped example for tests and smoke checks."""
    return {
        "state": np.zeros((STATE_DIM,), dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 480, 640), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 480, 640), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 480, 640), dtype=np.uint8),
        },
        "prompt": "Stack the three bowls together.",
    }


def _robot_vector(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or array.shape[-1] != STATE_DIM:
        expected = "[14]" if ndim == 1 else "[horizon, 14]"
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _image_hwc_uint8(value: object, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D image, got {image.shape}")

    if image.shape[-1] == 3:
        image_hwc = image
    elif image.shape[0] == 3:
        image_hwc = einops.rearrange(image, "c h w -> h w c")
    else:
        raise ValueError(f"{name} must be HWC or CHW with 3 channels, got {image.shape}")

    if np.issubdtype(image_hwc.dtype, np.floating):
        if not np.isfinite(image_hwc).all():
            raise ValueError(f"{name} contains non-finite values")
        if image_hwc.size and (image_hwc.min() < 0.0 or image_hwc.max() > 1.0):
            raise ValueError(f"Floating {name} must be in [0, 1]")
        image_hwc = np.rint(image_hwc * 255.0).astype(np.uint8)
    elif image_hwc.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 or floating [0, 1], got {image_hwc.dtype}")

    return np.ascontiguousarray(image_hwc)


@dataclasses.dataclass(frozen=True)
class RoboDojoArxX5Inputs(transforms.DataTransformFn):
    """Map one RoboDojo/LeRobot sample to the common Pi model interface."""

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        if "state" not in data:
            raise KeyError("RoboDojo input is missing 'state'")
        if "images" not in data or not isinstance(data["images"], dict):
            raise KeyError("RoboDojo input is missing the 'images' mapping")

        unknown_cameras = set(data["images"]) - set(self.EXPECTED_CAMERAS)
        missing_cameras = set(self.EXPECTED_CAMERAS) - set(data["images"])
        if unknown_cameras or missing_cameras:
            raise ValueError(
                "RoboDojo cameras do not match the contract: "
                f"missing={sorted(missing_cameras)}, unknown={sorted(unknown_cameras)}"
            )

        images = {
            name: _image_hwc_uint8(data["images"][name], name=name) for name in self.EXPECTED_CAMERAS
        }
        result = {
            "image": {
                "base_0_rgb": images["cam_high"],
                "left_wrist_0_rgb": images["cam_left_wrist"],
                "right_wrist_0_rgb": images["cam_right_wrist"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": _robot_vector(data["state"], name="state", ndim=1),
        }

        # Training samples contain an action chunk; inference observations do not.
        if "actions" in data:
            result["actions"] = _robot_vector(data["actions"], name="actions", ndim=2)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            result["actions_is_pad"] = data["actions_is_pad"]
        return result


@dataclasses.dataclass(frozen=True)
class RoboDojoArxX5Outputs(transforms.DataTransformFn):
    """Remove Pi model padding and return the 14 RoboDojo action dimensions."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            raise KeyError("Model output is missing 'actions'")
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < STATE_DIM:
            raise ValueError(f"Model actions must have shape [horizon, >=14], got {actions.shape}")
        return {"actions": actions[:, :STATE_DIM]}
