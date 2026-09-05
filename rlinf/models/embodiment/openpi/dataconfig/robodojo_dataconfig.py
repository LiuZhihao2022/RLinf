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
"""Data pipeline for RoboDojo's dual-arm ARX X5 joint-control datasets.

Mirrors ``LeRobotRoboDojoArxX5DataConfig`` in the openpi checkout so BC and
CFGRL see identical inputs — the two must stay in sync.

Differences from :class:`LeRobotX2robotDataConfig` (the real-robot pipeline):

* **Joint space, 14 dims** — RoboDojo scores policies on joint targets, so
  state and action are ``left arm(6) + left gripper(1) + right arm(6) +
  right gripper(1)``, not the 28-dim end-effector + master layout.
* **No master arms, no state history** — the benchmark's observation is a
  single current frame, so there is nothing to stack and no leader/follower
  split to drop.
* Datasets converted by ``convert_robodojo_dagger_data_to_lerobot_v1.py`` also
  carry ``observation.ee_pose``; the repack below simply does not select it, so
  it rides along for analysis without entering the model.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import (
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
)
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import robodojo_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRoboDojoArxX5DataConfig(DataConfigFactory):
    """LeRobot -> Pi pipeline for RoboDojo ARX X5 (joint space, 14 dims)."""

    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    action_dim: int = 14

    repack_transforms: _transforms.Group = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[robodojo_policy.RoboDojoArxX5Inputs()],
            outputs=[robodojo_policy.RoboDojoArxX5Outputs()],
        )
        if self.use_delta_joint_actions:
            # Six delta arm joints then one absolute gripper, per arm. The
            # gripper is normalized [0, 1] and effectively binary, so a delta
            # on it would be meaningless.
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(
                default_prompt=self.default_prompt
            )(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )
