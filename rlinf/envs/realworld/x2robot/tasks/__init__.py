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

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.x2robot.tcp_env import X2RobotTCPEnv


def create_x2robot_tcp_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    # No Franka wrappers here: the takeover flag comes from the slave over TCP
    # rather than from a SpacemouseIntervention wrapper, and the pose frame /
    # rotation conventions are already what the checkpoint was trained on.
    return X2RobotTCPEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )


register(
    id="X2RobotTCPEnv-v1",
    entry_point="rlinf.envs.realworld.x2robot.tasks:create_x2robot_tcp_env",
)
