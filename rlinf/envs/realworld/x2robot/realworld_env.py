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

"""Chunk-granular RealWorldEnv for the x2robot TCP bridge.

``RealWorldEnv.chunk_step`` is step-granular: it calls ``step()`` once per
sub-action and expects a fresh observation each time.  The x2robot slave
uploads **one observation per chunk** on the inference socket and executes the
whole chunk itself, so this subclass performs exactly one TCP round trip per
``chunk_step``.

Takeover data comes from the *second*, always-on upload channel
(:class:`~rlinf.envs.realworld.x2robot.upload_server.UploadServer`).  The
inference socket is mode-gated -- the slave drops it the moment
``/running_mode`` leaves 1 -- so it can never carry a takeover frame.  The
uploader streams every 20 Hz tick regardless of mode; those records supply the
per-sub-step **executed action** and ``is_takeover`` flag that RLinf's
``update_last_actions`` / ``extract_intervene_traj`` need.

Note that HG-DAgger replaces the *action* and keeps the observation the policy
actually conditioned on, so per-frame observations are not required here; only
the executed 28-d pose and the takeover flag are.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.utils import to_tensor

logger = logging.getLogger(__name__)


class X2RobotTCPRealWorldEnv(RealWorldEnv):
    """One ``chunk_step`` == one obs-upload / action-chunk round trip."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        # Downstream indexes chunk_* with [:, -1] and obs_list[-1]; keeping the
        # per-chunk length canonical avoids surprising any consumer that assumes
        # len(obs_list) == num_action_chunks.  The slave uploads one observation
        # per chunk on the inference socket, so these entries repeat that frame
        # -- HG-DAgger trains on the action, not on these.
        self.expand_chunk_obs = bool(self.override_cfg.get("expand_chunk_obs", True))
        self._last_obs = None
        self._pending_success = None

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        obs, infos = super().reset(
            reset_state_ids=reset_state_ids, seed=seed, options=options, env_idx=env_idx
        )
        self._last_obs = obs
        self._pending_success = None
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError("x2robot_tcp is chunk-granular; use chunk_step().")

    # ------------------------------------------------------------------ core
    def chunk_step(self, chunk_actions):
        """chunk_actions: ``[num_envs, chunk_steps, action_dim]`` (num_envs == 1)."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        action_dim = chunk_actions.shape[2]

        # num_envs == 1 is asserted by RealWorldEnv.__init__.
        results = self.env.call("chunk_round_trip", chunk_actions[0])
        raw_obs, _reward, terminated, truncated, info = results[0]

        self._elapsed_steps += 1

        if raw_obs is None:
            # Slave disconnected: takeover started (mode != 1) or the episode
            # ended.  Reuse the last good observation so the trajectory closes
            # out cleanly rather than propagating a None.
            obs = self._last_obs
            truncated = True
        else:
            batched = {
                "state": {k: v[None] for k, v in raw_obs["state"].items()},
                "frames": {k: v[None] for k, v in raw_obs["frames"].items()},
            }
            obs = self._wrap_obs(batched)
            self._last_obs = obs

        intervene_action, intervene_flag, n_recs = self._collect_intervene(
            chunk_size, action_dim, chunk_actions[0]
        )

        # An episode_end marker from the slave (a / c / Ctrl+C on the recording
        # script) is authoritative for both the boundary and the success label.
        end = self.env.call("take_episode_end")[0]
        if end is not None:
            truncated = True
            success = end.get("success")
            self._pending_success = success
            logger.info("episode_end from slave: %s (upload records=%d)", end, n_recs)

        n = chunk_size if self.expand_chunk_obs else 1
        obs_list = [obs] * n

        chunk_rewards = torch.zeros((self.num_envs, n), dtype=torch.float32)
        chunk_terminations = torch.zeros((self.num_envs, n), dtype=torch.bool)
        chunk_truncations = torch.zeros((self.num_envs, n), dtype=torch.bool)
        chunk_terminations[:, -1] = bool(terminated)
        chunk_truncations[:, -1] = bool(truncated)

        success_np = np.zeros(self.num_envs, dtype=bool)
        if self._pending_success:
            success_np[:] = True
        step_reward = success_np.astype(np.float32)

        infos = self._record_metrics(
            step_reward,
            np.array([bool(terminated)]),
            success_np,
            intervene_flag.any(dim=-1).cpu().numpy(),
            dict(info),
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(success_np)
            chunk_terminations[:, -1] = False

        # Shapes required by EmbodiedRolloutResult.update_last_actions:
        #   intervene_action [bsz, chunk * action_dim]
        #   intervene_flag   [bsz, chunk]
        infos["intervene_action"] = intervene_action
        infos["intervene_flag"] = intervene_flag
        infos["n_upload_records"] = n_recs

        infos_list = [infos] * n
        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    # ------------------------------------------------------------- uploads
    def _collect_intervene(self, chunk_size, action_dim, policy_chunk):
        """Per-sub-step executed action + takeover flag from the upload channel.

        Returns ``(intervene_action [1, chunk*action_dim],
        intervene_flag [1, chunk], n_records)``.

        The slave executes ``move_steps`` (15) of the policy's ``chunk_size``
        (20) steps per round trip, so the record count rarely equals
        ``chunk_size``: pad by repeating the last record (keeping its flag, so a
        fully-taken-over chunk stays fully taken over for
        ``extract_intervene_traj(mode="all")``) and keep the most recent records
        when there are too many.
        """
        recs = self.env.call("drain_uploads", chunk_size)[0]
        n_recs = len(recs)

        flags = np.zeros((chunk_size,), dtype=bool)
        actions = np.asarray(policy_chunk, dtype=np.float32).copy()

        if n_recs:
            if n_recs > chunk_size:
                recs = recs[-chunk_size:]
            for i, rec in enumerate(recs):
                a = np.asarray(rec["action_28"], dtype=np.float32)
                actions[i, : min(action_dim, a.shape[0])] = a[:action_dim]
                flags[i] = bool(rec["is_takeover"])
            if len(recs) < chunk_size:
                actions[len(recs) :] = actions[len(recs) - 1]
                flags[len(recs) :] = flags[len(recs) - 1]

        intervene_action = torch.from_numpy(
            actions.reshape(1, chunk_size * action_dim)
        )
        intervene_flag = torch.from_numpy(flags.reshape(1, chunk_size))
        return intervene_action, intervene_flag, n_recs
