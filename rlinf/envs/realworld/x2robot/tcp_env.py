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

"""TCP bridge env for the x2robot (自变量) dual-arm slave.

The slave arm already talks a fixed wire protocol to an inference server
(``scripts/x2robot_infer_seq_qiuyi.py`` on the openpi side,
``toolkits/standalone_eval_scripts/openpi/x2robot_realworld_infer.py`` on the
RLinf side).  This module turns that server into a **gym env** so RLinf's
rollout / actor / env workers can drive the real robot without the slave
client changing a single byte.

Wire protocol (unchanged, every frame is ``4-byte little-endian length +
payload``)::

    slave -> bridge :  JSON  {"follow1_pos": (h+1,7), "follow2_pos": (h+1,7)}
                       JPEG  left wrist
                       JPEG  face / front
                       JPEG  right wrist
    bridge -> slave :  JSON  {"follow1_pos": (move_steps+1,7),
                              "follow2_pos": (move_steps+1,7)}

Granularity note
----------------
The protocol is **chunk-granular**: one uploaded observation buys one whole
action chunk, which the slave interpolates and executes on its own before
uploading the next observation.  RLinf's stock ``RealWorldEnv.chunk_step``
is step-granular (it calls ``step()`` once per sub-action and expects a fresh
observation each time), so this env is driven through
:class:`~rlinf.envs.realworld.x2robot.realworld_env.X2RobotTCPRealWorldEnv`,
which overrides ``chunk_step`` to perform exactly one TCP round trip per
chunk.  ``step()`` is intentionally unsupported here.

Threading
---------
A daemon thread owns the listening socket and the ``master_queue``; it blocks
on ``accept``/``recv``.  The gym API talks to it through two size-1 queues, so
the caller blocks on the robot rather than spinning.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import queue
import socket
import struct
import threading
from collections import deque
from typing import Any, Mapping

import cv2
import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.x2robot.upload_server import UploadServer

logger = logging.getLogger(__name__)

# Sentinel pushed onto the observation queue when the slave disconnects.
_DISCONNECTED = object()


@dataclasses.dataclass
class X2RobotTCPConfig:
    """Bridge configuration.

    Defaults mirror the deployed openpi server so the slave client connects
    with no changes.  ``state_history_size`` / ``state_future_size`` /
    ``state_step`` **must** match both the slave's command line and the
    checkpoint's data config, otherwise the assembled state is misaligned.
    """

    host: str = "0.0.0.0"
    port: int = 57770

    # State-sequence geometry; state_seq_len = history + 1 + future.
    state_history_size: int = 3
    state_future_size: int = 2
    state_step: int = 1

    # Latency compensation: drop this many leading steps of the predicted
    # chunk.  ``None`` -> state_future_size (the deployed default).
    latency_step: int | None = None
    # Steps actually returned to the slave per round trip.
    move_steps: int = 15

    policy_mode: str = "sm2sm"  # s2s / s2m / sm2m / sm2sm

    # Chunk-boundary smoothing (off by default, matching deployment).
    blend_steps: int = 0
    blend_skip_dims: tuple[int, ...] = (6, 13)

    # Camera frames as uploaded by the slave; used only to declare the
    # observation space.  Frames of a different size are resized to match.
    image_height: int = 480
    image_width: int = 640

    task_description: str = ""

    # Always-on upload channel (doc 1.2.5). The inference socket is mode-gated
    # and drops on takeover, so per-frame records -- rollout AND takeover --
    # arrive here instead. Disabled by default: v1 runs inference only.
    upload_enabled: bool = False
    upload_port: int = 57772
    upload_maxlen: int = 4000

    # Seconds to wait for the slave before reset()/round-trip gives up.
    # ``None`` waits forever (the usual choice on a real robot).
    connect_timeout: float | None = None

    def __post_init__(self):
        if self.latency_step is None:
            self.latency_step = self.state_future_size
        if self.policy_mode not in ("s2s", "s2m", "sm2m", "sm2sm"):
            raise ValueError(f"unsupported policy_mode {self.policy_mode!r}")
        self.blend_skip_dims = tuple(self.blend_skip_dims)

    @property
    def state_seq_len(self) -> int:
        return self.state_history_size + 1 + self.state_future_size

    @property
    def latency_len(self) -> int:
        return self.state_history_size + 1 + self.latency_step


# ══════════════════════════════════════════════════════════════════════════
# wire helpers  (byte-for-byte identical to the deployed inference server)
# ══════════════════════════════════════════════════════════════════════════
def _recv_all(sock: socket.socket, count: int) -> bytes | None:
    buf = b""
    while count:
        chunk = sock.recv(count)
        if not chunk:
            return None
        buf += chunk
        count -= len(chunk)
    return buf


def _read_size(conn: socket.socket) -> int:
    header = _recv_all(conn, 4)
    if header is None:
        raise ConnectionError("client disconnected")
    return struct.unpack("<L", header)[0]


def _read_img(conn: socket.socket) -> np.ndarray:
    size = _read_size(conn)
    payload = _recv_all(conn, size)
    if payload is None:
        raise ConnectionError("client disconnected during image payload")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ConnectionError("failed to decode JPEG payload")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _blend_chunk_transition(
    action_pred: np.ndarray,
    master_queue: deque,
    blend_steps: int,
    skip_dims: tuple[int, ...] = (),
) -> np.ndarray:
    """Smooth the chunk seam by fading from the old velocity extrapolation.

    ``action_pred[0]`` is the anchor (= ``master_queue[-1]``).  Gripper dims
    are passed through untouched so grasp timing is not delayed.
    """
    if blend_steps <= 0 or len(action_pred) < 2 or len(master_queue) < 2:
        return action_pred
    out = action_pred.copy()
    window = min(blend_steps, len(action_pred) - 1)
    anchor = np.asarray(action_pred[0], dtype=np.float64)
    velocity = anchor - np.asarray(master_queue[-2], dtype=np.float64)
    for i in range(1, window + 1):
        alpha = i / (window + 1)
        old_extrap = anchor + velocity * i
        blended = (1.0 - alpha) * old_extrap + alpha * np.asarray(
            action_pred[i], dtype=np.float64
        )
        for d in skip_dims:
            if 0 <= d < blended.shape[-1]:
                blended[d] = action_pred[i][d]
        out[i] = blended
    return out


# ══════════════════════════════════════════════════════════════════════════
# the env
# ══════════════════════════════════════════════════════════════════════════
class X2RobotTCPEnv(gym.Env):
    """Gym facade over the x2robot slave's TCP inference protocol."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: Mapping[str, Any] | None = None,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ):
        super().__init__()
        cfg_fields = {f.name for f in dataclasses.fields(X2RobotTCPConfig)}
        raw = dict(override_cfg or {})
        self.config = X2RobotTCPConfig(
            **{k: v for k, v in raw.items() if k in cfg_fields}
        )
        self.env_idx = env_idx

        self._init_spaces()

        # server thread <-> gym API
        self._obs_q: queue.Queue = queue.Queue(maxsize=1)
        self._chunk_q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        # Set once per TCP connection; reset() consumes it as the episode start.
        self._connected = threading.Event()
        self._server_error: BaseException | None = None

        self.upload: UploadServer | None = None
        if self.config.upload_enabled:
            self.upload = UploadServer(
                host=self.config.host,
                port=self.config.upload_port,
                image_height=self.config.image_height,
                image_width=self.config.image_width,
                maxlen=self.config.upload_maxlen,
            )
            self.upload.start()

        self._sock: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._serve, name=f"x2robot-tcp-{env_idx}", daemon=True
        )
        self._thread.start()

    # ---------------------------------------------------------------- spaces
    def _init_spaces(self) -> None:
        c = self.config
        # 28-d absolute end-effector pose (slave 14 + master 14); the chunk is
        # produced by the policy, so the box is only a declaration.
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        img = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c.image_height, c.image_width, 3),
            dtype=np.uint8,
        )
        # RealWorldEnv._wrap_obs concatenates sorted(state) on the last axis and
        # stacks sorted(frames) minus main_image_key.  A single state key keeps
        # the concat a no-op; the frame names sort to
        # [face_view, left_wrist_view, right_wrist_view] so popping face_view
        # (the configured main_image_key) leaves [left, right] -- exactly the
        # order obs_processor expects for extra_view_images.
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "state_seq": gym.spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(c.state_seq_len, 32),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "face_view": img,
                        "left_wrist_view": copy.deepcopy(img),
                        "right_wrist_view": copy.deepcopy(img),
                    }
                ),
            }
        )

    @property
    def task_description(self) -> str:
        return self.config.task_description

    # ---------------------------------------------------------------- server
    def _serve(self) -> None:
        c = self.config
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((c.host, c.port))
            self._sock.listen(1)
            logger.info("X2RobotTCPEnv listening on %s:%d", c.host, c.port)
        except Exception as exc:  # noqa: BLE001 - surfaced to the gym caller
            self._server_error = exc
            self._obs_q.put(_DISCONNECTED)
            return

        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("slave connected from %s", addr)

            # A fresh connection means a fresh episode: the slave clears its
            # action queue on reconnect, so carrying master history across
            # would condition the policy on actions that were never executed.
            master_queue: deque = deque(maxlen=100)
            self._connected.set()

            try:
                while not self._stop.is_set():
                    slave_state_raw, frames = self._read_frame(conn)
                    state = self._assemble_state(slave_state_raw, master_queue)
                    self._obs_q.put({"state_seq": state, "frames": frames})

                    chunk = self._chunk_q.get()  # (action_horizon, 28)
                    if chunk is None:  # shutting down
                        break
                    payload = self._postprocess(chunk, master_queue)
                    conn.sendall(struct.pack("<L", len(payload)))
                    conn.sendall(payload)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                logger.info("slave disconnected: %s", exc)
            except Exception as exc:  # noqa: BLE001
                self._server_error = exc
                logger.exception("X2RobotTCPEnv server loop failed")
            finally:
                self._connected.clear()
                # Wake a caller blocked in reset()/round trip.
                try:
                    self._obs_q.put_nowait(_DISCONNECTED)
                except queue.Full:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass

    def _read_frame(self, conn: socket.socket):
        size = _read_size(conn)
        payload = _recv_all(conn, size)
        if payload is None:
            raise ConnectionError("client disconnected during JSON payload")
        data = json.loads(payload.decode("utf8"))
        left = np.asarray(data["follow1_pos"], dtype=np.float32)  # (h+1, 7)
        right = np.asarray(data["follow2_pos"], dtype=np.float32)  # (h+1, 7)

        # Fixed upload order: left wrist, face, right wrist.
        img_left = self._fit(_read_img(conn))
        img_face = self._fit(_read_img(conn))
        img_right = self._fit(_read_img(conn))
        frames = {
            "face_view": img_face,
            "left_wrist_view": img_left,
            "right_wrist_view": img_right,
        }
        return (left, right), frames

    def _fit(self, image: np.ndarray) -> np.ndarray:
        c = self.config
        if image.shape[0] != c.image_height or image.shape[1] != c.image_width:
            image = cv2.resize(
                image, (c.image_width, c.image_height), interpolation=cv2.INTER_AREA
            )
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _assemble_state(self, slave_state_raw, master_queue: deque) -> np.ndarray:
        """Build the ``(state_seq_len, 32)`` state exactly like the deployed server."""
        c = self.config
        left, right = slave_state_raw
        state = np.zeros((c.state_seq_len, 32), dtype=np.float32)

        slave_state = np.concatenate([left, right], axis=1)  # (h+1, 14)
        # Pad the future slots by repeating the current pose.
        slave_state = np.concatenate(
            [slave_state] + [slave_state[-1:]] * c.state_future_size
        )

        if not master_queue:
            # First frame of a connection: seed with the current pose so the
            # master channel is well defined before anything was commanded.
            master_queue.extend(
                [slave_state[-1]] * max(c.state_seq_len, c.latency_len)
            )

        master_list = list(master_queue)[-c.latency_len :]
        if c.latency_step < c.state_future_size:  # inpainting mode
            master_list = master_list + [master_list[-1]] * (
                c.state_future_size - c.latency_step
            )
            # Column 31 is the inpainting mask marking the to-be-filled tail.
            state[c.latency_step - c.state_future_size :, -1] = 1.0
        else:  # naive async
            master_list = master_list[: c.state_seq_len]
        master_state = np.asarray(master_list, dtype=np.float32)

        if c.policy_mode in ("s2s", "s2m"):
            state[:, :14] = slave_state
        else:
            state[:, :28] = np.concatenate([slave_state, master_state], axis=1)
        return state

    def _postprocess(self, chunk: np.ndarray, master_queue: deque) -> bytes:
        """(action_horizon, 28) policy chunk -> JSON payload for the slave."""
        c = self.config
        action_pred = np.asarray(chunk, dtype=np.float64)
        if action_pred.ndim != 2 or action_pred.shape[-1] < 28:
            raise ValueError(
                f"expected chunk of shape (T, 28), got {action_pred.shape}"
            )

        if c.policy_mode == "sm2sm":
            action_pred = action_pred[:, 14:28]  # master half is what we command
        elif c.policy_mode in ("s2m", "sm2m"):
            action_pred = action_pred[:, 14:28]
        else:  # s2s -> the slave half is the command
            action_pred = action_pred[:, :14]

        action_pred = action_pred[c.latency_step :]
        action_pred = action_pred[: c.move_steps, ...]
        # Prepend the anchor so the slave can interpolate continuously; it
        # drops this element right after interpolating.
        action_pred = np.concatenate([[np.asarray(master_queue[-1])], action_pred])
        if c.blend_steps > 0:
            action_pred = _blend_chunk_transition(
                action_pred, master_queue, c.blend_steps, skip_dims=c.blend_skip_dims
            )
        for action in action_pred[1:]:
            master_queue.append(action)

        data = {
            "follow1_pos": action_pred[:, :7].tolist(),
            "follow2_pos": action_pred[:, 7:].tolist(),
        }
        return json.dumps(data).encode("utf-8")

    # ------------------------------------------------------------- gym API
    def _next_obs(self, timeout: float | None):
        item = self._obs_q.get(timeout=timeout)
        if item is _DISCONNECTED:
            if self._server_error is not None:
                raise RuntimeError(
                    f"X2RobotTCPEnv server failed: {self._server_error}"
                ) from self._server_error
            return None
        return item

    @staticmethod
    def _to_raw_obs(item) -> dict:
        return {
            "state": {"state_seq": item["state_seq"]},
            "frames": item["frames"],
        }

    def reset(self, *, seed=None, options=None):
        """Block until the slave is connected and has uploaded a frame."""
        super().reset(seed=seed)
        # Drain anything stale from a previous connection.
        while True:
            try:
                self._obs_q.get_nowait()
            except queue.Empty:
                break
        item = self._next_obs(self.config.connect_timeout)
        while item is None:
            item = self._next_obs(self.config.connect_timeout)
        return self._to_raw_obs(item), {}

    def chunk_round_trip(self, chunk: np.ndarray):
        """Send one action chunk, then block for the next uploaded observation.

        Returns ``(raw_obs, reward, terminated, truncated, info)``.  A slave
        disconnect ends the episode as a truncation and returns the last
        observation unchanged, so the caller can close the trajectory out.
        """
        self._chunk_q.put(np.asarray(chunk))
        item = self._next_obs(self.config.connect_timeout)
        if item is None:
            # Disconnect: the slave either took over (mode != 1) or the
            # recording ended.  Episode boundary; observation is unavailable.
            return None, 0.0, False, True, {"disconnected": True}
        return self._to_raw_obs(item), 0.0, False, False, {}

    def step(self, action):
        raise NotImplementedError(
            "X2RobotTCPEnv is chunk-granular: the slave uploads one observation "
            "per action chunk. Drive it through X2RobotTCPRealWorldEnv.chunk_step "
            "(env_type: x2robot_tcp) instead of per-step step()."
        )

    def drain_uploads(self, n=None):
        """Per-frame records buffered since the last call (empty if disabled)."""
        return self.upload.drain(n) if self.upload is not None else []

    def take_episode_end(self):
        """Episode-end marker from the slave, or None."""
        return self.upload.take_episode_end() if self.upload is not None else None

    def close(self):
        self._stop.set()
        if self.upload is not None:
            self.upload.close()
        try:
            self._chunk_q.put_nowait(None)
        except queue.Full:
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
