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

"""Receiver for the x2robot slave's always-on trajectory upload channel.

The inference socket is mode-gated -- the slave drops it the instant
``/running_mode`` leaves 1, so it never carries a takeover frame.  The slave's
``dagger_uploader.DaggerUploader`` therefore opens a second, one-way socket
that survives mode transitions and streams every 20 Hz tick of the episode.
This class is the other end of it.

Records land in a bounded deque; :class:`~rlinf.envs.realworld.x2robot.tcp_env.X2RobotTCPEnv`
drains them per chunk so the trajectory handed to the actor contains the real
per-frame observations and executed actions, each tagged ``is_takeover``.

Wire format -- ``4-byte little-endian length + payload`` throughout::

    hello        JSON  {"type":"hello","proto":1,...}
    step         JSON  {"type":"step", seq,t,mode,is_takeover,
                        follow1_pos[7],follow2_pos[7],
                        master1_pos[7],master2_pos[7]}
                 JPEG  left wrist / face / right wrist   (may be zero-length)
    episode_end  JSON  {"type":"episode_end","success":bool|null,"n":int}
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
from collections import deque
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROTO = 1


def _recv_all(sock: socket.socket, count: int) -> bytes | None:
    buf = b""
    while count:
        chunk = sock.recv(count)
        if not chunk:
            return None
        buf += chunk
        count -= len(chunk)
    return buf


def _read_frame(conn: socket.socket) -> bytes:
    header = _recv_all(conn, 4)
    if header is None:
        raise ConnectionError("uploader disconnected")
    size = struct.unpack("<L", header)[0]
    if size == 0:
        return b""
    payload = _recv_all(conn, size)
    if payload is None:
        raise ConnectionError("uploader disconnected mid-payload")
    return payload


class UploadServer:
    """Background TCP server collecting per-frame records from the slave."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 57772,
        image_height: int = 480,
        image_width: int = 640,
        maxlen: int = 4000,
        decode_images: bool = True,
    ):
        self.host = host
        self.port = port
        self.image_height = image_height
        self.image_width = image_width
        self.decode_images = decode_images

        self._records: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._episode_end: dict[str, Any] | None = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._serve, name="x2robot-upload", daemon=True
        )
        self._n_recv = 0

    # ------------------------------------------------------------------ api
    def start(self) -> None:
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def drain(self, n: int | None = None) -> list[dict]:
        """Pop up to ``n`` buffered records (all of them when ``n`` is None)."""
        with self._lock:
            if n is None or n >= len(self._records):
                out = list(self._records)
                self._records.clear()
            else:
                out = [self._records.popleft() for _ in range(n)]
        return out

    def pending(self) -> int:
        with self._lock:
            return len(self._records)

    def take_episode_end(self) -> dict[str, Any] | None:
        """Consume the episode-end marker if the slave sent one."""
        with self._lock:
            end, self._episode_end = self._episode_end, None
        return end

    def stats(self) -> dict[str, Any]:
        return {
            "received": self._n_recv,
            "buffered": self.pending(),
            "connected": self.connected,
        }

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # --------------------------------------------------------------- server
    def _serve(self) -> None:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.listen(1)
            logger.info("upload channel listening on %s:%d", self.host, self.port)
        except Exception:
            logger.exception("upload channel failed to bind")
            return

        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("uploader connected from %s", addr)
            self._connected.set()
            try:
                self._session(conn)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                logger.info("uploader disconnected: %s", exc)
            except Exception:
                logger.exception("upload session failed")
            finally:
                self._connected.clear()
                try:
                    conn.close()
                except OSError:
                    pass

    def _session(self, conn: socket.socket) -> None:
        while not self._stop.is_set():
            header = json.loads(_read_frame(conn).decode("utf8"))
            kind = header.get("type")

            if kind == "hello":
                if header.get("proto") != PROTO:
                    logger.warning(
                        "uploader protocol %s != expected %s",
                        header.get("proto"),
                        PROTO,
                    )
                logger.info("uploader hello: %s", header)
                continue

            if kind == "episode_end":
                with self._lock:
                    self._episode_end = header
                logger.info("episode_end: %s", header)
                continue

            if kind != "step":
                logger.warning("unknown upload record type %r", kind)
                continue

            # Fixed 3 image frames follow every step record.
            blobs = [_read_frame(conn) for _ in range(3)]
            rec = self._build_record(header, blobs)
            with self._lock:
                self._records.append(rec)
            self._n_recv += 1

    def _build_record(self, header: dict, blobs: list[bytes]) -> dict:
        # slave 14 + master 14 = the 28-d absolute pose the checkpoints use.
        slave = np.asarray(
            header["follow1_pos"] + header["follow2_pos"], dtype=np.float32
        )
        master = np.asarray(
            header["master1_pos"] + header["master2_pos"], dtype=np.float32
        )
        rec = {
            "seq": int(header.get("seq", -1)),
            "t": float(header.get("t", 0.0)),
            "mode": int(header.get("mode", 1)),
            "is_takeover": bool(header.get("is_takeover", False)),
            "slave_state": slave,  # (14,)
            "action_28": np.concatenate([slave, master]),  # (28,)
        }
        if self.decode_images:
            names = ("left_wrist_view", "face_view", "right_wrist_view")
            frames = {}
            for name, blob in zip(names, blobs):
                frames[name] = self._decode(blob)
            rec["frames"] = frames
        return rec

    def _decode(self, blob: bytes) -> np.ndarray:
        if not blob:
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_height or img.shape[1] != self.image_width:
            img = cv2.resize(
                img,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(img, dtype=np.uint8)
