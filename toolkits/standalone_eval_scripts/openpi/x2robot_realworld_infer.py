"""Standalone TCP inference server: serve RLinf's CFG pi0 model
(``OpenPi0ForCFGActionPrediction``) over the x2robot socket protocol so it is a
drop-in replacement for openpi's ``scripts/x2robot_infer_seq_qiuyi.py``.

The wire protocol is byte-for-byte identical to that script, so the 从臂's
``socket2ros`` client needs no changes:

  recv (per step):
      [4B LE len][JSON  {"follow1_pos": (h+1, 7), "follow2_pos": (h+1, 7)}]
      [4B LE len][JPEG bytes]   # image1 = left  wrist
      [4B LE len][JPEG bytes]   # image2 = front / face
      [4B LE len][JPEG bytes]   # image3 = right wrist
  send (per step):
      [4B LE len][JSON  {"follow1_pos": (move+1, 7), "follow2_pos": (move+1, 7)}]

The only thing swapped vs. the openpi script is the inference call: instead of an
openpi ``Policy``/two-pass ``_cfg_infer``, we load the CFG model through RLinf's
``openpi_cfg`` stack (``OpenPi0ForCFGActionPrediction``) and call
``predict_action_batch(env_obs, mode="eval")``. State assembly, master_queue,
latency truncation and chunk-boundary blending are copied verbatim so behaviour
matches the deployed openpi server.

Run (inside the RLinf uv env):
    cd /home/user/qiuyi_projects/RLinf_active
    PYTHONPATH=$PWD .venv/bin/python \
        toolkits/standalone_eval_scripts/openpi/x2robot_realworld_infer.py \
        --host 192.168.120.153 --port 57770 --prompt "put the gift into the box"
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import os
import socket
import struct
from collections import deque
from typing import Any

import cv2
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class Args:
    # --- checkpoint / config ---
    model_path: str = (
        "/home/user/qiuyi_projects/openpi/checkpoints/open_giftbox/"
        "open_giftbox_steam_dagger/global_step_30000"
    )
    config_name: str = "open_giftbox_sm2sm"
    # assets/ subdir name inside model_path (underscore form; the config's repo_id
    # uses a comma form that does NOT match the on-disk dir, so we override it).
    asset_id: str = (
        "open_giftbox_xpc06230626062706280629_gqy0629_sby06290630_steam_sft_"
        "open_giftbox_gqy07010702_steam_dagger"
    )
    policy_mode: str = "sm2sm"  # s2s | s2m | sm2m | sm2sm

    # --- network ---
    host: str = "192.168.120.153"
    port: int = 57770

    # --- obs / action windowing (must match training config) ---
    prompt: str = "put the gift into the box"
    state_history_size: int = 3
    state_future_size: int = 2
    state_step: int = 1
    latency_step: int | None = None  # None -> state_future_size
    move_steps: int = 15
    only_right_arm: bool = False
    blend_steps: int = 0
    blend_skip_dims: tuple[int, ...] = (6, 13)  # per-arm gripper dims

    # --- CFG guidance (match the deployed openpi command:
    #     --cfg-enable --cfg-guidance-scale 2.5 --cfg-guidance-type positive) ---
    cfgrl_guidance_scale: float = 2.5
    guidance_type: str = "positive"
    positive_only_conditional: bool = True
    unconditional_prob: float = 0.1
    num_steps: int = 10

    device: str = "cuda"


# ═══════════════════════════════════════════════════════════════════════════
# Model loader — mirrors rlinf.models.embodiment.openpi_cfg.get_model but fixes
# three things get_model gets wrong for this deployment checkpoint:
#   1. passes the explicit underscore-form asset_id (config repo_id is comma form)
#   2. loads norm_stats from  <ckpt>/assets/<asset_id>  (get_model drops "assets/")
#   3. zeroes the train-time augmentations (random_pos_offset unconditionally
#      touches inputs["actions"], which is absent at inference -> KeyError)
# Validated to return actions of shape [B, 20, 28].
# ═══════════════════════════════════════════════════════════════════════════
def load_cfg_model(args: Args):
    import safetensors.torch
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    infer_data_kwargs = {
        "random_pos_offset": 0.0,
        "random_drop_master": 0.0,
        "random_drop_history": 0.0,
        "random_drop_future": 0.0,
    }
    tc = get_openpi_config(
        args.config_name,
        model_path=args.model_path,
        asset_id=args.asset_id,
        data_kwargs=infer_data_kwargs,
    )
    mc = OpenPi0Config(**tc.model.__dict__)
    overrides = dict(
        config_name=args.config_name,
        num_images_in_input=3,
        action_chunk=20,
        action_env_dim=28,
        num_steps=args.num_steps,
        train_expert_only=False,
        cfgrl_guidance_scale=args.cfgrl_guidance_scale,
        unconditional_prob=args.unconditional_prob,
        guidance_type=args.guidance_type,
        positive_only_conditional=args.positive_only_conditional,
    )
    for k, v in overrides.items():
        mc.__dict__[k] = v  # frozen dataclass: bypass via __dict__ (same as get_model)

    model = OpenPi0ForCFGActionPrediction(mc)
    weight_paths = sorted(glob.glob(os.path.join(args.model_path, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(args.model_path, "model.safetensors")]
    for wp in weight_paths:
        safetensors.torch.load_model(model, wp, strict=False)
    logging.info("loaded %d safetensors shard(s)", len(weight_paths))

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = tc.data.create(tc.assets_dirs, mc)
    norm_stats = _checkpoints.load_norm_stats(
        os.path.join(args.model_path, "assets"), data_config.asset_id
    )
    if norm_stats is None:
        raise RuntimeError(f"norm_stats not found for asset_id={data_config.asset_id}")

    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
    )
    model = model.to(args.device).eval()
    logging.info(
        "CFG model ready: guidance=%s scale=%s steps=%d",
        mc.guidance_type, mc.cfgrl_guidance_scale, mc.num_steps,
    )
    return model, norm_stats


# ═══════════════════════════════════════════════════════════════════════════
# Wire protocol helpers (copied verbatim from x2robot_infer_seq_qiuyi.py)
# ═══════════════════════════════════════════════════════════════════════════
def recv_all(sock: socket.socket, count: int):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def read_size(conn: socket.socket) -> int:
    header = recv_all(conn, 4)
    if header is None:
        raise ConnectionError("client disconnected")
    return struct.unpack("<L", header)[0]


def read_img(conn: socket.socket) -> np.ndarray:
    image_size = read_size(conn)
    image = recv_all(conn, image_size)
    if image is None:
        raise ConnectionError("client disconnected during image payload")
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _blend_chunk_transition(
    action_pred: np.ndarray,
    master_queue: deque,
    blend_steps: int,
    skip_dims: tuple[int, ...] = (),
) -> np.ndarray:
    """Smooth the chunk-boundary discontinuity (see openpi script for rationale).

    action_pred is (move_steps+1, 14) with action_pred[0] = anchor = master_queue[-1].
    """
    if blend_steps <= 0 or len(master_queue) < 2 or action_pred.shape[0] < 2:
        return action_pred
    W = min(blend_steps, action_pred.shape[0] - 1)
    out = action_pred.astype(np.float64).copy()
    anchor = out[0]
    prev = np.asarray(master_queue[-2], dtype=np.float64)
    velocity = anchor - prev
    skip_idx = list(skip_dims) if skip_dims else None
    for i in range(1, W + 1):
        alpha = i / (W + 1)
        old_extrap = anchor + velocity * i
        blended = (1.0 - alpha) * old_extrap + alpha * out[i]
        if skip_idx:
            blended[skip_idx] = out[i, skip_idx]
        out[i] = blended
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RLinf inference wrapper — replaces openpi policy.infer / _cfg_infer.
# Returns the full predicted chunk (action_horizon, 28).
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def rlinf_infer(model, camera_left, camera_front, camera_right, state, prompt) -> np.ndarray:
    env_obs = {
        "main_images": camera_front[None],  # face_view -> main [1, H, W, C]
        "extra_view_images": np.stack([camera_left, camera_right], axis=0)[None],  # [1,2,H,W,C]
        "wrist_images": None,
        "states": state[None].astype(np.float32),  # [1, state_seq_len, 32]
        "task_descriptions": [prompt],
    }
    actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
    return np.asarray(actions)[0]  # (action_horizon, 28)


# ═══════════════════════════════════════════════════════════════════════════
# Main serve loop (state assembly + post-processing copied from openpi script)
# ═══════════════════════════════════════════════════════════════════════════
def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.latency_step is None:
        args.latency_step = args.state_future_size

    model, norm_stats = load_cfg_model(args)

    state_seq_len = args.state_history_size + 1 + args.state_future_size
    latency_len = args.state_history_size + 1 + args.latency_step

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    logging.info("RLinf CFG inference server listening on %s:%d", args.host, args.port)

    try:
        while True:  # outer accept loop: survives takeover disconnect/reconnect
            conn, addr = sock.accept()
            logging.info("Connection from %s", addr)
            master_queue: deque = deque(maxlen=100)  # reset per (re)connect
            try:
                while True:
                    data_size = read_size(conn)
                    data = recv_all(conn, data_size)
                    if data is None:
                        raise ConnectionError("client disconnected during payload")
                    action_data = json.loads(data.decode("utf8"))

                    left_agent_data = np.asarray(action_data["follow1_pos"])   # (h+1, 7)
                    right_agent_data = np.asarray(action_data["follow2_pos"])  # (h+1, 7)

                    image1 = read_img(conn)  # left
                    image2 = read_img(conn)  # front / face
                    image3 = read_img(conn)  # right
                    h, w, c = np.array(image1).shape
                    camera_front = np.array(image2).reshape(h, w, c)
                    camera_left = np.array(image1).reshape(h, w, c)
                    camera_right = np.array(image3).reshape(h, w, c)

                    # ── build (state_seq_len, 32) state exactly like the openpi server ──
                    state = np.zeros((state_seq_len, 32), dtype=np.float32)
                    slave_state = np.concatenate([left_agent_data, right_agent_data], axis=1)  # (h+1,14)
                    slave_state = np.concatenate(
                        [slave_state] + [slave_state[-1:]] * args.state_future_size
                    )

                    if not master_queue:
                        master_queue.extend([slave_state[-1]] * max(state_seq_len, latency_len))

                    master_list = list(master_queue)[-latency_len:]
                    if args.latency_step < args.state_future_size:  # inpainting mode
                        master_list = master_list + [master_list[-1]] * (
                            args.state_future_size - args.latency_step
                        )
                        state[args.latency_step - args.state_future_size:, -1] = 1.0
                    else:  # naive async
                        master_list = master_list[:state_seq_len]
                    master_state = np.array(master_list)

                    if args.policy_mode in ["s2s", "s2m"]:
                        state[:, :14] = slave_state
                    else:
                        state[:, :28] = np.concatenate([slave_state, master_state], axis=1)

                    if args.only_right_arm:
                        mean = np.asarray(norm_stats["state"].mean)
                        state[:, 0:7] = mean[..., 0:7]
                        if args.policy_mode in ["sm2m", "sm2sm"]:
                            state[:, 14:21] = mean[..., 14:21]

                    # ── RLinf CFG inference ──
                    action_pred = rlinf_infer(
                        model, camera_left, camera_front, camera_right, state, args.prompt
                    )  # (action_horizon, 28)

                    if args.policy_mode == "sm2sm":
                        action_pred = action_pred[:, 14:28]  # master action (14-d)

                    # ── latency truncation + chunk-boundary blend + queue update ──
                    action_pred = action_pred[args.latency_step:]
                    action_pred = action_pred[: args.move_steps, ...]  # (move_steps, 14)
                    action_pred = np.concatenate([[master_queue[-1]], action_pred])
                    if args.blend_steps > 0:
                        action_pred = _blend_chunk_transition(
                            action_pred, master_queue, args.blend_steps,
                            skip_dims=args.blend_skip_dims,
                        )
                    for action in action_pred[1:]:
                        master_queue.append(action)

                    # ── response: split 14-d master action into two 7-d arms ──
                    data_dir = {
                        "follow1_pos": action_pred[:, :7].tolist(),
                        "follow2_pos": action_pred[:, 7:].tolist(),
                    }
                    data_bytes = json.dumps(data_dir).encode("utf-8")
                    conn.sendall(struct.pack("<L", len(data_bytes)))
                    conn.sendall(data_bytes)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                logging.info("Client disconnected: %s. Waiting for next connection.", exc)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    finally:
        sock.close()


def _parse_args() -> Args:
    d = Args()
    p = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(Args):
        default = getattr(d, f.name)
        if f.name == "blend_skip_dims":
            p.add_argument("--blend-skip-dims", type=int, nargs="*", default=list(default))
        elif isinstance(default, bool):
            p.add_argument(f"--{f.name.replace('_', '-')}", action="store_true", default=default)
        else:
            typ = type(default) if default is not None else str
            p.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=default)
    ns = p.parse_args()
    kwargs = {f.name: getattr(ns, f.name) for f in dataclasses.fields(Args)}
    kwargs["blend_skip_dims"] = tuple(kwargs["blend_skip_dims"])
    return Args(**kwargs)


if __name__ == "__main__":
    main(_parse_args())
