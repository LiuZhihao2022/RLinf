"""Render per-episode mp4 showing subtask + takeover + advantage together.

Layout per frame:
  * face_view video
  * overlay: current subtask (id + name), red TAKEOVER flash
  * bottom strip: the whole advantage curve (with subtask bands, threshold and
    takeover marks) drawn once, plus a moving playhead

Two schemas:
  --schema start : read `subtask_start_frames` from meta/episodes.jsonl
                   (current schema; None => sequence truncated)
  --schema end   : reconstruct the OLD `subtask_end_frames` from its source,
                   huzhipeng/.../dagger_subtask_segmentation/results/*.json
                   (that key was popped from episodes.jsonl on 07-16)

`--auto-good` (end schema) selects only the episodes Gemini labelled cleanly:
no None, coverage >= 0.9, no zero-length subtask.
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEROBOT = "/mnt/public/guqiuyi/huggingface/lerobot/open_giftbox_gqy07010702_steam_dagger"
TAG = "steam_k32_ensemble3_ckpt16000_wco_exp_and_rollout"
SEG = "/mnt/public/huzhipeng/stage_segmentation/dagger_subtask_segmentation/results"
FPS = 20
CW, CH = 640, 480
SH = 150            # advantage strip height
FLASH = 20
FONT = cv2.FONT_HERSHEY_SIMPLEX

PALETTE = [  # BGR, 11 subtasks
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
    (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23),
    (170, 170, 90),
]


# ---------------- schema handling ----------------
def end_frames_from_source(idx, length):
    """OLD end-key: clamp(round(end_sec*FPS)), None when observed=false."""
    stages = json.load(open(f"{SEG}/episode_{idx:06d}.json"))["stages"]
    out, prev = [], 0
    for st in stages:
        et = st.get("end_sec")
        if not st.get("observed") or et is None:
            out.append(None)
            continue
        ef = max(prev, min(round(float(et) * FPS), length - 1))
        out.append(ef)
        prev = ef
    return out


def spans_from_end(ends, length):
    """(start, end, idx, exact); unknown boundaries merge into the next known one."""
    spans, prev, pending = [], 0, []
    for i, e in enumerate(ends):
        if e is None:
            pending.append(i)
            continue
        spans.append((prev, e, (pending + [i])[0], not pending))
        prev, pending = e, []
    if pending:
        spans.append((prev, length, pending[0], False))
    return [s for s in spans if s[1] > s[0]]


def spans_from_start(starts, length):
    """(start, end, idx, exact); None => truncated, last observed span inexact."""
    obs = [(i, s) for i, s in enumerate(starts) if s is not None]
    trunc = len(obs) < len(starts)
    spans = []
    for j, (i, s) in enumerate(obs):
        e = obs[j + 1][1] if j + 1 < len(obs) else length
        spans.append((s, e, i, not (trunc and j == len(obs) - 1)))
    return [sp for sp in spans if sp[1] > sp[0]]


def end_key_is_clean(ends, length):
    if any(x is None for x in ends):
        return False
    known = [x for x in ends if x is not None]
    if max(known) / length < 0.9:
        return False
    prev = 0
    for x in ends:
        if x <= prev:
            return False
        prev = x
    return True


# ---------------- advantage strip ----------------
def make_strip(a, spans, tks, thr, width=CW, height=SH):
    """Render the whole advantage curve once -> BGR image; x maps linearly to frames."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    n = len(a)
    x = np.arange(n)
    for si, (s, e, i, exact) in enumerate(spans):
        ax.axvspan(s, e, color=np.array(PALETTE[i % 11][::-1]) / 255.0,
                   alpha=0.18 if exact else 0.08, lw=0)
    sm = pd.Series(a).rolling(20, center=True, min_periods=1).mean().to_numpy()
    ax.plot(x, a, lw=0.4, color="0.6")
    ax.plot(x, sm, lw=1.4, color="k")
    ax.axhline(0, color="0.35", lw=0.7, ls="--")
    if thr is not None:
        ax.axhline(thr, color="green", lw=0.8, ls=":")
    for t in tks:
        ax.axvline(t, color="red", lw=1.4)
    ax.set_xlim(0, max(1, n))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def render(idx, meta, spans, a, thr, lerobot, out_path, names):
    tks = meta["takeover_frames"]
    length = meta["length"]
    strip = make_strip(a, spans, tks, thr)
    cap = cv2.VideoCapture(f"{lerobot}/videos/chunk-000/face_view/episode_{idx:06d}.mp4")
    out_h = CH + 26 + SH
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CW, out_h))
    span_of = lambda f: next((s for s in spans if s[0] <= f < s[1]), None)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.zeros((out_h, CW, 3), np.uint8)
        canvas[:CH] = cv2.resize(frame, (CW, CH))
        sp = span_of(i)
        col = PALETTE[sp[2] % 11] if sp else (200, 200, 200)

        if any(t <= i < t + FLASH for t in tks):
            cv2.rectangle(canvas, (0, 0), (CW - 1, CH - 1), (0, 0, 255), 8)
            cv2.putText(canvas, "TAKEOVER", (CW - 300, 46), FONT, 1.2, (0, 0, 255), 4)

        # subtask banner
        if sp:
            txt = f"[{sp[2]+1}/11] {names[sp[2]]}" + ("" if sp[3] else "  (end unknown)")
        else:
            txt = "(no subtask)"
        cv2.rectangle(canvas, (0, CH - 34), (CW, CH), (0, 0, 0), -1)
        cv2.putText(canvas, txt[:58], (8, CH - 12), FONT, 0.52, col, 2)

        # info line
        av = a[i] if i < len(a) else float("nan")
        cv2.putText(canvas, f"ep{idx}  f{i}/{length}  t={i/FPS:5.1f}s  A={av:+.2f}",
                    (8, CH + 18), FONT, 0.5, (255, 255, 255), 1)
        # advantage strip + playhead
        canvas[CH + 26:] = strip
        px = int(CW * min(i, length - 1) / max(1, length))
        cv2.line(canvas, (px, CH + 26), (px, out_h - 1), (0, 255, 0), 2)

        writer.write(canvas)
        i += 1
    cap.release()
    writer.release()
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", choices=["start", "end"], required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=None)
    ap.add_argument("--auto-good", action="store_true",
                    help="end schema: only episodes Gemini labelled cleanly")
    ap.add_argument("--lerobot", default=LEROBOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    eps = {json.loads(l)["episode_index"]: json.loads(l)
           for l in open(f"{args.lerobot}/meta/episodes.jsonl")}
    df = pd.read_parquet(f"{args.lerobot}/meta/advantages_{args.tag}.parquet",
                         columns=["episode_index", "frame_index", "advantage_continuous"])
    thr = None
    mp = f"{args.lerobot}/meta/mixture_config.yaml"
    if os.path.exists(mp):
        import yaml
        thr = yaml.safe_load(open(mp)).get("tags", {}).get(args.tag, {}).get("positive_threshold")

    todo = args.episodes if args.episodes is not None else sorted(eps)
    if args.schema == "end" and args.auto_good:
        good = []
        for i in sorted(eps):
            e = end_frames_from_source(i, eps[i]["length"])
            if end_key_is_clean(e, eps[i]["length"]):
                good.append(i)
        todo = [i for i in todo if i in good]
        print(f"Gemini END key 干净的 episode: {len(good)}/{len(eps)} -> {good}\n")

    for idx in todo:
        m = eps[idx]
        names = m.get("subtask_names") or [f"subtask {i+1}" for i in range(11)]
        if args.schema == "start":
            sp = spans_from_start(m["subtask_start_frames"], m["length"])
        else:
            sp = spans_from_end(end_frames_from_source(idx, m["length"]), m["length"])
        a = (df[df.episode_index == idx].sort_values("frame_index")
             .advantage_continuous.to_numpy())
        out = os.path.join(args.output, f"episode_{idx:06d}_subtask_takeover_adv.mp4")
        n = render(idx, m, sp, a, thr, args.lerobot, out, names)
        print(f"  ep{idx:2d}: {n} frames, {len(sp)} spans, "
              f"{len(m['takeover_frames'])} takeovers -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
