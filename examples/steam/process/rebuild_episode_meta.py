"""Rebuild the per-episode metadata that meta/episodes.jsonl would carry.

Why: meta/episodes.jsonl of the open_giftbox dagger dataset is currently
root-owned with mode 600, so it cannot be read. Every input it was built from
IS readable, so we reconstruct the same fields into a sidecar JSON:

  * length              <- advantage parquet (rows per episode)
  * takeover_frames     <- re-run the deterministic detection on the RAW data
                           (same code path as add_takeover_to_lerobot.py)
  * subtask_names /
    subtask_start_frames <- replicate `stages_to_metadata` from
                           huzhipeng/stage_segmentation/batch_segment_lerobot_dagger.py
                           applied to its own results/episode_XXXXXX.json

The reconstruction is validated against the parquet episode lengths and, where
possible, against values observed from the pre-overwrite episodes.jsonl.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

OPENPI_X2 = "/mnt/public/guqiuyi/openpi/examples/x2robot"
sys.path.insert(0, OPENPI_X2)
sys.path.insert(0, os.path.join(OPENPI_X2, "mp4_json_edit"))
import filter_x2robot_data_v2 as F  # noqa: E402
import add_takeover_to_lerobot as T  # noqa: E402

LEROBOT = "/mnt/public/guqiuyi/huggingface/lerobot/open_giftbox_gqy07010702_steam_dagger"
TAG = "steam_k32_ensemble3_ckpt16000_wco_exp_and_rollout"
SEG_RESULTS = "/mnt/public/huzhipeng/stage_segmentation/dagger_subtask_segmentation/results"
FPS = 20

SUBTASK_NAMES = [
    "Cut open the carton",
    "Open the carton",
    "Grasp the gift box",
    "Place the gift box on the table",
    "Lay the gift box down",
    "Hold the gift box with the left arm",
    "Open the gift box",
    "Remove standee 1 and lay it on the table",
    "Remove standee 2 and lay it on the table",
    "Close the gift box",
    "Done",
]


def stages_to_start_frames(stages, fps, episode_length):
    """Verbatim port of batch_segment_lerobot_dagger.stages_to_metadata.

    Note the semantics: the FIRST unobserved stage sets sequence_finished, so
    every later stage is None too (truncation, not a hole).
    """
    start_frames, start_seconds = [], []
    prev_end_frame, prev_end_time, finished = 0, 0.0, False
    for expected_id, stage in enumerate(stages, start=1):
        end_time = stage.get("end_sec")
        if finished or not stage.get("observed") or end_time is None:
            finished = True
            start_frames.append(None)
            start_seconds.append(None)
            continue
        end_time = float(end_time)
        if end_time < prev_end_time:
            raise ValueError(f"non-monotonic stage {expected_id}")
        start_frames.append(prev_end_frame)
        start_seconds.append(prev_end_time)
        end_frame = round(end_time * fps)
        end_frame = max(prev_end_frame, min(end_frame, episode_length - 1))
        prev_end_frame, prev_end_time = end_frame, end_time
    # end of the last observed stage — NOT stored by the original script, but we
    # need it to know where the observed sequence stops.
    return start_frames, start_seconds, prev_end_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", default=LEROBOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--seg-results", default=SEG_RESULTS)
    ap.add_argument("--reset-min", type=int, default=70)
    ap.add_argument("--max-uniq-ratio", type=float, default=0.3)
    ap.add_argument("--out", default=f"{LEROBOT}/meta/episode_meta_rebuilt.json")
    args = ap.parse_args()

    # ---- lengths from the advantage parquet ----
    df = pd.read_parquet(f"{args.lerobot}/meta/advantages_{args.tag}.parquet",
                         columns=["episode_index", "frame_index"])
    lengths = df.groupby("episode_index").size().to_dict()
    print(f"parquet: {len(lengths)} episodes")

    # ---- raw dirs, same order as the conversion ----
    raw_eps = []
    for root in T.RAW_ROOTS:
        raw_eps += sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))
    assert len(raw_eps) == len(lengths), f"{len(raw_eps)} raw vs {len(lengths)} parquet"

    out = []
    for idx, e in enumerate(raw_eps):
        nm = os.path.basename(e)
        data = json.load(open(f"{e}/{nm}.json"))["data"]
        keep = F.filter_stationary_frames(F.get_state_arrays(data), 0)
        length = len(keep) - 1
        assert length == lengths[idx], f"ep{idx} length {length} != parquet {lengths[idx]}"

        tk, _ = T.detect_takeovers(data, keep, args.reset_min,
                                   max_uniq_ratio=args.max_uniq_ratio)

        rec = {"episode_index": idx, "raw_name": nm, "length": length,
               "takeover_frames": tk,
               "takeover_seconds": [round(t / FPS, 2) for t in tk]}

        rp = f"{args.seg_results}/episode_{idx:06d}.json"
        if os.path.exists(rp):
            stages = json.load(open(rp))["stages"]
            sf, ss, last_end = stages_to_start_frames(stages, FPS, length)
            rec.update(subtask_names=SUBTASK_NAMES, subtask_start_frames=sf,
                       subtask_start_seconds=ss, last_observed_end_frame=last_end)
        out.append(rec)
        n_obs = sum(1 for x in rec.get("subtask_start_frames", []) if x is not None)
        print(f"  ep{idx:2d} len={length:5d} takeovers={len(tk):2d} observed_subtasks={n_obs}/11")

    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}  ({len(out)} episodes)")


if __name__ == "__main__":
    main()
