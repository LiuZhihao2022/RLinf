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
"""Precompute per-frame DW-BC weights (proposal M11).

    w_i = (1 + d_{sigma(i)} / d_avg) * max(A_i, 0),   then  w_i /= mean(w)

* d_s     stage hazard from a DAgger set's meta (episodes with a takeover in
          stage s + 1) / (episodes that reached s + 2)  -- Laplace alpha=beta=1.
* sigma(i) frame -> stage, from each dataset's meta subtask_start_frames.
          Episodes without labels get the corpus-mean difficulty factor 2
          (sum_s rho_s (1 + d_s/d_avg) == 2 identically), i.e. neutral.
* A_i     advantage_continuous from meta/advantages_{advantage_tag}.parquet.
* The final division by the corpus mean lets training use a plain weighted
  MEAN over the batch: no per-batch sum(w) denominator (micro-batches are
  small, their sum(w) is noisy), and the effective lr matches vanilla BC.

Weights go to meta/dwbc_weights_{out_tag}.parquet per dataset
(episode_index, frame_index, weight, stage_index), plus a sidecar
meta/dwbc_weights_{out_tag}.json recording d_s / d_avg / normalizer, so a
weight file is reproducible and auditable against the M12 closed form.

Ablations (E5 A10) are alternative --scheme values; training never changes:
    default            (1 + d/d_avg) * max(A, 0)
    fixed_lambda:<v>   (1 + v*d)     * max(A, 0)
    failure_share      (f_s/rho_s)   * max(A, 0)
    no_difficulty      max(A, 0)                    (RA-BC-like quality only)

Usage:
    python compute_dwbc_weights.py \
        --datasets /path/robodojo_stack_bowls_sft_ep100 \
                   /path/robodojo_stack_bowls_dagger_ep175 \
        --hazard-from /path/robodojo_stack_bowls_dagger_ep175 \
        --advantage-tag steam2_sb_k32_ckpt8000 \
        --out-tag dwbc_steam2_sb_k32_ckpt8000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def read_meta(ds):
    path = Path(ds) / "meta" / "episodes.jsonl"
    return {int(r["episode_index"]): r
            for r in map(json.loads, path.read_text().splitlines()) if r}


def spans(rec):
    """(stage_name, start, end) list for one labelled episode."""
    starts = rec.get("subtask_start_frames") or []
    names = rec.get("subtask_names") or []
    if not starts or not names or all(s is None for s in starts):
        return []
    obs = [(n, int(s)) for n, s in zip(names, starts) if s is not None]
    out = []
    for j, (n, s) in enumerate(obs):
        e = obs[j + 1][1] if j + 1 < len(obs) else int(rec["length"])
        if e > s:
            out.append((n, s, e))
    return out


def compute_hazard(dagger_meta):
    """Episode-level conditional failure rate per stage, Laplace 1/1."""
    n_reach, n_fail = defaultdict(int), defaultdict(int)
    for rec in dagger_meta.values():
        sp = spans(rec)
        tks = sorted(int(t) for t in (rec.get("takeover_frames") or []))
        for name, s, e in sp:
            n_reach[name] += 1
            if any(s <= t < e for t in tks):
                n_fail[name] += 1
    return {n: (n_fail[n] + 1.0) / (n_reach[n] + 2.0) for n in n_reach}, \
        dict(n_reach), dict(n_fail)


def load_adv(ds, tag):
    p = Path(ds) / "meta" / f"advantages_{tag}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run compute_advantages first")
    df = pd.read_parquet(p)
    return {(int(e), int(f)): float(a) for e, f, a in
            zip(df["episode_index"], df["frame_index"],
                df["advantage_continuous"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--hazard-from", required=True)
    ap.add_argument("--advantage-tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--scheme", default="default",
                    help="default | fixed_lambda:<v> | failure_share | no_difficulty")
    args = ap.parse_args()

    hazard, n_reach, n_fail = compute_hazard(read_meta(args.hazard_from))
    stages = sorted(hazard)
    sid = {n: i for i, n in enumerate(stages)}

    # ---- pass 1: rho over the corpus being weighted (labelled frames only)
    frames_per_stage = defaultdict(int)
    total_labelled = 0
    metas = {}
    for ds in args.datasets:
        metas[ds] = read_meta(ds)
        for rec in metas[ds].values():
            for name, s, e in spans(rec):
                if name in hazard:
                    frames_per_stage[name] += e - s
                    total_labelled += e - s
    rho = {n: frames_per_stage[n] / total_labelled for n in stages}
    d_avg = sum(rho[n] * hazard[n] for n in stages)

    def difficulty(name):
        """Difficulty factor u for one stage (or None -> corpus mean)."""
        if args.scheme == "no_difficulty":
            return 1.0
        if args.scheme == "failure_share":
            f = (n_fail.get(name, 0) + 1.0) / sum(v + 1.0 for v in n_fail.values())
            return f / rho[name] if name else 1.0
        if args.scheme.startswith("fixed_lambda:"):
            lam = float(args.scheme.split(":", 1)[1])
            return 1.0 + lam * (hazard[name] if name else d_avg)
        return 1.0 + (hazard[name] if name else d_avg) / d_avg   # default

    neutral_u = difficulty(None)

    # ---- pass 2: raw weights
    per_ds = {}
    all_w = []
    stage_w_sum = defaultdict(float)
    for ds in args.datasets:
        adv = load_adv(ds, args.advantage_tag)
        rows = []
        for ep, rec in sorted(metas[ds].items()):
            length = int(rec["length"])
            stage_of = np.full(length, -1, dtype=np.int32)
            u = np.full(length, neutral_u, dtype=np.float64)
            for name, s, e in spans(rec):
                if name in hazard:
                    stage_of[s:e] = sid[name]
                    u[s:e] = difficulty(name)
            for f in range(length):
                a = adv.get((ep, f))
                if a is None:
                    # terminal frames get a default-negative score upstream;
                    # anything truly missing means tag/dataset mismatch
                    raise KeyError(f"{ds} ep{ep} frame{f} missing advantage")
                q = max(a, 0.0)
                w = u[f] * q
                rows.append((ep, f, w, int(stage_of[f])))
                if stage_of[f] >= 0:
                    stage_w_sum[stages[stage_of[f]]] += w
        per_ds[ds] = rows
        all_w.extend(r[2] for r in rows)

    mean_w = float(np.mean(all_w))
    if mean_w <= 0:
        raise ValueError("mean raw weight is 0 -- every frame has A<=0?")

    # ---- write
    for ds, rows in per_ds.items():
        df = pd.DataFrame(rows, columns=["episode_index", "frame_index",
                                         "weight", "stage_index"])
        df["weight"] = (df["weight"] / mean_w).astype(np.float32)
        out = Path(ds) / "meta" / f"dwbc_weights_{args.out_tag}.parquet"
        df.to_parquet(out, index=False)
        z = float((df["weight"] == 0).mean())
        print(f"{out}\n    {len(df):,} frames | zero {z:.1%} | "
              f"mean {df['weight'].mean():.4f} | p90 "
              f"{df['weight'].quantile(0.9):.3f} | max {df['weight'].max():.3f}")

    sidecar = {
        "scheme": args.scheme, "advantage_tag": args.advantage_tag,
        "hazard_from": args.hazard_from, "datasets": args.datasets,
        "stages": stages, "hazard": hazard, "n_reach": n_reach,
        "n_fail": n_fail, "rho": rho, "d_avg": d_avg,
        "neutral_difficulty": neutral_u, "normalizer_mean_raw_w": mean_w,
    }
    for ds in args.datasets:
        (Path(ds) / "meta" / f"dwbc_weights_{args.out_tag}.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2))

    # ---- diagnostics: realized share vs M12 closed form vs failure share
    print(f"\nscheme={args.scheme}   d_avg={d_avg:.4f}   "
          f"lambda_eff={1 / d_avg:.2f}   normalizer={mean_w:.4f}")
    tot_w = sum(stage_w_sum.values())
    tot_fail = sum(v + 1.0 for v in n_fail.values())
    print(f"{'stage':<22}{'rho':>7}{'d_s':>7}{'份额(实际)':>10}"
          f"{'M12闭式':>9}{'失败份额':>9}")
    print("-" * 66)
    for n in stages:
        closed = rho[n] / 2 * (1 + hazard[n] / d_avg)
        print(f"{n:<22}{rho[n]:>7.3f}{hazard[n]:>7.3f}"
              f"{stage_w_sum[n] / tot_w:>10.3f}{closed:>9.3f}"
              f"{(n_fail.get(n, 0) + 1) / tot_fail:>9.3f}")


if __name__ == "__main__":
    main()
