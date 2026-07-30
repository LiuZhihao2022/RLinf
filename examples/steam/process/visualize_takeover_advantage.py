"""Visualize STEAM advantage against human-takeover moments (open_giftbox dagger).

Answers two questions:
  1. Per episode: what does the advantage curve look like around each takeover?
  2. Does the ensemble (min-aggregation) actually identify "entering failure" and
     "recovering from failure"? -> per-member curves + takeover-aligned profile
     + AUROC per member vs. the min aggregate.

Outputs PNGs to --output.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEROBOT = "/mnt/public/guqiuyi/huggingface/lerobot/open_giftbox_gqy07010702_steam_dagger"
TAG = "steam_k32_ensemble3_ckpt16000_wco_exp_and_rollout"
FPS = 20

# Fixed 11-subtask taxonomy (identical across all 35 episodes), produced by
# Gemini in /mnt/public/huzhipeng/stage_segmentation. `subtask_end_frames[i]`
# is the END frame of subtask i+1; None == Gemini reported observed=false.
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


def auroc(pos, neg):
    """P(pos scored lower than neg) — we detect failure via LOW advantage."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    a = (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return 1 - a  # low advantage => failure


def load(lerobot, tag):
    df = pd.read_parquet(f"{lerobot}/meta/advantages_{tag}.parquet")
    eps = [json.loads(l) for l in open(f"{lerobot}/meta/episodes.jsonl")]
    return df, eps


def episode_arrays(df, idx):
    d = df[df.episode_index == idx].sort_values("frame_index")
    a = d.advantage_continuous.to_numpy()
    lab = d.advantage.to_numpy().astype(bool)
    mv = d.member_values.to_numpy()
    mv = np.stack([np.asarray(x, dtype=np.float32) for x in mv]) if len(mv) else None
    return a, lab, mv


def _smooth(x, w=20):
    """Centred rolling mean (w frames = w/FPS seconds)."""
    if len(x) < w:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def subtask_spans(starts, length):
    """Turn `subtask_start_frames` into drawable spans.

    START schema (batch_segment_lerobot_dagger.stages_to_metadata): entry i is
    the START frame of subtask i, so subtask i runs to the next observed start.
    A None marks TRUNCATION — the first unobserved subtask sets sequence_finished,
    so every later entry is None too. When truncated, the last observed subtask's
    end is unknown, so its span (running to the episode end) is flagged inexact.

    Returns: list of (start, end, subtask_index, exact).
    """
    obs = [(i, s) for i, s in enumerate(starts) if s is not None]
    truncated = len(obs) < len(starts)
    spans = []
    for j, (i, s) in enumerate(obs):
        e = obs[j + 1][1] if j + 1 < len(obs) else length
        exact = not (truncated and j == len(obs) - 1)
        spans.append((s, e, i, exact))
    return [sp for sp in spans if sp[1] > sp[0]]


def subtask_of(frame, starts, length):
    """Which subtask a frame falls in -> (index, exact) or (None, False)."""
    for s, e, i, exact in subtask_spans(starts, length):
        if s <= frame < e:
            return i, exact
    return None, False


_T20 = plt.get_cmap("tab20").colors
SUB_COLORS = [_T20[i] for i in range(0, 20, 2)] + [_T20[1]]  # 11 distinct


def subtask_thumbnails(lerobot, idx, spans):
    """One representative frame (the MIDDLE of the span) per subtask -> {i: RGB}."""
    import cv2
    cap = cv2.VideoCapture(f"{lerobot}/videos/chunk-000/face_view/episode_{idx:06d}.mp4")
    out = {}
    for s, e, i, _ in spans:
        cap.set(cv2.CAP_PROP_POS_FRAMES, (s + e) // 2)
        ok, fr = cap.read()
        if ok:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def plot_episode(idx, meta, a, lab, mv, thr, out, lerobot,
                 a2=None, thr2=None, label2="STEAM 2.0"):
    tks = meta["takeover_frames"]
    starts = meta.get("subtask_start_frames")
    names = meta.get("subtask_names") or SUBTASK_NAMES
    t = np.arange(len(a)) / FPS
    ymin, ymax = -1.05, 1.05
    spans = subtask_spans(starts, len(a)) if starts else []

    # ---- figure: 3 rows of subtask thumbnails (1-4 / 5-8 / 9-11) + advantage ----
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 2.1], hspace=0.32, wspace=0.06)
    thumbs = subtask_thumbnails(lerobot, idx, spans)
    span_by_i = {i: (s, e, exact) for s, e, i, exact in spans}
    for k in range(len(names)):
        axk = fig.add_subplot(gs[k // 4, k % 4])
        axk.set_xticks([])
        axk.set_yticks([])
        col = SUB_COLORS[k % len(SUB_COLORS)]
        if k in thumbs:
            axk.imshow(thumbs[k])
            s, e, exact = span_by_i[k]
            sub = f"{s/FPS:.0f}-{e/FPS:.0f}s" + ("" if exact else "  end unknown")
        else:
            axk.text(0.5, 0.5, "not observed", ha="center", va="center",
                     fontsize=11, color="crimson", transform=axk.transAxes)
            sub = "—"
        for sp in axk.spines.values():
            sp.set_edgecolor(col)
            sp.set_linewidth(3.5)
        axk.set_title(f"{k+1}. {names[k]}\n{sub}", fontsize=8, color="black", pad=3)

    ax = fig.add_subplot(gs[3, :])

    # ---- background: subtask bands, colour-matched to the thumbnail borders ----
    for s, e, i, exact in spans:
        ax.axvspan(s / FPS, e / FPS, color=SUB_COLORS[i % len(SUB_COLORS)],
                   alpha=0.20, lw=0, zorder=0)
        ax.axvline(s / FPS, color="dimgray", lw=0.9, ls="-", alpha=0.55, zorder=1)
        disp = f"{i+1}" + ("" if exact else "+?")
        ax.annotate(disp, ((s + e) / 2 / FPS, ymax - 0.05), ha="center", va="top",
                    fontsize=9, color="black", fontweight="bold", zorder=2)
        if not exact:  # end of this subtask unknown (sequence truncated)
            ax.axvspan(s / FPS, e / FPS, facecolor="none", hatch="//",
                       edgecolor="gray", lw=0, zorder=0)

    # ---- ensemble member spread: a NARROW band means the members agree ----
    comparing = a2 is not None
    if mv is not None and mv.ndim == 2 and not comparing:
        ax.fill_between(t, mv.min(1), mv.max(1), color="tab:blue", alpha=0.22, lw=0,
                        label=f"ensemble members ({mv.shape[1]}) min-max spread", zorder=3)
    prim_lbl = "old predictor, 1s avg" if comparing else "advantage, 1s moving average"
    ax.plot(t, a, lw=0.5, color="gray", alpha=0.4,
            label="old predictor, raw" if comparing else "advantage, raw (ensemble-min)",
            zorder=4)
    ax.plot(t, _smooth(a), lw=2.0, color="k", label=prim_lbl, zorder=5)
    if comparing:
        t2 = np.arange(len(a2)) / FPS
        ax.plot(t2, a2, lw=0.5, color="tab:orange", alpha=0.35, zorder=4)
        ax.plot(t2, _smooth(a2), lw=2.0, color="tab:orange", zorder=5,
                label=f"{label2}, 1s avg")
    ax.axhline(0, color="dimgray", lw=1.0, ls="--", zorder=2)
    if thr is not None:
        ax.axhline(thr, color="green", lw=1.0, ls=":", zorder=2,
                   label=f"{'old ' if comparing else ''}optimal threshold {thr:.3f}")
    if comparing and thr2 is not None:
        ax.axhline(thr2, color="darkorange", lw=1.0, ls=":", zorder=2,
                   label=f"{label2} optimal threshold {thr2:.3f}")

    # ---- takeovers, annotated with the subtask they fall in ----
    tk_sub = []
    for j, tk in enumerate(tks):
        i, exact = subtask_of(tk, starts, len(a)) if starts else (None, False)
        lb_disp = "?" if i is None else f"{i+1}" + ("" if exact else "+?")
        tk_sub.append(lb_disp)
        ax.axvline(tk / FPS, color="red", lw=1.8, alpha=0.9, zorder=6,
                   label="human takeover" if j == 0 else None)
        ax.annotate(f"takeover {tk/FPS:.0f}s (sub {lb_disp})", (tk / FPS, ymin + 0.05),
                    rotation=90, ha="right", va="bottom", fontsize=7, color="red", zorder=6)

    # ---- optimal ribbon at the very bottom (keeps the plot area clean) ----
    ax.fill_between(t, ymin, ymin + 0.05, where=lab, color="green", alpha=0.65, step="mid",
                    label="labelled optimal (o=1)", zorder=3)

    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time in LeRobot (static-filtered) timeline [s]")
    ax.set_ylabel("STEAM advantage")
    n_obs = sum(1 for x in (starts or []) if x is not None)
    ax.set_title(f"{len(tks)} takeovers, in subtask(s) {tk_sub}   |   "
                 f"{100*lab.mean():.0f}% of frames labelled optimal   |   "
                 f"subtasks observed {n_obs}/11   |   band colour = the thumbnail "
                 f"border above;  '+?'+hatch = end unknown (truncated)", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), fontsize=7, ncol=6,
              framealpha=0.9)
    suptitle = (f"ep{idx}  —  subtask thumbnails + advantage: "
                f"old predictor (black) vs {label2} (orange)"
                if comparing else
                f"ep{idx}  —  subtask thumbnails (middle frame of each subtask) "
                f"+ STEAM advantage")
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.savefig(f"{out}/episode_{idx:06d}_advantage_takeover.png", dpi=100,
                bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", default=LEROBOT)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--output", default="/mnt/public/guqiuyi/RLinf_active/logs/viz_takeover_adv")
    ap.add_argument("--window", type=int, default=60, help="±frames around takeover")
    ap.add_argument("--episodes", type=int, nargs="+", default=None,
                    help="Only render these episode indices (default: all).")
    ap.add_argument("--compare-tag", default=None,
                    help="Second advantage tag to overlay for comparison "
                         "(e.g. a STEAM 2.0 dagger model).")
    ap.add_argument("--compare-label", default="STEAM 2.0",
                    help="Legend label for the --compare-tag curve.")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    df, eps = load(args.lerobot, args.tag)
    if args.episodes is not None:
        keep = set(args.episodes)
        eps = [e for e in eps if e["episode_index"] in keep]
        print(f"rendering {len(eps)} episodes: {sorted(keep)}")

    def _threshold_for(tag):
        mixp = f"{args.lerobot}/meta/mixture_config.yaml"
        if not os.path.exists(mixp):
            return None
        import yaml
        return yaml.safe_load(open(mixp)).get("tags", {}).get(tag, {}).get(
            "positive_threshold"
        )

    thr = _threshold_for(args.tag)
    df2 = None
    thr2 = None
    if args.compare_tag:
        import pandas as pd
        df2 = pd.read_parquet(
            f"{args.lerobot}/meta/advantages_{args.compare_tag}.parquet"
        )
        thr2 = _threshold_for(args.compare_tag)
        print(f"overlaying compare tag={args.compare_tag} (thr={thr2}) as "
              f"'{args.compare_label}'")

    W = args.window
    n_mem = None
    pre_m, far_m, prof_m = {}, {}, {}   # per member
    pre_a, far_a, prof_a = [], [], {}   # min aggregate

    for e in eps:
        i = e["episode_index"]
        a, lab, mv = episode_arrays(df, i)
        if len(a) == 0:
            continue
        if mv is not None and mv.ndim == 2:
            n_mem = mv.shape[1]
        a2 = None
        if df2 is not None:
            a2, _lab2, _mv2 = episode_arrays(df2, i)
            if len(a2) == 0:
                a2 = None
        plot_episode(i, e, a, lab, mv, thr, args.output, args.lerobot,
                     a2=a2, thr2=thr2, label2=args.compare_label)

        n = len(a)
        near = np.zeros(n, bool)
        for tk in e["takeover_frames"]:
            near[max(0, tk - W):min(n, tk + W)] = True
            pre_a.append(a[max(0, tk - W):tk])
            for off in range(-100, 101, 10):
                j = tk + off
                if 0 <= j < n:
                    prof_a.setdefault(off, []).append(a[j])
                    if n_mem:
                        for m in range(n_mem):
                            prof_m.setdefault(m, {}).setdefault(off, []).append(mv[j, m])
            if n_mem:
                for m in range(n_mem):
                    pre_m.setdefault(m, []).append(mv[max(0, tk - W):tk, m])
        far_a.append(a[~near])
        if n_mem:
            for m in range(n_mem):
                far_m.setdefault(m, []).append(mv[~near, m])

    cat = lambda L: np.concatenate([x for x in L if len(x)])
    pre_a, far_a = cat(pre_a), cat(far_a)

    # ---- aggregate figure: takeover-aligned profile, per member + min ----
    offs = sorted(prof_a)
    fig, ax = plt.subplots(figsize=(9, 5))
    if n_mem:
        for m in range(n_mem):
            ax.plot([o / FPS for o in offs], [np.mean(prof_m[m][o]) for o in offs],
                    marker="o", ms=3, lw=1.2, alpha=0.7, label=f"member {m}")
    ax.plot([o / FPS for o in offs], [np.mean(prof_a[o]) for o in offs],
            marker="s", ms=4, lw=2.4, color="k", label="ensemble-min (the one actually used)")
    ax.axvline(0, color="red", lw=2, label="human takeover")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    if thr is not None:
        ax.axhline(thr, color="green", ls=":", lw=1.2, label=f"optimal threshold {thr:.3f}")
    ax.set_xlabel("time relative to human takeover [s]   (<0 entering failure, >0 human recovery)")
    ax.set_ylabel("mean STEAM advantage")
    ax.set_title("Takeover-aligned advantage profile: ensemble members vs min\n"
                 f"open_giftbox dagger, {len(eps)} episodes / "
                 f"{sum(len(e['takeover_frames']) for e in eps)} takeovers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.output}/00_takeover_aligned_profile.png", dpi=130)
    plt.close(fig)

    # ---- discrimination table ----
    print(f"\n{'='*74}\n判别力: 用低 advantage 识别'接管前{W}帧失败态' vs '远离接管的正常帧'\n{'='*74}")
    print(f"{'估计器':<22}{'AUROC':>8}{'接管前均值':>12}{'正常均值':>10}{'接管时刻均值':>13}")
    rows = []
    if n_mem:
        for m in range(n_mem):
            p, f = cat(pre_m[m]), cat(far_m[m])
            rows.append((f"member {m}", auroc(p, f), p.mean(), f.mean(), np.mean(prof_m[m][0])))
    rows.append(("ensemble-min", auroc(pre_a, far_a), pre_a.mean(), far_a.mean(), np.mean(prof_a[0])))
    for nm, au, pm, fm, at in rows:
        print(f"{nm:<22}{au:>8.3f}{pm:>12.3f}{fm:>10.3f}{at:>13.3f}")
    print(f"\n图已存到 {args.output}/")


if __name__ == "__main__":
    main()
