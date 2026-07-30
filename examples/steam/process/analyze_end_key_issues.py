"""Audit the problems with the OLD `subtask_end_frames` schema.

The old key is reconstructed from its own source of truth —
huzhipeng/stage_segmentation/dagger_subtask_segmentation/results/episode_*.json —
as  end_frame[i] = clamp(round(end_sec[i] * FPS), prev, length-1),  None when
Gemini reported observed=false. (Verified against values read from the
pre-overwrite episodes.jsonl: ep0/ep1/ep17 match exactly.)

Episode lengths come from the advantage parquet; takeover frames are re-derived
with the same deterministic detector used to write them.
"""
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
SEG = "/mnt/public/huzhipeng/stage_segmentation/dagger_subtask_segmentation/results"
FPS = 20
N_SUB = 11

# Values read from episodes.jsonl BEFORE it was overwritten (schema change at 14:09).
KNOWN_OLD = {
    0: [240, 520, 640, 840, 1020, 1150, 1280, 1400, 1620, 1840, 1892],
    1: [240, 540, 860, 1080, 1280, 1480, 1780, 2000, 2260, 2380, 2459],
    17: [280, 600, 800, 1040, 1220, 1300, 1560, None, None, 2880, 2922],
}


def old_end_frames(stages, length):
    """Reconstruct the OLD end-key: no truncation, holes stay holes."""
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


def main():
    df = pd.read_parquet(f"{LEROBOT}/meta/advantages_{TAG}.parquet",
                         columns=["episode_index", "frame_index"])
    lengths = df.groupby("episode_index").size().to_dict()

    raw_eps = []
    for root in T.RAW_ROOTS:
        raw_eps += sorted(d for d in glob.glob(f"{root}/*") if os.path.isdir(d))

    eps = []
    for idx in range(len(raw_eps)):
        length = lengths[idx]
        stages = json.load(open(f"{SEG}/episode_{idx:06d}.json"))["stages"]
        ends = old_end_frames(stages, length)
        e = raw_eps[idx]
        nm = os.path.basename(e)
        data = json.load(open(f"{e}/{nm}.json"))["data"]
        keep = F.filter_stationary_frames(F.get_state_arrays(data), 0)
        tk, _ = T.detect_takeovers(data, keep, 70, max_uniq_ratio=0.3)
        eps.append(dict(idx=idx, length=length, ends=ends, takeovers=tk))

    # ---- 0. validate the reconstruction ----
    print("=" * 78)
    print("0. 重建校验(对比覆写前从 episodes.jsonl 亲眼读到的值)")
    print("=" * 78)
    ok = True
    for i, exp in KNOWN_OLD.items():
        got = eps[i]["ends"]
        m = got == exp
        ok &= m
        print(f"  ep{i:2d}: {'✓ 一致' if m else '✗ 不一致'}")
        if not m:
            print(f"        期望 {exp}\n        重建 {got}")
    print(f"  => 重建{'可信' if ok else '不可信'},以下统计基于该重建\n")

    # ---- 1. completeness ----
    print("=" * 78)
    print("1. 完整性:哪些 episode 没标完整")
    print("=" * 78)
    bad = [e for e in eps if any(x is None for x in e["ends"])]
    tot_none = sum(sum(1 for x in e["ends"] if x is None) for e in eps)
    print(f"  含 None 的 episode: {len(bad)}/{len(eps)}    None 总数: {tot_none}/{len(eps)*N_SUB} "
          f"({100*tot_none/(len(eps)*N_SUB):.1f}%)\n")
    per_idx = [sum(1 for e in eps if e["ends"][i] is None) for i in range(N_SUB)]
    print("  每个子任务缺失次数(1-based id):")
    for i, c in enumerate(per_idx):
        if c:
            print(f"    subtask {i+1:2d}: {c:2d}/{len(eps)}  {'#'*c}")
    print(f"\n  => 缺失高度集中在后期子任务 8-11: "
          f"{sum(per_idx[7:])}/{tot_none} 占 {100*sum(per_idx[7:])/tot_none:.0f}%\n")
    print("  有问题的 episode 明细:")
    for e in bad:
        miss = [i + 1 for i, x in enumerate(e["ends"]) if x is None]
        print(f"    ep{e['idx']:2d} (len={e['length']:5d}): 缺 subtask {miss}")
        print(f"          ends = {e['ends']}")

    # ---- 2. hole vs truncation ----
    print("\n" + "=" * 78)
    print("2. 【致命】'空洞'语义:None 之后又出现数值 -> 中间那段无法归属")
    print("=" * 78)
    holes = []
    for e in eps:
        ends = e["ends"]
        for i, x in enumerate(ends):
            if x is None and any(y is not None for y in ends[i + 1:]):
                holes.append(e)
                break
    print(f"  存在'空洞'的 episode: {len(holes)}/{len(eps)} -> {[e['idx'] for e in holes]}")
    lost = 0
    for e in holes:
        ends = e["ends"]
        prev_known = 0
        run = []
        for i, x in enumerate(ends):
            if x is None:
                run.append(i + 1)
                continue
            if run:
                span = x - prev_known
                lost += span
                print(f"    ep{e['idx']:2d}: subtask {run} 边界未知 -> "
                      f"帧 [{prev_known}, {x}) 共 {span} 帧 ({span/FPS:.1f}s) 归属不明")
                run = []
            prev_known = x
    print(f"  => 因空洞而无法归属的帧: {lost} 帧 ({lost/FPS:.0f}s)")

    # ---- 3. tail ----
    print("\n" + "=" * 78)
    print("3. 尾部未覆盖:最后一个已知边界之后的帧")
    print("=" * 78)
    tail_frames = 0
    tail_eps = []
    for e in eps:
        known = [x for x in e["ends"] if x is not None]
        last = max(known) if known else 0
        gap = e["length"] - last
        if gap > FPS:  # >1s uncovered
            tail_frames += gap
            tail_eps.append((e["idx"], last, gap))
    for idx, last, gap in sorted(tail_eps, key=lambda x: -x[2])[:8]:
        print(f"    ep{idx:2d}: 最后已知边界={last:5d}, 之后还有 {gap:5d} 帧 ({gap/FPS:5.1f}s) 没有子任务归属")
    print(f"  => {len(tail_eps)} 条 episode 有尾部空白,共 {tail_frames} 帧 ({tail_frames/FPS:.0f}s)")

    # ---- 4. degenerate spans ----
    print("\n" + "=" * 78)
    print("4. 退化边界:零长度 / 塌缩到 0 的子任务")
    print("=" * 78)
    for e in eps:
        ends, prev, zero = e["ends"], 0, []
        for i, x in enumerate(ends):
            if x is None:
                continue
            if x <= prev:
                zero.append(i + 1)
            prev = x
        if zero:
            print(f"    ep{e['idx']:2d}: subtask {zero} 时长=0  ends={ends}")

    # ---- 5. impact on the difficulty axis ----
    print("\n" + "=" * 78)
    print("5. 对难度轴的实际影响:接管无法归属到子任务")
    print("=" * 78)

    def assign(tk, ends, length):
        prev, pend = 0, []
        for i, x in enumerate(ends):
            if x is None:
                pend.append(i + 1)
                continue
            if prev <= tk < x:
                return (i + 1) if not pend else f"{pend[0]}-{i+1}(不确定)"
            prev = x
            pend = []
        return "tail(无归属)"

    from collections import Counter
    c = Counter()
    for e in eps:
        for tk in e["takeovers"]:
            c[assign(tk, e["ends"], e["length"])] += 1
    total = sum(c.values())
    clean = sum(v for k, v in c.items() if isinstance(k, int))
    print(f"  接管总数: {total}")
    print(f"  能干净归属到单个子任务: {clean} ({100*clean/total:.0f}%)")
    print(f"  归属不明(落在空洞/尾部): {total-clean} ({100*(total-clean)/total:.0f}%)")
    print("\n  不明的明细:")
    for k, v in sorted(c.items(), key=lambda x: str(x[0])):
        if not isinstance(k, int):
            print(f"    {k}: {v} 次")
    print("\n  => 这 %d 次接管若被强行计入某个子任务,就会把 Gemini 的分割失败" % (total - clean))
    print("     误记成机器人的任务失败,直接污染难度统计。")


if __name__ == "__main__":
    main()
