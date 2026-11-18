#!/usr/bin/env python3
"""
离线评测：对 logs/ds 里每个样本跑 检测 -> 求解，用真值打分。

  .venv/bin/python tools/eval_offline.py
  .venv/bin/python tools/eval_offline.py --tag enforce   # 结果打标签，便于对比
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from pipeline import analyze          # noqa: E402
from solver import solve_best, verify  # noqa: E402

DS = ROOT / "logs" / "ds"
TRUTH_TO_SIG = {1: -1, -1: 1, 0: 0}   # 引擎 +1 实际是凹


def match_to_truth(an, truth, geo):
    """检测碎片 -> 真值碎片 的贪心最近匹配（按当前位置中心）。"""
    cand = []
    for i, p in enumerate(an["pieces"]):
        for j, t in enumerate(truth):
            d = math.hypot(t["x"] + geo["pw"] / 2 - p["cx"],
                           t["y"] + geo["ph"] / 2 - p["cy"])
            cand.append((d, i, j))
    cand.sort()
    ui, uj, pairs = set(), set(), {}
    for d, i, j in cand:
        if i in ui or j in uj or d > geo["pw"] * 0.6:
            continue
        ui.add(i); uj.add(j); pairs[i] = j
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = []
    for canvas in sorted(DS.glob("*_canvas.png")):
        stem = canvas.name[:-len("_canvas.png")]
        tpath = DS / f"{stem}_truth.json"
        if not tpath.exists():
            continue
        img = cv2.imread(str(canvas), cv2.IMREAD_UNCHANGED)
        tr = json.loads(tpath.read_text())
        geo, truth = tr["geom"], tr["pieces"]
        n = len(truth)
        s = int(round(math.sqrt(n)))

        an = analyze(img, n)
        if len(an["pieces"]) != n:
            rows.append({"sample": stem, "err": f"检出 {len(an['pieces'])}/{n}"})
            continue
        pairs = match_to_truth(an, truth, geo)

        # 边标签
        ok = tot = unk = 0
        for i, j in pairs.items():
            for side in ("top", "right", "bottom", "left"):
                tl = TRUTH_TO_SIG[truth[j]["edges"][side]]
                pr = an["pieces"][i]["sides"][side][0]
                tot += 1
                if pr is None:
                    unk += 1
                elif pr == tl:
                    ok += 1
        # 求解
        grid, viol, nodes, info = solve_best([p["sides"] for p in an["pieces"]], s)
        correct = 0
        if grid is not None:
            for r in range(s):
                for c in range(s):
                    j = pairs.get(grid[r][c])
                    if j is not None and truth[j]["row"] == r and truth[j]["col"] == c:
                        correct += 1
        rows.append({"sample": stem, "n": n, "edge_ok": ok, "edge_tot": tot - unk,
                     "edge_acc": round(ok / max(1, tot - unk), 3), "unknown": unk,
                     "viol": viol, "cells_correct": correct, "cells": n,
                     "cell_acc": round(correct / n, 3)})

    good = [r for r in rows if "err" not in r]
    print(f"样本 {len(rows)}（失败 {len(rows) - len(good)}）{'tag=' + args.tag if args.tag else ''}")
    for r in rows:
        if "err" in r:
            print(f"  {r['sample']:10s} {r['err']}")
        else:
            print(f"  {r['sample']:10s} n={r['n']:2d} 边准确 {r['edge_acc']:6.1%} "
                  f"(未知{r['unknown']:2d}) 违背{r['viol']:2d} 位置正确 "
                  f"{r['cells_correct']:2d}/{r['cells']:2d}")
    if good:
        for key, label in (("edge_acc", "边准确率"), ("cell_acc", "整格正确率")):
            v = [r[key] for r in good]
            print(f"  平均{label}: {np.mean(v):.1%}   完全解出: "
                  f"{sum(1 for r in good if r['cells_correct'] == r['cells'])}/{len(good)}")
    out = ROOT / "logs" / f"eval_{args.tag or 'run'}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
