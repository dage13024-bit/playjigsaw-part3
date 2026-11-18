#!/usr/bin/env python3
"""
离线模拟"混淆碎片"：往真实画面里贴 N 块随机卡扣的白碎片，检验
求解器能否从 (cells + N) 块里挑出正确的 cells 块。

  .venv/bin/python tools/sim_decoys.py --n-decoys 5 --difficulty 25

官方站规则：decoyCount = max(1, round(cells * 0.2)) → 3×3:2, 4×4:3, 5×5:5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import white_vision as wv          # noqa: E402
from pipeline import analyze, detect_board   # noqa: E402
from solver import solve_best, solve_with_decoys, verify  # noqa: E402

DS = ROOT / "logs" / "ds"
TRUTH_TO_SIG = {1: -1, -1: 1, 0: 0}


def add_decoys(img: np.ndarray, board, pw: float, ph: float, n: int, seed: int = 0):
    """贴 n 块白碎片（随机卡扣组合）到棋盘上下的空白带里。"""
    rng = random.Random(seed)
    out = img.copy()
    templates, _, tab = wv._build_templates(pw, ph)
    H, W = out.shape[:2]
    # 棋盘上下两条空白带（左右两侧被托盘占了）
    bands = [(0, board["y"] - int(tab), board["x"], board["x"] + board["w"]),
             (board["y"] + board["h"] + int(tab), H, board["x"], board["x"] + board["w"])]
    # 用固定槽位摆放，避免互相重叠或贴边被裁
    placed = 0
    slots = []
    for y_lo, y_hi, x_lo, x_hi in bands:
        for tpl_h in [max(t["mask"].shape[0] for t in templates)]:
            y = y_lo + max(0, (y_hi - y_lo - tpl_h) // 2)
            x = x_lo + 4
            while x < x_hi - 20 and len(slots) < 12:
                slots.append((x, y))
                x += max(max(t["mask"].shape[1] for t in templates) for _ in [0]) + 6
    rng.shuffle(slots)
    for tpl in [rng.choice(templates) for _ in range(n)]:
        if placed >= len(slots):
            break
        m = tpl["mask"]
        th, tw = m.shape
        x, y = slots[placed]
        if y + th > H or x + tw > W or y < 0 or x < 0:
            continue
        roi = out[y:y + th, x:x + tw]
        sel = m > 0
        roi[sel, :3] = 255
        roi[sel, 3] = 255
        placed += 1
    return out, placed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=25)
    ap.add_argument("--n-decoys", type=int, default=5)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    samples = sorted(DS.glob(f"p{args.difficulty}_*_canvas.png"))[:args.trials]
    if not samples:
        print(f"没有 p{args.difficulty}_*_canvas.png，先跑 tools/collect.py")
        return 1

    print(f"=== {args.difficulty} 格 + {args.n_decoys} 块混淆 ===")
    ok_decoy = ok_cell = total = 0
    for canvas in samples:
        stem = canvas.name[:-len("_canvas.png")]
        truth = json.loads((DS / f"{stem}_truth.json").read_text())
        geo, tp = truth["geom"], truth["pieces"]
        img = cv2.imread(str(canvas), cv2.IMREAD_UNCHANGED)
        b = detect_board(img)
        s = int(round(math.sqrt(len(tp))))
        pw, ph = b["w"] / s, b["h"] / s

        img2, added = add_decoys(img, b, pw, ph, args.n_decoys, seed=hash(stem) % 1000)
        an = analyze(img2, len(tp))
        n_pieces = len(an["pieces"])
        if n_pieces != len(tp) + added:
            print(f"  {stem}: ❌ 检出 {n_pieces} 块，期望 {len(tp) + added}")
            continue

        # 检测碎片 -> 真值碎片 的匹配（用真值当前位置；混淆块匹配不上）
        cand = []
        for i, p in enumerate(an["pieces"]):
            for j, t in enumerate(tp):
                d = math.hypot(t["x"] + geo["pw"] / 2 - p["cx"], t["y"] + geo["ph"] / 2 - p["cy"])
                cand.append((d, i, j))
        cand.sort()
        ui, uj, pair = set(), set(), {}
        for d, i, j in cand:
            if i in ui or j in uj or d > geo["pw"] * 0.6:
                continue
            ui.add(i); uj.add(j); pair[i] = j

        grid, viol, info = solve_with_decoys([p["sides"] for p in an["pieces"]], an["s"])
        used = {i for row in grid for i in row} if grid else set()
        decoys_ok = sum(1 for i in range(len(an["pieces"])) if i not in pair and i not in used)
        real_used = sum(1 for i in pair if i in used)
        cells_ok = 0
        if grid:
            for r in range(s):
                for c in range(s):
                    j = pair.get(grid[r][c])
                    if j is not None and tp[j]["row"] == r and tp[j]["col"] == c:
                        cells_ok += 1
        # 对照：不做混淆剔除的旧求解器
        grid0, viol0, _ = solve_best([p["sides"] for p in an["pieces"]], an["s"]) if False else (None, None, None)
        print(f"  {stem}: 检出{n_pieces}块(真{len(pair)}+混{len(an['pieces'])-len(pair)}) "
              f"| 混淆块未被使用 {decoys_ok}/{len(an['pieces'])-len(pair)} "
              f"| 真块被选中 {real_used}/{len(pair)} "
              f"| 位置正确 {cells_ok}/{s*s} | 违背 {viol}")
        ok_decoy += decoys_ok
        ok_cell += cells_ok
        total += len(pair)

    print(f"\n汇总：混淆剔除 {ok_decoy}/{args.n_decoys * len(samples)}   "
          f"整格正确 {ok_cell}/{args.difficulty * len(samples)} "
          f"= {ok_cell / max(1, args.difficulty * len(samples)):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
