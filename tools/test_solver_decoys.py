#!/usr/bin/env python3
"""
求解器的算法级测试：随机生成拼图 + 混淆块 + 注入检测噪声，检验
  1) 能否从 (cells + 20%) 块里把混淆块挑出来
  2) 能否把真块放到正确的行列

为什么不用真图测：贴图受布局干扰，测不出求解器本身的好坏。
这里直接喂"边标签"，可以跑几百次、还能精确控制噪声水平。

  .venv/bin/python tools/test_solver_decoys.py --trials 30
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from solver import SIDES, solve_best, solve_with_decoys  # noqa: E402


def make_puzzle(s: int, rng: random.Random):
    """按引擎的规则造真值边矩阵，返回 (pieces, truth_rc)。"""
    h = [[rng.choice((1, -1)) for _ in range(s)] for _ in range(s - 1)]
    v = [[rng.choice((1, -1)) for _ in range(s - 1)] for _ in range(s)]
    pieces, truth = [], {}
    for r in range(s):
        for c in range(s):
            engine = {
                "top": 0 if r == 0 else -h[r - 1][c],
                "right": 0 if c == s - 1 else v[r][c],
                "bottom": 0 if r == s - 1 else h[r][c],
                "left": 0 if c == 0 else -v[r][c - 1],
            }
            # 引擎 +1 实际渲染成"凹"，本模块语义取反
            sem = {k: (0 if val == 0 else -val) for k, val in engine.items()}
            truth[len(pieces)] = (r, c)
            pieces.append({k: (sem[k], 1.0) for k in SIDES})
    return pieces, truth


def add_decoys(pieces, n, rng):
    out = list(pieces)
    for _ in range(n):
        out.append({k: (rng.choice((1, -1, 0)), 0.6) for k in SIDES})
    return out


def add_noise(pieces, p_flip, p_unknown, rng):
    """模拟检测误差：p_flip 概率把标签判反，p_unknown 概率判成 None。"""
    out = []
    for p in pieces:
        q = dict(p)
        for k in SIDES:
            lab, conf = q[k]
            if lab is None:
                continue
            u = rng.random()
            if u < p_unknown:
                q[k] = (None, 0.0)
            elif u < p_unknown + p_flip:
                q[k] = (-lab if lab != 0 else rng.choice((1, -1)), max(0.35, conf * 0.6))
        out.append(q)
    return out


def run(s: int, trials: int, p_flip: float, p_unknown: float, seed: int = 0):
    rng = random.Random(seed)
    cells = s * s
    n_decoy = max(1, round(cells * 0.2))
    ok_cell, ok_decoy, ok_full, viols = [], [], 0, []
    for t in range(trials):
        pieces, truth = make_puzzle(s, rng)
        noisy = add_noise(pieces, p_flip, p_unknown, rng)
        allp = add_decoys(noisy, n_decoy, rng)
        order = list(range(len(allp)))
        rng.shuffle(order)
        shuffled = [allp[i] for i in order]
        # shuffled[j] 原来指向 order[j]；真值映射跟着换
        truth2 = {}
        for j, orig in enumerate(order):
            if orig in truth:
                truth2[j] = truth[orig]

        grid, viol, info = solve_with_decoys(shuffled, s)
        if grid is None:
            continue
        used = {i for row in grid for i in row}
        decoys = [j for j in range(len(shuffled)) if j not in truth2]
        ok_decoy.append(sum(1 for j in decoys if j not in used))
        correct = 0
        for r in range(s):
            for c in range(s):
                j = grid[r][c]
                if truth2.get(j) == (r, c):
                    correct += 1
        ok_cell.append(correct / cells)
        viols.append(viol)
        if correct == cells:
            ok_full += 1
    return {
        "s": s, "cells": cells, "decoys": n_decoy, "trials": trials,
        "cell_acc": statistics.mean(ok_cell) if ok_cell else 0,
        "full": ok_full,
        "decoy_rm": statistics.mean(ok_decoy) / n_decoy if ok_decoy else 0,
        "viol": statistics.median(viols) if viols else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("难度 | 噪声(判反/未知) | 整格正确率 | 完全解出 | 混淆剔除率 | 中位违背")
    print("-----|-----------------|-----------|---------|-----------|--------")
    for s in (3, 4, 5):
        for p_flip, p_unk in ((0.0, 0.0), (0.03, 0.02), (0.06, 0.04)):
            r = run(s, args.trials, p_flip, p_unk, seed=args.seed)
            print(f" {s}×{s} | {p_flip:.0%} / {p_unk:.0%}      | "
                  f"{r['cell_acc']:8.1%} | {r['full']:2d}/{r['trials']:2d}  | "
                  f"{r['decoy_rm']:8.1%} | {r['viol']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
