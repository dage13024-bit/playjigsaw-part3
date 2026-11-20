#!/usr/bin/env python3
"""
第 4 课补充：阈值到底是怎么定出来的？

不是拍脑袋，是**看分布**。把一帧里 120 条边的 (外侧密度, 内侧密度) 全算出来，
画成直方图，你会看到两个峰：直边挤在 0 附近，凸边挤在 0.5~0.9；
凹边的内侧挤在 0.3 附近，实心的挤在 1.0。阈值就放在两个峰之间的谷底。

这也解释了一个事实：**阈值和取样窗口是绑定的**。
窗口一变（上一课的 a/b、d1），密度整体平移，原来的阈值就不成立了。

运行：
    .venv/bin/python teaching/l4_thresholds.py

产物：
    teaching/out/l4_thresholds.png   两张直方图 + 阈值线
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)

from teaching.l4_edges import windows, density, SIDES   # noqa: E402


def hist_rows(edges):
    """edges: [(out, inn)] → 10 个区间的计数。"""
    bins = np.zeros(10, int)
    for o, i in edges:
        bins[min(9, int(o * 10))] += 1
    return bins


def draw_panel(canvas, x0, title, counts, lines, color):
    base = 330
    top = 60
    scale = 250 / max(1, counts.max())
    for b in range(10):
        x = x0 + b * 40
        h = int(counts[b] * scale)
        cv2.rectangle(canvas, (x, base - h), (x + 34, base), color, -1)
        cv2.putText(canvas, f"{b/10:.1f}", (x + 2, base + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        if counts[b]:
            cv2.putText(canvas, str(counts[b]), (x + 6, base - h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
    cv2.putText(canvas, title, (x0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for v, name in lines:
        x = int(x0 + v * 400)
        cv2.line(canvas, (x, top), (x, base), (0, 0, 255), 2)
        cv2.putText(canvas, f"{name} {v:.2f}", (x + 4, top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)


def main() -> None:
    f = sorted(glob.glob(str(ROOT / "logs" / "*_canvas.png")))[-1]
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    mask = (img[:, :, 3] >= 253).astype(np.uint8)
    from pipeline import detect_board
    from slots import occupied_slots
    board = detect_board(img)
    s = 5
    pw, ph = board["w"] / s, board["h"] / s
    tab = 0.22 * min(pw, ph)
    occ = occupied_slots(img, board, pw, ph, expect=30)

    edges = []
    for slot in occ:
        for side in SIDES:
            o, i = windows(slot, pw, ph, tab, side)
            edges.append((density(mask, *o), density(mask, *i)))

    out_counts = hist_rows(edges)
    out_counts_inn = np.zeros(10, int)
    for o, i in edges:
        out_counts_inn[min(9, int(i * 10))] += 1

    canvas = np.full((400, 940, 3), 255, np.uint8)
    draw_panel(canvas, 40, "outer density (out)", out_counts,
               [(0.15, "flat"), (0.50, "tab")], (120, 160, 230))
    draw_panel(canvas, 520, "inner density (inn)", out_counts_inn,
               [(0.70, "slot")], (180, 210, 150))
    cv2.imwrite(str(OUT / "l4_thresholds.png"), canvas)

    # ---- 决策树与计数 ----
    slot_n = tab_n = flat_n = unknown = 0
    for o, i in edges:
        if i < 0.70:
            slot_n += 1
        elif o > 0.50:
            tab_n += 1
        elif o < 0.15:
            flat_n += 1
        else:
            unknown += 1
    print(f"120 条边：凹 {slot_n} / 凸 {tab_n} / 直 {flat_n} / 拿不准 {unknown}")
    print("\n判定顺序很重要（先凹，再凸，再直）：")
    print("  inn < 0.70           → 凹 (-)    [凹边的 out 也接近 0，不能靠 out 区分直/凹]")
    print("  out > 0.50           → 凸 (+)    [凸边的 inn 也是 1，不能靠 inn 区分凸/直]")
    print("  out < 0.15           → 直 (0)")
    print("  其余                 → 拿不准 (?)")
    print("\n产物：", OUT / "l4_thresholds.png")


if __name__ == "__main__":
    main()
