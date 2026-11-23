#!/usr/bin/env python3
"""
第 6 课：参考图 + NCC（本项目的心脏）

问题：给你一块碎片和 20 个格子，怎么判断它属于哪一格？

答案只有三步：
    1. 把"碎片"和"格子"都裁成同样的视野（中心 62%）、缩成同样大小；
    2. 各自减去平均值，再除以自己的长度（变成单位向量）；
    3. 点积。

点积的结果就是 NCC（归一化互相关），范围 -1~1：
    1  = 图案完全一样（亮度、对比度不同也算一样）
    0  = 没关系
   -1  = 反着像（亮的地方暗、暗的地方亮）

为什么减均值：整体变亮/变暗不改变"图案"，减掉均值就把它消掉了。
为什么除以长度：整体对比度变强/变弱不改变"图案"，归一化就把它消掉了。

运行：
    .venv/bin/python teaching/l6_ncc.py

产物：
    teaching/out/l6_matrix.png   25×20 的相似度矩阵（白圈=真实格子）
    teaching/out/l6_pairs.png    几组"碎片 vs 格子"对照
    teaching/out/l6_hist.png     正确对 / 错误对 的分数分布
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)

TAG = "final_g3"          # 25 次拖拽、0 失败的一局：它的指派就是真值


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """最朴素的 NCC：两个一维数组。"""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-6 else 0.0


def prep(img, size=(96, 72)):
    """缩放 → 轻模糊 → Lab。和 play_p3._prep 一样。"""
    g = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)


def sim_lab(piece, cell):
    pa, ca = prep(piece), prep(cell)
    return float(np.mean([ncc(pa[..., i], ca[..., i]) for i in range(3)]))


def sim_gray(piece, cell):
    a = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    return ncc(cv2.resize(a, (96, 72)), cv2.resize(b, (96, 72)))


def main() -> None:
    import play_p3 as P
    import slots as S
    from pipeline import detect_board

    canvas = cv2.imread(str(ROOT / "logs" / f"{TAG}_canvas.png"), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(ROOT / "logs" / f"{TAG}_ref.png"), cv2.IMREAD_COLOR)
    rep = json.loads((ROOT / "logs" / f"{TAG}_report.json").read_text())

    # ---- 0. 先用手算例子建立直觉 ----
    print("=" * 66)
    print("手算三个例子（a 是原图，b 是另一张图）：")
    a = np.array([1, 2, 3, 4])
    for b in ([2, 4, 6, 8], [4, 3, 2, 1], [2, -1, -4, 3]):
        print(f"  a={a.tolist()}  b={b}  →  NCC = {ncc(a, np.array(b)):+.2f}")
    print("  第一组：b 是 a 的两倍（对比度不同）→ 1.00，NCC 不在乎")
    print("  第二组：b 是 a 倒过来 → -1.00")
    print("  第三组：b 和 a 没有关系 → 0.00")

    # ---- 1. 取棋盘、30 块碎片、25 格参考图 ----
    board = detect_board(canvas)
    s = 5
    pw, ph = board["w"] / s, board["h"] / s
    occ = S.occupied_slots(canvas, board, pw, ph, expect=30)

    pieces = []
    for k, slot in enumerate(occ):
        crop = P.core_crop(canvas, slot, pw, ph, P.MATCH_FOV)
        if crop is None:
            continue
        pieces.append({"k": k, "slot": slot, "crop": crop,
                       "white": P.is_white_slot(canvas, slot, pw, ph)})

    cells = P.split_cells(ref, s)
    photo_pieces = [p for p in pieces if not p["white"]]
    photo_cells = [c for c in cells if not c["white"]]
    white_cells = [c for c in cells if c["white"]]
    print(f"\n碎片 {len(pieces)}（白 {len(pieces) - len(photo_pieces)}），"
          f"照片格 {len(photo_cells)}，白格 {len(white_cells)}")

    # ---- 2. 真值：0 失败那局的指派 ----
    truth = {}
    for key, rc in rep["assign"].items():
        k = int(key[1:])
        for j, c in enumerate(photo_cells):
            if [c["r"], c["c"]] == rc:
                truth[k] = j
                break

    # ---- 3. 算矩阵：25 照片碎片 × 20 照片格 ----
    M = np.zeros((len(photo_pieces), len(photo_cells)), np.float32)
    Mg = np.zeros_like(M)
    for i, p in enumerate(photo_pieces):
        for j, c in enumerate(photo_cells):
            M[i, j] = sim_lab(p["crop"], P.cell_crop(c["img"]))
            Mg[i, j] = sim_gray(p["crop"], P.cell_crop(c["img"]))

    # ---- 4. 分开看"正确对"和"错误对" ----
    correct, wrong, margins = [], [], []
    margin_by_k = {}
    for i, p in enumerate(photo_pieces):
        k = p["k"]
        if k not in truth:
            continue
        t = truth[k]
        correct.append(M[i, t])
        row = np.delete(M[i], t)
        wrong.extend(row.tolist())
        margins.append(M[i, t] - row.max())
        margin_by_k[k] = M[i, t] - row.max()
        if M[i].argmax() != t:
            print(f"  ⚠ 片{k} top1 不是真值（{M[i].max():.3f} vs {M[i, t]:.3f}）")

    correct = np.array(correct); wrong = np.array(wrong); margins = np.array(margins)
    top1 = sum(1 for i, p in enumerate(photo_pieces)
               if p["k"] in truth and M[i].argmax() == truth[p["k"]])
    print(f"\n正确对 {len(correct)} 个：中位 {np.median(correct):.3f}  "
          f"最小 {correct.min():.3f}")
    print(f"错误对 {len(wrong)} 个：中位 {np.median(wrong):.3f}  "
          f"最大 {wrong.max():.3f}")
    print(f"正确 - 最佳错误（margin）：中位 {np.median(margins):.3f}  "
          f"最小 {margins.min():.3f}")
    print(f"top-1 正确：{top1}/{len(correct)}")
    print(f"灰度版 top-1：{sum(1 for i, p in enumerate(photo_pieces) if p['k'] in truth and Mg[i].argmax() == truth[p['k']])}/{len(correct)}"
          f"（Lab 只是让 margin 更大，不是魔法）")

    # ---- 5. 和 OpenCV 的 matchTemplate 对一下 ----
    p0 = next(p for p in photo_pieces if p["k"] in truth)
    j0 = truth[p0["k"]]
    cell_img = photo_cells[j0]["img"]
    ch_, cw_ = cell_img.shape[:2]
    # 注意尺度：碎片中心 62% 对应格子中心 62%，所以模板要先缩到同样大小
    t = cv2.resize(p0["crop"], (int(cw_ * P.MATCH_FOV), int(ch_ * P.MATCH_FOV)))
    g_t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
    g_c = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    r = cv2.matchTemplate(g_c, g_t, cv2.TM_CCOEFF_NORMED)
    print(f"\ncv2.matchTemplate(TM_CCOEFF_NORMED)：最大 {r.max():.3f}，"
          f"位置 {cv2.minMaxLoc(r)[3]}（格子中心附近）")
    print("  它和 NCC 是同一个公式，只是会滑动找最佳位置；我们已知对齐，所以直接算")

    # ---- 6. 可视化 ----
    # 矩阵热力图
    bh, bw = 20, 22
    heat = np.zeros((M.shape[0] * bh + 40, M.shape[1] * bw + 70, 3), np.uint8)
    Mn = ((M - M.min()) / (M.max() - M.min() + 1e-6) * 255).astype(np.uint8)
    color = cv2.applyColorMap(Mn, cv2.COLORMAP_TURBO)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            heat[30 + i * bh:30 + (i + 1) * bh, 60 + j * bw:60 + (j + 1) * bw] = color[i, j]
    for i, p in enumerate(photo_pieces):
        if p["k"] in truth:
            j = truth[p["k"]]
            cv2.circle(heat, (60 + j * bw + bw // 2, 30 + i * bh + bh // 2),
                       4, (255, 255, 255), -1)
        cv2.putText(heat, f"p{p['k']}", (2, 30 + i * bh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for j, c in enumerate(photo_cells):
        cv2.putText(heat, f"{c['r']}{c['c']}", (60 + j * bw + 2, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / "l6_matrix.png"), heat)

    # 几组对照图
    order = sorted(photo_pieces, key=lambda p: margin_by_k.get(p["k"], 9))
    cards = []
    for p in order[:6]:
        if p["k"] not in truth:
            continue
        i = photo_pieces.index(p)
        j = truth[p["k"]]
        a_img = cv2.resize(p["crop"], (120, 90))
        b_img = cv2.resize(P.cell_crop(photo_cells[j]["img"]), (120, 90))
        card = np.hstack([a_img, b_img])
        cv2.putText(card, f"{M[i, j]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cards.append(card)
    if cards:
        cv2.imwrite(str(OUT / "l6_pairs.png"), np.vstack(cards))

    # 直方图（两组各自归一化成"占比"，否则 20 vs 380 会把绿色压没）
    bins = np.arange(0, 1.05, 0.05)
    hc, _ = np.histogram(correct, bins)
    hw, _ = np.histogram(wrong, bins)
    fc = hc / max(1, hc.sum())
    fw = hw / max(1, hw.sum())
    hist = np.full((420, 780, 3), 255, np.uint8)
    scale = 280 / max(fc.max(), fw.max())
    for b in range(len(bins) - 1):
        x = 50 + b * 34
        cv2.rectangle(hist, (x, 350 - int(fc[b] * scale)), (x + 14, 350),
                      (0, 150, 0), -1)
        cv2.rectangle(hist, (x + 15, 350 - int(fw[b] * scale)), (x + 29, 350),
                      (0, 0, 220), -1)
        if b % 2 == 0:
            cv2.putText(hist, f"{bins[b]:.1f}", (x, 372),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
    # 阈值线：MIN_ASSIGN
    x = 50 + int(0.45 / 0.05) * 34
    cv2.line(hist, (x, 60), (x, 350), (255, 0, 0), 2)
    cv2.putText(hist, "0.45 assign gate", (x + 4, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(hist, "green = correct pairs, red = wrong pairs (each normalized)",
                (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(hist, "NCC score", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(OUT / "l6_hist.png"), hist)

    print("\n产物：")
    for f in ("l6_matrix.png", "l6_pairs.png", "l6_hist.png"):
        print(f"  {OUT / f}")
    print("\n练习：")
    print("  1) 手算 a=[1,2,3,4] 和 b=[2,0,6,0] 的 NCC（先减均值，再点积）")
    print("  2) 如果碎片裁中心 62%、格子裁整格，NCC 会怎样？为什么？")
    print("  3) 为什么减均值能消掉'整体变亮'，除以长度能消掉'对比度变强'？")


if __name__ == "__main__":
    main()
