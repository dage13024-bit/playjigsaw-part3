#!/usr/bin/env python3
"""
第 2 课：从画面里把棋盘"抠"出来

上一课你得到了 alpha 掩码：它告诉你"哪些像素有东西"。
但棋盘和碎片都"有东西"，所以 alpha 还不够。这一课加两个条件：

    棋盘 = 不透明(alpha≈1) 且 暗(gray ≤ 80) 且 是一整块实心矩形

前两个条件用一次布尔与就能做到；"一整块实心矩形"要靠**连通域**：
给每个连在一起的暗像素块编号，量它的外接框、面积、填充率。

    填充率 fill = 连通域面积 / 外接框面积
    score      = 面积 × 填充率

棋盘是画面里"又大又实"的暗块，score 最高。
注意 fill 的阈值只能取 0.25 左右，不能取 0.9：碎片压在棋盘上会"挖洞"，
棋盘的填充率会掉到 0.5 甚至更低，但它的外接框仍然是完整的棋盘。

运行：
    .venv/bin/python teaching/l2_board.py

产物：
    teaching/out/l2_dark.png        暗色 + 不透明的掩码
    teaching/out/l2_components.png  所有候选暗块（按 score 排序、标注）
    teaching/out/l2_board.png       最终选中的棋盘
"""
from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)

BOARD_DARK_MAX = 200      # 和 src/pipeline.py 保持一致
MIN_SIDE = 40            # 太小的暗块直接忽略（噪点、碎片上的暗花纹）


def pick_canvas() -> Path:
    files = sorted(glob.glob(str(ROOT / "logs" / "*_canvas.png")))
    if not files:
        raise SystemExit("logs/ 里没有 *_canvas.png，先跑一次 "
                         ".venv/bin/python play_p3.py --difficulty 25 --dry-run")
    return Path(files[-1])


def main() -> None:
    path = pick_canvas()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片：{path}")
    h, w = img.shape[:2]
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(OUT / "l2_gray.png"), gray)

    

    print("=" * 64)
    print(f"文件：{path.relative_to(ROOT)}   {w}×{h}")

    # ---- 1. 两个条件：不透明 + 暗 ----
    opaque = alpha >= 0.99 * 255
    dark = (opaque & (gray <= BOARD_DARK_MAX)).astype(np.uint8)
    print(f"不透明像素 {int(opaque.sum())}，其中暗的 {int(dark.sum())}")
    print(f"棋盘颜色的灰度实测：{int(np.median(gray[opaque & (gray <= BOARD_DARK_MAX)]))}")
    # 通过灰色像素的中位数知道棋盘的灰度值，如果比初始的BOARD_DARK_MAX小很多，那就把它设置小

    cv2.imwrite(str(OUT / "l2_dark.png"), dark * 255)
    # 把零一放大到0，255，方便人眼核实

    # ---- 2. 连通域：给每个连在一起的暗块编号 ----
    # connectivity=8：斜着挨着也算连在一起
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    print(f"\n连通域总数（含背景）：{n}")
    # 背景编号是0，前景编号是1~n-1

    rows = []
    for i in range(1, n):                     # 0 是背景，跳过
        x, y, ww, hh, area = (int(stats[i, cv2.CC_STAT_LEFT]),
                              int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]),
                              int(stats[i, cv2.CC_STAT_HEIGHT]),
                              int(stats[i, cv2.CC_STAT_AREA]))
        if ww < MIN_SIDE or hh < MIN_SIDE:
            continue
        fill = area / float(ww * hh)
        score = area * fill
        # 既奖励"大"，也奖励"方/满"。这样大而规整的矩形得分最高。
        rows.append({"i": i, "x": x, "y": y, "w": ww, "h": hh,
                     "area": area, "fill": fill, "score": score})
    print(f"\n候选暗块：",len(rows))
    rows.sort(key=lambda r: -r["score"])
    print("\n候选暗块（按 score 排序）：")
    print("  序号   位置(x,y)     尺寸(w×h)    面积     填充率   score")
    for k, r in enumerate(rows[:8]):
        print(f"  {k:>2}   ({r['x']:>4},{r['y']:>4})   "
              f"{r['w']:>4}×{r['h']:<4}  {r['area']:>7}  "
              f"{r['fill']:.3f}   {r['score']:>10.0f}")

    # ---- 3. 选棋盘：score 最高 且 fill 过线 ----
    best = None
    for r in rows:
        if r["fill"] > 0.9:
            best = r
            break
    if best is None:
        print("\n❌ 没有暗块通过 fill > 0.25，找不到棋盘")
        return

    print(f"\n✅ 棋盘：({best['x']},{best['y']})  {best['w']}×{best['h']}  "
          f"fill={best['fill']:.3f}  score={best['score']:.0f}")
    print(f"   5×5 时每格 {best['w'] / 5:.1f}×{best['h'] / 5:.1f} 像素")
    print("   注意：这一帧是开局，棋盘还空着，所以 fill=1.0。")
    print("   等碎片铺上去，暗区被挖洞，fill 会掉下来 —— 下面模拟给你看。")

    # ---- 3.5 模拟：棋盘上放 12 块碎片，fill 会掉到多少？ ----
    sim = dark.copy()
    cw, chh = best["w"] / 5.0, best["h"] / 5.0
    for r in range(3):
        for c in range(4):
            x0 = best["x"] + int(c * cw) + 4
            x1 = best["x"] + int((c + 1) * cw) - 4
            y0 = best["y"] + int(r * chh) + 4
            y1 = best["y"] + int((r + 1) * chh) - 4
            sim[y0:y1, x0:x1] = 0          # 这一格被碎片盖住 → 不再是暗像素
    cv2.imwrite(str(OUT / "l2_sim.png"), sim*255)

    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(sim, connectivity=8)
    sim_fill = None
    for i in range(1, n2):
        x, y, ww, hh, area = (int(st2[i, cv2.CC_STAT_LEFT]), int(st2[i, cv2.CC_STAT_TOP]),
                              int(st2[i, cv2.CC_STAT_WIDTH]), int(st2[i, cv2.CC_STAT_HEIGHT]),
                              int(st2[i, cv2.CC_STAT_AREA]))
        if abs(x - best["x"]) <= 3 and abs(y - best["y"]) <= 3 \
                and abs(ww - best["w"]) <= 3 and abs(hh - best["h"]) <= 3:
            sim_fill = area / float(ww * hh)
            break
    if sim_fill is not None:
        print(f"\n   模拟 12 块碎片压上去后：fill = {sim_fill:.3f}")
        print(f"   → 如果阈值设成 0.90，这块棋盘就会被扔掉；"
              f"所以阈值只能取 0.25 这种低值")

    # ---- 4. 画出来看 ----
    vis_all = bgr.copy()
    for k, r in enumerate(rows[:8]):
        color = (0, 255, 0) if r is best else (0, 165, 255)
        cv2.rectangle(vis_all, (r["x"], r["y"]),
                      (r["x"] + r["w"], r["y"] + r["h"]), color, 2)
        cv2.putText(vis_all, f"#{k} s={r['score']:.0f} f={r['fill']:.2f}",
                    (r["x"] + 4, r["y"] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    cv2.imwrite(str(OUT / "l2_components.png"), vis_all)

    vis_board = bgr.copy()
    cv2.rectangle(vis_board, (best["x"], best["y"]),
                  (best["x"] + best["w"], best["y"] + best["h"]), (0, 255, 0), 3)
    # 把 5×5 的格子线也画上，提前感受下一课的"格"
    for k in range(1, 5):
        x = best["x"] + round(k * best["w"] / 5)
        y = best["y"] + round(k * best["h"] / 5)
        cv2.line(vis_board, (x, best["y"]), (x, best["y"] + best["h"]), (0, 255, 0), 1)
        cv2.line(vis_board, (best["x"], y), (best["x"] + best["w"], y), (0, 255, 0), 1)
    cv2.imwrite(str(OUT / "l2_board.png"), vis_board)

    # ---- 5. 和项目里的函数对一下答案 ----
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    # 把src加入到模块搜索路径前面，不然导入不了pipeline
    from pipeline import detect_board
    got = detect_board(img)
    print(f"\n项目里的 detect_board() 给出：{got}")
    ok = got and all(abs(got[k] - best[k]) <= 1 for k in "xywh")
    print("两边一致 ✅" if ok else "两边不一致 ❌（差 1px 以内算一致）")

    print("\n产物：")
    for f in ("l2_dark.png", "l2_components.png", "l2_board.png"):
        print(f"  {OUT / f}")
    print("\n练习（改完重跑，观察数字怎么变）：")
    print("  1) BOARD_DARK_MAX 从 80 改成 30：棋盘还在吗？为什么？")
    print("  2) 改成 200：哪些暗色照片碎片会被并进来？")
    print("  3) 把 fill 阈值从 0.25 改成 0.90：结合上面的模拟数字，说说为什么不行")


if __name__ == "__main__":
    main()
