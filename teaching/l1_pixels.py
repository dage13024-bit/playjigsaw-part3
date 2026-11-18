#!/usr/bin/env python3
"""
第 1 课：一张截图到底是什么？（像素 / 通道 / alpha / 坐标）

这一课只干一件事：把"图像"从抽象概念变成你亲手摸过的数字。
你会看到：
  1. 截图是一个 (高, 宽, 4) 的 uint8 数组，第 4 个通道叫 alpha（透明度）；
  2. 画布背景是**透明**的，碎片是**不透明**的 —— 所以 alpha 天生就是"碎片掩码"，
     根本不需要颜色阈值。这是整个项目"找碎片"的地基；
  3. 像素坐标 (x, y) 和数组下标 [y, x] 是反的 —— 这是新手最容易错的地方；
  4. 透明区域的 RGB 值是没有意义的（可能全是 0，也可能全是 255），
     看颜色之前必须先看 alpha。

运行：
    .venv/bin/python teaching/l1_pixels.py

产物（用图片浏览器打开看）：
    teaching/out/l1_channels.png   四个通道分别长什么样
    teaching/out/l1_alpha.png      alpha 当掩码：白=碎片，黑=背景
    teaching/out/l1_grid.png       画上 100px 网格和坐标，感受像素坐标
"""
from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def pick_canvas() -> Path:
    """找一张现成的画布截图（运行过 play_p3 就会有）。"""
    files = sorted(glob.glob(str(ROOT / "logs" / "*_canvas.png")))
    if not files:
        raise SystemExit("logs/ 里没有 *_canvas.png，先跑一次："
                         ".venv/bin/python play_p3.py --difficulty 25 --dry-run")
    return Path(files[-1])


def main() -> None:
    path = pick_canvas()
    # IMREAD_UNCHANGED：保留第 4 个通道，否则 alpha 会被丢掉
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    print("=" * 60)
    print(f"文件：{path.relative_to(ROOT)}")
    print(f"形状：{img.shape}    数据类型：{img.dtype}")
    h, w = img.shape[:2]
    print(f"高 {h} 像素，宽 {w} 像素 —— 注意 numpy 的顺序是 (高, 宽)，不是 (宽, 高)")

    if img.shape[2] == 3:
        raise SystemExit("这张图没有 alpha 通道，换一张 canvas 截图")

    b, g, r, a = cv2.split(img)

    # ---- 1. alpha：哪些像素是"有东西"的 ----
    opaque = a >= 250
    print(f"\n不透明像素：{int(opaque.sum())} / {h * w} = {opaque.mean():.1%}")
    print(f"透明  像素：{int((~opaque).sum())} / {h * w} = {(~opaque).mean():.1%}")
    print(f"不透明区域的 BGR 均值：{img[opaque][:, :3].mean(axis=0).round(1)}")
    if (~opaque).any():
        print(f"透明  区域的 BGR 均值：{img[~opaque][:, :3].mean(axis=0).round(1)}"
              "   ← 看，透明区的颜色是没意义的")

    # 不透明像素的外接框：这就是"画面里所有碎片一共占了多大地方"
    ys, xs = np.where(opaque)
    print(f"所有碎片的外接框：x {xs.min()}~{xs.max()}，y {ys.min()}~{ys.max()}")

    # ---- 2. 把四个通道拼成一张图看 ----
    panels = []
    for name, ch in (("B", b), ("G", g), ("R", r), ("A", a)):
        p = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
        cv2.putText(p, name, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        panels.append(p)
    cv2.imwrite(str(OUT / "l1_channels.png"), np.hstack(panels))

    # ---- 3. alpha 当掩码：白=碎片，黑=背景 ----
    mask = (a >= 250).astype(np.uint8) * 255
    cv2.imwrite(str(OUT / "l1_alpha.png"), mask)

    # ---- 4. 画网格，建立"像素坐标"的直觉 ----
    vis = img[:, :, :3].copy()
    # 透明背景在 PNG 里是黑的，叠到深灰底上好看一点
    alpha = (a.astype(np.float32) / 255.0)[..., None]
    vis = (vis * alpha + np.full_like(vis, 40) * (1 - alpha)).astype(np.uint8)
    for x in range(0, w, 100):
        cv2.line(vis, (x, 0), (x, h), (0, 180, 255), 1)
        cv2.putText(vis, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
    for y in range(0, h, 100):
        cv2.line(vis, (0, y), (w, y), (0, 180, 255), 1)
        cv2.putText(vis, str(y), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
    cv2.imwrite(str(OUT / "l1_grid.png"), vis)

    # ---- 5. 一个具体的像素：下标是 [y, x] ----
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    bright = (a >= 250) & (gray >= 200)          # 碎片上的亮像素
    dark = (a >= 250) & (gray <= 40)             # 棋盘/阴影这类暗像素
    print(f"\n不透明像素里：亮像素(碎片) {int(bright.sum())}，"
          f"暗像素(棋盘等) {int(dark.sum())} —— alpha 只说'有没有东西'，"
          f"分不出棋盘和碎片，下一步才用亮度/颜色分")
    cand = np.argwhere(bright)
    cand = cand[(cand[:, 1] < h) & (cand[:, 0] < w)]     # 保证 x/y 互换也不越界
    if len(cand):
        py, px = cand[len(cand) // 2]
        print(f"取一个碎片上的像素：x={px}, y={py}")
        print(f"  img[{py}, {px}] 的 BGRA = {img[py, px]}   ← 数组下标是 [y, x]")
        print(f"  img[{px}, {py}] 的 BGRA = {img[px, py]}   ← 写反了读到的是别处")
    ys2, xs2 = np.where(~opaque)
    print(f"取一个透明背景像素：x={xs2[0]}, y={ys2[0]}  BGRA = {img[ys2[0], xs2[0]]}")

    print("\n产物：")
    for f in ("l1_channels.png", "l1_alpha.png", "l1_grid.png"):
        print(f"  {OUT / f}")
    print("\n看完后回答自己三个问题：")
    print("  1) 为什么这个项目找碎片不靠颜色阈值，而靠 alpha？")
    print("  2) 如果把 alpha 阈值从 250 改成 200，掩码会变多还是变少？为什么？")
    print("  3) alpha 掩码里同时有棋盘和碎片，怎么把棋盘单独挑出来？"
          "（提示：棋盘是暗的、且是一整块实心矩形）")


if __name__ == "__main__":
    main()
