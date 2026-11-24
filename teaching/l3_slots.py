#!/usr/bin/env python3
"""
第 3 课：碎片不是"分割"出来的，是"算"出来的

为什么不能用阈值分割照片碎片？
  1. 一块碎片内部颜色千变万化（狗有白毛黑毛、背景有草地天空），
     一个亮度阈值会把同一块碎片切成十几块；
  2. 相邻碎片的卡扣会伸进对方的格子里，连通域会粘成一坨；
  3. 碎片之间的阴影还会让 alpha 变成 0.5~0.9，边界更糊。

但游戏引擎把碎片**按固定间距摆在左右两条托盘里**。间距是碎片尺寸的 1.16 倍，
所以只要知道棋盘位置和格数，每个槽位的坐标可以用一条公式直接算出来：

    step_x = 碎片宽 × 1.16
    step_y = 碎片高 × 1.16
    左侧第 col 列、第 row 行的槽位：
        x = 0 + col × step_x + (step_x - 碎片宽) / 2
        y = 0 + row × step_y + (step_y - 碎片高) / 2
    右侧同理，只是原点从 board.x + board.w 开始。

槽位总数可能比碎片多（5×5 有 32 个槽、30 块碎片），所以还要用 alpha 覆盖率
判断"这个槽位上到底有没有碎片"。

运行：
    .venv/bin/python teaching/l3_slots.py

产物：
    teaching/out/l3_slots.png       全部槽位 + 有碎片的槽位
    teaching/out/l3_zoom.png        左托盘的放大图（看清槽位间距）
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)

STEP = 1.16          # 引擎常量：槽距 = 1.16 × 碎片尺寸
FILL_MIN = 0.55      # 槽位 alpha 覆盖率高于此值 → 认为有碎片


def pick_canvas() -> Path:
    files = sorted(glob.glob(str(ROOT / "logs" / "*_canvas.png")))
    if not files:
        raise SystemExit("logs/ 里没有 *_canvas.png，先跑一次 "
                         ".venv/bin/python play_p3.py --difficulty 25 --dry-run")
    return Path(files[-1])


def build_slots(board, pw, ph, cw, ch, step=STEP):
    """完全按引擎公式手写一遍（不调用 src/slots.py），方便你对照。"""
    step_x, step_y = pw * step, ph * step
    band = max(0.0, float(board["x"]))          # 棋盘左边到画布左边的宽度
    cols_per_side = int(math.floor(band / step_x))
    rows = int(math.floor(ch / step_y)) #向下取整
    slots = []
    for side in (0, 1):
        origin_x = 0.0 if side == 0 else board["x"] + board["w"]
        for col in range(cols_per_side):
            for row in range(rows):
                x = origin_x + col * step_x + (step_x - pw) / 2.0
                y = row * step_y + (step_y - ph) / 2.0
                if x < 0 or y < 0 or x + pw > cw or y + ph > ch:
                    continue
                slots.append({"side": side, "col": col, "row": row,
                              "x": x, "y": y})
    return slots, step_x, step_y, band, cols_per_side, rows


def main() -> None:
    path = pick_canvas()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片：{path}")
    ch, cw = img.shape[:2]
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    from pipeline import detect_board
    board = detect_board(img)
    if board is None:
        raise SystemExit("没找到棋盘")

    s = 5
    pw, ph = board["w"] / s, board["h"] / s
    slots, step_x, step_y, band, cols, rows = build_slots(
        board, pw, ph, cw, ch)
    print("=" * 64)
    print(f"棋盘 ({board['x']},{board['y']}) {board['w']}×{board['h']}")
    print(f"5×5 → 碎片 {pw:.1f}×{ph:.1f}")
    print(f"槽距 step_x = {pw:.2f} × 1.16 = {step_x:.2f}")
    print(f"槽距 step_y = {ph:.2f} × 1.16 = {step_y:.2f}")
    print(f"左侧带宽 band = 棋盘左边 x = {band:.0f}")
    print(f"每侧列数 = floor({band:.0f} / {step_x:.2f}) = {cols}")
    print(f"每列行数 = floor({ch} / {step_y:.2f}) = {rows}")
    print(f"槽位总数 = 2 侧 × {cols} 列 × {rows} 行 = {len(slots)}（碎片 30 块）")

    # ---- 每个槽位有没有碎片：alpha 覆盖率 ----
    mask = (alpha >= 253).astype(np.uint8)
    for k, slot in enumerate(slots):
        x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
        x1, y1 = int(round(slot["x"] + pw)), int(round(slot["y"] + ph))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(cw, x1), min(ch, y1)
        slot["fill"] = float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0
        slot["occupied"] = slot["fill"] >= FILL_MIN

    occ = [x for x in slots if x["occupied"]]
    print(f"\n有碎片的槽位：{len(occ)}（预期 30）")
    print("空的槽位：", [(x["side"], x["col"], x["row"]) for x in slots
                          if not x["occupied"]])

    # ---- 手算一个槽位，和公式对答案 ----
    demo = next(x for x in slots if x["side"] == 0 and x["col"] == 1 and x["row"] == 3)
    print(f"\n手算示例：右侧 col=1 row=3")
    print(f"  x = 0 + 1×{step_x:.2f} + ({step_x:.2f}-{pw:.1f})/2"
          f" = {demo['x']:.1f}")
    print(f"  y = 0 + 3×{step_y:.2f} + ({step_y:.2f}-{ph:.1f})/2 = {demo['y']:.1f}")
    print(f"  脚本算出来 = ({demo['x']:.1f}, {demo['y']:.1f})  "
          f"alpha 覆盖率 {demo['fill']:.2f}")

    # ---- 和项目函数对答案 ----
    from slots import tray_slots, occupied_slots
    t = tray_slots(board, pw, ph, cw, ch)
    o = occupied_slots(img, board, pw, ph, expect=30)
    print(f"\n项目 tray_slots() 槽位数 = {len(t)}，"
          f"occupied_slots() 有碎片 = {len(o)}")

    # ---- 画图 ----
    vis = bgr.copy()
    cv2.rectangle(vis, (board["x"], board["y"]),
                  (board["x"] + board["w"], board["y"] + board["h"]),
                  (255, 120, 0), 2)
    for k, slot in enumerate(slots):
        x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
        x1, y1 = int(round(slot["x"] + pw)), int(round(slot["y"] + ph))
        if slot["occupied"]:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, str(k), (x0 + 4, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (120, 120, 120), 1)
    cv2.imwrite(str(OUT / "l3_slots.png"), vis)

    zoom = vis[0:min(ch, int(3 * step_y)), 0:min(cw, int(2 * step_x) + 40)]
    cv2.imwrite(str(OUT / "l3_zoom.png"), zoom)

    print("\n产物：")
    for f in ("l3_slots.png", "l3_zoom.png"):
        print(f"  {OUT / f}")
    print("\n练习：")
    print("  1) 手算左侧 col=1, row=3 的 x,y，和图上编号对一对")
    print("  2) 把 STEP 改成 1.0：槽位会重叠吗？为什么引擎要留 0.16 的缝？")
    print("  3) 槽位有 32 个、碎片只有 30 块，为什么不能直接假设'每个槽都有碎片'？")


if __name__ == "__main__":
    main()
