#!/usr/bin/env python3
"""
第 4 课：一条边是凸、是凹、还是直的？

纯白碎片没有任何图案，能用的信息只有"形状"。而形状又只有一种问法：

    这条边的外面，有没有东西鼓出来？   （凸 tab）
    这条边的里面，有没有东西凹进去？   （凹 slot）
    外面没有、里面是实的 → 直边          （外框）

具体做法：在每条边的"外侧"和"内侧"各取一个窄窗口，数 alpha 覆盖率：

        外侧窗口            内侧窗口
    ┌──────────────┐   ┌──────────────┐
    │  xR+d0 ~ d1  │   │  xR-d1 ~ d0  │
    └──────────────┘   └──────────────┘
                  ↑ 边线 xR

    d0 = 0.25 × tab，d1 = 1.00 × tab
    tab = 0.22 × min(碎片宽, 碎片高)   ← 卡扣伸出的长度

窗口只取边的中间 34%~66%，因为卡扣长在中间，四个角永远是实心的。

判定规则（来自 Part 2 的实测）：
    内侧 < 0.70        → 凹 (-)
    外侧 > 0.50        → 凸 (+)
    外侧 < 0.15        → 直 (0)
    其余               → 拿不准 (?)   ← 被邻片压住 / 阴影

运行：
    .venv/bin/python teaching/l4_edges.py

产物：
    teaching/out/l4_windows.png   一块碎片四条边的取样窗口（红=外，蓝=内）
    teaching/out/l4_edges.png     所有碎片的四边判读结果
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "teaching" / "out"
OUT.mkdir(parents=True, exist_ok=True)

SIDES = ("top", "right", "bottom", "left")
CHAR = {1: "+", -1: "-", 0: "0", None: "?"}
COLOR = {1: (0, 200, 0), -1: (0, 0, 255), 0: (160, 160, 160), None: (0, 165, 255)}

# 找到最后的（最新的）canvas图片路径
def pick_canvas() -> Path:
    files = sorted(glob.glob(str(ROOT / "logs" / "*_canvas.png")))
    if not files:
        raise SystemExit("logs/ 里没有 *_canvas.png，先跑一次 "
                         ".venv/bin/python play_p3.py --difficulty 25 --dry-run")
    return Path(files[-1])

# 计算掩码图像中某个矩形区域内前景像素的占比，返回 0.0～1.0判断某个区域够够不够密
def density(mask, x0, y0, x1, y1) -> float:
    h, w = mask.shape[:2]
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    # max,min防止越界
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((mask[y0:y1, x0:x1] > 0).mean())

# 返回某条边两边的两个矩形区域，后面统计像素量判断凸边凹边
def windows(slot, pw, ph, tab, side, a=0.34, b=0.66):
    """返回 (外侧窗口, 内侧窗口)，每个窗口是 (x0,y0,x1,y1)。"""
    xL, yT = slot["x"], slot["y"]
    xR, yB = xL + pw, yT + ph
    d0, d1 = 0.25 * tab, 1.00 * tab
    if side == "right":
        return ((xR + d0, yT + a * ph, xR + d1, yT + b * ph),
                (xR - d1, yT + a * ph, xR - d0, yT + b * ph))
    if side == "left":
        return ((xL - d1, yT + a * ph, xL - d0, yT + b * ph),
                (xL + d0, yT + a * ph, xL + d1, yT + b * ph))
    if side == "bottom":
        return ((xL + a * pw, yB + d0, xL + b * pw, yB + d1),
                (xL + a * pw, yB - d1, xL + b * pw, yB - d0))
    return ((xL + a * pw, yT - d1, xL + b * pw, yT - d0),
            (xL + a * pw, yT + d0, xL + b * pw, yT + d1))

# 通过内外框的像素密度判边
def label_of(out_d, in_d):
    """复刻 src/slots.py:label_from_density 的三个阈值。"""
    if in_d < 0.70:
        return -1
    if out_d > 0.50:
        return 1
    if out_d < 0.15:
        return 0
    return None


def main() -> None:
    path = pick_canvas()
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片：{path}")
    bgr = img[:, :, :3].copy()
    mask = (img[:, :, 3] >= 253).astype(np.uint8)
    cv2.imwrite(str(OUT / "l4_mask.png"), mask * 255)

    ch, cw = mask.shape[:2]

    from pipeline import detect_board
    from slots import occupied_slots

    board = detect_board(img)
    if board is None:
        raise SystemExit(f"无法检测到棋盘：{path}")
    s = 5
    pw, ph = board["w"] / s, board["h"] / s
    tab = 0.22 * min(pw, ph)
    occ = occupied_slots(img, board, pw, ph, expect=30)
    print("=" * 64)
    print(f"棋盘 ({board['x']},{board['y']}) {board['w']}×{board['h']}  "
          f"碎片 {pw:.1f}×{ph:.1f}  卡扣 {tab:.1f}px")
    print(f"有碎片槽位 {len(occ)} 块")

    # ---- 1. 逐块判四条边 ----
    results = []
    for k, slot in enumerate(occ):
        labs = {}
        for side in SIDES:
            outer, inner = windows(slot, pw, ph, tab, side)
            o = density(mask, *outer)
            i = density(mask, *inner)
            labs[side] = (label_of(o, i), o, i)
        results.append(labs)
# results：每个槽位一个字典 {side: (label, o, i)}
    print("\n前 8 块的判读（out=外侧密度, inn=内侧密度）：")
    print("  片   top          right        bottom       left")
    for k in range(min(8, len(results))):
        cells = []
        for side in SIDES:
            lab, o, i = results[k][side]
            cells.append(f"{CHAR[lab]}({o:.2f}/{i:.2f})")
        print(f"  {k:>2}   " + "  ".join(cells))

    from collections import Counter
    # 用 Counter 统计所有槽位所有边上 +/-/0/? 各出现多少次，打印分布，便于发现异常（如 ? 很多说明取样有问题）。
    cnt = Counter(CHAR[r[s][0]] for r in results for s in SIDES)
    print(f"\n四边标签分布：{dict(cnt)}")

    # ---- 2. 和项目函数对答案（前 3 块） ----
    from slots import edges_at_slot, label_from_density
    same = True
    for k in range(min(3, len(occ))):
        d = edges_at_slot(img, occ[k], pw, ph, tab)
        for side in SIDES:
            ref = label_from_density(*d[side])[0]
            if ref != results[k][side][0]:
                same = False
    print(f"与项目 edges_at_slot/label_from_density 对照：{'一致 ✅' if same else '不一致 ❌'}")

    # ---- 3. 画取样窗口（第一块）----
    slot = occ[0]
    pad = int(tab * 2)
    x0 = max(0, int(slot["x"] - pad)); y0 = max(0, int(slot["y"] - pad))
    x1 = min(cw, int(slot["x"] + pw + pad)); y1 = min(ch, int(slot["y"] + ph + pad))
    zoom = bgr[y0:y1, x0:x1].copy()
    for side in SIDES:
        outer, inner = windows(slot, pw, ph, tab, side)
        for rect, color in ((outer, (0, 0, 255)), (inner, (255, 120, 0))):
            cv2.rectangle(zoom,
                          (int(rect[0] - x0), int(rect[1] - y0)),
                          (int(rect[2] - x0), int(rect[3] - y0)), color, 1)
    cv2.imwrite(str(OUT / "l4_windows.png"), cv2.resize(zoom, None, fx=3, fy=3,
                                                        interpolation=cv2.INTER_NEAREST))

    # ---- 4. 把所有判读写到图上 ----
    vis = bgr.copy()
    for k, (slot, labs) in enumerate(zip(occ, results)):
        for side in SIDES:
            lab = labs[side][0]
            if side == "top":
                p = (int(slot["x"] + pw / 2), int(slot["y"] - tab * 0.35))
            elif side == "bottom":
                p = (int(slot["x"] + pw / 2), int(slot["y"] + ph + tab * 0.55))
            elif side == "left":
                p = (int(slot["x"] - tab * 0.55), int(slot["y"] + ph / 2))
            else:
                p = (int(slot["x"] + pw + tab * 0.25), int(slot["y"] + ph / 2))
            cv2.putText(vis, CHAR[lab], p, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, COLOR[lab], 2)
    cv2.imwrite(str(OUT / "l4_edges.png"), vis)

    print("\n产物：")
    for f in ("l4_windows.png", "l4_edges.png"):
        print(f"  {OUT / f}")
    print("\n练习：")
    print("  1) 把 a/b 从 0.34/0.66 改成 0.05/0.95：窗口覆盖整条边，")
    print("     卡扣只占中间 1/3，密度被稀释（实测 out 0.51→0.19），大片标签变 ?")
    print("  2) 把 d1 改成 1.5×tab：既越过卡扣尖端（稀释），又伸进邻片（污染），")
    print("     实测 30 条边变化，既有 +→? 也有 ?→+")
    print("  3) 找一块 '?'：先判断是'自己被压住'还是'窗口里混进了别人的卡扣'，")
    print("     提层（点一下碎片）只对前者有效")


if __name__ == "__main__":
    main()
