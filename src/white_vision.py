#!/usr/bin/env python3
"""
Part 2（纯白碎片）视觉核心 - 分割 + 卡扣边分类

设计要点（全部来自实测）：
  * canvas 是带 alpha 的：render() 会 clearRect，背景 alpha=0，所以 toDataURL 拿到的图
    里 alpha>=0.99 就是"有东西"的区域，不需要任何颜色阈值。
  * 碎片本体是纯白 255；碎片之间的重影/阴影落在 26~250（实测直方图双峰）。
    所以 alpha>=0.99 且 亮度>=240  -> 恰好切出 25 个连通域（=5x5）。
    阈值放宽到 200 时相邻碎片会粘成 12 个，这是最关键的参数。
  * 边分类不用"中位基线"，而是先估出**核心矩形**（四个角的直线段交点），
    再对每条边取带符号的偏差：向外鼓 0.22*min(W,H) => 凸(+1)，向内凹 => 凹(-1)。
    bbox 包含卡扣，直接按 bbox 分带会被邻近卡扣污染（这是第一版失败的原因）。
  * 贴 canvas 边界的边可能被裁切，标记 unknown，交给约束求解去补。

用法:
  .venv/bin/python src/white_vision.py --canvas logs/p2_d25_canvas.png \
      --truth logs/p2_d25_truth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WHITE_MIN = 240          # 碎片本体纯白 255；阴影/重影 26~250 -> 排除
ALPHA_MIN = 0.99
TAB_RATIO = 0.22         # 与 PieceCutter.tabSize 一致：tabSize = 0.22*min(pw,ph)
# 语义：本模块内部 1=凸(向碎片外鼓) / -1=凹 / 0=直边
# 注意引擎的约定是反的：PieceCutter.drawEdge 的法向量在四条边上都指向碎片内部，
# 所以 edges=+1 实际渲染成"凹"、edges=-1 渲染成"凸"。比对真值时必须翻一次。
TRUTH_TO_SIGN = {1: -1, -1: 1, 0: 0}
SIDES = ("top", "right", "bottom", "left")
LBL = {1: "+", -1: "-", 0: "0", None: "?"}


# ---------------------------------------------------------------- segmentation
def piece_mask(img: np.ndarray) -> np.ndarray:
    """alpha + 亮度双条件取碎片本体（对纯白题）。"""
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img.shape[:2], np.float32)
    gray = img[:, :, :3].astype(np.int16).mean(axis=2)
    return (((alpha >= ALPHA_MIN) & (gray >= WHITE_MIN)).astype(np.uint8)) * 255


def piece_mask_generic(img: np.ndarray) -> np.ndarray:
    """通用版本（Part1/3 用）：只看 alpha，不看颜色。"""
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img.shape[:2], np.float32)
    return ((alpha >= ALPHA_MIN).astype(np.uint8)) * 255


def extract_pieces(mask: np.ndarray, min_area_ratio: float = 0.15):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
    if not areas:
        return []
    med = float(np.median([a for a, _ in areas]))
    out = []
    for area, i in areas:
        if area < med * min_area_ratio:
            continue
        m = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        out.append({"mask": m, "contour": c, "area": int(area),
                    "bbox": cv2.boundingRect(c),
                    "centroid": (float(cents[i][0]), float(cents[i][1]))})
    out.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    return out


# ------------------------------------------------------------- core rectangle
def core_rect(piece):
    """投影平台法估核心矩形。

    碎片 = 矩形 + 卡扣凸起/凹陷。沿 x 方向统计每列像素数：
      * 卡扣凸起的列只有很矮的一条（球头）-> 计数低
      * 凹槽所在列只少了槽高的一半左右 -> 计数仍 > 0.45*max
    所以 "计数 >= 0.45*max" 的连续区段 = 核心区段。比 approxPolyDP 的角点匹配稳得多。
    """
    m = piece["mask"]
    col = (m > 0).sum(axis=0).astype(np.float32)
    row = (m > 0).sum(axis=1).astype(np.float32)

    def plateau(v, frac=0.45, pad=6):
        thr = max(1.0, frac * float(v.max()))
        ok = np.flatnonzero(v >= thr)
        if ok.size == 0:
            return 0, len(v) - 1
        # 去掉两端 padding 造成的小尖刺：取最长连续段
        runs, start = [], ok[0]
        for a, b in zip(ok, ok[1:]):
            if b - a > pad:
                runs.append((start, a)); start = b
        runs.append((start, ok[-1]))
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        return int(lo), int(hi)

    xL, xR = plateau(col)
    yT, yB = plateau(row)
    return float(xL), float(yT), float(xR - xL), float(yB - yT)


def classify_edges(piece, canvas_wh, band=(0.20, 0.80)):
    """返回 {side: (label, out, inn)}，label ∈ {1,0,-1,None}。"""
    W, H = canvas_wh
    xL, yT, cw, ch = core_rect(piece)
    xR, yB = xL + cw, yT + ch
    if cw < 10 or ch < 10:
        return {s: (None, 0.0, 0.0) for s in SIDES} | {"_rect": (xL, yT, cw, ch, 0.0)}
    t = TAB_RATIO * min(cw, ch)
    thr = 0.5 * t
    pts = piece["contour"].reshape(-1, 2).astype(np.float64)
    lo, hi = band

    # 每条边只看"该边附近 1.5t 内 + 中间段"，防止相邻边的卡扣污染
    y_in = (pts[:, 1] >= yT + lo * ch) & (pts[:, 1] <= yT + hi * ch)
    x_in = (pts[:, 0] >= xL + lo * cw) & (pts[:, 0] <= xL + hi * cw)
    near = 1.5 * t
    sel = {
        "right": pts[y_in & (pts[:, 0] >= xR - near)],
        "left": pts[y_in & (pts[:, 0] <= xL + near)],
        "bottom": pts[x_in & (pts[:, 1] >= yB - near)],
        "top": pts[x_in & (pts[:, 1] <= yT + near)],
    }
    ref = {"right": (0, xR, +1.0), "left": (0, xL, -1.0),
           "bottom": (1, yB, +1.0), "top": (1, yT, -1.0)}
    out = {}
    for s in SIDES:
        axis, r0, sign = ref[s]
        q = sel[s]
        if q.size == 0:
            out[s] = (0, 0.0, 0.0)
            continue
        dev = (q[:, axis] - r0) * sign
        o, i = float(dev.max()), float(-dev.min())
        label = 1 if o > thr else (-1 if i > thr else 0)
        bx, by, bw, bh = piece["bbox"]
        if ((s == "left" and bx <= 1) or (s == "right" and bx + bw >= W - 1)
                or (s == "top" and by <= 1) or (s == "bottom" and by + bh >= H - 1)):
            label = None
        out[s] = (label, o, i)
    out["_rect"] = (xL, yT, cw, ch, t)
    return out


# --------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--vis", default="logs/white_edges_vis.png")
    args = ap.parse_args()

    img = cv2.imread(args.canvas, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"读不到 {args.canvas}")
    m = piece_mask(img)
    pieces = extract_pieces(m)
    print(f"碎片数: {len(pieces)}")
    for p in pieces:
        p["edges"] = classify_edges(p, (img.shape[1], img.shape[0]))

    # 全局核心尺寸一致性：被别的碎片压住的碎片，核心矩形会一边变短，
    # 那一边读成"直边"其实是假直边 -> 改成 unknown 交给约束求解
    cws = [p["edges"]["_rect"][2] for p in pieces]
    chs = [p["edges"]["_rect"][3] for p in pieces]
    med_cw, med_ch = float(np.median(cws)), float(np.median(chs))
    for p in pieces:
        xL, yT, cw, ch, t = p["edges"]["_rect"]
        bx, by, bw, bh = p["bbox"]
        if ch < 0.92 * med_ch and by > 1 and by + bh < img.shape[0] - 1:
            for side in ("top", "bottom"):
                if p["edges"][side][0] == 0:
                    p["edges"][side] = (None,) + p["edges"][side][1:]
        if cw < 0.92 * med_cw and bx > 1 and bx + bw < img.shape[1] - 1:
            for side in ("left", "right"):
                if p["edges"][side][0] == 0:
                    p["edges"][side] = (None,) + p["edges"][side][1:]
        xL, yT, cw, ch, t = p["edges"]["_rect"]
        print(f"  core=({xL:6.1f},{yT:6.1f}) {cw:5.1f}x{ch:5.1f} t={t:4.1f} "
              + " ".join(f"{s}:{LBL[p['edges'][s][0]]}({p['edges'][s][1]:+4.0f}/{p['edges'][s][2]:+4.0f})"
                         for s in SIDES))

    vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(pieces):
        e = p["edges"]
        xL, yT, cw, ch, t = e["_rect"]
        cv2.rectangle(vis, (int(xL), int(yT)), (int(xL + cw), int(yT + ch)), (0, 165, 255), 1)
        cv2.drawContours(vis, [p["contour"]], -1, (60, 200, 60), 1)
        txt = "".join(LBL[e[s][0]] for s in SIDES)
        cv2.putText(vis, f"{i}:{txt}", (int(xL) + 1, int(yT) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(args.vis, vis)
    print(f"可视化: {args.vis}  (标注顺序 T R B L; +凸 -凹 0直 ?未知; 橙框=核心矩形)")

    if not args.truth:
        return 0

    truth = json.loads(Path(args.truth).read_text())
    geo = truth["pieces"] and truth["geometry"]
    tp = truth["pieces"]
    cand = []
    for i, p in enumerate(pieces):
        x, y, w, h = p["bbox"]
        pcx, pcy = x + w / 2, y + h / 2
        for j, t_ in enumerate(tp):
            # 真值里 x/y 是"当前位置"左上角；cx/cy 是正确位置（别混）
            d = ((t_["x"] + geo["pw"] / 2 - pcx) ** 2 +
                 (t_["y"] + geo["ph"] / 2 - pcy) ** 2) ** 0.5
            cand.append((d, i, j))
    cand.sort()
    used_i, used_j, pairs = set(), set(), []
    for d, i, j in cand:
        if i in used_i or j in used_j or d > geo["pw"] * 0.5:
            continue
        used_i.add(i); used_j.add(j); pairs.append((i, j, d))

    ok = tot = unk = 0
    conf = np.zeros((3, 3), dtype=int)
    order = {-1: 0, 0: 1, 1: 2}
    wrong = []
    for i, j, d in pairs:
        for s in SIDES:
            tl, pred = TRUTH_TO_SIGN[tp[j]["edges"][s]], pieces[i]["edges"][s][0]
            tot += 1
            if pred is None:
                unk += 1
                continue
            conf[order[tl], order[pred]] += 1
            if pred == tl:
                ok += 1
            else:
                wrong.append((pieces[i]["centroid"], s, tl, pred,
                              pieces[i]["edges"][s][1], pieces[i]["edges"][s][2]))
    print(f"\n匹配 {len(pairs)}/{len(pieces)} 块；边标签 {ok}/{tot - unk} "
          f"= {ok / max(1, tot - unk):.1%}（{unk} 条贴边记 unknown）")
    print("混淆矩阵 (行=真值 列=预测, 顺序 -1/0/+1):\n", conf)
    if wrong:
        print("错例前 8 条 (center, side, truth, pred, out, inn):")
        for w_ in wrong[:8]:
            print("  ", np.round(w_[0], 1), w_[1], w_[2], w_[3], f"{w_[4]:.1f}/{w_[5]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ======================================================================
# 模板 IoU 分类器（第二版）：用 PieceCutter 的精确 Bézier 几何合成模板
# ======================================================================
# 为什么需要它：投影/平台法只能把核心矩形估到 ±10px，遇到"凸起被邻片压住"
# 或卡扣颈部顶点时会把凸读成直、把直读成凹。棋盘给出了精确的 pw/ph 之后，
# 就可以把 81 种边组合全部合成出来直接比 IoU，不依赖任何启发式阈值。
COMBOS = [(t, r, b, l)
          for t in (0, 1, -1) for r in (0, 1, -1)
          for b in (0, 1, -1) for l in (0, 1, -1)]
_TEMPLATE_CACHE: dict = {}


def _cubic(p0, p1, p2, p3, n=10):
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def _edge_poly(x1, y1, x2, y2, label, tab):
    """复刻 PieceCutter.drawEdge 的路径。label: 1=凸(向外) -1=凹 0=直。

    引擎里 type=+1 实际渲染成"凹"（法向量总指向碎片内部），所以这里取 sg=-label。
    """
    if label == 0:
        return [(x2, y2)]
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    if L == 0:
        return [(x2, y2)]
    nx, ny = -dy / L, dx / L
    sg = -label

    def pt(u, off):
        return (x1 + dx * u + nx * tab * off * sg,
                y1 + dy * u + ny * tab * off * sg)

    p1, p2, p3, p4 = pt(0.38, 0.0), pt(0.42, 0.30), pt(0.35, 1.05), pt(0.50, 1.15)
    p5, p6, p7 = pt(0.65, 1.05), pt(0.58, 0.30), pt(0.62, 0.0)
    pts = [p1]
    pts += _cubic(p1, p2, p3, p4)
    pts += _cubic(p4, p5, p6, p7)
    pts.append((x2, y2))
    return pts


def _build_templates(pw: float, ph: float):
    key = (round(pw, 2), round(ph, 2))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    tab = TAB_RATIO * min(pw, ph)
    pad = int(np.ceil(tab * 1.3)) + 2
    W = int(np.ceil(pw)) + 2 * pad
    H = int(np.ceil(ph)) + 2 * pad
    ox, oy = pad, pad
    out = []
    for combo in COMBOS:
        t, r, b, l = combo
        poly = [(ox, oy)]
        poly += _edge_poly(ox, oy, ox + pw, oy, t, tab)
        poly += _edge_poly(ox + pw, oy, ox + pw, oy + ph, r, tab)
        poly += _edge_poly(ox + pw, oy + ph, ox, oy + ph, b, tab)
        poly += _edge_poly(ox, oy + ph, ox, oy, l, tab)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 255)
        ys, xs = np.nonzero(m)
        mo = cv2.moments(m, binaryImage=True)
        out.append({"combo": combo, "mask": m,
                    "bbox": (int(xs.min()), int(ys.min()),
                             int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                    "cen": (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]),
                    "core": (pad, pad)})
    _TEMPLATE_CACHE[key] = (out, (W, H), tab)
    return _TEMPLATE_CACHE[key]


def classify_by_template(mask, bbox, pw, ph, centroid):
    """对单块碎片，在 81 种模板里找 IoU 最高者（**质心对齐**）。

    bbox 对齐会被"卡扣被裁掉/被邻片压住"整体带偏（实测 IoU 掉到 0.75、margin≈0）；
    质心对齐在同一套卡扣下天然重合，真值组合的 IoU 中位数 0.92。
    返回 (labels, best_iou, margin)：margin 小说明有歧义（遮挡），调用方应记为 unknown。
    """
    templates, _, tab = _build_templates(pw, ph)
    H, W = mask.shape[:2]
    bx, by, bw, bh = bbox
    m = int(np.ceil(2 * tab)) + 6   # ROI 要能装下模板，否则正确组合会被裁掉
    x0, y0 = max(0, bx - m), max(0, by - m)
    x1, y1 = min(W, bx + bw + m), min(H, by + bh + m)
    roi = mask[y0:y1, x0:x1]
    cx, cy = centroid[0] - x0, centroid[1] - y0

    def iou(tpl):
        dx, dy = cx - tpl["cen"][0], cy - tpl["cen"][1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        w = cv2.warpAffine(tpl["mask"], M, (roi.shape[1], roi.shape[0]),
                           flags=cv2.INTER_NEAREST)
        inter = np.count_nonzero((w > 0) & (roi > 0))
        union = np.count_nonzero((w > 0) | (roi > 0))
        return inter / union if union else 0.0

    scored = []
    for tpl in templates:
        tw, th = tpl["bbox"][2], tpl["bbox"][3]
        if abs(tw - bw) > 2.0 * tab or abs(th - bh) > 2.0 * tab:
            continue
        scored.append((iou(tpl), tpl["combo"]))
    if len(scored) < 2:
        scored = [(iou(tpl), tpl["combo"]) for tpl in templates]
    scored.sort(reverse=True)
    best_iou, best_combo = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    return dict(zip(SIDES, best_combo)), float(best_iou), float(best_iou - second)


# ======================================================================
# 分区密度分类器（第三版）：bbox 定"哪几条边凸"，密度定"凹还是直"
# ======================================================================
# 为什么不用 IoU：碎片外沿有 1~2px 的高光描边 + 抗锯齿晕圈(≈540px)，
# 而一个卡扣只有 ~700px，整形状 IoU 里两者只差 0.02，判不开。
# bbox 的宽度/高度只被"凸起"改变（凹槽和直边都不变），所以：
#   nx = round((bw - pw) / (1.15*tab))  -> 左右两条边里有几条是凸
# 再对每条边取两个小窗口的密度，就能把 凸/凹/直 分开，且对晕圈不敏感。
def _dens(mask, x0, y0, x1, y1):
    H, W = mask.shape[:2]
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask[y0:y1, x0:x1]
    return float(np.count_nonzero(roi)) / float(roi.size)


def _side_evidence(mask, core, t, side, a=0.34, b=0.66):
    xL, yT, W, H = core
    xR, yB = xL + W, yT + H
    d0, d1 = 0.25 * t, 1.00 * t
    if side == "right":
        out = _dens(mask, xR + d0, yT + a * H, xR + d1, yT + b * H)
        inn = _dens(mask, xR - d1, yT + a * H, xR - d0, yT + b * H)
    elif side == "left":
        out = _dens(mask, xL - d1, yT + a * H, xL - d0, yT + b * H)
        inn = _dens(mask, xL + d0, yT + a * H, xL + d1, yT + b * H)
    elif side == "bottom":
        out = _dens(mask, xL + a * W, yB + d0, xL + b * W, yB + d1)
        inn = _dens(mask, xL + a * W, yB - d1, xL + b * W, yB - d0)
    else:  # top
        out = _dens(mask, xL + a * W, yT - d1, xL + b * W, yT - d0)
        inn = _dens(mask, xL + a * W, yT + d0, xL + b * W, yT + d1)
    return out, inn


def _label_from_evidence(out, inn):
    """(标签, 置信度)。

    实测统计（用真值核心位置量出来的）：
        真值=凹: out=0.00, inn=0.33
        真值=直: out=0.00, inn=1.00
        真值=凸: out=0.55, inn=1.00
    所以判据只需要两个阈值：inn 先分"内侧空不空"，out 再分"外侧鼓不鼓"。
    落在模糊带的（多半是卡扣被邻片压住）一律返回 None，交给求解器。
    """
    if inn < 0.70:
        return -1, float(min(1.0, (0.70 - inn) / 0.37 + 0.4))
    if out > 0.50:
        return 1, float(min(1.0, (out - 0.50) / 0.10 + 0.6))
    if out < 0.15:
        return 0, float(min(1.0, (inn - 0.70) / 0.30 + 0.5))
    return None, 0.0


def classify_zones(mask, bbox, pw, ph, canvas_wh):
    """返回 ({side: (label, conf)}, core, nx, ny)。"""
    tab = TAB_RATIO * min(pw, ph)
    ext = 1.15 * tab
    bx, by, bw, bh = bbox
    nx = int(round((bw - pw) / ext))
    ny = int(round((bh - ph) / ext))
    nx, ny = max(0, min(2, nx)), max(0, min(2, ny))

    # 每条轴最多 2 个假设（哪一侧是凸起）；凸起数由 bbox 决定，轴间独立。
    # 关键：核心位置错了，证据会变糊（直边 inn 掉到 0.8、凹边 inn 涨到 0.5）。
    # 所以用"证据锐度"来选核心位置，而不是靠 bbox 的 tab 计数硬推。
    def sharpness(core):
        sc = 0.0
        out_ev = {}
        for side in SIDES:
            o, i = _side_evidence(mask, core, tab, side)
            out_ev[side] = (o, i)
            sc += abs(i - 0.62) / 0.38          # 离模糊带越远越好
            sc += min(o, 1.0) * 0.8             # 有鼓包就是强证据
        return sc, out_ev

    best = None
    for ltab in (0, 1):
        if not (0 <= nx - ltab <= 1):
            continue
        for ttab in (0, 1):
            if not (0 <= ny - ttab <= 1):
                continue
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    core = (bx + ext * ltab + ddx, by + ext * ttab + ddy, pw, ph)
                    sc, ev = sharpness(core)
                    if best is None or sc > best[0]:
                        best = (sc, None, None, core, nx, ny, ev)
    _, _, _, core, nx, ny, ev = best
    labels, confs = {}, {}
    for side in SIDES:
        lab, cf = _label_from_evidence(*ev[side])
        labels[side], confs[side] = lab, cf

    # ---- bbox 卡扣计数硬约束 ----
    # bbox 的宽/高只被"凸起"改变（凹槽和直边都不改），所以
    #   nx = 左右两边里有几条是凸， ny = 上下两边里有几条是凸
    # 这是很强的先验：当一侧已判为直/凹、另一侧未知时，未知那侧必然是凸（可推导）。
    xL, yT, W, H = core
    clipped = {"left": bx <= 1, "top": by <= 1,
               "right": bx + bw >= canvas_wh[0] - 1,
               "bottom": by + bh >= canvas_wh[1] - 1}
    for axis_sides, need in ((("left", "right"), nx), (("top", "bottom"), ny)):
        if any(clipped[s] for s in axis_sides):
            continue                      # 该轴可能被画布裁掉，计数不可信
        tabs = [s for s in axis_sides if labels[s] == 1]
        if len(tabs) > need:
            tabs.sort(key=lambda s: ev[s][0])          # 外侧证据最弱的先撤
            for s in tabs[:len(tabs) - need]:
                labels[s], confs[s] = None, 0.0
        elif len(tabs) < need:
            unknown = [s for s in axis_sides if labels[s] is None]
            miss = need - len(tabs)
            if unknown:
                # 未知里挑外侧证据最强的补成凸；证据全无时按顺序补（统计上更有利）
                unknown.sort(key=lambda s: -ev[s][0])
                for s in unknown[:miss]:
                    labels[s] = 1
                    confs[s] = 0.30       # 推导出来的凸，置信度低于实测
    # 贴画布边界 -> 不可信
    xL, yT, W, H = core
    bx2, by2 = bx, by
    if bx2 <= 1:
        labels["left"], confs["left"] = None, 0.0
    if by2 <= 1:
        labels["top"], confs["top"] = None, 0.0
    if bx2 + bw >= canvas_wh[0] - 1:
        labels["right"], confs["right"] = None, 0.0
    if by2 + bh >= canvas_wh[1] - 1:
        labels["bottom"], confs["bottom"] = None, 0.0
    return ({s: (labels[s], confs[s]) for s in SIDES}, core, nx, ny)


# ======================================================================
# 核心位置拟合（第四版）：坐标下降 + 模板 IoU
# ======================================================================
# 前三版都卡在同一件事：核心矩形放错 20px（恰好一个卡扣长度）时，4 条边全错。
# bbox 只能给出"凸起总数"，给不出"哪一侧凸"（凸起被压住时 bbox 根本不延伸）。
# 这里直接按 IoU 拟合：给定核心位置 -> 由证据得出标签 -> 取该标签的合成模板
# -> 与碎片掩码算 IoU。IoU 最高的位置就是最自洽的位置。
def fit_piece(mask, bbox, pw, ph, canvas_wh):
    templates, _, tab = _build_templates(pw, ph)
    tmap = {t["combo"]: t for t in templates}
    ext = 1.15 * tab
    H, W = mask.shape[:2]
    bx, by, bw, bh = bbox
    m = int(np.ceil(2.6 * tab)) + 6
    x0, y0 = max(0, bx - m), max(0, by - m)
    x1, y1 = min(W, bx + bw + m), min(H, by + bh + m)
    roi = mask[y0:y1, x0:x1]

    def evaluate(xL, yT):
        core = (xL, yT, pw, ph)
        labels, confs, ev = {}, {}, {}
        for side in SIDES:
            o, i = _side_evidence(mask, core, tab, side)
            lab, cf = _label_from_evidence(o, i)
            labels[side], confs[side], ev[side] = lab, cf, (o, i)
        combo = tuple(labels[s] if labels[s] is not None else 0 for s in SIDES)
        tpl = tmap[combo]
        # 模板里核心左上角在 tpl["core"]，把核心对齐到 (xL, yT)
        core_tx = tpl["core"][0]      # 模板里核心左上角 x
        core_ty = tpl["core"][1]
        M = np.float32([[1, 0, (xL - x0) - core_tx], [0, 1, (yT - y0) - core_ty]])
        w = cv2.warpAffine(tpl["mask"], M, (roi.shape[1], roi.shape[0]),
                           flags=cv2.INTER_NEAREST)
        inter = np.count_nonzero((w > 0) & (roi > 0))
        union = np.count_nonzero((w > 0) | (roi > 0))
        iou = inter / union if union else 0.0
        return iou, labels, confs, ev, core

    # 初始：bbox 推的 4 个假设；然后 x/y 各做"粗扫 + 细化"
    best = None
    for ltab in (0, 1):
        for ttab in (0, 1):
            r = evaluate(bx + ext * ltab, by + ext * ttab)
            if best is None or r[0] > best[0]:
                best = r
    span = int(ext) + 4
    for step, half in ((3, span), (1, 2)):
        cur = best
        for axis in (0, 1):
            base = cur[4][axis]
            for delta in range(-half, half + 1, step):
                if delta == 0:
                    continue
                xL = base + delta if axis == 0 else cur[4][0]
                yT = base + delta if axis == 1 else cur[4][1]
                r = evaluate(xL, yT)
                if r[0] > best[0]:
                    best = r
    iou, labels, confs, ev, core = best
    # 贴画布边界 -> 不可信
    for side, cond in (("left", core[0] <= 1), ("top", core[1] <= 1),
                       ("right", core[0] + pw >= canvas_wh[0] - 1),
                       ("bottom", core[1] + ph >= canvas_wh[1] - 1)):
        if cond:
            labels[side], confs[side] = None, 0.0
    return ({s: (labels[s], confs[s]) for s in SIDES}, core, float(iou))
