#!/usr/bin/env python3
"""
检测管线：把一张 canvas 图变成"可求解的棋盘状态"

  analyze(img, n_expected) -> {
      board: {x,y,w,h},            # 棋盘暗色矩形（画面上可测）
      rows, cols, pw, ph,          # 由棋盘与格数推出，不读任何内部状态
      pieces: [{cx, cy, bbox, core, sides:{side:(label,conf)}, sig}]
  }

坐标全部是 **canvas 位图坐标**；换算成页面 CSS 坐标由调用方乘 bitmap/css。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from white_vision import (SIDES, TAB_RATIO, classify_zones, extract_pieces,
                          fit_piece, piece_mask)

BOARD_DARK_MAX = 80
LATTICE_SNAP = False   # 实测：格点吸附没有提升（94.5% vs 94.2%），默认关闭     # 棋盘 #0f172a 灰度约 26.7；碎片纯白 255


def detect_board(img: np.ndarray):
    """棋盘 = 唯一的"不透明 + 暗色 + 实心矩形"。"""
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img.shape[:2], np.float32)
    gray = img[:, :, :3].astype(np.int16).mean(axis=2)
    dark = ((alpha >= 0.99) & (gray <= BOARD_DARK_MAX)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None
    best, best_score = None, -1.0
    for i in range(1, n):
        x, y, w, h, area = (int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                            int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT]),
                            int(st[i, cv2.CC_STAT_AREA]))
        if w < 40 or h < 40:
            continue
        fill = area / float(w * h)
        score = area * fill              # 越大越方越像棋盘
        # 棋盘上有碎片后填充率会掉到 0.5 甚至更低（只剩外框），不能用 0.9
        if fill > 0.25 and score > best_score:
            best, best_score = {"x": x, "y": y, "w": w, "h": h}, score
    return best


def _sig(sides) -> tuple:
    return tuple(sides[s][0] for s in SIDES)


def analyze(img: np.ndarray, n_expected: int, mode: str = "white"):
    """一帧画面 -> 可求解的棋盘状态。"""
    H, W = img.shape[:2]
    s = int(round(math.sqrt(n_expected)))
    board = detect_board(img)
    mask = piece_mask(img)
    raw = extract_pieces(mask)

    out_pieces = []
    pw = ph = None
    if board:
        pw, ph = board["w"] / s, board["h"] / s
        for p in raw:
            # 主用核心拟合（模板 IoU），失败再回退到分区密度
            try:
                sides, core, fit_iou = fit_piece(p["mask"], p["bbox"], pw, ph, (W, H))
            except Exception:
                zs, zcore, _, _ = classify_zones(p["mask"], p["bbox"], pw, ph, (W, H))
                sides, core, fit_iou = zs, zcore, 0.0
            xL, yT, cw, ch = core
            on_board = (xL + cw / 2.0 > board["x"] and xL + cw / 2.0 < board["x"] + board["w"]
                        and yT + ch / 2.0 > board["y"] and yT + ch / 2.0 < board["y"] + board["h"])
            bxc, byc = p["bbox"][0] + p["bbox"][2] / 2.0, p["bbox"][1] + p["bbox"][3] / 2.0
            out_pieces.append({
                "on_board": on_board,
                "cx": xL + cw / 2.0, "cy": yT + ch / 2.0,
                "grab": (bxc, byc),   # 受包围盒保证：这个点一定在本碎片的格子里
                "bbox": p["bbox"], "core": core, "t": TAB_RATIO * min(pw, ph),
                "sides": sides, "sig": _sig(sides),
                "unknown": sum(1 for k in SIDES if sides[k][0] is None),
                "conf_mean": float(np.mean([sides[k][1] for k in SIDES])),
                "fit_iou": float(fit_iou),
                "_mask": p["mask"],
            })
        if LATTICE_SNAP:
            out_pieces = snap_cores_to_lattice(out_pieces, pw, ph)
        # 吸附后核心变了，用新核心重判四边（这一步很便宜）
        from white_vision import _label_from_evidence, _side_evidence
        t_lab = TAB_RATIO * min(pw, ph)
        for op in out_pieces:
            sides = {}
            for side in SIDES:
                o, i = _side_evidence(op["_mask"], op["core"], t_lab, side)
                sides[side] = _label_from_evidence(o, i)
            op["sides"] = sides
            op["sig"] = _sig(sides)
        out_pieces.sort(key=lambda d: (round(d["cy"] / 10), d["cx"]))

    geom = None
    if board and pw:
        geom = {"board": board, "rows": s, "cols": s, "pw": pw, "ph": ph}
    return {"geom": geom, "pieces": out_pieces, "s": s, "raw_pieces": raw}


def draw_overlay(img: np.ndarray, an: dict, grid=None) -> np.ndarray:
    vis = img[:, :, :3].copy() if img.shape[2] >= 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.35 + 40).astype(np.uint8)      # 压暗，便于看标注
    if an.get("geom"):
        b = an["geom"]["board"]
        cv2.rectangle(vis, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (90, 90, 90), 1)
        pw, ph = an["geom"]["pw"], an["geom"]["ph"]
        for r in range(an["s"] + 1):
            cv2.line(vis, (b["x"], int(b["y"] + r * ph)), (b["x"] + b["w"], int(b["y"] + r * ph)),
                     (70, 70, 70), 1)
        for c in range(an["s"] + 1):
            cv2.line(vis, (int(b["x"] + c * pw), b["y"]), (int(b["x"] + c * pw), b["y"] + b["h"]),
                     (70, 70, 70), 1)
    lbl = {1: "+", -1: "-", 0: "0", None: "?"}
    for i, p in enumerate(an["pieces"]):
        x, y, w, h = p["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 255), 1)
        txt = "".join(lbl[p["sides"][k][0]] for k in SIDES)
        cv2.putText(vis, f"{i}{txt}", (x + 1, max(10, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 230, 60), 1, cv2.LINE_AA)
    if grid:
        for r, row in enumerate(grid):
            for c, idx in enumerate(row):
                if idx is None or idx < 0:
                    continue
                cx, cy = an["pieces"][idx]["cx"], an["pieces"][idx]["cy"]
                cv2.putText(vis, str(idx), (int(cx) - 4, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 80, 80), 1, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="logs/pipeline_vis.png")
    a = ap.parse_args()
    im = cv2.imread(a.canvas, cv2.IMREAD_UNCHANGED)
    an = analyze(im, a.n)
    print(json.dumps({"geom": an["geom"], "n_pieces": len(an["pieces"]),
                      "unknowns": sum(p["unknown"] for p in an["pieces"])}, indent=1))
    cv2.imwrite(a.out, draw_overlay(im, an))
    print("vis ->", a.out)


# ---------------------------------------------------------------- 局部重分类
def analyze_piece_at(img: np.ndarray, core, pw: float, ph: float):
    """已知碎片核心位置时的局部重分类（用"点一下提到最上层"之后的画面）。

    不做全图分割：只在核心附近做一次小 ROI 阈值 + 连通域，取包含核心中心的
    那一个连通域当碎片本体。碎片在最上层时，它和邻片之间有阴影隔开，
    所以连通域天然分离 —— 这正是解决"卡扣被压住"的关键。
    """
    from white_vision import TAB_RATIO, SIDES, _label_from_evidence, _side_evidence

    t = TAB_RATIO * min(pw, ph)
    xL, yT = core[0], core[1]
    cx, cy = xL + pw / 2.0, yT + ph / 2.0
    m = int(np.ceil(2.2 * t)) + 4
    H, W = img.shape[:2]
    x0, y0 = max(0, int(xL - m)), max(0, int(yT - m))
    x1, y1 = min(W, int(xL + pw + m)), min(H, int(yT + ph + m))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = img[y0:y1, x0:x1]
    if roi.shape[2] == 4:
        alpha = roi[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(roi.shape[:2], np.float32)
    gray = roi[:, :, :3].astype(np.int16).mean(axis=2)
    binm = ((alpha >= 0.99) & (gray >= 240)).astype(np.uint8)
    n, lab = cv2.connectedComponents(binm, connectivity=8)
    if n <= 1:
        return None
    px, py = int(cx - x0), int(cy - y0)
    px = min(max(px, 0), binm.shape[1] - 1)
    py = min(max(py, 0), binm.shape[0] - 1)
    target = lab[py, px]
    if target == 0:                       # 核心中心不在碎片本体上（被盖住了）
        ids, counts = np.unique(lab[lab > 0], return_counts=True)
        if ids.size == 0:
            return None
        target = int(ids[counts.argmax()])
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1][lab == target] = 255

    sides = {}
    for side in SIDES:
        o, i = _side_evidence(full, core, t, side)
        sides[side] = _label_from_evidence(o, i)
    return sides, (x0, y0, x1, y1)


# ------------------------------------------------------------ 格点吸附
def snap_cores_to_lattice(pieces, pw, ph):
    """托盘槽位是等距格点，且「槽位原点 == 碎片核心原点」。

    核心拟合偶尔会偏一个卡扣长度（≈20px），但格点约束可以把它们拉回来。
    做法：把各片核心的 x / y 分别一维聚类，再吸附到最近格线上。
    """
    if len(pieces) < 4:
        return pieces
    def cluster(vals, tol):
        vals = sorted(vals)
        groups, cur = [], [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                groups.append(cur); cur = [v]
        groups.append(cur)
        return [float(np.median(g)) for g in groups if len(g) >= 1]

    xs = cluster([p["core"][0] for p in pieces], pw * 0.45)
    ys = cluster([p["core"][1] for p in pieces], ph * 0.45)
    if len(xs) < 2 and len(ys) < 2:
        return pieces
    snapped = 0
    for p in pieces:
        xL, yT, cw, ch = p["core"]
        nx = min(xs, key=lambda v: abs(v - xL))
        ny = min(ys, key=lambda v: abs(v - yT))
        if abs(nx - xL) < pw * 0.45 and abs(nx - xL) > 0.5:
            xL = nx; snapped += 1
        if abs(ny - yT) < ph * 0.45 and abs(ny - yT) > 0.5:
            yT = ny
        p["core"] = (xL, yT, cw, ch)
        p["cx"], p["cy"] = xL + cw / 2.0, yT + ch / 2.0
    p_any = pieces[0]
    p_any["_lat_cols"] = len(xs)
    p_any["_lat_rows"] = len(ys)
    p_any["_snapped"] = snapped
    return pieces
