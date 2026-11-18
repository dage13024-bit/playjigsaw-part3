#!/usr/bin/env python3
"""
槽位锚定的碎片提取（Part 1 / Part 3 用）

为什么不用"分割"：照片碎片的颜色千变万化，
  * 按亮度阈值分 → 一块照片被拆成十几个小块（白墙、高光都成了碎片）
  * 只看 alpha   → 相邻碎片互相粘连成一坨
但游戏把碎片**摆在等距槽位**上（间距 1.16×碎片尺寸），槽位坐标可以由棋盘几何
解析算出来。于是根本不需要分割：直接按槽位取图、按槽位量卡扣。

槽位公式来自引擎 PuzzleBoard.traySlots()：
    stepX = pw*1.16, stepY = ph*1.16
    左右两侧各 floor(boardX/stepX) 列，整幅画布高 floor(ch/stepY) 行
"""
from __future__ import annotations

import math

import numpy as np

STEP = 1.16          # 槽位间距 = 1.16 × 碎片尺寸（引擎常量）
FILL_MIN = 0.55      # 槽位内 alpha 覆盖率高于此值就认为"这里有碎片"


def tray_slots(board, pw: float, ph: float, canvas_w: int, canvas_h: int):
    """复刻引擎的槽位布局。返回 [{x, y, side, col, row}]。"""
    step_x, step_y = pw * STEP, ph * STEP
    band = max(0.0, float(board["x"]))
    cols_per_side = int(math.floor(band / step_x))
    slot_rows = int(math.floor(canvas_h / step_y))
    slots = []
    for side in (0, 1):
        origin_x = 0.0 if side == 0 else board["x"] + board["w"]
        for col in range(cols_per_side):
            for row in range(slot_rows):
                x = origin_x + col * step_x + (step_x - pw) / 2.0
                y = row * step_y + (step_y - ph) / 2.0
                if x < 0 or y < 0 or x + pw > canvas_w or y + ph > canvas_h:
                    continue
                slots.append({"x": float(x), "y": float(y),
                              "side": side, "col": col, "row": row})
    return slots


def alpha_mask(img: np.ndarray) -> np.ndarray:
    """碎片本体 = 不透明像素（照片碎片也适用，跟颜色无关）。"""
    if img.shape[2] == 4:
        return ((img[:, :, 3] >= 253).astype(np.uint8)) * 255
    return np.full(img.shape[:2], 255, np.uint8)


def slot_fill(mask: np.ndarray, slot, pw: float, ph: float) -> float:
    """返回槽位核心矩形内"不透明"的比例，用来判断这个槽位有没有碎片。"""
    h, w = mask.shape[:2]
    x0, y0 = int(round(slot["x"])), int(round(slot["y"]))
    x1, y1 = int(round(slot["x"] + pw)), int(round(slot["y"] + ph))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((mask[y0:y1, x0:x1] > 0).mean())


def occupied_slots(img: np.ndarray, board, pw: float, ph: float, expect: int | None = None):
    """找出真正有碎片的槽位（按覆盖率降序，可选截断到 expect 个）。"""
    mask = alpha_mask(img)
    h, w = mask.shape[:2]
    slots = tray_slots(board, pw, ph, w, h)
    for s in slots:
        s["fill"] = slot_fill(mask, s, pw, ph)
    occ = [s for s in slots if s["fill"] >= FILL_MIN]
    occ.sort(key=lambda s: -s["fill"])
    if expect is not None and len(occ) > expect:
        occ = occ[:expect]
    return occ


def crop_piece(img: np.ndarray, slot, pw: float, ph: float, inset: int = 3,
               shrink: float = 0.55):
    """按槽位裁出碎片内容，默认只取中心 72%。

    为什么要 shrink：槽位矩形的四条边中间会被"凹槽"挖掉（那是透明区，
    存成 RGB 就变成黑块），会干扰分类和"是否纯白"的判断。
    目标通常画在格子的中心，所以取中心区域既避开凹槽，也避开邻片渗色。
    """
    h, w = img.shape[:2]
    cx = slot["x"] + pw / 2.0
    cy = slot["y"] + ph / 2.0
    half_w = pw * shrink / 2.0
    half_h = ph * shrink / 2.0
    x0 = int(round(cx - half_w))
    y0 = int(round(cy - half_h))
    x1 = int(round(cx + half_w))
    y1 = int(round(cy + half_h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1, :3].copy()


def _dens(mask: np.ndarray, x0, y0, x1, y1) -> float:
    h, w = mask.shape[:2]
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask[y0:y1, x0:x1]
    return float((roi > 0).mean())


def edges_at_slot(img: np.ndarray, slot, pw: float, ph: float, tab: float,
                  a: float = 0.34, b: float = 0.66):
    """在槽位上量四条边的 (外侧密度, 内侧密度)。

    判据与 Part 2 相同：内侧空=凹，外侧有鼓包=凸，外空内实=直。
    注意：相邻碎片可能压过来，所以这里只作为初值，必要时再"提层重判"。
    """
    mask = alpha_mask(img)
    xL, yT = slot["x"], slot["y"]
    xR, yB = xL + pw, yT + ph
    d0, d1 = 0.25 * tab, 1.00 * tab
    out = {}
    out["right"] = (_dens(mask, xR + d0, yT + a * ph, xR + d1, yT + b * ph),
                    _dens(mask, xR - d1, yT + a * ph, xR - d0, yT + b * ph))
    out["left"] = (_dens(mask, xL - d1, yT + a * ph, xL - d0, yT + b * ph),
                   _dens(mask, xL + d0, yT + a * ph, xL + d1, yT + b * ph))
    out["bottom"] = (_dens(mask, xL + a * pw, yB + d0, xL + b * pw, yB + d1),
                     _dens(mask, xL + a * pw, yB - d1, xL + b * pw, yB - d0))
    out["top"] = (_dens(mask, xL + a * pw, yT - d1, xL + b * pw, yT - d0),
                  _dens(mask, xL + a * pw, yT + d0, xL + b * pw, yT + d1))
    return out


def label_from_density(out: float, inn: float):
    """(标签, 置信度)：1 凸 / -1 凹 / 0 直 / None 拿不准。阈值同 Part 2 实测。"""
    if inn < 0.70:
        return -1, min(1.0, (0.70 - inn) / 0.37 + 0.4)
    if out > 0.50:
        return 1, min(1.0, (out - 0.50) / 0.10 + 0.6)
    if out < 0.15:
        return 0, min(1.0, (inn - 0.70) / 0.30 + 0.5)
    return None, 0.0


def is_white_piece(img: np.ndarray, slot, pw: float, ph: float,
                   bright_min: float = 248.0, std_max: float = 5.0) -> bool:
    """判断这个槽位上的碎片是不是"纯白碎片"。

    只在 **alpha>0 的碎片像素** 上统计亮度，凹槽（透明）自然被排除，
    所以不会像"裁矩形取均值"那样被凹槽或背景干扰。
    """
    if img.shape[2] < 4:
        return False
    h, w = img.shape[:2]
    x0 = max(0, int(round(slot["x"]))); y0 = max(0, int(round(slot["y"])))
    x1 = min(w, int(round(slot["x"] + pw))); y1 = min(h, int(round(slot["y"] + ph)))
    if x1 <= x0 or y1 <= y0:
        return False
    alpha = img[y0:y1, x0:x1, 3]
    sel = alpha >= 253
    if int(sel.sum()) < 200:
        return False
    pix = img[y0:y1, x0:x1, :3][sel].astype(np.float32)
    return bool(pix.mean() >= bright_min and pix.std() <= std_max)


def crop_piece_for_cls(img: np.ndarray, slot, pw: float, ph: float, tab: float):
    """给分类器用的碎片图：**完整碎片 + 外侧填黑**，和训练时一致。

    为什么不能取中心小图：5×5 时碎片只有 97×73，取中心 55% 只剩 53×40，
    细节全丢了（实测分类直接崩）。训练用的合成图是"整个碎片、外面填黑"，
    所以推理也必须用同样的预处理。
    """
    m = int(np.ceil(tab * 1.2)) + 2
    h, w = img.shape[:2]
    x0 = max(0, int(round(slot["x"] - m)))
    y0 = max(0, int(round(slot["y"] - m)))
    x1 = min(w, int(round(slot["x"] + pw + m)))
    y1 = min(h, int(round(slot["y"] + ph + m)))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    rgb = roi[:, :, :3].astype(np.float32)
    if roi.shape[2] == 4:
        a = (roi[:, :, 3:4].astype(np.float32) / 255.0)
        # 外侧填"图像自身均值色"，不要填黑：填黑会让"台式机/显示器"这类
        # 本身大片黑色的类别被无限放大（实测 desktop 从 6 个涨到 19 个）。
        bg = rgb[a[..., 0] > 0.5].mean(axis=0) if (a > 0.5).any() else np.array([128, 128, 128])
        rgb = rgb * a + bg * (1 - a)
        # 紧贴碎片裁掉多余背景
        ys, xs = np.nonzero(a[..., 0] > 0.5)
        if xs.size:
            rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgb.astype(np.uint8)
