#!/usr/bin/env python3
"""
Part 3（照片 + 纯白混合）自动解题 —— 官方考核站 puzzle.moyunliuyin.me

和 Part 2 / 旧 Part 3 方案的关键差别
------------------------------------
旧方案认为"参考图每格是同类别的另一张照片，只能靠类别猜位置"，于是整条链路
压在分类器上，类别一抖就崩。

读官方站 build 20261115r 源码后确认：三部分的参考图**不是一回事**
  * Part 1 `renderAnimalGrid()`：参考格用 `alternatePhoto()` 换成同类别的
    **另一张照片** → 确实不能像素比对，只能认类别；
  * Part 3 `renderMixedPuzzle()`：`updatePreviewImage(canvas.toDataURL())`，
    预览就是**拼图源画布本身**（照片格 + 纯白格 + 3px 描边）。
    碎片就是从这张画布上切下来的 → **同一块碎片和它那一格的中心区域几乎逐像素一致**。

所以本文件的定位主证据是"逐格比对"，YOLO 类别只做交叉校验/兜底
（考核要求照片格用 YOLO，类别仍是整条链路的一部分，只是不再拿它当唯一证据）。

流水线
------
  1. 等页面初始渲染安静（否则默认 animals 主题的回调会把题目覆盖掉）
  2. 开 Part 3 · 5×5，校验预览里恰好 5 个纯白格（= 混合题真的放出来了）
  3. 槽位锚定 + 逐块 bringToTop 截图 → 30 块：中心裁剪 / 四边卡扣 / 白片判定
  4. 预览原生分辨率切 25 格 → 白格判定
  5. 照片碎片 × 照片格 相似度矩阵（同视野中心区，Lab 三通道 NCC）
  6. 指派 + 混淆块剔除（5 块），纯白碎片按"已知邻边互补"落到 5 个白格
  7. 环形顺序（外圈→内圈，满足锚点规则）逐格拖拽；吸附失败回拖 + 换候选
  8. 完成检测 + 落盘日志/截图

用法
----
  .venv/bin/python play_p3.py --difficulty 25 --dry-run --headed   # 只看识别结果
  .venv/bin/python play_p3.py --difficulty 25 --headed             # 真拖
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import slots as S                                       # noqa: E402
from net import open_page                                # noqa: E402
from pipeline import detect_board                        # noqa: E402
from play_part2 import canvas_geometry, drag, grab_canvas, read_progress  # noqa: E402
from solver import SIDES                                 # noqa: E402

OFFICIAL = "https://puzzle.moyunliuyin.me/?mode=exam"
LOGS = ROOT / "logs"
WHITE = "__white__"

# 匹配参数：碎片和参考格都只取"核心矩形的中心 62%"，
# 这样两者的视野一致（卡扣/描边都在外面），尺度差只剩分辨率。
MATCH_FOV = 0.62
MATCH_SIZE = (96, 72)
MIN_ASSIGN = 0.45          # 低于这个分数的一律不认，留给"拖上去试"


# ------------------------------------------------------------------ 页面
def settle(page, timeout_s: float = 40.0) -> None:
    """等页面初始 animals 渲染跑完。

    页面构造时 `loadTheme('animals')` 是异步的；它加载完会重绘 sourceCanvas
    并 `startNewGame()`。如果这时候我们已经开了混合题，就会被它覆盖成照片题。
    判据：参考图的 src 连续 4 次（2 秒）不变。
    """
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except Exception:                                   # noqa: BLE001
        pass
    last, stable, t0 = None, 0, time.time()
    while time.time() - t0 < timeout_s:
        cur = page.evaluate(
            "() => {const i = document.getElementById('preview-img');"
            " return i && i.naturalWidth ? i.src.length + ':' + i.naturalWidth : '';}")
        stable = stable + 1 if cur and cur == last else 0
        last = cur
        if stable >= 4:
            return
        page.wait_for_timeout(500)
    raise RuntimeError("页面初始渲染迟迟不安静，网络可能太慢")


def preview_native(page) -> np.ndarray | None:
    """取参考图的原生像素。

    优先用 `<img>` 的 src（data URL，和显示出来的参考图是同一张，1600×1200），
    拿不到就退回元素截图（只有 ~340px 宽，匹配精度会降）。
    """
    src = page.evaluate("() => document.getElementById('preview-img').src || ''")
    if src.startswith("data:image"):
        buf = base64.b64decode(src.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    try:
        shot = page.locator("#preview-img").screenshot()
        return cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR)
    except Exception:                                   # noqa: BLE001
        return None


def split_cells(ref: np.ndarray, s: int):
    """参考图切成 s×s，返回每格的图像和是否纯白。"""
    h, w = ref.shape[:2]
    cells = []
    for r in range(s):
        for c in range(s):
            x0, x1 = int(round(c * w / s)), int(round((c + 1) * w / s))
            y0, y1 = int(round(r * h / s)), int(round((r + 1) * h / s))
            img = ref[y0:y1, x0:x1]
            cells.append({"r": r, "c": c, "img": img, "white": is_white_cell(img)})
    return cells


def is_white_cell(img: np.ndarray, margin: float = 0.2) -> bool:
    h, w = img.shape[:2]
    core = img[int(h * margin):int(h * (1 - margin)),
               int(w * margin):int(w * (1 - margin))]
    if core.size == 0:
        return False
    g = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    return bool(g.mean() > 252 and g.std() < 1.5)


def start_part3(page, s: int, attempts: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """开一局混合题，直到预览里纯白格数量正确。返回 (预览图, 首帧画布)。"""
    want_white = max(1, round(s * s * 0.2))
    for k in range(attempts):
        try:
            page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
            page.check('input[name="exam-part"][value="3"]')
            page.check(f'input[name="exam-difficulty"][value="{s * s}"]')
            page.click("#btn-exam-start")
            # 混合题要现场下载 20 张图，弱网下"准备中"可能持续十几秒
            page.wait_for_function(
                "() => document.querySelector('#exam-start').hidden"
                " && ((document.getElementById('exam-state')||{}).textContent||'').includes('进行中')",
                timeout=120_000)
            page.wait_for_timeout(600)
        except Exception as e:                             # noqa: BLE001
            print(f"  ⚠ 第 {k + 1} 次开题失败：{type(e).__name__}，重载重试")
            try:
                page.reload()
                settle(page)
            except Exception:                              # noqa: BLE001
                pass
            continue

        ref = preview_native(page)
        if ref is None:
            raise RuntimeError("拿不到参考图")
        whites = sum(1 for c in split_cells(ref, s) if c["white"])
        if whites == want_white:
            img = wait_canvas_ready(page)
            print(f"  开题成功（第 {k + 1} 次）：预览 {ref.shape[1]}×{ref.shape[0]}，"
                  f"纯白格 {whites}")
            return ref, img
        print(f"  ⚠ 第 {k + 1} 次开题异常：纯白格 {whites}（应为 {want_white}），重开")
        page.reload()
        settle(page)
    raise RuntimeError("连续几次都没开到正确的混合题")


def wait_canvas_ready(page, tries: int = 30) -> np.ndarray:
    """等画布真的画出东西、且尺寸稳定。"""
    last, img = None, None
    for _ in range(tries):
        size = page.evaluate("() => {const c = document.getElementById('puzzle-canvas');"
                             " return [c.width, c.height];}")
        img = grab_canvas(page)
        has = img is not None and img.shape[2] == 4 and int((img[:, :, 3] > 0).sum()) > 5000
        if has and size == last:
            return img
        last = size
        page.wait_for_timeout(200)
    return img


# ------------------------------------------------------------------ 碎片
def _coverage(img, slot, pw, ph, fov=0.62) -> float:
    """核心中心区里"不透明像素"的占比，用来判断这块碎片有没有被压住。"""
    h, w = img.shape[:2]
    cx, cy = slot["x"] + pw / 2.0, slot["y"] + ph / 2.0
    hw, hh = pw * fov / 2.0, ph * fov / 2.0
    x0, y0 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
    x1, y1 = min(w, int(round(cx + hw))), min(h, int(round(cy + hh)))
    if x1 <= x0 or y1 <= y0 or img.shape[2] < 4:
        return 0.0
    a = img[y0:y1, x0:x1, 3]
    return float((a >= 250).mean())


def capture_pieces(page, geom, img, board, pw, ph, s, lift=False):
    """按槽位裁出 30 块碎片。

    lift=False（默认）：整局只截一帧，全部从这一帧裁 —— 因为托盘槽距 1.16×碎片，
    卡扣只够伸到邻片边缘，中心 62% 永远属于自己这块。
    （逐块点一下会触发 bringsToTop，但游戏把每次点击也记成一步：
      实测多出 30 步，胜利面板 31/33 很难看。）

    只有中心区不透明度异常（被压住/裁到画布外）的那一块才补点一次并重截。
    """
    tab = 0.22 * min(pw, ph)
    occ = S.occupied_slots(img, board, pw, ph, expect=s * s + max(1, round(s * s * 0.2)))
    pieces = []
    lifted = 0
    for k, slot in enumerate(occ):
        shot = img
        if lift or _coverage(shot, slot, pw, ph) < 0.90:
            cx, cy = slot["x"] + pw / 2.0, slot["y"] + ph / 2.0
            ax = geom["left"] + cx * geom["sx"]
            ay = geom["top"] + cy * geom["sy"]
            page.mouse.move(ax, ay)
            page.mouse.down()
            page.mouse.up()
            shot = grab_canvas(page)
            lifted += 1
        crop = core_crop(shot, slot, pw, ph, MATCH_FOV)
        if crop is None:
            continue
        core = core_crop(shot, slot, pw, ph, 1.0, rgba=True)
        dens = S.edges_at_slot(shot, slot, pw, ph, tab)
        pieces.append({
            "k": k, "slot": slot, "pos": (slot["x"] + pw / 2.0, slot["y"] + ph / 2.0),
            "crop": crop,
            "core": on_gray(core) if core is not None else None,
            "sides": {kk: S.label_from_density(*v)[0] for kk, v in dens.items()},
            "white": is_white_slot(shot, slot, pw, ph),
        })
    if lifted:
        print(f"    提层补截 {lifted} 块（其余直接从首帧裁）")
    return pieces


def on_gray(rgba: np.ndarray, bg: int = 128) -> np.ndarray:
    """把带 alpha 的碎片合成到灰底上：分类器训练/推理都用这个域。"""
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    return np.clip(rgb * a + bg * (1.0 - a), 0, 255).astype(np.uint8)


def is_white_slot(shot, slot, pw, ph, fov: float = 0.60) -> bool:
    """纯白碎片判定：核心中心区、不透明像素上"亮度≈255 且几乎没有起伏"。

    不能用整个槽位矩形：卡扣凹槽会把邻居/背景带进来，std 一高就误判成照片。
    """
    h, w = shot.shape[:2]
    cx, cy = slot["x"] + pw / 2.0, slot["y"] + ph / 2.0
    hw, hh = pw * fov / 2.0, ph * fov / 2.0
    x0, y0 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
    x1, y1 = min(w, int(round(cx + hw))), min(h, int(round(cy + hh)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    reg = shot[y0:y1, x0:x1]
    alpha = reg[:, :, 3]
    sel = alpha >= 250
    if int(sel.sum()) < 0.3 * sel.size:
        return False
    pix = reg[:, :, :3][sel].astype(np.float32)
    return bool(pix.mean() >= 250 and pix.std() <= 3.0)


def core_crop(img, slot, pw, ph, fov, rgba: bool = False):
    h, w = img.shape[:2]
    cx, cy = slot["x"] + pw / 2.0, slot["y"] + ph / 2.0
    hw, hh = pw * fov / 2.0, ph * fov / 2.0
    x0, y0 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
    x1, y1 = min(w, int(round(cx + hw))), min(h, int(round(cy + hh)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1, :].copy() if rgba else img[y0:y1, x0:x1, :3].copy()


# ------------------------------------------------------------------ YOLO
def load_model(weights):
    """加载自训练的 YOLO 分类器；没装/没权重就返回 None，流程退回纯 NCC。"""
    path = Path(weights) if weights else None
    if not path or not path.exists():
        print(f"  ⚠ 找不到分类模型 {weights}，本次只用像素比对")
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(path))
    except Exception as e:                              # noqa: BLE001
        print(f"  ⚠ 模型加载失败（{e}），本次只用像素比对")
        return None


def classify(model, imgs):
    """批量分类，返回每张图的 top1 / top3 / top5。"""
    out = []
    for r in model.predict(imgs, verbose=False, batch=16):
        pr = r.probs
        idx = np.atleast_1d(pr.top5)
        names = [r.names[int(i)] for i in idx]
        out.append({"key": names[0], "conf": float(pr.top1conf),
                    "top3": set(names[:3]), "top5": set(names)})
    return out


def apply_prior(M, p_cls, c_cls, same=0.10, top3=0.04, miss=-0.06):
    """YOLO 类别先验：同类别 +，top3 内小 +，top5 外 -。

    权重刻意压得比 NCC 的实际差距（正确对 ~0.9、错误对 <0.5）小一个量级：
    YOLO 负责"类别约束"，NCC 负责"同类别里到底是哪一格"。
    """
    M = M.copy()
    for i, p in enumerate(p_cls):
        for j, c in enumerate(c_cls):
            if c["key"] == p["key"]:
                M[i, j] += same
            elif c["key"] in p["top3"]:
                M[i, j] += top3
            elif c["key"] not in p["top5"]:
                M[i, j] += miss
    return M


# ------------------------------------------------------------------ 匹配
def _prep(img, size=MATCH_SIZE):
    g = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)


def _ncc(a, b):
    a = a.reshape(-1) - a.mean()
    b = b.reshape(-1) - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-6 else 0.0


def similarity(piece_img, cell_img) -> float:
    """同视野、同尺寸、Lab 三通道相关系数平均。"""
    pa, ca = _prep(piece_img), _prep(cell_img)
    return float(np.mean([_ncc(pa[..., i], ca[..., i]) for i in range(3)]))


def cell_crop(cell_img, fov=MATCH_FOV):
    h, w = cell_img.shape[:2]
    dy, dx = int(h * (1 - fov) / 2), int(w * (1 - fov) / 2)
    return cell_img[dy:h - dy, dx:w - dx]


def _hungarian(cost: np.ndarray):
    """标准 O(n³) 匈牙利（e-maxx 版），行数 ≤ 列数，返回 (row, col) 配对。

    自己实现是为了不引入 scipy：交付仓库只依赖 numpy/opencv/ultralytics。
    """
    n, m = cost.shape
    assert n <= m
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = 第 j 列匹配到的行（1-based，0 表示空）
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]


def assign(score: np.ndarray):
    """照片碎片 → 照片格的最优一一指派。

    score: (n_pieces, n_cells)，n_pieces = 20 照片格 + 5 混淆（5×5）。
    求"总分最大"的 20 对；没被选中的 5 块就是混淆块。
    低分对仍然保留（拖上去试比直接放弃更划算），只是标记 confidence。
    """
    n, m = score.shape
    if n > m:                                   # 碎片比格子多：转置成"行≤列"
        pairs = _hungarian(-score.T)
        pairs = [(j, i) for i, j in pairs]
    else:
        pairs = _hungarian(-score)
    out = {i: j for i, j in pairs}
    conf = {i: float(score[i, j]) >= MIN_ASSIGN for i, j in out.items()}
    return out, conf


# ------------------------------------------------------------------ 纯白
def white_plan(white_pieces, white_cells, photo_map, pieces, cells, s):
    """纯白碎片 → 纯白格：5! = 120 种排列直接穷举。

    评分只用"能被确定"的边：
      * 邻接棋盘外框 → 必须是直边 0
      * 邻接照片格   → 必须和那块照片碎片的对边互补
      * 两块纯白格相邻 → 两块碎片的对边也要互补（这一项能解决纯白互换）
    """
    import itertools

    by_k = {p["k"]: p for p in pieces}
    cell2piece = {}
    for pk, cj in photo_map.items():
        cell2piece[(cells[cj]["r"], cells[cj]["c"])] = by_k[pk]

    def required(r, c, side):
        if side == "top":
            nr, nc, opp = r - 1, c, "bottom"
        elif side == "bottom":
            nr, nc, opp = r + 1, c, "top"
        elif side == "left":
            nr, nc, opp = r, c - 1, "right"
        else:
            nr, nc, opp = r, c + 1, "left"
        if not (0 <= nr < s and 0 <= nc < s):
            return 0
        other = cell2piece.get((nr, nc))
        if other is None:
            return None
        val = other["sides"][opp]
        return None if val is None else -val

    wc = [(c["r"], c["c"]) for c in white_cells]
    n = min(len(white_pieces), len(wc))
    if n == 0:
        return {}, {}
    req = [{(r_, c_): {sd: required(r_, c_, sd) for sd in SIDES} for (r_, c_) in wc}][0]

    def pen_two(i, j):
        """两块白片放在 wc[i] / wc[j] 时，相邻边的互补惩罚。"""
        (r1, c1), (r2, c2) = wc[i], wc[j]
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return 0
        if r1 == r2:
            sd1, sd2 = ("right", "left") if c1 < c2 else ("left", "right")
        else:
            sd1, sd2 = ("bottom", "top") if r1 < r2 else ("top", "bottom")
        return (sd1, sd2, i, j)

    adj = []
    for i in range(n):
        for j in range(i + 1, n):
            t = pen_two(i, j)
            if t:
                adj.append(t)

    best_pen, best_perm = None, None
    for perm in itertools.permutations(range(len(white_pieces)), n):
        pen = 0
        for i in range(n):
            sd_map = white_pieces[perm[i]]["sides"]
            for sd in SIDES:
                need = req[wc[i]][sd]
                got = sd_map[sd]
                if need is None or got is None:
                    continue
                pen += 0 if got == need else 1
        for sd1, sd2, i, j in adj:
            a = white_pieces[perm[i]]["sides"][sd1]
            b = white_pieces[perm[j]]["sides"][sd2]
            if a is not None and b is not None:
                pen += 0 if a == -b else 1
        if best_pen is None or pen < best_pen:
            best_pen, best_perm = pen, perm

    out = {best_perm[i]: wc[i] for i in range(n)}
    return out, {"penalty": best_pen, "cells": wc}


# ------------------------------------------------------------------ 执行
def piece_here(page, x, y) -> bool:
    im = grab_canvas(page)
    m = S.alpha_mask(im)
    h, w = m.shape[:2]
    xi, yi = int(x), int(y)
    if not (0 <= xi < w and 0 <= yi < h):
        return False
    patch = m[max(0, yi - 8):yi + 8, max(0, xi - 8):xi + 8]
    return patch.size > 0 and float((patch > 0).mean()) > 0.5


def ring_order(s: int):
    """顺时针一圈一圈，从外圈到内圈（锚点规则：内部格子必须有邻居已放）。"""
    order = []
    for k in range((s + 1) // 2):
        lo, hi = k, s - 1 - k
        if lo == hi:
            order.append((lo, lo))
            continue
        for c in range(lo, hi):
            order.append((lo, c))
        for r in range(lo, hi):
            order.append((r, hi))
        for c in range(hi, lo, -1):
            order.append((hi, c))
        for r in range(hi, lo, -1):
            order.append((r, lo))
    return order


def _crop_at(img, x, y, pw, ph, fov=MATCH_FOV):
    h, w = img.shape[:2]
    hw, hh = pw * fov / 2.0, ph * fov / 2.0
    x0, y0 = max(0, int(round(x - hw))), max(0, int(round(y - hh)))
    x1, y1 = min(w, int(round(x + hw))), min(h, int(round(y + hh)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1, :3]


def _looks_like(e, img, x, y, pw, ph, thr=0.86) -> float:
    c = _crop_at(img, x, y, pw, ph)
    if c is None:
        return 0.0
    return similarity(e["crop"], c)


def locate(page, e, cands, pw, ph):
    """碎片跑哪去了？在所有槽位/格子中心里找和它最像的位置。"""
    img = grab_canvas(page)
    best, best_pos = 0.0, None
    for (x, y) in cands:
        sc = _looks_like(e, img, x, y, pw, ph)
        if sc > best:
            best, best_pos = sc, (x, y)
    return (best_pos, best) if best >= 0.80 else (None, best)


def execute(page, geom, board, pw, ph, s, pieces, photo_map, wmap, part_title,
            max_tries=400):
    """逐格拖拽。

    候选池只放"被指派到某一格"的 25 块：5 块混淆是**永远吸不上**的，
    留在池子里只会被反复试（实测能把一局从 25 次拖拽抬到 80+ 次）。
    如果最后仍有空格，再把它们当兜底候选放回来（防止指派阶段误删真碎片）。

    状态以画面为准：每次失败回拖后先看原位有没有碎片，没有就全画布找它。
    """
    cells_n = s * s
    target = {}
    for k, j in photo_map.items():
        target[(j // s, j % s)] = k
    for k, rc in wmap.items():
        target[rc] = k

    assigned = set(target.values())
    decoy_k = {p["k"] for p in pieces} - assigned
    pool = [{"i": p["k"], "pos": p["pos"], "sides": p["sides"],
             "crop": p["crop"], "white": p["white"]}
            for p in pieces if p["k"] in assigned]
    by_k = {p["i"]: p for p in pool}
    cands = [p["pos"] for p in pieces]
    cands += [(board["x"] + (c + 0.5) * pw, board["y"] + (r + 0.5) * ph)
              for r in range(s) for c in range(s)]
    placed, tries, fails = {}, 0, 0

    def there(x, y) -> bool:
        img = grab_canvas(page)
        m = S.alpha_mask(img)
        h, w = m.shape[:2]
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        patch = m[max(0, yi - 8):yi + 8, max(0, xi - 8):xi + 8]
        return patch.size > 0 and float((patch > 0).mean()) > 0.5

    def rank(r, c, k):
        want = target.get((r, c))
        sc = []
        for e in pool:
            v = 6.0 if e["i"] == want else 0.0
            for (nr, nc, mine, theirs) in ((r, c - 1, "left", "right"),
                                           (r, c + 1, "right", "left"),
                                           (r - 1, c, "top", "bottom"),
                                           (r + 1, c, "bottom", "top")):
                j = placed.get((nr, nc))
                if j is None:
                    continue
                a, b = e["sides"][mine], j["sides"][theirs]
                if a is None or b is None:
                    continue
                v += 2.0 if a == -b else -2.0
            if r in (0, s - 1) or c in (0, s - 1):
                for sd, on_edge in (("top", r == 0), ("bottom", r == s - 1),
                                    ("left", c == 0), ("right", c == s - 1)):
                    if on_edge:
                        v += 3.0 if e["sides"][sd] == 0 else -3.0
            sc.append((v, e["i"]))
        sc.sort(reverse=True)
        out = [i for _, i in sc]
        return out if k >= 900 else out[:k]

    t0 = time.perf_counter()

    def run(k_list) -> bool:
        """跑一轮/几轮，返回是否已完成。"""
        nonlocal tries, fails
        for k_round in k_list:
            if len(placed) == cells_n or tries > max_tries:
                break
            for (r, c) in ring_order(s):
                if (r, c) in placed or tries > max_tries:
                    continue
                tcx = board["x"] + (c + 0.5) * pw
                tcy = board["y"] + (r + 0.5) * ph
                for k in rank(r, c, k_round):
                    e = by_k.get(k)
                    if e is None:
                        continue
                    geom = canvas_geometry(page)      # 页面一滚动缓存坐标就全错
                    cur = page.evaluate(
                        "() => (document.getElementById('exam-part-title')||{}).textContent")
                    if cur != part_title:
                        print("  ⛔ 部分被切换，停手")
                        return True
                    gx, gy = e["pos"]
                    if not there(gx, gy):
                        pos, _ = locate(page, e, cands, pw, ph)
                        if pos is None:
                            pool.remove(e)
                            by_k.pop(k, None)
                            continue
                        e["pos"] = pos
                        gx, gy = pos
                    before = read_progress(page)
                    drag(page, geom, gx, gy, tcx, tcy)
                    tries += 1
                    if read_progress(page) > before:
                        placed[(r, c)] = e
                        e["pos"] = (tcx, tcy)
                        pool.remove(e)
                        by_k.pop(k, None)
                        if page.evaluate(
                                "() => !!document.querySelector('#victory-modal.show')"):
                            return True
                        break
                    # 没吸上 → 拖回原位，再用画面确认它到底在哪
                    drag(page, geom, tcx, tcy, gx, gy)
                    if not there(gx, gy):
                        if there(tcx, tcy):
                            e["pos"] = (tcx, tcy)
                        else:
                            pos, _ = locate(page, e, cands, pw, ph)
                            if pos is None:
                                pool.remove(e)
                                by_k.pop(k, None)
                                continue
                            e["pos"] = pos
                    fails += 1
                    kind = "白片" if e.get("white") else "照片"
                    print(f"    ✗ 片{e['i']}({kind}) → ({r},{c}) 没吸上（第 {fails} 次失败）")
        return False

    done = run((1, 2, 4, 999))

    # 兜底：还有空格，说明"混淆块"的判定可能误删了真碎片 → 把它们放回候选再试
    if not done and len(placed) < cells_n:
        placed_k = {e["i"] for e in placed.values()}
        added = 0
        for p in pieces:
            if p["k"] in decoy_k and p["k"] not in placed_k and p["k"] not in by_k:
                e = {"i": p["k"], "pos": p["pos"], "sides": p["sides"],
                     "crop": p["crop"], "white": p["white"]}
                pool.append(e)
                by_k[p["k"]] = e
                added += 1
        if added:
            print(f"  ↻ 兜底：把 {added} 块原判混淆放回候选")
            done = run((999,))
        done = done or len(placed) == cells_n

    return {"placed": len(placed), "tries": tries, "fails": fails, "done": done,
            "drag_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ------------------------------------------------------------------ 主流程
def solve_once(page, s: int, tag: str, dry: bool = False, save_data: bool = False,
               out_dir: Path | None = None, model=None) -> dict:
    """一局完整流程：取画面 → 识别 → 指派 → 拖拽。返回统计。"""
    expect = s * s + max(1, round(s * s * 0.2))
    out = Path(out_dir) if out_dir else LOGS
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    t = time.perf_counter()

    ref = preview_native(page)
    if ref is None:
        raise RuntimeError("拿不到参考图")
    img = wait_canvas_ready(page)
    cv2.imwrite(str(out / f"{tag}_ref.png"), ref)
    cv2.imwrite(str(out / f"{tag}_canvas.png"), img)

    part_title = page.evaluate(
        "() => (document.getElementById('exam-part-title')||{}).textContent")
    geom = canvas_geometry(page)
    board = detect_board(img)
    if board is None:
        raise RuntimeError("没找到棋盘")
    pw, ph = board["w"] / s, board["h"] / s
    print(f"  棋盘 {board['w']}×{board['h']} @({board['x']},{board['y']})  碎片 {pw:.1f}×{ph:.1f}")

    # ---- 碎片 ----
    t = time.perf_counter()
    pieces = capture_pieces(page, geom, img, board, pw, ph, s)
    cap_ms = (time.perf_counter() - t) * 1000
    whites = [p for p in pieces if p["white"]]
    print(f"  碎片 {len(pieces)} 块（纯白 {len(whites)}）预期 {expect}，截取 {cap_ms:.0f}ms")
    if len(pieces) < expect:
        raise RuntimeError(f"只抓到 {len(pieces)} 块碎片（预期 {expect}）")

    cells = split_cells(ref, s)
    photo_cells = [c for c in cells if not c["white"]]
    white_cells = [c for c in cells if c["white"]]
    print(f"  参考格：照片 {len(photo_cells)} + 纯白 {len(white_cells)}")
    if len(whites) != len(white_cells):
        print(f"  ⚠ 纯白碎片 {len(whites)} ≠ 纯白格 {len(white_cells)}，靠吸附兜底")

    # ---- 匹配 ----
    photo_pieces = [p for p in pieces if not p["white"]]
    t = time.perf_counter()
    M = np.zeros((len(photo_pieces), len(photo_cells)), np.float32)
    for i, p in enumerate(photo_pieces):
        for j, c in enumerate(photo_cells):
            M[i, j] = similarity(p["crop"], cell_crop(c["img"]))
    yolo_agree = None
    p_cls = c_cls = None
    if model is not None:
        try:
            p_cls = classify(model, [p["core"] for p in photo_pieces])
            c_cls = classify(model, [c["img"] for c in photo_cells])
            M = apply_prior(M, p_cls, c_cls)
            print(f"  YOLO：碎片 top1 置信中位 "
                  f"{float(np.median([c['conf'] for c in p_cls])):.2f}，"
                  f"格子 {float(np.median([c['conf'] for c in c_cls])):.2f}")
            report["p_cls"] = {photo_pieces[i]["k"]: p_cls[i]["key"] for i in range(len(p_cls))}
            report["c_cls"] = {f"{c['r']},{c['c']}": c_cls[j]["key"]
                               for j, c in enumerate(photo_cells)}
        except Exception as e:                          # noqa: BLE001
            print(f"  ⚠ YOLO 推理失败（{e}），退回纯 NCC")
            p_cls = c_cls = None
    pmap, pconf = assign(M)
    mat_ms = (time.perf_counter() - t) * 1000

    photo_map = {photo_pieces[i]["k"]: photo_cells[j]["r"] * s + photo_cells[j]["c"]
                 for i, j in pmap.items()}
    used = set(photo_map)
    decoys = [p for p in photo_pieces if p["k"] not in used]
    low = [photo_pieces[i]["k"] for i, j in pmap.items() if not pconf[i]]
    print(f"  匹配：照片格 {len(photo_map)}/{len(photo_cells)}  混淆 {len(decoys)}  "
          f"低置信 {len(low)}  用时 {mat_ms:.0f}ms")

    for i, p in enumerate(photo_pieces):
        j = pmap.get(i)
        if j is None:
            print(f"    片{p['k']:2d} → 混淆/放弃 {float(M[i].max()):.3f}")
        else:
            flag = "" if pconf[i] else "  ← 低置信"
            print(f"    片{p['k']:2d} → ({photo_cells[j]['r']},{photo_cells[j]['c']}) "
                  f"{M[i, j]:.3f}{flag}")

    wmap, wneed = white_plan(whites, white_cells, photo_map, pieces, cells, s)
    wmap_k = {whites[i]["k"]: rc for i, rc in wmap.items()}
    print("  纯白格：", [(c["r"], c["c"]) for c in white_cells])
    print("  纯白碎片指派：", wmap_k)

    if p_cls is not None and c_cls is not None:
        yolo_agree = sum(1 for i, j in pmap.items() if p_cls[i]["key"] == c_cls[j]["key"])
    report.update({"cap_ms": round(cap_ms, 1), "mat_ms": round(mat_ms, 1),
                   "n_pieces": len(pieces), "n_white": len(whites),
                   "n_decoys": len(decoys), "low_conf": low, "yolo_agree": yolo_agree,
                   "assign": {f"p{k}": [j // s, j % s] for k, j in photo_map.items()},
                   "white": {f"p{k}": list(rc) for k, rc in wmap_k.items()}})

    if dry:
        for p in pieces:
            cv2.imwrite(str(out / f"{tag}_piece{p['k']:02d}.png"), p["crop"])
            if save_data and p.get("core") is not None:
                cv2.imwrite(str(out / f"{tag}_core{p['k']:02d}.png"), p["core"])
        if save_data:
            for c in cells:
                cv2.imwrite(str(out / f"{tag}_cell_{c['r']}_{c['c']}.png"), c["img"])
        (out / f"{tag}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        np.save(out / f"{tag}_match.npy", M)
        page.screenshot(path=str(out / f"{tag}_page.png"))
        return report

    # ---- 执行 ----
    res = execute(page, geom, board, pw, ph, s, pieces, photo_map, wmap_k, part_title)
    report.update(res)
    try:
        page.wait_for_selector("#victory-modal.show", timeout=5000)
        report["victory"] = True
    except Exception:                                    # noqa: BLE001
        report["victory"] = False
    (out / f"{tag}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    page.screenshot(path=str(out / f"{tag}_final.png"))
    return report


def next_game(page) -> None:
    """点"再来一局"：同一张源图重新切块（新卡扣/新槽位），比整页重载快得多。"""
    page.click("#btn-play-again")
    page.wait_for_selector("#victory-modal.show", state="hidden", timeout=10_000)
    page.wait_for_timeout(250)
    wait_canvas_ready(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=25, choices=[9, 16, 25])
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只识别+打印指派，不拖拽")
    ap.add_argument("--games", type=int, default=1, help="同一浏览器里连跑几局")
    ap.add_argument("--url", default=OFFICIAL)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--weights", default=str(ROOT / "models" / "p3cls_piece" / "weights" / "best.pt"),
                    help="自训练 YOLO 分类模型；传空字符串则关闭")
    ap.add_argument("--record", default=None, help="把浏览器画面录成视频（目录）")
    args = ap.parse_args()

    s = int(round(math.sqrt(args.difficulty)))
    tag0 = args.tag or f"p3_d{args.difficulty}"
    LOGS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    results = []
    model = load_model(args.weights)

    with sync_playwright() as pw:
        browser, page, net_mode = open_page(pw, args.url, headed=args.headed,
                                            record_dir=args.record)
        print(f"  网络: {net_mode}")
        t = time.perf_counter()
        settle(page)
        start_part3(page, s)
        print(f"  开题耗时 {(time.perf_counter() - t) * 1000:.0f}ms")

        for g in range(1, args.games + 1):
            tag = tag0 if args.games == 1 else f"{tag0}_g{g}"
            print(f"\n—— 第 {g}/{args.games} 局 ——")
            rep = solve_once(page, s, tag, dry=args.dry_run, model=model)
            results.append(rep)
            if rep.get("victory") is not None and not args.dry_run:
                print(f"  本局：{rep['placed']}/{s * s}  拖拽 {rep['tries']}（失败 {rep['fails']}）"
                      f"  {'✅ 完成' if rep.get('victory') else '❌ 未完成'}")
            if args.dry_run or g == args.games:
                break
            next_game(page)
        if args.record:
            try:
                page.context.close()          # 录像要等 context 关闭才写完
            except Exception:                  # noqa: BLE001
                pass
        browser.close()

    ok = sum(1 for r in results if r.get("victory"))
    print(f"\n汇总：{ok}/{len(results)} 局完成")
    for i, r in enumerate(results, 1):
        print(f"  局{i}: {r.get('placed', '-')}/25  拖拽 {r.get('tries', '-')}"
              f"（失败 {r.get('fails', '-')}）  {'✅' if r.get('victory') else '❌'}")
    print(f"总耗时 {(time.perf_counter() - t0) * 1000:.0f}ms  日志 logs/{tag0}_*")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
