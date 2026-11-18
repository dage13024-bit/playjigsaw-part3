#!/usr/bin/env python3
"""
play_part2.py —— 单文件教学版：自动玩「Part 2 · 纯白拼图」

设计原则：只用画面像素 + OpenCV 几何 + 约束求解 + 真鼠标。
不读取网页任何内部状态（不碰 window.app、不读正确坐标），符合考核规则。

运行：
    .venv/bin/python play_part2.py --difficulty 4 --headed     # 看得见浏览器
    .venv/bin/python play_part2.py --difficulty 4              # 后台跑

整条流水线（7 步）：
    1. 开考          点 UI 选 Part/难度，点 "Start this part"
    2. 取画面        canvas.toDataURL() 拿一张带 alpha 的 PNG（约 7ms）
    3. 分割碎片      alpha>=0.99 且 亮度>=240  → 连通域 = 每块碎片
    4. 找棋盘        不透明 + 暗色 + 实心矩形（棋盘 #0f172a 灰度约 27）
    5. 判四条边      拟合"核心矩形"，再看每条边外侧有没有鼓包 / 内侧有没有凹槽
    6. 求解          相邻边凸凹互补 + 外圈必须是直边 → 每块碎片落在哪个格子
    7. 拖拽          从碎片包围盒中心抓起，按"目标中心−核心中心"位移放下
最后读进度条判断是否完成。

这个文件刻意写成"一条直线"，方便从上往下读。进阶/实验版本的代码在 src/ 里。
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

# ============================== 可调参数 ==============================
URL = "https://f27e491f.cv-animal-puzzle.pages.dev/?mode=exam"

ALPHA_MIN = 0.99     # 碎片本体是"不透明"
WHITE_MIN = 240      # 碎片本体是纯白 255；阴影/重影在 26~250。放低到 200 相邻碎片会粘连
BOARD_MAX_GRAY = 80  # 棋盘 #0f172a 灰度约 27
TAB_RATIO = 0.22     # 卡扣长度 = 0.22 × min(碎片宽, 碎片高)（与引擎一致）
TAB_EXT = 1.15       # 卡扣顶端伸出 = 1.15 × 卡扣长度

LOGS = Path(__file__).resolve().parent / "logs"
SIDES = ("top", "right", "bottom", "left")

# 每帧调试图；想看得更清楚就把 True 打开
SAVE_DEBUG = True


# ============================== 1. 开考 ==============================
def start_exam(page, part: int, difficulty: int):
    page.goto(URL, wait_until="load", timeout=60_000)
    page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
    page.check(f'input[name="exam-part"][value="{part}"]')
    page.check(f'input[name="exam-difficulty"][value="{difficulty}"]')
    page.click("#btn-exam-start")
    # 碎片构建完、面板收起，就说明这一局开始了
    page.wait_for_function(
        "() => window.exam && window.exam.running"
        " && document.querySelector('#exam-start').hidden", timeout=30_000)
    page.wait_for_timeout(350)


# ============================== 2. 取画面 ==============================
def grab_canvas(page) -> np.ndarray:
    """把 canvas 的像素拿下来。

    为什么不用截图？canvas.toDataURL() 只要 ~7ms（CDP 截图要 48ms），
    而且它保留了 **alpha 通道**：canvas 背景是 clearRect 出来的透明区，
    于是"碎片"天然就是 alpha=1 的那些像素，分割几乎不用做。
    """
    raw = page.evaluate("() => document.getElementById('puzzle-canvas').toDataURL('image/png')")
    buf = base64.b64decode(raw.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码 canvas 图像")
    return img


def canvas_geometry(page) -> dict:
    """canvas 位图尺寸 vs 它在页面上的 CSS 尺寸。

    鼠标事件用的是 **CSS 像素**，而检测用的是 **位图坐标**，两者差一个比例
    （实测 1043/1039.2 = 1.0036，缩放不同浏览器时差别更大）。所以必须换算。
    """
    return page.evaluate("""() => {
        const c = document.getElementById('puzzle-canvas');
        const r = c.getBoundingClientRect();
        return {left: r.left, top: r.top,
                sx: r.width / c.width, sy: r.height / c.height,
                cssW: r.width, cssH: r.height};
    }""")


# ============================== 3. 分割碎片 ==============================
def segment_pieces(img: np.ndarray):
    """alpha + 亮度 双条件 → 连通域 = 每块碎片。

    这一步是整个方案的地基，两个阈值都很关键：
      * alpha >= 0.99：排除碎片投下的柔和阴影（半透明）
      * 亮度 >= 240：碎片本体是纯白 255，但"碎片压在碎片上"的地方会被阴影
        压暗到 26~250。阈值放到 200，25 块碎片会粘成 12 块连在一起。
    返回每块碎片的 (外接矩形, 二值掩码, 质心)。
    """
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img.shape[:2], np.float32)
    gray = img[:, :, :3].astype(np.int16).mean(axis=2)
    mask = ((alpha >= ALPHA_MIN) & (gray >= WHITE_MIN)).astype(np.uint8)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pieces = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200:                       # 噪点
            continue
        one = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        pieces.append({
            "mask": one,
            "bbox": cv2.boundingRect(c),     # (x, y, w, h)，含卡扣
            "centroid": tuple(float(v) for v in centroids[i]),
            "area": int(area),
        })
    return pieces


# ============================== 4. 找棋盘 ==============================
def find_board(img: np.ndarray):
    """棋盘 = 唯一"不透明 + 暗色 + 实心矩形"。

    碎片是白的，背景是透明的，所以暗色不透明区只有棋盘。
    注意填充率阈值不能定太高：棋盘上放几块碎片后，暗区就只剩外框了（实测会掉到 0.5）。
    """
    if img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones(img.shape[:2], np.float32)
    gray = img[:, :, :3].astype(np.int16).mean(axis=2)
    dark = ((alpha >= 0.99) & (gray <= BOARD_MAX_GRAY)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    best = None
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
                            int(stats[i, cv2.CC_STAT_AREA]))
        if w < 100 or h < 100:
            continue
        fill = area / float(w * h)
        if fill > 0.25 and (best is None or area > best["area"]):
            best = {"x": x, "y": y, "w": w, "h": h, "area": area}
    return best


# ============================== 5. 判四条边 ==============================
def _sample_edge(x1, y1, x2, y2, label, tab, n=8):
    """把引擎（PieceCutter.drawEdge）的卡扣曲线采样成折线。

    label: 1=凸(往外鼓) / -1=凹(往里凹) / 0=直边
    引擎里法向量在四条边上都指向碎片内部，所以这里取 sg=-label。
    """
    if label == 0:
        return [(x2, y2)]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return [(x2, y2)]
    nx, ny = -dy / L, dx / L
    sg = -label

    def pt(u, off):
        return (x1 + dx * u + nx * tab * off * sg, y1 + dy * u + ny * tab * off * sg)

    p1, p2, p3, p4 = pt(0.38, 0.0), pt(0.42, 0.30), pt(0.35, 1.05), pt(0.50, 1.15)
    p5, p6, p7 = pt(0.65, 1.05), pt(0.58, 0.30), pt(0.62, 0.0)

    def cubic(a, b, c, d):
        out = []
        for i in range(1, n + 1):
            t = i / n
            u = 1 - t
            out.append((u**3 * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t**3 * d[0],
                        u**3 * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t**3 * d[1]))
        return out

    pts = [p1] + cubic(p1, p2, p3, p4) + cubic(p4, p5, p6, p7)
    pts.append((x2, y2))
    return pts


_TEMPLATE_CACHE = {}


def template_for(combo, pw, ph):
    """按"四条边是什么"合成一块标准碎片（用来跟真碎片比形状）。"""
    key = (combo, round(pw, 1), round(ph, 1))
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    tab = TAB_RATIO * min(pw, ph)
    pad = int(math.ceil(tab * 1.3)) + 2
    W, H = int(math.ceil(pw)) + 2 * pad, int(math.ceil(ph)) + 2 * pad
    ox, oy = pad, pad
    poly = [(ox, oy)]
    poly += _sample_edge(ox, oy, ox + pw, oy, combo[0], tab)
    poly += _sample_edge(ox + pw, oy, ox + pw, oy + ph, combo[1], tab)
    poly += _sample_edge(ox + pw, oy + ph, ox, oy + ph, combo[2], tab)
    poly += _sample_edge(ox, oy + ph, ox, oy, combo[3], tab)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.array(poly)).astype(np.int32)], 255)
    _TEMPLATE_CACHE[key] = (m, (pad, pad))
    return m, (pad, pad)


def _density(mask, x0, y0, x1, y1):
    """一个矩形小窗口里"有碎片"的比例。"""
    H, W = mask.shape[:2]
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(W, int(round(x1))), min(H, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask[y0:y1, x0:x1]
    return float(np.count_nonzero(roi)) / float(roi.size)


def side_densities(mask, core, tab, side, a=0.34, b=0.66):
    """量一条边的两个小窗口：

      * out（外侧鼓包区）：在核心边外面 0.25t~1.0t 的一条带子里有多少碎片
      * in （内侧凹槽区）：在核心边里面 0.25t~1.0t 的一条带子里有多少碎片

    实测统计（用真值核心位置量的）：
        直边: out=0.00  in=1.00     ← 外侧空、内侧实
        凹边: out=0.00  in=0.33     ← 内侧被挖掉一块
        凸边: out=0.55  in=1.00     ← 外侧多出一个鼓包
    """
    xL, yT, W, H = core
    xR, yB = xL + W, yT + H
    d0, d1 = 0.25 * tab, 1.00 * tab
    if side == "right":
        return (_density(mask, xR + d0, yT + a * H, xR + d1, yT + b * H),
                _density(mask, xR - d1, yT + a * H, xR - d0, yT + b * H))
    if side == "left":
        return (_density(mask, xL - d1, yT + a * H, xL - d0, yT + b * H),
                _density(mask, xL + d0, yT + a * H, xL + d1, yT + b * H))
    if side == "bottom":
        return (_density(mask, xL + a * W, yB + d0, xL + b * W, yB + d1),
                _density(mask, xL + a * W, yB - d1, xL + b * W, yB - d0))
    return (_density(mask, xL + a * W, yT - d1, xL + b * W, yT - d0),
            _density(mask, xL + a * W, yT + d0, xL + b * W, yT + d1))


def label_from_densities(out: float, inn: float):
    """两个密度 → 边的标签(1 凸 / -1 凹 / 0 直 / None 拿不准)。"""
    if inn < 0.70:                 # 内侧被挖掉 → 凹
        return -1, min(1.0, (0.70 - inn) / 0.37 + 0.4)
    if out > 0.50:                 # 外侧鼓包 → 凸
        return 1, min(1.0, (out - 0.50) / 0.10 + 0.6)
    if out < 0.15:                 # 外空内实 → 直边
        return 0, min(1.0, (inn - 0.70) / 0.30 + 0.5)
    return None, 0.0               # 卡在中间（多半是卡扣被压住）→ 未知


def fit_piece(piece, pw, ph, canvas_wh):
    """找这块碎片的"核心矩形"，并给出四条边的标签。

    核心矩形 = 去掉卡扣之后那个矩形本体。为什么必须精确？
      实测：核心偏一个卡扣长度(≈20px)时，四条边会一起判错。
    为什么不能直接用包围盒推？
      凸起被别的碎片压住时包围盒根本不延伸，`bx + 1.15t` 这几个假设会全部落空。
    所以这里做坐标下降：移动核心位置 → 由密度得到标签 → 合成同标签的标准碎片
    → 和真碎片算 IoU（重合度）。IoU 最高的那个位置就是最自洽的位置。
    """
    mask, (bx, by, bw, bh) = piece["mask"], piece["bbox"]
    tab = TAB_RATIO * min(pw, ph)
    ext = TAB_EXT * tab

    def evaluate(xL, yT):
        core = (xL, yT, pw, ph)
        labels = {}
        for side in SIDES:
            labels[side], _ = label_from_densities(*side_densities(mask, core, tab, side))
        combo = tuple(labels[s] if labels[s] is not None else 0 for s in SIDES)
        tpl, (core_x, core_y) = template_for(combo, pw, ph)
        M = np.array([[1, 0, xL - core_x], [0, 1, yT - core_y]], dtype=np.float32)
        warped = cv2.warpAffine(tpl, M, (mask.shape[1], mask.shape[0]),
                                flags=cv2.INTER_NEAREST)
        inter = np.count_nonzero((warped > 0) & (mask > 0))
        union = np.count_nonzero((warped > 0) | (mask > 0))
        return (inter / union if union else 0.0), labels, core

    # 初值：包围盒推的 4 个假设里挑最好的
    best = None
    for dx in (0, ext):
        for dy in (0, ext):
            r = evaluate(bx + dx, by + dy)
            if best is None or r[0] > best[0]:
                best = r
    # 粗扫（步长 3）+ 细化（步长 1），先水平后垂直
    for step, half in ((3, int(ext) + 4), (1, 2)):
        for axis in (0, 1):
            base = best[2][axis]
            for delta in range(-half, half + 1, step):
                if delta == 0:
                    continue
                xL = base + delta if axis == 0 else best[2][0]
                yT = base + delta if axis == 1 else best[2][1]
                r = evaluate(xL, yT)
                if r[0] > best[0]:
                    best = r

    iou, labels, core = best
    # 碎片贴着画布边界时那一条边可能被裁掉，标签不可信
    W, H = canvas_wh
    for side, cut in (("left", core[0] <= 1), ("top", core[1] <= 1),
                      ("right", core[0] + pw >= W - 1), ("bottom", core[1] + ph >= H - 1)):
        if cut:
            labels[side] = None
    return labels, core, iou


# ============================== 6. 求解 ==============================
def solve(labels_list, s):
    """把碎片排进 s×s 网格。

    约束只有两条，但很强：
      1. 相邻边互补：A 的右边是凸，B 的左边就必须是凹
      2. 外圈必须是直边：第 0 行上边=0、最后一行下边=0、第 0 列左边=0、最后一列右边=0
    由于碎片不能旋转/翻转，满足这两条约束的排布基本唯一。

    搜索方式：best-first 生长。每一步在所有"空位 × 未用碎片"里挑得分最高的放。
    （行优先贪心的问题是第 0 行错一块、后面全错且无法回头。）
    """
    n = s * s
    grid = [[-1] * s for _ in range(s)]
    used = [False] * n

    def edge_score(a, b):
        """两条互补边的吻合分：正=吻合，负=冲突，0=未知。"""
        if a is None or b is None:
            return 0.0
        return 2.0 if a == -b else -2.0

    def cell_score(r, c, i):
        lab = labels_list[i]
        sc = 0.0
        if c == 0:
            sc += 1.2 if lab["left"] == 0 else (-1.2 if lab["left"] is not None else 0)
        if c == s - 1:
            sc += 1.2 if lab["right"] == 0 else (-1.2 if lab["right"] is not None else 0)
        if r == 0:
            sc += 1.2 if lab["top"] == 0 else (-1.2 if lab["top"] is not None else 0)
        if r == s - 1:
            sc += 1.2 if lab["bottom"] == 0 else (-1.2 if lab["bottom"] is not None else 0)
        if c > 0 and grid[r][c - 1] >= 0:
            sc += edge_score(lab["left"], labels_list[grid[r][c - 1]]["right"])
        if r > 0 and grid[r - 1][c] >= 0:
            sc += edge_score(lab["top"], labels_list[grid[r - 1][c]]["bottom"])
        return sc

    def placed_neighbors(r, c):
        k = 0
        if c > 0 and grid[r][c - 1] >= 0:
            k += 1
        if c < s - 1 and grid[r][c + 1] >= 0:
            k += 1
        if r > 0 and grid[r - 1][c] >= 0:
            k += 1
        if r < s - 1 and grid[r + 1][c] >= 0:
            k += 1
        return k

    for _ in range(n):
        best = None
        for r in range(s):
            for c in range(s):
                if grid[r][c] >= 0:
                    continue
                nb = placed_neighbors(r, c)
                for i in range(n):
                    if used[i]:
                        continue
                    key = (cell_score(r, c, i) + 0.35 * nb, nb)
                    if best is None or key > best[0]:
                        best = (key, r, c, i)
        if best is None:
            return None
        _, r, c, i = best
        grid[r][c] = i
        used[i] = True
    return grid


# ============================== 7. 拖拽 ==============================
def drag(page, geom, fx, fy, tx, ty):
    """从画布坐标 (fx,fy) 拖到 (tx,ty)。

    注意抓取点用 **包围盒中心** 而不是核心中心：核心可能估偏 20px，
    而包围盒中心一定在这块碎片的格子里（否则会抓错碎片）。
    落点用位移：把"目标中心 − 核心中心"加到抓取点上，误差正好被吸附半径吸收。
    """
    ax, ay = geom["left"] + fx * geom["sx"], geom["top"] + fy * geom["sy"]
    bx, by = geom["left"] + tx * geom["sx"], geom["top"] + ty * geom["sy"]
    page.mouse.move(ax, ay)
    page.mouse.down()
    page.mouse.move(bx, by)
    page.mouse.up()


def read_progress(page) -> int:
    """读页面上的进度文字（可见 UI，不是内部状态）。"""
    txt = page.evaluate("() => (document.getElementById('progress-display')||{}).textContent || ''")
    try:
        return int(txt.split("/")[0].strip())
    except Exception:
        return -1


# ============================== 主流程 ==============================
def main() -> int:
    ap = argparse.ArgumentParser(description="自动玩 Part 2 · 纯白拼图")
    ap.add_argument("--difficulty", type=int, default=4, choices=[4, 9, 16, 25])
    ap.add_argument("--part", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--headed", action="store_true", help="显示浏览器窗口")
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    t_all = time.perf_counter()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_context(viewport={"width": 1440, "height": 900},
                                   device_scale_factor=1).new_page()

        # ---- 1. 开考 ----
        start_exam(page, args.part, args.difficulty)
        progress0 = read_progress(page)
        t_boot = (time.perf_counter() - t_all) * 1000   # 浏览器启动 + 页面加载（一次性开销）

        # ---- 2. 取画面 ----
        t = time.perf_counter()
        img = grab_canvas(page)
        t_capture = (time.perf_counter() - t) * 1000
        geom = canvas_geometry(page)
        cv2.imwrite(str(LOGS / f"play_d{args.difficulty}_1_canvas.png"), img)

        # ---- 3. 分割 + 4. 找棋盘 ----
        t = time.perf_counter()
        pieces = segment_pieces(img)
        board = find_board(img)
        if board is None or len(pieces) != args.difficulty:
            print(f"❌ 检出 {len(pieces)} 块碎片 / 棋盘={bool(board)}，期望 {args.difficulty} 块")
            browser.close()
            return 1
        s = int(round(math.sqrt(args.difficulty)))
        pw, ph = board["w"] / s, board["h"] / s

        # ---- 5. 判四条边 ----
        labels_list, cores, ious = [], [], []
        for p in pieces:
            lab, core, iou = fit_piece(p, pw, ph, (img.shape[1], img.shape[0]))
            labels_list.append(lab)
            cores.append(core)
            ious.append(iou)
        t_detect = (time.perf_counter() - t) * 1000

        # ---- 6. 求解 ----
        t = time.perf_counter()
        grid = solve(labels_list, s)
        t_solve = (time.perf_counter() - t) * 1000
        if grid is None:
            print("❌ 求解失败")
            browser.close()
            return 1

        # 打印一下让用户能看到中间结果
        print(f"  棋盘 {board['w']}×{board['h']}  碎片核心 {pw:.1f}×{ph:.1f}  "
              f"卡扣 {TAB_RATIO * min(pw, ph):.1f}px")
        for i, (lab, iou) in enumerate(zip(labels_list, ious)):
            tag = "".join({1: "+", -1: "-", 0: "0", None: "?"}[lab[k]] for k in SIDES)
            print(f"  碎片{i:2d} 边(T R B L)={tag}  IoU={iou:.3f}")
        if SAVE_DEBUG:
            _save_debug(img, pieces, cores, grid, args.difficulty)

        # ---- 7. 拖拽 ----
        plan = []
        for r in range(s):
            for c in range(s):
                i = grid[r][c]
                gx = pieces[i]["bbox"][0] + pieces[i]["bbox"][2] / 2.0
                gy = pieces[i]["bbox"][1] + pieces[i]["bbox"][3] / 2.0
                tcx = board["x"] + (c + 0.5) * pw
                tcy = board["y"] + (r + 0.5) * ph
                plan.append((i, (r, c), (gx, gy),
                             (gx + tcx - cores[i][0] - pw / 2.0,
                              gy + tcy - cores[i][1] - ph / 2.0)))

        t = time.perf_counter()
        ok = 0
        for i, cell, (gx, gy), (tx, ty) in plan:
            before = read_progress(page)
            drag(page, geom, gx, gy, tx, ty)
            if read_progress(page) > before:
                ok += 1
        t_drag = (time.perf_counter() - t) * 1000

        # ---- 判定 ----
        try:
            page.wait_for_selector("#victory-modal.show", timeout=2000)
            done = True
        except Exception:
            done = False          # 超时=弹窗没出现，就是没完成
        final = read_progress(page)
        page.screenshot(path=str(LOGS / f"play_d{args.difficulty}_3_final.png"))
        browser.close()

    total = (time.perf_counter() - t_all) * 1000
    print(f"\n结果：放置正确 {ok}/{args.difficulty}   进度 {progress0} → {final}   "
          f"完成提示={'有' if done else '无'}")
    print(f"耗时：启动+加载 {t_boot:.0f}ms（一次性） | 取画面 {t_capture:.0f}ms + 检测 {t_detect:.0f}ms "
          f"+ 求解 {t_solve:.1f}ms + 拖拽 {t_drag:.0f}ms = 解题 {t_capture + t_detect + t_solve + t_drag:.0f}ms")
    print(f"调试图：logs/play_d{args.difficulty}_*.png")
    return 0


def _save_debug(img, pieces, cores, grid, difficulty):
    """把检测结果画出来，方便理解每一步在干什么。"""
    # 把透明背景合成成深色底：碎片保持纯白，方便看清
    rgb = img[:, :, :3].astype(np.float32)
    if img.shape[2] == 4:
        a = (img[:, :, 3:4].astype(np.float32) / 255.0)
        bg = np.array([28, 32, 44], np.float32)
        vis = (rgb * a + bg * (1 - a)).astype(np.uint8)
    else:
        vis = rgb.astype(np.uint8)
    for i, (p, core) in enumerate(zip(pieces, cores)):
        x, y, w, h = p["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (80, 80, 80), 1)
        cv2.rectangle(vis, (int(core[0]), int(core[1])),
                      (int(core[0] + core[2]), int(core[1] + core[3])), (0, 170, 255), 1)
        cv2.putText(vis, str(i), (int(core[0]) + 2, int(core[1]) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 230, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(LOGS / f"play_d{difficulty}_2_detect.png"), vis)


if __name__ == "__main__":
    sys.exit(main())
