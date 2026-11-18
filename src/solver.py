#!/usr/bin/env python3
"""
拼图约束求解器（Part 2 用；Part 1/3 复用同一套）

模型：s×s 网格，每个碎片 4 条边 ∈ {凸=1, 凹=-1, 直=0, 未知=None}。
约束：
  1. 相邻边互补：     A.right + B.left = 0，A.bottom + B.top = 0
  2. 外圈必须直边：   第 0 行 top=0、第 s-1 行 bottom=0、第 0 列 left=0、第 s-1 列 right=0
  3. 未知边对任何取值都成立（代价 0）

碎片不可旋转/翻转（页面里就是正着画的），所以镜像/旋转 180° 一般会破坏约束，
解基本唯一 —— 不需要额外的方向消歧。

检测会有错（遮挡导致的假直边、临界误判），所以不是"硬解"而是**带权最小违背**：
每条边带一个置信度 conf=|偏差|/t，违背代价 = 双方置信度的均值。
用 DFS + 分支限界找最小代价解；节点数封顶，超了返回当前最优。

离线自测：
  .venv/bin/python src/solver.py --selftest logs/p2_d25_truth.json logs/p2_d25_canvas.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SIDES = ("top", "right", "bottom", "left")
OPP = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


# ------------------------------------------------------------------ 核心求解
def _place_viol(r, c, i, grid, pieces, s):
    """把 i 放到 (r,c)：返回 (违背数, 违背权重和)，只看已放置的邻居/边界。"""
    p = pieces[i]
    v = 0
    w = 0.0

    def bump(a, b=None):
        nonlocal v, w
        v += 1
        w += a[1] if b is None else 0.5 * (a[1] + b[1])

    if c == 0 and p["left"][0] not in (0, None):
        bump(p["left"])
    if c == s - 1 and p["right"][0] not in (0, None):
        bump(p["right"])
    if r == 0 and p["top"][0] not in (0, None):
        bump(p["top"])
    if r == s - 1 and p["bottom"][0] not in (0, None):
        bump(p["bottom"])
    if c > 0:
        a, b = p["left"], pieces[grid[r][c - 1]]["right"]
        if a[0] is not None and b[0] is not None and a[0] != -b[0]:
            bump(a, b)
    if r > 0:
        a, b = p["top"], pieces[grid[r - 1][c]]["bottom"]
        if a[0] is not None and b[0] is not None and a[0] != -b[0]:
            bump(a, b)
    return v, w


def solve(pieces, s: int, max_viol: int = 0, node_cap: int = 400_000):
    """DFS + 硬上限 max_viol（违背条数）。

    返回 (grid, viol, nodes)；无解时 grid=None。
    必须先按 max_viol=0 搜，再逐层放宽 —— 否则会在错误分支上耗尽节点。
    """
    cells = s * s
    n = len(pieces)
    assert n >= cells, f"碎片数 {n} 少于格子数 {cells}"
    grid = [[-1] * s for _ in range(s)]
    used = [False] * n
    nodes = 0
    flat_cnt = [sum(1 for k in SIDES if pieces[i][k][0] == 0) for i in range(n)]
    order = sorted(range(n), key=lambda i: -flat_cnt[i])

    def dfs(pos, viol):
        nonlocal nodes
        if nodes >= node_cap:
            return None
        if pos == cells:
            return [row[:] for row in grid]
        r, c = divmod(pos, s)
        cands = []
        for i in order:
            if used[i]:
                continue
            v, w = _place_viol(r, c, i, grid, pieces, s)
            if viol + v <= max_viol:
                cands.append((v, w, i))
        cands.sort()                      # 先试违背少的，尽早收敛
        for v, w, i in cands:
            nodes += 1
            grid[r][c] = i
            used[i] = True
            res = dfs(pos + 1, viol + v)
            used[i] = False
            grid[r][c] = -1
            if res is not None:
                return res
        return None

    g = dfs(0, 0)
    viol = None
    if g is not None:
        viol = len(verify(g, pieces))
    return g, viol, nodes


# --------------------------------------------------- 贪心 + 局部修复 + 限界
def _edge_score(a, b):
    """两条互补边的匹配分数：正=吻合，负=冲突，0=未知/中性。"""
    la, ca = a
    lb, cb = b
    if la is None or lb is None:
        return 0.0
    if la == -lb:
        return 0.6 + 0.4 * min(ca, cb)       # 吻合：置信度越高越可信
    return -1.0 - min(ca, cb)               # 冲突


def _cell_score(r, c, i, grid, pieces, s):
    """把 i 放 (r,c) 的局部得分（越大越好）。"""
    p = pieces[i]
    sc = 0.0
    if c == 0:
        sc += 1.2 if p["left"][0] == 0 else (-1.2 if p["left"][0] is not None else 0.0)
    if c == s - 1:
        sc += 1.2 if p["right"][0] == 0 else (-1.2 if p["right"][0] is not None else 0.0)
    if r == 0:
        sc += 1.2 if p["top"][0] == 0 else (-1.2 if p["top"][0] is not None else 0.0)
    if r == s - 1:
        sc += 1.2 if p["bottom"][0] == 0 else (-1.2 if p["bottom"][0] is not None else 0.0)
    if c > 0:
        sc += _edge_score(p["left"], pieces[grid[r][c - 1]]["right"])
    if r > 0:
        sc += _edge_score(p["top"], pieces[grid[r - 1][c]]["bottom"])
    return sc


def greedy(pieces, s: int):
    """按"约束最多"的顺序贪心铺满；返回 grid 或 None。"""
    n = len(pieces)
    # 角 -> 边 -> 内部：先放直边多的，边界约束能立刻生效
    cells = sorted(((r, c) for r in range(s) for c in range(s)),
                   key=lambda rc: -(sum(1 for k, ok in
                                        (("left", rc[1] == 0), ("right", rc[1] == s - 1),
                                         ("top", rc[0] == 0), ("bottom", rc[0] == s - 1)) if ok)))
    order = []
    for r, c in cells:
        order.append((r, c))
    # 实际放置顺序仍要求邻居先放：按行优先，但行内可换向。这里用"行优先 + 每行依直边数排序"
    grid = [[-1] * s for _ in range(s)]
    used = [False] * n
    for r in range(s):
        cols = range(s)
        for c in cols:
            best, best_sc = -1, -1e9
            for i in range(n):
                if used[i]:
                    continue
                sc = _cell_score(r, c, i, grid, pieces, s)
                if sc > best_sc:
                    best, best_sc = i, sc
            if best < 0:
                return None
            grid[r][c] = best
            used[best] = True
    return grid


def grow(pieces, s: int):
    """best-first 生长：每步挑"约束最多 + 匹配分最高"的一处放。

    行优先贪心的问题是：第 0 行放错一块，后面全错且无法回头。
    生长法先放置信度最高的位置（角块 2 条边界直边 + 邻居），错误不会一步扩散。
    """
    n = len(pieces)              # 碎片可能比格子多（混淆块）
    cells = s * s
    grid = [[-1] * s for _ in range(s)]
    used = [False] * n
    remaining = cells

    def neighbors_placed(r, c):
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

    def full_score(r, c, i):
        p = pieces[i]
        sc, hard = 0.0, 0.0
        if c == 0:
            sc += 1.2 if p["left"][0] == 0 else (-1.2 if p["left"][0] is not None else 0)
        if c == s - 1:
            sc += 1.2 if p["right"][0] == 0 else (-1.2 if p["right"][0] is not None else 0)
        if r == 0:
            sc += 1.2 if p["top"][0] == 0 else (-1.2 if p["top"][0] is not None else 0)
        if r == s - 1:
            sc += 1.2 if p["bottom"][0] == 0 else (-1.2 if p["bottom"][0] is not None else 0)
        if c > 0 and grid[r][c - 1] >= 0:
            sc += _edge_score(p["left"], pieces[grid[r][c - 1]]["right"])
        if c < s - 1 and grid[r][c + 1] >= 0:
            sc += _edge_score(p["right"], pieces[grid[r][c + 1]]["left"])
        if r > 0 and grid[r - 1][c] >= 0:
            sc += _edge_score(p["top"], pieces[grid[r - 1][c]]["bottom"])
        if r < s - 1 and grid[r + 1][c] >= 0:
            sc += _edge_score(p["bottom"], pieces[grid[r + 1][c]]["top"])
        return sc

    while remaining:
        best = None
        for r in range(s):
            for c in range(s):
                if grid[r][c] >= 0:
                    continue
                nb = neighbors_placed(r, c)
                for i in range(n):
                    if used[i]:
                        continue
                    sc = full_score(r, c, i) + 0.35 * nb
                    key = (sc, nb)
                    if best is None or key > best[0]:
                        best = (key, r, c, i)
        if best is None:
            return None
        (_, _), r, c, i = best
        grid[r][c] = i
        used[i] = True
        remaining -= 1
    return grid


def repair(pieces, s: int, grid, rounds: int = 40):
    """成对交换下降：只接受让总违背数下降的交换。"""
    def total(g):
        return len(verify(g, pieces))

    best_t = total(grid)
    for _ in range(rounds):
        improved = False
        for a in range(s * s):
            for b in range(a + 1, s * s):
                ra, ca = divmod(a, s)
                rb, cb = divmod(b, s)
                grid[ra][ca], grid[rb][cb] = grid[rb][cb], grid[ra][ca]
                t = total(grid)
                if t < best_t:
                    best_t, improved = t, True
                else:
                    grid[ra][ca], grid[rb][cb] = grid[rb][cb], grid[ra][ca]
        if not improved:
            break
    return grid, best_t


def solve_best(pieces, s: int, node_cap: int = 20_000):
    """贪心 -> 局部修复 -> 如果还有违背再试限界 DFS。

    返回 (grid, viol, nodes, info)。
    """
    g = grow(pieces, s)
    stage = "grow"
    if g is None:
        g, stage = greedy(pieces, s), "greedy"
    if g is None:
        return None, None, 0, {"stage": "no_construction"}
    g, v = repair(pieces, s, g)
    info = {"stage": stage, "construct_viol": v}
    if v > 0:
        # 用贪心解作上界，做一次带界的 DFS（可能进一步降低违背）
        best_v = v
        grid0 = [row[:] for row in g]
        gg, vv, nodes = solve(pieces, s, max_viol=max(0, v - 1), node_cap=node_cap)
        info["bnb_nodes"] = nodes
        if gg is not None and vv is not None and vv < best_v:
            g, best_v = gg, vv
            info["bnb_improved"] = True
        else:
            g = grid0
        v = best_v
    return g, v, info.get("bnb_nodes", 0), info


def verify(grid, pieces):
    """统计最小代价解里被违背的约束（用于日志/告警）。"""
    s = len(grid)
    bad = []
    for r in range(s):
        for c in range(s):
            i = grid[r][c]
            if c + 1 < s:
                a, b = pieces[i]["right"], pieces[grid[r][c + 1]]["left"]
                if a[0] is not None and b[0] is not None and a[0] != -b[0]:
                    bad.append((r, c, "h", a[0], b[0]))
            if r + 1 < s:
                a, b = pieces[i]["bottom"], pieces[grid[r + 1][c]]["top"]
                if a[0] is not None and b[0] is not None and a[0] != -b[0]:
                    bad.append((r, c, "v", a[0], b[0]))
            for side, ok in (("left", c == 0), ("right", c == s - 1),
                             ("top", r == 0), ("bottom", r == s - 1)):
                if ok and pieces[i][side][0] not in (0, None):
                    bad.append((r, c, side, pieces[i][side][0], 0))
    return bad


# ------------------------------------------------------------------ 离线自测
def _edges_from_truth(piece):
    """真值符号 -> 本模块语义（引擎 +1 实际是凹）"""
    m = {1: -1, -1: 1, 0: 0}
    return {s: (m[piece["edges"][s]], 1.0) for s in SIDES}


def selftest(truth_path: str, canvas_path: str | None):
    truth = json.loads(Path(truth_path).read_text())
    geo, tp = truth["geometry"], truth["pieces"]
    s = int(round(math.sqrt(len(tp))))
    ok_all = True

    def run(name, edges_list, expect_grid):
        grid, cost, nodes, info = solve_best(edges_list, s)
        if grid is None:
            print(f"  {name}: 无解 info={info}")
            return False
        got = [tp[grid[r][c]]["row"] * s + tp[grid[r][c]]["col"]
               for r in range(s) for c in range(s)]
        good = sum(1 for k, g in enumerate(got) if g == k)
        bad = verify(grid, edges_list)
        print(f"  {name}: 位置正确 {good}/{s*s}  违背={len(bad)} info={info}")
        return good == s * s

    print(f"[1] 纯真值边（无 unknown、无噪声）s={s}")
    ok_all &= run("truth", [_edges_from_truth(p) for p in tp], None)

    if canvas_path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline import analyze  # noqa: E402
        import cv2
        img = cv2.imread(canvas_path, cv2.IMREAD_UNCHANGED)
        an = analyze(img, len(tp))
        print(f"[2] 真实检测（含 unknown 与临界错误）：检出 {len(an['pieces'])} 片")
        # 检测碎片 -> 真值碎片 的最近中心匹配
        import numpy as np
        cand = []
        for i, p in enumerate(an["pieces"]):
            for j, t in enumerate(tp):
                d = math.hypot(t["x"] + geo["pw"] / 2 - p["cx"],
                               t["y"] + geo["ph"] / 2 - p["cy"])
                cand.append((d, i, j))
        cand.sort()
        ui, uj, pairs = set(), set(), {}
        for d, i, j in cand:
            if i in ui or j in uj or d > geo["pw"] * 0.6:
                continue
            ui.add(i); uj.add(j); pairs[i] = j
        # 用检测边重排成"真值顺序"的列表，便于按真值行列表验证
        det_edges = [None] * len(tp)
        for i, j in pairs.items():
            det_edges[j] = an["pieces"][i]["sides"]
        missing = [j for j in range(len(tp)) if det_edges[j] is None]
        if missing:
            print(f"  !! 有 {len(missing)} 片未匹配到真值，跳过真实检测自测")
            return ok_all
        # 参照：把"真值排列"直接代入检测到的边，看它有多少违背（可达下界）
        truth_arr = [None] * len(tp)
        for r in range(s):
            for c in range(s):
                truth_arr[r * s + c] = det_edges[r * s + c]
        print(f"  参照：真值排列本身的违背数 = {len(verify([list(range(k*s, k*s+s)) for k in range(s)], det_edges))}")
        ok_all &= run("detected", det_edges, None)
    print("\n自测结果:", "全部通过" if ok_all else "存在问题")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", nargs="+")
    args = ap.parse_args()
    if args.selftest:
        tp = args.selftest[0]
        cv = args.selftest[1] if len(args.selftest) > 1 else None
        return 0 if selftest(tp, cv) else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ======================================================================
# 混淆感知求解（官方站：25 格会给你 30 块，其中 5 块永远吸不上）
# ======================================================================
# 思路：grow() 本来就会"用掉多少块算多少块"，30 块填 25 格自然剩 5 块没用上。
# 但它可能先抓了混淆块占住格子，所以再加一步**换出修复**：
#   反复尝试「把已放的一格换成一块没用过的」，只要总违背数下降就接受。
# 混淆块的边是随机的，跟邻居几乎必然冲突，所以会被换出去。
def solve_with_decoys(pieces, s: int, node_cap: int = 250_000, max_layers: int = 8):
    """碎片数 > 格子数时的求解（官方站 5×5 是 30 块填 25 格）。

    策略：从"零违背"开始逐层放宽允许的违背条数，让 DFS 自己决定
    用哪 cells 块、怎么排。没被选中的碎片就是混淆块。
    实在搜不出再用「生长 + 换出修复」兜底。
    """
    n = len(pieces)
    cells = s * s
    assert n >= cells, f"碎片数 {n} 少于格子数 {cells}"
    info = {"n_pieces": n, "extras": n - cells, "layers": []}

    spent = 0
    for mv in range(0, max_layers + 1):
        left = node_cap - spent
        if left <= 0:
            break
        g, viol, nodes = solve(pieces, s, max_viol=mv, node_cap=left)
        spent += nodes
        info["layers"].append((mv, nodes, g is not None))
        if g is not None:
            used = {i for row in g for i in row}
            info.update({"stage": "dfs", "viol": len(verify(g, pieces)),
                         "unused": sorted(i for i in range(n) if i not in used),
                         "nodes": spent})
            return g, info["viol"], info

    # 兜底：生长 + 换出修复
    g = grow(pieces, s) or greedy(pieces, s)
    if g is None:
        return None, None, {**info, "stage": "failed"}
    g, v = repair(pieces, s, g)
    used = {i for row in g for i in row}
    info.update({"stage": "grow+repair", "viol": v, "nodes": spent,
                 "unused": sorted(i for i in range(n) if i not in used)})
    return g, v, info


# ---------------------------------------------------------- 锚点顺序（官方规则）
# checkSnap 的 hasAnchor：外圈格子永远可放；内部格子必须已有邻居 isPlaced。
# 所以拖拽顺序必须是「外圈 -> 一层层往里」。
def placement_order(grid, s: int):
    """按「到边界的距离」升序给出放置顺序：(r, c, level)。

    level = min(r, c, s-1-r, s-1-c)，即离外圈几层。
    同一层内部任意顺序都满足锚点要求（只要外层已放完）。
    """
    order = []
    for r in range(s):
        for c in range(s):
            if grid[r][c] < 0:
                continue
            order.append((min(r, c, s - 1 - r, s - 1 - c), r, c))
    order.sort()
    return [(r, c, lvl) for lvl, r, c in order]


def cell_candidates(pieces, grid, r, c, s, limit=4):
    """给某格排候选碎片（供"吸不上就换一块试"的兜底用）。

    评分 = 与已放邻居的边吻合（权重 2）+ 该格若是外圈则奖励直边。
    返回 [(piece_index, score), ...]，分数从高到低。
    """
    cands = []
    for i in range(len(pieces)):
        sc = 0.0
        p = pieces[i]
        if c == 0:
            sc += 1.2 if p["left"][0] == 0 else -1.2
        if c == s - 1:
            sc += 1.2 if p["right"][0] == 0 else -1.2
        if r == 0:
            sc += 1.2 if p["top"][0] == 0 else -1.2
        if r == s - 1:
            sc += 1.2 if p["bottom"][0] == 0 else -1.2
        if c > 0 and grid[r][c - 1] >= 0 and grid[r][c - 1] != i:
            sc += _edge_score(p["left"], pieces[grid[r][c - 1]]["right"])
        if c < s - 1 and grid[r][c + 1] >= 0 and grid[r][c + 1] != i:
            sc += _edge_score(p["right"], pieces[grid[r][c + 1]]["left"])
        if r > 0 and grid[r - 1][c] >= 0 and grid[r - 1][c] != i:
            sc += _edge_score(p["top"], pieces[grid[r - 1][c]]["bottom"])
        if r < s - 1 and grid[r + 1][c] >= 0 and grid[r + 1][c] != i:
            sc += _edge_score(p["bottom"], pieces[grid[r + 1][c]]["top"])
        cands.append((i, sc))
    cands.sort(key=lambda t: -t[1])
    return cands[:limit]
