#!/usr/bin/env python3
"""
Part 2（纯白）自动解题 —— 官方考核站

流程：
  1. 反复"开始这一部分"，直到拿到**稳定是白色**的一局（页面会用默认照片主题
     异步渲染一盘，加载完会覆盖白题并把计时器清零，所以必须等它安静下来）
  2. 槽位锚定：用引擎的等距槽位算每块碎片的位置（实测误差 2.7~3.2px）
  3. 判四条边：外侧/内侧两个密度窗口 → 凸/凹/直
  4. 求解：相邻边互补 + 外圈直边 + 混淆块剔除
  5. 执行：位置池 + 顺时针环形顺序 + 吸附反馈 + 失败归位（双向校验）

用法：
  .venv/bin/python play_white.py --difficulty 25 --headed
"""
from __future__ import annotations

import argparse
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

import slots as S                                   # noqa: E402
from net import open_page                           # noqa: E402
from play_p3 import settle                          # noqa: E402
from pipeline import detect_board                   # noqa: E402
from play_part2 import canvas_geometry, drag, grab_canvas, read_progress  # noqa: E402


def drag_with(page, geom, x1, y1, x2, y2, backend="page"):
    """两种拖拽后端：CDP 鼠标（默认） / 系统级 pyautogui。"""
    if backend == "pyautogui":
        import pyautogui
        ox = geom.get("scr_x", 0)
        oy = geom.get("scr_y", 0)
        pyautogui.moveTo(ox + (geom["left"] + x1 * geom["sx"]),
                         oy + (geom["top"] + y1 * geom["sy"]))
        pyautogui.mouseDown()
        pyautogui.moveTo(ox + (geom["left"] + x2 * geom["sx"]),
                         oy + (geom["top"] + y2 * geom["sy"]), duration=0.06)
        pyautogui.mouseUp()
    else:
        drag(page, geom, x1, y1, x2, y2)
from solver import SIDES, solve_with_decoys         # noqa: E402

OFFICIAL = "https://puzzle.moyunliuyin.me/?mode=exam"
LOGS = ROOT / "logs"
PART_TITLE = "纯白图形"


# ------------------------------------------------------------------ 布局
def ring_order(s: int):
    """顺时针一圈一圈，从外圈到内圈：每次放置都贴着上一块，至少两条边约束。"""
    order = []
    for k in range((s + 1) // 2):
        lo, hi = k, s - 1 - k
        if lo > hi:
            break
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


def level_order(s: int):
    """按层 + 行优先（对照组）。"""
    o = [(min(r, c, s - 1 - r, s - 1 - c), r, c) for r in range(s) for c in range(s)]
    o.sort()
    return [(r, c) for _, r, c in o]


# ------------------------------------------------------------------ 开局
def start_part2_once(page, url: str, difficulty: int, part: int = 2):
    """严格按工作流开一局：

        打开页面 → 选择 part/难度 → 点"开始这一部分"
        → 等状态"准备中" → 等状态"进行中" → 立刻抓画面。

    状态文本是页面自己给的唯一权威信号：
      * 只看到"进行中"不够（可能是上一局留下的）
      * 必须先观察到"准备中"，再观察到"进行中"，才说明这一局真的放出来了
    """
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)

    # 关键：页面加载时会先跑一次 loadTheme('animals')（默认主题，异步等图片）。
    # 它在 await 之后才继续执行 —— 如果我们这时已经开了 Part 2，
    # 它回来时会把 sourceCanvas 覆盖成照片、重建碎片、清空计时器。
    # 判据用"参考图 src 连续 2 秒不变"，比"大于 50KB"可靠：
    # 50KB 时图片可能只下载了一半，回调还没跑完。
    t_init = time.time()
    settle(page)
    print(f"  初始渲染就绪（等 {(time.time()-t_init)*1000:.0f}ms）")

    page.check(f'input[name="exam-part"][value="{part}"]')
    page.check(f'input[name="exam-difficulty"][value="{difficulty}"]')
    got = page.evaluate("() => (document.querySelector('input[name=\"exam-part\"]:checked')||{}).value")
    if str(got) != str(part):
        print(f"  ⚠️ Part {part} 没勾上（当前 {got}）")
        return None

    page.click("#btn-exam-start")

    def state():
        return page.evaluate("() => (document.getElementById('exam-state')||{}).textContent || ''")

    t0 = time.time()
    while '准备中' not in state():                 # ① 等它进入"准备中"
        if time.time() - t0 > 15:
            print(f"  ⚠️ 15 秒内没进入准备中（当前 {state()!r}）")
            return None
        page.wait_for_timeout(50)
    print(f"  准备中... ({(time.time() - t0) * 1000:.0f}ms)")

    t1 = time.time()
    while '进行中' not in state():                 # ② 等它变成"进行中"
        if time.time() - t1 > 120:
            print("  ⚠️ 120 秒内没进入进行中")
            return None
        page.wait_for_timeout(50)
    print(f"  进行中！(等待 {(time.time() - t1) * 1000:.0f}ms)")

    # ③ 等画布尺寸/内容稳定再抓：横幅进场会改变画布高度，
    #    在它稳定前算出来的碎片尺寸会让所有落点整体偏移（全失败）。
    from play_p3 import wait_canvas_ready
    page.wait_for_timeout(200)
    return wait_canvas_ready(page)


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=25, choices=[9, 16, 25])
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--order", default="ring", choices=["ring", "level"])
    ap.add_argument("--fast-start", action="store_true",
                    help="跳过'等默认主题图下载完'的等待（更快，但 5×5 有概率被覆盖重开）")
    ap.add_argument("--url", default=OFFICIAL)
    ap.add_argument("--cdp", default=None,
                    help="挂到已开调试端口的 Chrome（例如 http://127.0.0.1:9222），不自己启动浏览器")
    ap.add_argument("--no-start", action="store_true",
                    help="不点任何按钮：假定你已经在页面上开好题了，直接开始解题")
    ap.add_argument("--countdown", type=int, default=None,
                    help="开始拖拽前的倒计时秒数（用 pyautogui 时默认 5 秒，好让你切到浏览器）")
    ap.add_argument("--drag", default="page", choices=["page", "pyautogui"],
                    help="page=CDP 鼠标（默认，不需窗口焦点） / pyautogui=系统级鼠标")
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    tag = f"white_d{args.difficulty}"
    rep = {"args": vars(args)}
    t0 = time.perf_counter()

    with sync_playwright() as pw:
        if args.cdp:
            browser = pw.chromium.connect_over_cdp(args.cdp)
            ctx = browser.contexts[0] if browser.contexts else browser.new_context()
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            print(f"  已挂到 {args.cdp}  当前页面: {page.url[:60]}")
            net_mode = "cdp"
        else:
            browser, page, net_mode = open_page(pw, args.url, headed=args.headed)
            print(f"  网络方式: {net_mode}")
            page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
            # 等默认 animals 主题渲染完再开题，否则它加载完会把白题覆盖掉
            settle(page)

        # 盾牌：禁止误点工具栏/面板（纯样式，不碰游戏状态）
        page.add_style_tag(content="""
            .app-header, .game-toolbar, .exam-banner, .exam-start,
            .modal-backdrop, .reference-panel, .app-footer { pointer-events: none !important; }
        """)

        if args.no_start:
            # 假定你已经把题开好了：只等画面里出现足够多的白色碎片
            img = None
            for _ in range(300):
                img = grab_canvas(page)
                if img is not None and img.shape[2] == 4 and int((img[:, :, 3] > 0).sum()) > 2000:
                    break
                page.wait_for_timeout(100)
            print("  直接解题模式（不点任何按钮）")
        else:
            img = None
            for attempt in range(1, 4):
                img = start_part2_once(page, args.url, args.difficulty, part=2)
                if img is not None:
                    break
                print(f"  {attempt} 次开题失败，重载重来")
            if img is None:
                print("❌ 开不出稳定的纯白题")
                browser.close()
                return 1
        s = int(round(math.sqrt(args.difficulty)))
        expect = args.difficulty + max(1, round(args.difficulty * 0.2))
        board = detect_board(img)
        pw, ph = board["w"] / s, board["h"] / s
        tab = 0.22 * min(pw, ph)
        geom = canvas_geometry(page)
        # 画布在**屏幕**上的原点（pyautogui 需要绝对屏幕坐标）
        if args.drag == "pyautogui":
            win = page.evaluate("""() => ({sx: window.screenX, sy: window.screenY,
                ow: window.outerWidth, iw: window.innerWidth,
                oh: window.outerHeight, ih: window.innerHeight})""")
            # 窗口边框/标题栏的偏移：用"外框-内框"的差值估
            geom["scr_x"] = win["sx"] + (win["ow"] - win["iw"]) / 2
            geom["scr_y"] = win["sy"] + (win["oh"] - win["ih"])
            print(f"  屏幕坐标原点(估): ({geom['scr_x']:.0f}, {geom['scr_y']:.0f})"
                  f"  画布框: left={geom['left']:.0f} top={geom['top']:.0f}")
        print("  横幅:", page.evaluate("() => (document.getElementById('exam-part-title')||{}).textContent"),
              "|", page.evaluate("() => (document.getElementById('exam-part-sub')||{}).textContent"))

        # ---- 槽位 + 判边 ----
        t = time.perf_counter()
        occ = S.occupied_slots(img, board, pw, ph, expect=expect)
        sides, sigs = [], []
        for o in occ:
            dens = S.edges_at_slot(img, o, pw, ph, tab)
            lab = {k: S.label_from_density(*v) for k, v in dens.items()}
            sides.append(lab)
            sigs.append(tuple(lab[k][0] for k in SIDES))
        det_ms = (time.perf_counter() - t) * 1000
        print(f"  槽位 {len(occ)} 块（预期 {expect}）  碎片 {pw:.1f}×{ph:.1f}  "
              f"吸附 {max(pw * 0.14, 10):.1f}px  检测 {det_ms:.0f}ms")

        # ---- 求解 ----
        t = time.perf_counter()
        grid, viol, info = solve_with_decoys(sides, s, node_cap=60_000, max_layers=3)
        sol_ms = (time.perf_counter() - t) * 1000
        if grid is None:
            print("❌ 求解失败")
            browser.close()
            return 1
        print(f"  求解：违背 {viol}  混淆块 {len(occ) - len({i for row in grid for i in row})} 块"
              f"  {sol_ms:.0f}ms")
        rep.update({"n": len(occ), "viol": viol, "det_ms": round(det_ms, 1),
                    "solve_ms": round(sol_ms, 1)})
        if args.dry_run:
            browser.close()
            return 0

        # ---- 执行：位置池 ----
        slot_center = [(o["x"] + pw / 2, o["y"] + ph / 2) for o in occ]
        pool = [{"pos": slot_center[i], "labels": sides[i], "sig": sigs[i]} for i in range(len(occ))]
        placed = {}
        tries = fails = popped = 0
        order = ring_order(s) if args.order == "ring" else level_order(s)

        def piece_here(x, y):
            im = grab_canvas(page)
            m = S.alpha_mask(im)
            h, w = m.shape[:2]
            xi, yi = int(x), int(y)
            if not (0 <= xi < w and 0 <= yi < h):
                return False
            patch = m[max(0, yi - 8):yi + 8, max(0, xi - 8):xi + 8]
            return patch.size > 0 and float((patch > 0).mean()) > 0.5

        def in_canvas(x, y):
            """这个点是否**真的落在画布元素上**。

            不只看数值范围：还用 elementFromPoint 确认该坐标最上层的元素就是
            画布本身。这样即使布局/滚动出意外，也不可能点到工具栏或面板上。
            """
            px, py = x * geom["sx"], y * geom["sy"]
            if not (0.5 <= px <= geom["cssW"] - 0.5 and 0.5 <= py <= geom["cssH"] - 0.5):
                return False
            return bool(page.evaluate(
                "([x, y]) => { const e = document.elementFromPoint(x, y);"
                " return e && e.id === 'puzzle-canvas'; }",
                [geom["left"] + px, geom["top"] + py]))

        def rank(r, c, k):
            sc = []
            for idx, e in enumerate(pool):
                lab = e["labels"]
                v = 0.0
                if r == 0 and lab["top"][0] not in (0, None):
                    v -= 6
                if r == s - 1 and lab["bottom"][0] not in (0, None):
                    v -= 6
                if c == 0 and lab["left"][0] not in (0, None):
                    v -= 6
                if c == s - 1 and lab["right"][0] not in (0, None):
                    v -= 6
                for (nr, nc, mine, theirs) in ((r, c - 1, "left", "right"),
                                               (r, c + 1, "right", "left"),
                                               (r - 1, c, "top", "bottom"),
                                               (r + 1, c, "bottom", "top")):
                    j = placed.get((nr, nc))
                    if j is None:
                        continue
                    a, b = lab[mine][0], j["labels"][theirs][0]
                    if a is None or b is None:
                        continue
                    v += 2.5 if a == -b else -2.5
                sc.append((v, idx))
            sc.sort(reverse=True)
            o = [pool[i] for _, i in sc]
            return o if k >= 900 else o[:k]

        # pyautogui 是在真实屏幕上动手，必须先让浏览器在前台
        wait_s = args.countdown if args.countdown is not None else (5 if args.drag == "pyautogui" else 0)
        if wait_s > 0:
            print(f"\n▶ {wait_s} 秒后开始拖拽 —— 请现在切到浏览器窗口，别动鼠标")
            for i in range(wait_s, 0, -1):
                print(f"   {i}…", flush=True)
                time.sleep(1)
            print("   开始！")

        last_prog = 0
        done_early = False
        t = time.perf_counter()
        for k_round in (4, 8, 999):
            if done_early:
                break
            for (r, c) in order:
                if done_early or (r, c) in placed or tries > 320:
                    continue
                tcx = board["x"] + (c + 0.5) * pw
                tcy = board["y"] + (r + 0.5) * ph
                for entry in rank(r, c, k_round):
                    if entry not in pool:
                        continue
                    geom = canvas_geometry(page)
                    st = page.evaluate("""() => ({
                        panel: !document.querySelector('#exam-start').hidden,
                        state: (document.getElementById('exam-state')||{}).textContent || '',
                        title: (document.getElementById('exam-part-title')||{}).textContent || '',
                        prog: (document.getElementById('progress-display')||{}).textContent || ''})""")
                    # 状态守卫：页面只有"进行中"时才允许动手；"准备中"说明题目还没放出来
                    if '进行中' not in st["state"]:
                        print(f"  ⏸ 状态={st['state']!r}，等它回到进行中")
                        for _ in range(100):
                            page.wait_for_timeout(100)
                            if '进行中' in (page.evaluate(
                                    "() => (document.getElementById('exam-state')||{}).textContent || ''")):
                                break
                        continue
                    if st["panel"] or PART_TITLE not in st["title"] or f"/ {args.difficulty}" not in st["prog"].replace("  ", " "):
                        print(f"  ⛔ 局面变了 panel={st['panel']} title={st['title']!r} prog={st['prog']!r}")
                        cv2.imwrite(str(LOGS / "abort_frame.png"), grab_canvas(page))
                        done_early = True
                        break
                    gx, gy = entry["pos"]
                    if not (in_canvas(gx, gy) and in_canvas(tcx, tcy)):
                        print(f"  ⛔ 坐标出界 ({gx:.0f},{gy:.0f})->({tcx:.0f},{tcy:.0f})")
                        pool.remove(entry)
                        continue
                    if not piece_here(gx, gy):
                        pool.remove(entry); popped += 1
                        continue
                    before = read_progress(page)
                    if before < last_prog:
                        snap = page.evaluate("""() => ({
                            state: (document.getElementById('exam-state')||{}).textContent || '',
                            timer: (document.getElementById('timer-display')||{}).textContent || '',
                            title: (document.getElementById('exam-part-title')||{}).textContent || '',
                            sub: (document.getElementById('exam-part-sub')||{}).textContent || '',
                            prev: ((document.getElementById('preview-img')||{}).src || '').length})""")
                        print(f"  ⛔ 进度回退 {last_prog}->{before}  状态={snap['state']!r} "
                              f"计时={snap['timer']} 标题={snap['title']!r} 预览长度={snap['prev']}")
                        done_early = True
                        break
                    drag_with(page, geom, gx, gy, tcx, tcy, args.drag)
                    tries += 1
                    now = read_progress(page)
                    last_prog = max(last_prog, now)
                    if now > before:
                        placed[(r, c)] = entry
                        pool.remove(entry)
                        if page.evaluate("() => !!document.querySelector('#victory-modal.show')"):
                            done_early = True
                        break
                    # 失败：拖回原位 + 双向校验
                    drag_with(page, geom, tcx, tcy, gx, gy, args.drag)
                    if piece_here(gx, gy):
                        entry["pos"] = (gx, gy)
                    elif piece_here(tcx, tcy):
                        entry["pos"] = (tcx, tcy)
                    else:
                        pool.remove(entry); popped += 1
                        break
                    fails += 1
            print(f"  第{k_round if k_round < 900 else 'ALL'}轮: 已放 {len(placed)}/{s*s}, "
                  f"池 {len(pool)}, 尝试 {tries}, 失败 {fails}, 判空 {popped}")
            if len(placed) == s * s:
                break
        drag_ms = (time.perf_counter() - t) * 1000

        try:
            page.wait_for_selector("#victory-modal.show", timeout=3000)
            done = True
        except Exception:
            done = False
        page.screenshot(path=str(LOGS / f"{tag}_final.png"))
        rep.update({"placed": len(placed), "tries": tries, "fails": fails,
                    "drag_ms": round(drag_ms, 1), "done": done,
                    "total_ms": round((time.perf_counter() - t0) * 1000, 1)})
        browser.close()

    print(f"\n结果：格子正确 {len(placed)}/{args.difficulty}  拖拽 {tries} 次（失败 {fails}）"
          f"  完成={'有' if done else '无'}")
    print(f"耗时：检测 {det_ms:.0f} + 求解 {sol_ms:.0f} + 拖拽 {drag_ms:.0f} ms")
    (LOGS / f"{tag}_run.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if done else 2


if __name__ == "__main__":
    sys.exit(main())
