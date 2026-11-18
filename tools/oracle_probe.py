#!/usr/bin/env python3
"""
M0 - 环境探针 + 真值 oracle（**仅开发期使用，不参与考核提交**）

做四件事：
  1. 打通 Playwright(channel=chrome) -> 打开考核页 -> 开始指定 Part/难度
  2. 导出真值状态（pieces/board 几何/canvas 尺寸），写 logs/state_*.json
  3. 实测三条截图路径与一条拖拽路径的耗时，替换掉拍脑袋的估算
  4. 用 L2（真鼠标）拖 1 块验证坐标换算，再用 L0（改状态）收尾验证胜利链路

用法:
  .venv/bin/python tools/oracle_probe.py --part 1 --difficulty 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://f27e491f.cv-animal-puzzle.pages.dev/?mode=exam"
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def bench(fn, n: int = 5) -> tuple[float, list[float]]:
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs), xs


STATE_JS = """
() => {
  const b = window.app.board;
  const c = b.canvas;
  const r = c.getBoundingClientRect();
  return {
    exam: { part: window.exam && window.exam.part && window.exam.part.id,
            difficulty: window.exam && window.exam.difficulty,
            running: !!(window.exam && window.exam.running) },
    canvas: { bitmapW: c.width, bitmapH: c.height,
              cssW: r.width, cssH: r.height,
              left: r.left, top: r.top,
              dpr: window.devicePixelRatio,
              scale: c.width / r.width },
    board: { rows: b.rows, cols: b.cols, total: b.totalPieces,
             x: b.boardX, y: b.boardY, w: b.boardWidth, h: b.boardHeight,
             pieceW: b.pieceWidth, pieceH: b.pieceHeight,
             snapThreshold: Math.max(26, b.pieceWidth * 0.28),
             placed: b.placedCount, moves: b.moveCount,
             ghost: b.showGhost, ghostLocked: b.ghostLocked },
    pieces: b.pieces.map(p => ({ id: p.id, row: p.row, col: p.col,
      x: p.x, y: p.y, cx: p.correctX, cy: p.correctY,
      isPlaced: p.isPlaced, edges: p.edges,
      margin: p.canvasMargin })),
  };
}
"""

L2_DRAG_JS = """
() => {
  // 页面内合成事件：验证拖拽链路本身（不经过 CDP 往返），返回参考耗时
  const b = window.app.board, c = b.canvas, r = c.getBoundingClientRect();
  const p = b.pieces.find(q => !q.isPlaced);
  if (!p) return null;
  const sx = r.left + p.x + b.pieceWidth / 2;
  const sy = r.top + p.y + b.pieceHeight / 2;
  const tx = r.left + p.correctX + b.pieceWidth / 2;
  const ty = r.top + p.correctY + b.pieceHeight / 2;
  const ev = (t, x, y, node) => node.dispatchEvent(new MouseEvent(t, {
    clientX: x, clientY: y, bubbles: true, cancelable: true }));
  const t0 = performance.now();
  ev('mousedown', sx, sy, c);
  ev('mousemove', tx, ty, window);
  ev('mouseup', tx, ty, window);
  return { pieceId: p.id, ms: performance.now() - t0, placed: p.isPlaced };
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--difficulty", type=int, default=4, choices=[4, 9, 16, 25])
    ap.add_argument("--headed", action="store_true", help="调试时看得到窗口")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    tag = f"part{args.part}_d{args.difficulty}"
    report: dict = {"url": args.url, "part": args.part, "difficulty": args.difficulty}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=not args.headed)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                                  device_scale_factor=1)
        page = ctx.new_page()
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))

        page.goto(args.url, wait_until="load", timeout=60_000)
        page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)

        page.check(f'input[name="exam-part"][value="{args.part}"]')
        page.check(f'input[name="exam-difficulty"][value="{args.difficulty}"]')
        page.click("#btn-exam-start")

        # 碎片构建是同步的，但动物图是异步加载的
        page.wait_for_function(
            "n => window.app && window.app.board.pieces.length === n"
            " && window.exam && window.exam.running === true",
            arg=args.difficulty,
            timeout=30_000,
        )
        page.wait_for_timeout(300)  # 让 render 稳定

        state = page.evaluate(STATE_JS)
        (LOGS / f"state_{tag}.json").write_text(json.dumps(state, indent=1))
        report["state"] = state

        # --- 截图三条路径实测 ---
        page.locator("#puzzle-canvas").scroll_into_view_if_needed()
        m_full, xs_full = bench(lambda: page.screenshot(), 5)
        m_clip, xs_clip = bench(lambda: page.locator("#puzzle-canvas").screenshot(), 5)
        m_dataurl = page.evaluate("""() => {
            const c = window.app.board.canvas;
            const t = [];
            for (let i = 0; i < 5; i++) {
              const t0 = performance.now();
              c.toDataURL('image/png');
              t.push(performance.now() - t0);
            }
            return t.sort((a,b)=>a-b)[2];
        }""")
        m_eval, _ = bench(lambda: page.evaluate("() => 1"), 5)
        report["timing"] = {
            "screenshot_fullpage_ms": round(m_full, 1),
            "screenshot_canvas_clip_ms": round(m_clip, 1),
            "canvas_toDataURL_ms": round(m_dataurl, 1),
            "evaluate_roundtrip_ms": round(m_eval, 2),
        }

        # --- L1：页面内合成事件拖 1 块（验证吸附/链路，同时给下界耗时）---
        report["l1"] = page.evaluate(L2_DRAG_JS)
        page.wait_for_timeout(100)

        # --- L2：CDP 真鼠标拖 1 块（验证坐标换算）---
        state = page.evaluate(STATE_JS)
        target = next((p for p in state["pieces"] if not p["isPlaced"]), None)
        if target:
            cv, bd = state["canvas"], state["board"]
            sx = cv["left"] + target["x"] + bd["pieceW"] / 2
            sy = cv["top"] + target["y"] + bd["pieceH"] / 2
            tx = cv["left"] + target["cx"] + bd["pieceW"] / 2
            ty = cv["top"] + target["cy"] + bd["pieceH"] / 2
            t0 = time.perf_counter()
            page.mouse.move(sx, sy)
            page.mouse.down()
            page.mouse.move(tx, ty)
            page.mouse.up()
            t_l2 = (time.perf_counter() - t0) * 1000
            after = page.evaluate(
                "id => { const p = window.app.board.pieces.find(q => q.id === id);"
                " return { isPlaced: p.isPlaced, x: p.x, y: p.y,"
                " dx: p.x - p.correctX, dy: p.y - p.correctY }; }", target["id"])
            report["l2"] = {"piece": target["id"], "ms": round(t_l2, 1), "after": after}
            page.screenshot(path=str(LOGS / f"l2_check_{tag}.png"))

        # --- L0：改状态收尾（oracle 基线耗时）---
        t0 = time.perf_counter()
        page.evaluate("""() => {
            const b = window.app.board;
            const t0 = performance.now();
            b.pieces.forEach(p => { p.x = p.correctX; p.y = p.correctY; p.isPlaced = true; });
            b.placedCount = b.totalPieces;
            b.moveCount = b.totalPieces;
            b.render();
            window.app.handleWin();
            return performance.now() - t0;
        }""")
        report["l0"] = {"wall_ms": round((time.perf_counter() - t0) * 1000, 1)}

        page.wait_for_selector("#victory-modal.show", timeout=5000)
        done = page.evaluate("""() => ({
            timer: document.getElementById('timer-display').textContent,
            winTime: document.getElementById('win-time').textContent,
            winMoves: document.getElementById('win-moves').textContent,
            progress: document.getElementById('progress-display').textContent,
            examState: document.getElementById('exam-state').textContent,
            pageErrors: null,
        })""")
        page.screenshot(path=str(LOGS / f"solved_{tag}.png"))
        report["win"] = done
        report["page_errors"] = errs
        browser.close()

    st = report["state"]
    print(json.dumps({k: v for k, v in report.items() if k != "state"}, indent=1, ensure_ascii=False))
    print("\ncanvas:", st["canvas"])
    print("board :", st["board"])
    print("piece0:", st["pieces"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
