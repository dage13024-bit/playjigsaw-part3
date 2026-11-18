#!/usr/bin/env python3
"""
M0 基准 - 用 25 片把"速度预算"量出来（仅开发期）

关注三件事：
  1. 截图到底走哪条路：Playwright screenshot / 原生 CDP captureScreenshot / canvas.toDataURL
  2. 游戏自己的 render() 有多贵（拖拽一次会触发 3 次 render）
  3. L2 真鼠标拖满 25 片需要多久（"视觉完美"时的动作下界）

用法: .venv/bin/python tools/bench_m0.py --part 2 --difficulty 25
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://f27e491f.cv-animal-puzzle.pages.dev/?mode=exam"
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"


def med(fn, n=5):
    xs = []
    for _ in range(n):
        t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(xs), 1), [round(x, 1) for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=2)
    ap.add_argument("--difficulty", type=int, default=25)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    tag = f"p{args.part}_d{args.difficulty}"
    out: dict = {}

    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="chrome", headless=not args.headed)
        ctx = br.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(URL, wait_until="load", timeout=60_000)
        page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
        page.check(f'input[name="exam-part"][value="{args.part}"]')
        page.check(f'input[name="exam-difficulty"][value="{args.difficulty}"]')
        page.click("#btn-exam-start")
        page.wait_for_function(
            "n => window.app && window.app.board.pieces.length === n && window.exam.running",
            arg=args.difficulty, timeout=30_000)
        page.wait_for_timeout(400)

        cdp = ctx.new_cdp_session(page)

        g = page.evaluate("""() => {
          const b = window.app.board, c = b.canvas, r = c.getBoundingClientRect();
          return { bitW: c.width, bitH: c.height, cssW: r.width, cssH: r.height,
                   left: r.left, top: r.top, scrollY: window.scrollY,
                   dpr: devicePixelRatio, rows: b.rows, cols: b.cols,
                   bw: b.boardWidth, bh: b.boardHeight, bx: b.boardX, by: b.boardY,
                   pw: b.pieceWidth, ph: b.pieceHeight,
                   snap: Math.max(26, b.pieceWidth * 0.28),
                   preview: (() => { const i = document.getElementById('preview-img');
                     return { w: i.naturalWidth, h: i.naturalHeight,
                              rect: i.getBoundingClientRect().toJSON() }; })() };
        }""")
        out["geometry"] = g

        # 1) render 成本
        out["render_ms"], _ = med(lambda: page.evaluate(
            "() => { const t = performance.now(); window.app.board.render();"
            " return performance.now() - t; }"), 3)
        # 2) 三条截图路径
        page.locator("#puzzle-canvas").scroll_into_view_if_needed()
        out["pw_screenshot_ms"], _ = med(lambda: page.screenshot(), 3)
        out["pw_locator_screenshot_ms"], _ = med(
            lambda: page.locator("#puzzle-canvas").screenshot(), 3)

        def cdp_shot(clip=None):
            p = {"format": "png", "fromSurface": True, "captureBeyondViewport": False}
            if clip:
                p["clip"] = {**clip, "scale": 1}
            d = cdp.send("Page.captureScreenshot", p)
            return base64.b64decode(d["data"])

        out["cdp_shot_full_ms"], _ = med(lambda: cdp_shot(), 3)
        rect = page.locator("#puzzle-canvas").bounding_box()
        out["cdp_shot_clip_ms"], _ = med(lambda: cdp_shot(
            {"x": rect["x"], "y": rect["y"], "width": rect["width"], "height": rect["height"]}), 3)
        out["toDataURL_ms"], _ = med(lambda: page.evaluate(
            "() => { const t = performance.now(); window.app.board.canvas.toDataURL('image/png');"
            " return performance.now() - t; }"), 3)

        # 落盘：白板原图（给 M1 用）
        raw = page.evaluate("() => window.app.board.canvas.toDataURL('image/png')")
        (LOGS / f"{tag}_canvas.png").write_bytes(base64.b64decode(raw.split(",", 1)[1]))
        st = page.evaluate("() => ({ pieces: window.app.board.pieces.map(p => ({id:p.id,row:p.row,"
                           "col:p.col,x:p.x,y:p.y,cx:p.correctX,cy:p.correctY,"
                           "isPlaced:p.isPlaced,edges:p.edges,margin:p.canvasMargin})) })")
        (LOGS / f"{tag}_truth.json").write_text(json.dumps({"geometry": g, **st}, indent=1))

        # 3) L2 拖满全部碎片：每次抓 z 序最上面的未放置碎片（保证可点中）
        sx = g["cssW"] / g["bitW"]; sy = g["cssH"] / g["bitH"]
        t_all = time.perf_counter()
        per = []
        while True:
            nxt = page.evaluate("""() => {
              const b = window.app.board;
              for (let i = b.pieces.length - 1; i >= 0; i--) {
                const p = b.pieces[i];
                if (!p.isPlaced) return { id: p.id, x: p.x, y: p.y, cx: p.correctX, cy: p.correctY };
              }
              return null; }""")
            if nxt is None:
                break
            t0 = time.perf_counter()
            fx = g["left"] + (nxt["x"] + g["pw"] / 2) * sx
            fy = g["top"] + (nxt["y"] + g["ph"] / 2) * sy
            tx = g["left"] + (nxt["cx"] + g["pw"] / 2) * sx
            ty = g["top"] + (nxt["cy"] + g["ph"] / 2) * sy
            page.mouse.move(fx, fy)
            page.mouse.down()
            page.mouse.move(tx, ty)
            page.mouse.up()
            ok = page.evaluate("id => window.app.board.pieces.find(p => p.id === id).isPlaced", nxt["id"])
            per.append({"id": nxt["id"], "ms": round((time.perf_counter() - t0) * 1000, 1), "ok": ok})
            if not ok:  # 点空了就再抓一次，避免死循环
                page.evaluate("() => { window.app.board.pieces.forEach(p => {"
                              " if (!p.isPlaced) { p.x = p.correctX; p.y = p.correctY;"
                              " p.isPlaced = true; } }); }")
        total = (time.perf_counter() - t_all) * 1000
        out["l2_all"] = {"pieces": len(per),
                         "total_ms": round(total, 1),
                         "per_drag_median_ms": round(statistics.median([p["ms"] for p in per]), 1),
                         "fails": sum(1 for p in per if not p["ok"]),
                         "detail": per}

        won = page.evaluate("""() => ({ progress: document.getElementById('progress-display').textContent,
            placed: window.app.board.placedCount, total: window.app.board.totalPieces,
            won: window.app.isWon, moves: window.app.board.moveCount })""")
        out["after"] = won
        if not won["won"]:
            page.evaluate("() => { const b = window.app.board;"
                          " b.pieces.forEach(p => { p.x = p.correctX; p.y = p.correctY; p.isPlaced = true; });"
                          " b.placedCount = b.totalPieces; b.render(); window.app.handleWin(); }")
            page.wait_for_selector("#victory-modal.show", timeout=5000)
        out["win"] = page.evaluate("""() => ({
            winTime: document.getElementById('win-time').textContent,
            winMoves: document.getElementById('win-moves').textContent,
            examState: document.getElementById('exam-state').textContent })""")
        page.screenshot(path=str(LOGS / f"{tag}_solved.png"))
        br.close()

    print(json.dumps(out, indent=1, ensure_ascii=False))
    (LOGS / f"{tag}_bench.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
