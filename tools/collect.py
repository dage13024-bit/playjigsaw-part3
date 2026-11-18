#!/usr/bin/env python3
"""
采样器：连续开 N 局，把 canvas 原图 + 真值存下来，供离线评测。

  .venv/bin/python tools/collect.py --part 2 --difficulty 25 --trials 4

产物：
  logs/ds/p{d}_{i}_canvas.png   纯白板画面（RGBA，含 alpha）
  logs/ds/p{d}_{i}_truth.json   棋盘几何 + 每片当前位置/正确位置/边矩阵
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

URL = "https://f27e491f.cv-animal-puzzle.pages.dev/?mode=exam"
ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "logs" / "ds"

STATE_JS = """() => {
  const b = window.app.board, c = b.canvas, r = c.getBoundingClientRect();
  return {
    geom: { bitW: c.width, bitH: c.height, cssW: r.width, cssH: r.height,
            rows: b.rows, cols: b.cols, boardX: b.boardX, boardY: b.boardY,
            boardW: b.boardWidth, boardH: b.boardHeight,
            pw: b.pieceWidth, ph: b.pieceHeight },
    pieces: b.pieces.map(p => ({ id: p.id, row: p.row, col: p.col,
      x: p.x, y: p.y, cx: p.correctX, cy: p.correctY,
      isPlaced: p.isPlaced, edges: p.edges, margin: p.canvasMargin }))
  };
}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=2)
    ap.add_argument("--difficulty", type=int, default=25)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    DS.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="chrome", headless=not a.headed)
        pg = br.new_context(viewport={"width": 1440, "height": 900},
                            device_scale_factor=1).new_page()
        for i in range(a.trials):
            pg.goto(URL, wait_until="load", timeout=60_000)
            pg.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
            pg.check(f'input[name="exam-part"][value="{a.part}"]')
            pg.check(f'input[name="exam-difficulty"][value="{a.difficulty}"]')
            pg.click("#btn-exam-start")
            pg.wait_for_function(
                "n => window.exam && window.exam.running && window.app.board.pieces.length===n",
                arg=a.difficulty, timeout=30_000)
            pg.wait_for_timeout(400)
            raw = pg.evaluate("() => window.app.board.canvas.toDataURL('image/png')")
            img = cv2.imdecode(
                __import__("numpy").frombuffer(base64.b64decode(raw.split(",", 1)[1]),
                                               __import__("numpy").uint8),
                cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(DS / f"p{a.difficulty}_{i}_canvas.png"), img)
            (DS / f"p{a.difficulty}_{i}_truth.json").write_text(
                json.dumps(pg.evaluate(STATE_JS)))
            print(f"  saved p{a.difficulty}_{i}  canvas={img.shape}")
        br.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
