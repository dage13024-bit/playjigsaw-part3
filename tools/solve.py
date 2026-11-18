#!/usr/bin/env python3
"""
端到端解题入口（当前支持 Part 2 · 纯白）

  .venv/bin/python tools/solve.py --part 2 --difficulty 4 --headed

流程：开考 -> 截图 -> 检测(分区密度) -> 求解(生长+修复) -> 逐块拖拽(带吸附校验)
      -> 未完成则重检测重解 -> 落日志

坐标系：检测全在 canvas 位图坐标里做；拖拽时再乘 bitmap/css 换算成页面坐标。
不使用任何网页内部状态（不读 window.app / 不读正确坐标）。
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import analyze, draw_overlay  # noqa: E402
from solver import solve_best, verify           # noqa: E402

URL = "https://f27e491f.cv-animal-puzzle.pages.dev/?mode=exam"
LOGS = ROOT / "logs"

PROGRESS_JS = "() => (document.getElementById('progress-display')||{}).textContent || ''"
DONE_JS = "() => !!document.querySelector('#victory-modal.show')"


class Session:
    def __init__(self, page):
        self.page = page
        self.t = {}

    # ---------------------------------------------------------------- 感知
    def capture(self, mode: str = "canvas"):
        t0 = time.perf_counter()
        if mode == "canvas":
            raw = self.page.evaluate(
                "() => window.app.board.canvas.toDataURL('image/png')")
            img = cv2.imdecode(np.frombuffer(base64.b64decode(raw.split(",", 1)[1]),
                                             np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            rect = self.page.evaluate(
                "() => { const r = document.getElementById('puzzle-canvas')"
                ".getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
            shot = self.page.locator("#puzzle-canvas").screenshot()
            img = cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR)
        self.t["capture_ms"] = (time.perf_counter() - t0) * 1000
        return img

    def geom(self):
        return self.page.evaluate("""() => {
            const c = document.getElementById('puzzle-canvas');
            const r = c.getBoundingClientRect();
            return {left:r.left, top:r.top, cssW:r.width, cssH:r.height,
                    bitW:c.width, bitH:c.height,
                    sx:r.width/c.width, sy:r.height/c.height};
        }""")

    # ---------------------------------------------------------------- 动作
    def drag(self, g, fx, fy, tx, ty):
        """canvas 位图坐标 -> 真鼠标拖拽"""
        ax, ay = g["left"] + fx * g["sx"], g["top"] + fy * g["sy"]
        bx, by = g["left"] + tx * g["sx"], g["top"] + ty * g["sy"]
        t0 = time.perf_counter()
        self.page.mouse.move(ax, ay)
        self.page.mouse.down()
        self.page.mouse.move(bx, by)
        self.page.mouse.up()
        return (time.perf_counter() - t0) * 1000

    def progress(self):
        txt = self.page.evaluate(PROGRESS_JS)
        try:
            return int(txt.split("/")[0].strip())
        except Exception:
            return -1


def start_part(page, part: int, difficulty: int):
    page.goto(URL, wait_until="load", timeout=60_000)
    page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
    page.check(f'input[name="exam-part"][value="{part}"]')
    page.check(f'input[name="exam-difficulty"][value="{difficulty}"]')
    page.click("#btn-exam-start")
    page.wait_for_function(
        "n => window.exam && window.exam.running && document.querySelectorAll('#exam-start:not([hidden])').length===0",
        arg=difficulty, timeout=30_000)
    page.wait_for_timeout(350)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--difficulty", type=int, default=4, choices=[4, 9, 16, 25])
    ap.add_argument("--capture", default="canvas", choices=["canvas", "cdp"])
    ap.add_argument("--rounds", type=int, default=1, help="重检测重解的最大轮数")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只检测+求解，不拖拽")
    args = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    tag = f"p{args.part}_d{args.difficulty}"
    report: dict = {"args": vars(args), "rounds": []}

    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="chrome", headless=not args.headed)
        pg = br.new_context(viewport={"width": 1440, "height": 900},
                            device_scale_factor=1).new_page()
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        start_part(pg, args.part, args.difficulty)
        S = Session(pg)

        cached_geom = None
        for rnd in range(args.rounds):
            t_round = time.perf_counter()
            img = S.capture(args.capture)
            cv2.imwrite(str(LOGS / f"{tag}_r{rnd}_raw.png"), img)
            g = S.geom()
            t_det = time.perf_counter()
            an = analyze(img, args.difficulty)
            if an["geom"] is None and cached_geom is not None:
                an["geom"] = cached_geom          # 棋盘被碎片盖住时复用上一轮几何
            elif an["geom"] is not None:
                cached_geom = an["geom"]
            det_ms = (time.perf_counter() - t_det) * 1000
            cv2.imwrite(str(LOGS / f"{tag}_r{rnd}_detect.png"), draw_overlay(img, an))

            n_det = len(an["pieces"])
            if n_det != args.difficulty or not an["geom"]:
                report["rounds"].append({"round": rnd, "error": f"检出 {n_det} 片 / 棋盘 {bool(an['geom'])}"})
                break

            t_solv = time.perf_counter()
            grid, viol, nodes, info = solve_best([p["sides"] for p in an["pieces"]], an["s"])
            solv_ms = (time.perf_counter() - t_solv) * 1000
            if grid is None:
                report["rounds"].append({"round": rnd, "error": "求解失败"})
                break

            b, pw, ph = an["geom"]["board"], an["geom"]["pw"], an["geom"]["ph"]
            plan = []
            for r in range(an["s"]):
                for c in range(an["s"]):
                    i = grid[r][c]
                    p = an["pieces"][i]
                    tgt = (b["x"] + (c + 0.5) * pw, b["y"] + (r + 0.5) * ph)
                    gx, gy = p["grab"]
                    plan.append({"cell": (r, c), "piece": i,
                                 "grab": (gx, gy),
                                 # 位移 = 目标中心 - 核心中心；核心估偏也能被吸附半径吸收
                                 "to": (gx + tgt[0] - p["cx"], gy + tgt[1] - p["cy"]),
                                 "sig": p["sig"]})
            cv2.imwrite(str(LOGS / f"{tag}_r{rnd}_plan.png"), draw_overlay(img, an, grid))

            if args.dry_run:
                report["rounds"].append({"round": rnd, "viol": viol, "info": info,
                                         "det_ms": round(det_ms, 1), "solve_ms": round(solv_ms, 1),
                                         "plan": [(c["cell"], c["piece"]) for c in plan]})
                break

            # 逐块拖拽 + 吸附校验
            prog0 = S.progress()
            drags = []
            for c in plan:
                before = S.progress()
                ms = S.drag(g, c["grab"][0], c["grab"][1], c["to"][0], c["to"][1])
                after = S.progress()
                drags.append({**{k: c[k] for k in ("cell", "piece")},
                              "ms": round(ms, 1), "ok": after > before,
                              "progress": [before, after]})
                if pg.evaluate(DONE_JS):
                    break
            done = pg.evaluate(DONE_JS)
            report["rounds"].append({
                "round": rnd, "det_ms": round(det_ms, 1), "solve_ms": round(solv_ms, 1),
                "capture_ms": round(S.t.get("capture_ms", 0), 1),
                "viol": viol, "info": info,
                "placed_before": prog0, "placed_after": S.progress(),
                "n_drags": len(drags), "snap_ok": sum(1 for d in drags if d["ok"]),
                "drag_ms_median": round(statistics.median([d["ms"] for d in drags]), 1) if drags else None,
                "done": done, "rnd_wall_ms": round((time.perf_counter() - t_round) * 1000, 1),
                "drags": drags,
            })
            if done:
                break

        report["page_errors"] = errs
        report["final"] = {
            "progress": S.progress(),
            "done": pg.evaluate(DONE_JS),
            "win_time": pg.evaluate("() => (document.getElementById('win-time')||{}).textContent"),
            "win_moves": pg.evaluate("() => (document.getElementById('win-moves')||{}).textContent"),
            "exam_state": pg.evaluate("() => (document.getElementById('exam-state')||{}).textContent"),
        }
        pg.screenshot(path=str(LOGS / f"{tag}_final.png"))
        br.close()

    (LOGS / f"{tag}_run.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rounds"}, indent=1, ensure_ascii=False))
    for r in report["rounds"]:
        print(json.dumps({k: v for k, v in r.items() if k != "drags"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
