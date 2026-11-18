#!/usr/bin/env python3
"""
Part 3 数据采集：把一局里的「碎片」和「参考图每格」都切出来存盘，
供 YOLO 标注/训练/评测。

  .venv/bin/python tools/collect_part3.py --difficulty 25 --runs 2

产物（logs/part3/<时间戳>/）：
  pieces/piece_XX.png      每块碎片的"核心矩形"（去掉卡扣，只剩照片内容）
  ref/cell_R_C.png         参考图按格子切开（已扣掉 object-fit 的黑边）
  ref_full.png             参考图整张（调试用）
  canvas.png               棋盘画面（含 alpha）
  meta.json                几何、每块碎片的边标签、哪些是纯白格
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from net import open_page                                 # noqa: E402
from pipeline import analyze, detect_board               # noqa: E402
import slots as S                                       # noqa: E402
from playwright.sync_api import sync_playwright          # noqa: E402
from play_part2 import canvas_geometry, grab_canvas      # noqa: E402

OFFICIAL = "https://puzzle.moyunliuyin.me/?mode=exam"
OUT = ROOT / "logs" / "part3"


def ref_content_rect(page):
    """参考图在 <img> 里是 object-fit: contain，上下/左右会有黑边，要扣掉。"""
    return page.evaluate("""() => {
        const i = document.getElementById('preview-img');
        const r = i.getBoundingClientRect();
        const s = Math.min(r.width / i.naturalWidth, r.height / i.naturalHeight);
        return {w: i.naturalWidth * s, h: i.naturalHeight * s,
                ox: (r.width - i.naturalWidth * s) / 2,
                oy: (r.height - i.naturalHeight * s) / 2,
                natW: i.naturalWidth, natH: i.naturalHeight,
                boxW: r.width, boxH: r.height};
    }""")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=3)
    ap.add_argument("--difficulty", type=int, default=25, choices=[9, 16, 25])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--proxy", default="http://127.0.0.1:1082")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    stamp = time.strftime("%m%d_%H%M%S")
    outdir = OUT / f"p{args.part}_d{args.difficulty}_{stamp}"
    (outdir / "pieces").mkdir(parents=True, exist_ok=True)
    (outdir / "ref").mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser, page, _net = open_page(pw, OFFICIAL, headed=args.headed)
        for run in range(args.runs):
            if run > 0:
                page.goto(OFFICIAL, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)
            page.check(f'input[name="exam-part"][value="{args.part}"]')
            page.check(f'input[name="exam-difficulty"][value="{args.difficulty}"]')
            page.click("#btn-exam-start")
            page.wait_for_function(
                "() => document.querySelector('#exam-start').hidden"
                " && (document.getElementById('exam-state')||{}).textContent === '进行中'",
                timeout=45_000)
            page.wait_for_timeout(800)

            img = grab_canvas(page)
            cv2.imwrite(str(outdir / f"canvas_{run}.png"), img)
            board = detect_board(img)
            s_ = int(round(math.sqrt(args.difficulty)))
            pw_, ph_ = board["w"] / s_, board["h"] / s_
            tab = 0.22 * min(pw_, ph_)
            expect = args.difficulty + max(1, round(args.difficulty * 0.2))
            geom = canvas_geometry(page)

            # 槽位锚定：先找"有碎片"的槽位，再逐个"提层"取干净的图
            occ = S.occupied_slots(img, board, pw_, ph_, expect=expect)
            pieces = []
            for k, slot in enumerate(occ):
                cx = slot["x"] + pw_ / 2
                cy = slot["y"] + ph_ / 2
                ax = geom["left"] + cx * geom["sx"]
                ay = geom["top"] + cy * geom["sy"]
                page.mouse.move(ax, ay)
                page.mouse.down()
                page.mouse.up()                       # 原地点击 = bringToTop
                img2 = grab_canvas(page)
                crop = S.crop_piece(img2, slot, pw_, ph_)
                if crop is not None:
                    cv2.imwrite(str(outdir / f"pieces/run{run}_{k:02d}.png"), crop)
                dens = S.edges_at_slot(img2, slot, pw_, ph_, tab)
                sides = {kk: S.label_from_density(*v)[0] for kk, v in dens.items()}
                white = S.is_white_piece(img2, slot, pw_, ph_)
                pieces.append({
                    "k": k, "slot": [round(slot["x"], 1), round(slot["y"], 1)],
                    "fill": round(slot["fill"], 3),
                    "sides": sides,
                    "likely_white": white,
                    "crop_mean": None if crop is None else round(float(crop.mean()), 1),
                    "crop_std": None if crop is None else round(float(crop.std()), 1),
                })

            # ---- 参考图 ----
            rc = ref_content_rect(page)
            shot = page.locator("#preview-img").screenshot()
            ref = cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(outdir / f"ref_full_{run}.png"), ref)
            x0 = int(round(rc["ox"])); y0 = int(round(rc["oy"]))
            x1 = int(round(rc["ox"] + rc["w"])); y1 = int(round(rc["oy"] + rc["h"]))
            content = ref[y0:y1, x0:x1]
            cells = []
            ch = content.shape[0] / s_
            cw = content.shape[1] / s_
            for r in range(s_):
                for c in range(s_):
                    cell = content[int(r * ch):int((r + 1) * ch), int(c * cw):int((c + 1) * cw)]
                    cv2.imwrite(str(outdir / f"ref_{run}_{r}_{c}.png"), cell)
                    cells.append({"r": r, "c": c,
                                  "mean": float(cell.mean()),
                                  "std": float(cell.std())})

            meta = {"run": run, "part": args.part, "difficulty": args.difficulty, "s": s_,
                    "board": board, "pw": round(pw_, 2), "ph": round(ph_, 2),
                    "n_slots": len(occ), "cells": args.difficulty,
                    "ref_content": rc, "ref_cells": cells, "pieces": pieces}
            (outdir / f"meta_{run}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
            n_white = sum(1 for p in pieces if p["likely_white"])
            print(f"  run{run}: 槽位碎片 {len(pieces)}（疑似纯白 {n_white}）格子 {s_}×{s_} "
                  f"参考图内容 {int(rc['w'])}×{int(rc['h'])}")
        browser.close()

    print(f"\n已保存到 {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
