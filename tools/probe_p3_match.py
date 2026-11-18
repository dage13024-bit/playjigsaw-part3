#!/usr/bin/env python3
"""
探针：验证 Part 3 的参考图到底能不能做"逐格像素比对"。

背景（读官方站源码 build 20261115r 得到）：
  * Part 1（animals）的参考格故意换成**同类别的另一张照片**（alternatePhoto），
    所以只能按类别匹配 —— 这是 STATUS.md 里"参考图不能像素比对"的来源；
  * Part 2（white）没有照片；
  * Part 3（mixed）的 `renderMixedPuzzle()` 里，
        this.controls.updatePreviewImage(canvas.toDataURL('image/png'), ...)
    预览用的就是**拼图源画布本身**（照片格 + 纯白格 + 3px 描边）。
    也就是说 Part 3 的参考图和碎片是同像素来源，**逐格比对是可行的**。

这个探针只做一件事：开一局 Part 3，逐块截碎片、逐格截参考图，
算一个 30×25 的相似度矩阵，看
  1) 每块照片碎片是不是都有一个明显更高的格子；
  2) 每格是不是恰好被一块碎片认领（匈牙利匹配一一对应）；
  3) 混淆块是不是整体分数偏低。

用法：
  .venv/bin/python tools/probe_p3_match.py --difficulty 25
产物：logs/probe3/*.png + match_matrix.npz
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import slots as S                                       # noqa: E402
from net import open_page                                # noqa: E402
from pipeline import detect_board                        # noqa: E402
from play_part2 import canvas_geometry, grab_canvas      # noqa: E402

OFFICIAL = "https://puzzle.moyunliuyin.me/?mode=exam"


def ref_image(page, out_dir: Path):
    """截参考图 `<img>`，并按 object-fit: contain 扣掉黑边。"""
    info = page.evaluate("""() => {
        const i = document.getElementById('preview-img');
        const r = i.getBoundingClientRect();
        const s = Math.min(r.width / i.naturalWidth, r.height / i.naturalHeight);
        return {nw: i.naturalWidth, nh: i.naturalHeight,
                cw: r.width, ch: r.height,
                w: i.naturalWidth * s, h: i.naturalHeight * s,
                ox: (r.width - i.naturalWidth * s) / 2,
                oy: (r.height - i.naturalHeight * s) / 2};
    }""")
    box = page.locator("#preview-img").bounding_box()
    shot = page.screenshot(clip={"x": box["x"], "y": box["y"],
                                 "width": box["width"], "height": box["height"]})
    raw = cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR)
    h, w = raw.shape[:2]
    sx, sy = w / info["cw"], h / info["ch"]
    x0 = int(round(info["ox"] * sx)); y0 = int(round(info["oy"] * sy))
    x1 = int(round((info["ox"] + info["w"]) * sx))
    y1 = int(round((info["oy"] + info["h"]) * sy))
    return raw[y0:y1, x0:x1].copy(), info


def split_cells(ref: np.ndarray, s: int):
    h, w = ref.shape[:2]
    cells = []
    for r in range(s):
        for c in range(s):
            x0, x1 = int(round(c * w / s)), int(round((c + 1) * w / s))
            y0, y1 = int(round(r * h / s)), int(round((r + 1) * h / s))
            cells.append({"r": r, "c": c, "img": ref[y0:y1, x0:x1]})
    return cells


def is_white(img: np.ndarray, margin: float = 0.18) -> bool:
    h, w = img.shape[:2]
    core = img[int(h * margin):int(h * (1 - margin)), int(w * margin):int(w * (1 - margin))]
    return bool(core.size and core.mean() > 250 and core.std() < 3)


def prep(img: np.ndarray, size=(96, 72)) -> np.ndarray:
    """统一到固定尺寸，转 Lab，轻度模糊压掉重采样的锯齿。"""
    g = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1); b = b.reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / d) if d > 1e-6 else 0.0


def score(piece: np.ndarray, cell: np.ndarray) -> float:
    """三通道相关系数取平均。"""
    pa, ca = prep(piece), prep(cell)
    return float(np.mean([ncc(pa[..., i], ca[..., i]) for i in range(3)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=int, default=25, choices=[9, 16, 25])
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--url", default=OFFICIAL)
    ap.add_argument("--out", default=str(ROOT / "logs" / "probe3"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    s = int(round(math.sqrt(args.difficulty)))
    expect = args.difficulty + max(1, round(args.difficulty * 0.2))

    with sync_playwright() as pw:
        browser, page, mode = open_page(pw, args.url, headed=args.headed)
        print(f"网络: {mode}")
        page.wait_for_selector("#exam-start:not([hidden])", timeout=30_000)

        # 关键：页面构造时会先 loadTheme('animals')，加载完会覆盖 sourceCanvas
        # 并 startNewGame。必须等它彻底跑完，否则开出来的题会被换掉。
        try:
            page.wait_for_load_state("networkidle", timeout=30_000)
        except Exception:                                  # noqa: BLE001
            pass
        last, stable = None, 0
        for _ in range(60):
            cur = page.evaluate("() => {const i=document.getElementById('preview-img');"
                                "return i && i.naturalWidth ? i.src.length + ':' + i.naturalWidth : '';}")
            stable = stable + 1 if cur == last else 0
            last = cur
            if stable >= 4:
                break
            page.wait_for_timeout(500)
        print(f"初始渲染已安静（preview {last}）")

        page.check('input[name="exam-part"][value="3"]')
        page.check(f'input[name="exam-difficulty"][value="{args.difficulty}"]')
        page.click("#btn-exam-start")
        page.wait_for_function(
            "() => document.querySelector('#exam-start').hidden"
            " && ((document.getElementById('exam-state')||{}).textContent||'').includes('进行中')",
            timeout=90_000)
        page.wait_for_timeout(1200)

        ref, info = ref_image(page, out)
        cells = split_cells(ref, s)
        whites = [c for c in cells if is_white(c["img"])]
        print(f"预览 natural={info['nw']}x{info['nh']} 截到 {ref.shape[1]}x{ref.shape[0]}  "
              f"纯白格 {len(whites)}（理论 {round(args.difficulty * 0.2)}）")
        if len(whites) == 0:
            print("⚠ 预览里没有纯白格 —— 可能被默认照片主题覆盖了，这局不可用")
        cv2.imwrite(str(out / "ref.png"), ref)

        geom = canvas_geometry(page)
        img = grab_canvas(page)
        board = detect_board(img)
        if board is None:
            print("❌ 没找到棋盘"); return 1
        pwd, phd = board["w"] / s, board["h"] / s
        occ = S.occupied_slots(img, board, pwd, phd, expect=expect)
        print(f"棋盘 {board['w']}x{board['h']} 碎片 {pwd:.1f}x{phd:.1f} 检到槽位 {len(occ)}（预期 {expect}）")

        pieces = []
        t = time.perf_counter()
        for k, slot in enumerate(occ):
            cx, cy = slot["x"] + pwd / 2, slot["y"] + phd / 2
            ax = geom["left"] + cx * geom["sx"]
            ay = geom["top"] + cy * geom["sy"]
            page.mouse.move(ax, ay); page.mouse.down(); page.mouse.up()   # bringToTop
            img2 = grab_canvas(page)
            crop = S.crop_piece(img2, slot, pwd, phd, shrink=0.62)
            if crop is None:
                continue
            pieces.append({"slot": slot, "img": crop,
                           "white": S.is_white_piece(img2, slot, pwd, phd)})
            cv2.imwrite(str(out / f"piece_{k:02d}.png"), crop)
        print(f"取碎片 {len(pieces)} 块，耗时 {(time.perf_counter()-t)*1000:.0f}ms")

        photo_pieces = [p for p in pieces if not p["white"]]
        white_pieces = [p for p in pieces if p["white"]]
        photo_cells = [c for c in cells if not is_white(c["img"])]
        print(f"照片碎片 {len(photo_pieces)} / 纯白碎片 {len(white_pieces)}；"
              f"照片格 {len(photo_cells)} / 白格 {len(whites)}")

        M = np.zeros((len(photo_pieces), len(photo_cells)), np.float32)
        for i, p in enumerate(photo_pieces):
            for j, c in enumerate(photo_cells):
                M[i, j] = score(p["img"], c["img"])

        np.savez_compressed(out / "match_matrix.npz", M=M,
                            cells=np.array([[c["r"], c["c"]] for c in photo_cells]))

        rows = []
        for i, p in enumerate(photo_pieces):
            order = np.argsort(-M[i])
            rows.append({"i": i, "white": False,
                         "top": [{"cell": [int(photo_cells[j]["r"]), int(photo_cells[j]["c"])],
                                  "s": round(float(M[i, j]), 3)} for j in order[:3]],
                         "best": round(float(M[i, order[0]]), 3),
                         "margin": round(float(M[i, order[0]] - M[i, order[1]]), 3)})
        for p in rows:
            print(f"  片{p['i']:2d} best={p['best']:.3f} margin={p['margin']:.3f} "
                  f"→ {p['top'][0]['cell']}  {p['top'][1]['cell']}({p['top'][1]['s']})")

        try:
            from scipy.optimize import linear_sum_assignment
            big = np.full((len(photo_pieces), len(photo_pieces)), -2.0, np.float32)
            big[:, :len(photo_cells)] = M
            ri, ci = linear_sum_assignment(-big)
            assign = {int(i): int(j) for i, j in zip(ri, ci) if j < len(photo_cells)}
            used = set(assign.values())
            print(f"匈牙利：{len(assign)} 对，覆盖格子 {len(used)}/{len(photo_cells)}")
            for i, j in sorted(assign.items()):
                print(f"    片{i:2d} → 格({photo_cells[j]['r']},{photo_cells[j]['c']}) "
                      f"score={M[i, j]:.3f}")
            vals = [M[i, j] for i, j in assign.items()]
            print(f"  分配分数 min/median = {min(vals):.3f}/{np.median(vals):.3f}")
        except ImportError:
            print("（没装 scipy，跳过匈牙利）")

        (out / "probe.json").write_text(json.dumps(
            {"info": info, "rows": rows, "n_photo_pieces": len(photo_pieces),
             "n_white_pieces": len(white_pieces), "n_white_cells": len(whites)},
            ensure_ascii=False, indent=2))
        print(f"产物：{out}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
