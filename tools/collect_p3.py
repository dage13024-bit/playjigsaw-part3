#!/usr/bin/env python3
"""
Part 3 真实画面数据采集（开发期工具）

每开一局就保存：
  * 参考图原生 25 格（320×240）—— 分类器的"格子域"
  * 30 块碎片的核心区（灰底合成）—— 分类器的"碎片域"
  * 该局的指派 / 白片标记 / 卡扣标签（meta.json）

标签怎么来？`tools/label_p3.py` 用题库逐张比对自动打标（实测 20/20 与人工核对一致）。
采集时**不拖拽**，只提层截图，所以对局不会被弄乱。

用法：
  .venv/bin/python tools/collect_p3.py --games 15 --out datasets/p3_raw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from net import open_page                     # noqa: E402
import play_p3 as P                            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=15)
    ap.add_argument("--start", type=int, default=1, help="起始局号（不覆盖已有 gNNN）")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "p3_raw"))
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--url", default=P.OFFICIAL)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ok = 0

    with sync_playwright() as pw:
        browser, page, net_mode = open_page(pw, args.url, headed=args.headed)
        print(f"网络: {net_mode}")
        for i in range(args.games):
            idx = args.start + i
            if i > 0:
                page.reload()
            P.settle(page)
            P.start_part3(page, 5)
            gdir = out / f"g{idx:03d}"
            try:
                rep = P.solve_once(page, 5, "g", dry=True, save_data=True, out_dir=gdir)
                ok += 1
                print(f"  [{i + 1}/{args.games}] 采集 {gdir.name}："
                      f"白片 {rep['n_white']}，照片格 {len(rep['assign'])}，"
                      f"混淆 {rep['n_decoys']}")
            except Exception as e:                     # noqa: BLE001
                print(f"  [{i + 1}/{args.games}] 失败：{type(e).__name__}: {e}")
        browser.close()

    print(f"\n采集完成 {ok}/{args.games} 局，{time.perf_counter() - t0:.0f}s → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
