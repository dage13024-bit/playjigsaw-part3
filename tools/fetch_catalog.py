#!/usr/bin/env python3
"""
下载考核站公开的题库图片，构建带标签的分类数据集。

来源：https://puzzle.moyunliuyin.me/src/data/catalog.js
      assets/catalog/<key>/<key>-<n>.jpg
15 类共 388 张。这些图就是拼图碎片用的同一批图，所以拿来做训练集最合适。

用法：.venv/bin/python tools/fetch_catalog.py
产物：datasets/catalog/<key>/<key>-<n>.jpg  +  datasets/catalog/labels.csv
"""
from __future__ import annotations

import csv
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://puzzle.moyunliuyin.me/"
OUT = Path(__file__).resolve().parent.parent / "datasets" / "catalog"
HDR = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str, tries: int = 3) -> bytes:
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:                       # noqa: BLE001
            last = e
    raise RuntimeError(f"{url}: {last}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = fetch(BASE + "src/data/catalog.js").decode("utf-8", "replace")
    rows = re.findall(r"category\('([a-z]+)',\s*'([^']+)',\s*(\d+),\s*(true|false)\)", cat)
    if not rows:
        print("解析 catalog.js 失败")
        return 1
    labels = []
    total = 0
    for key, label, count, pretrained in rows:
        d = OUT / key
        d.mkdir(exist_ok=True)
        ok = 0
        for i in range(1, int(count) + 1):
            url = f"{BASE}assets/catalog/{key}/{key}-{i}.jpg"
            dst = d / f"{key}-{i}.jpg"
            if dst.exists() and dst.stat().st_size > 1000:
                ok += 1
                labels.append((key, label, str(dst.relative_to(OUT)), pretrained))
                continue
            try:
                data = fetch(url)
                if len(data) < 1000:
                    continue
                dst.write_bytes(data)
                ok += 1
                labels.append((key, label, str(dst.relative_to(OUT)), pretrained))
            except Exception as e:                   # noqa: BLE001
                print(f"  ! {key}-{i}: {e}")
        print(f"  {key:10s} {label:16s} {ok}/{count}")
        total += ok

    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "path", "pretrained"])
        w.writerows(labels)
    print(f"\n共 {total} 张，标签写进 {OUT/'labels.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
