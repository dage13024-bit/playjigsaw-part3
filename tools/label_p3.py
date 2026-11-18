#!/usr/bin/env python3
"""
给采集到的 Part 3 画面自动打类别标签（开发期工具）

原理：官方题库的 388 张原图就是拼图源图的来源。把"碎片/格子"和题库逐张比，
取相似度最高的一张，它的类别就是标签。实测：
  * 参考格 top1 = 20/20
  * 碎片   top1 = 20/20（同视野中心区，Lab-NCC）
所以这批标签可以直接当训练真值用。

输出：datasets/p3_labels.csv
  game,kind,file,key,path,score
  kind ∈ {cell, piece}

用法：
  .venv-yolo/bin/python tools/label_p3.py --raw datasets/p3_raw
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import play_p3 as P                       # noqa: E402

SIZE = (96, 72)
FOV = 0.62


def prep(img: np.ndarray) -> np.ndarray:
    """缩放 → 模糊 → Lab → 展平 → 每通道零均值单位方差。"""
    g = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    lab = cv2.cvtColor(g, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    v = lab.reshape(-1)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * (1 - FOV) / 2), int(w * (1 - FOV) / 2)
    return img[dy:h - dy, dx:w - dx]


def build_catalog(cat_dir: Path):
    rows = list(csv.DictReader(open(cat_dir / "labels.csv")))
    keys, labels, paths, vecs = [], [], [], []
    for row in rows:
        img = cv2.imread(str(cat_dir / row["path"]))
        if img is None:
            continue
        # 源图被 drawImage 拉伸到格子尺寸，这里复现同样的几何
        img = cv2.resize(img, (320, 240))
        keys.append(row["key"])
        labels.append(row["label"])
        paths.append(row["path"])
        vecs.append(prep(center_square(img)))
    return keys, labels, paths, np.stack(vecs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "datasets" / "p3_raw"))
    ap.add_argument("--catalog", default=str(ROOT / "datasets" / "catalog"))
    ap.add_argument("--out", default=str(ROOT / "datasets" / "p3_labels.csv"))
    ap.add_argument("--min-score", type=float, default=0.55)
    args = ap.parse_args()

    keys, labels, paths, V = build_catalog(Path(args.catalog))
    print(f"题库 {len(keys)} 张，{len(set(keys))} 类")

    out_rows = []
    for gdir in sorted(Path(args.raw).glob("g*")):
        rep_path = gdir / "g_report.json"
        if not rep_path.exists():
            continue
        rep = json.loads(rep_path.read_text())
        white_pieces = set(rep.get("white", {}))
        n_ok = 0

        for p in sorted(gdir.glob("g_cell_*.png")):
            parts = p.stem.split("_")
            r, c = parts[-2], parts[-1]
            img = cv2.imread(str(p))
            if img is None:
                continue
            if P.is_white_cell(img):
                continue
            q = prep(center_square(img))
            sims = (V * q).sum(axis=1)
            best = int(np.argmax(sims))
            out_rows.append([gdir.name, "cell", p.name, keys[best], paths[best],
                             f"{sims[best]:.4f}"])
            n_ok += 1

        for p in sorted(gdir.glob("g_piece*.png")):
            if "_core" in p.name:
                continue
            k = p.stem.split("piece")[-1]
            if f"p{k}" in white_pieces:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            q = prep(img)
            sims = (V * q).sum(axis=1)
            best = int(np.argmax(sims))
            out_rows.append([gdir.name, "piece", p.name, keys[best], paths[best],
                             f"{sims[best]:.4f}"])
            n_ok += 1

        print(f"  {gdir.name}: 打标 {n_ok} 张")

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["game", "kind", "file", "key", "path", "score"])
        wr.writerows(out_rows)
    print(f"完成：{len(out_rows)} 条 → {args.out}")

    low = [r for r in out_rows if float(r[5]) < args.min_score]
    print(f"低分（<{args.min_score}）{len(low)} 条，建议剔除")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
