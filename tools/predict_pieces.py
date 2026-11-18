#!/usr/bin/env python3
"""
用训练好的分类器给"碎片图"和"参考格图"打标签，并检查两类是否对得上。

  .venv-yolo/bin/python tools/predict_pieces.py --dir logs/part3/p3_d25_xxxx

输出：每块碎片的预测类别 + 每个参考格的预测类别 + 每格可选的碎片候选。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def predict(model, path: str):
    r = model.predict(path, verbose=False)[0]
    names = r.names
    probs = r.probs
    if probs is None:
        return None, 0.0
    top = int(probs.top1)
    return names[top], float(probs.top1conf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "cls_v1" / "weights" / "best.pt"))
    args = ap.parse_args()

    d = Path(args.dir)
    model = YOLO(args.weights)

    pieces = sorted(glob.glob(str(d / "pieces" / "*.png")))
    refs = sorted(glob.glob(str(d / "ref_*.png")))
    print(f"碎片 {len(pieces)}  参考格 {len(refs)}")

    pcls = []
    for f in pieces:
        c, conf = predict(model, f)
        pcls.append((Path(f).name, c, round(conf, 3)))
    print("\n碎片预测:")
    for n, c, cf in pcls:
        print(f"  {n:16s} {c:10s} {cf:.2f}")

    rcls = []
    for f in refs:
        c, conf = predict(model, f)
        rcls.append((Path(f).name, c, round(conf, 3)))
    print("\n参考格预测:")
    for n, c, cf in rcls:
        print(f"  {n:16s} {c:10s} {cf:.2f}")

    out = {"pieces": pcls, "refs": rcls}
    (d / "predictions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> {d/'predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
