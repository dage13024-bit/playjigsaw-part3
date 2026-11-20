#!/usr/bin/env python3
"""
第 5 课：开环 vs 闭环——为什么"识别 95%"还不够，而"闭环"能救回来

假设有 25 个格子，每次"第一候选就选对"的概率是 p。

开环（选一次就放，不看结果）：
    整局全对 = p^25
    p=0.95 → 0.95^25 = 0.277      （只有 28% 的局能全对）
    p=0.90 → 0.072                （7%）

闭环（放错了能发现，再试下一块）：
    每格最多试 K 块，只要正确那块在候选里，就一定能试到：
    单格成功率 = 1 - (1-p)^K
    整局全对   = [1 - (1-p)^K]^25
    p=0.90, K=3 → 0.999^25 = 0.975  （97.5%）

这就是这个项目真正的护城河：
    不是"识别得更准"，而是"错了能被发现，并且能改"。

运行：
    .venv/bin/python teaching/l5_closed_loop.py
"""
from __future__ import annotations

import numpy as np


def open_loop(p: float, n: int = 25) -> float:
    return p ** n


def closed_loop(p: float, k: int, n: int = 25) -> float:
    return (1.0 - (1.0 - p) ** k) ** n


def expected_moves(p: float, k: int, n: int = 25) -> float:
    """闭环下平均要拖几次：每格第 1 次就中的概率 p，第 2 次 (1-p)p ..."""
    total = 0.0
    for i in range(k):
        # 第 i+1 次才中的概率（前 i 次都错，这次对），最后仍失败则算 k 次
        total += (1 - p) ** i * (p if i < k - 1 else 1.0) * (i + 1)
    return total * n


def main() -> None:
    print("=" * 66)
    print("开环：选一次就放（p^25）")
    print("闭环：每格最多试 K 块（[1-(1-p)^K]^25）")
    print()
    header = f"{'p':>6} | {'开环':>8} | " + " | ".join(f"闭环K={k:<2}" for k in (2, 3, 4))
    print(header)
    print("-" * len(header))
    for p in (0.70, 0.80, 0.90, 0.95, 0.99):
        row = f"{p:>6.2f} | {open_loop(p):>7.1%} | "
        row += " | ".join(f"{closed_loop(p, k):>8.1%}" for k in (2, 3, 4))
        print(row)

    print("\n代价：闭环要多拖几次（含失败回拖）")
    print(f"{'p':>6} | " + " | ".join(f"K={k} 平均拖拽" for k in (2, 3, 4)))
    for p in (0.70, 0.90, 0.95):
        print(f"{p:>6.2f} | " + " | ".join(f"{expected_moves(p, k):>10.1f}" for k in (2, 3, 4)))

    # ---- 蒙特卡洛验证公式 ----
    rng = np.random.default_rng(0)
    trials, n = 4000, 25
    print("\n蒙特卡洛（4000 局，验证公式没写错）：")
    for p, k in ((0.90, 1), (0.90, 3), (0.95, 1), (0.95, 3)):
        ok = 0
        for _ in range(trials):
            good = True
            for _cell in range(n):
                hit = False
                for _try in range(k):
                    if rng.random() < p:      # 这一块猜对了
                        hit = True
                        break
                if not hit:
                    good = False
                    break
            ok += good
        print(f"  p={p:.2f} K={k}: {ok / trials:.1%}")

    print("\n注意：这个模型假设'错了能安全回退、且正确那块一定在候选里'。")
    print("真实项目里这两条靠的就是：回拖 + 画面校验 + 候选池排序。")


if __name__ == "__main__":
    main()
