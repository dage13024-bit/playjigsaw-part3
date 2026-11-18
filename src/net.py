#!/usr/bin/env python3
"""
网络辅助：这台机器的 DNS 走 Clash 的 fake-IP（198.18.x.x），Chromium 直连会
ERR_TUNNEL_CONNECTION_FAILED，而且代理时通时不通。所以：
  * 启动时显式指定代理（默认 127.0.0.1:1082）
  * 打开页面带重试与退避
  * 失败时给出明确提示，而不是抛一堆 Playwright 栈
"""
from __future__ import annotations

import time

DEFAULT_PROXY = "http://127.0.0.1:1082"


def launch_with_retry(pw, proxy: str | None = DEFAULT_PROXY, headed: bool = False,
                      tries: int = 3):
    last = None
    for i in range(tries):
        try:
            kwargs = {"channel": "chrome", "headless": not headed}
            if proxy:
                kwargs["proxy"] = {"server": proxy}
            return pw.chromium.launch(**kwargs)
        except Exception as e:                     # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"浏览器启动失败：{last}")


def goto_with_retry(page, url: str, tries: int = 4, timeout: int = 45_000) -> bool:
    for i in range(tries):
        try:
            page.goto(url, wait_until="load", timeout=timeout)
            return True
        except Exception:                          # noqa: BLE001
            time.sleep(2.0 * (i + 1))
    return False


def check_reachable(page, url: str) -> bool:
    """轻量连通性检查，用来看是"网站挂了"还是"代理不通"。"""
    try:
        r = page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        return bool(r and r.ok)
    except Exception:                              # noqa: BLE001
        return False


STRATEGIES = (
    ("direct", {"proxy": None, "args": ["--no-proxy-server"]}),
    ("proxy", {"proxy": {"server": DEFAULT_PROXY}}),
)


def open_page(pw, url: str, headed: bool = False, viewport=(1440, 900),
              tries_per_strategy: int = 2, record_dir: str | None = None):
    """开一个能真正打开目标的浏览器。

    这台机器的网络会变：Clash TUN 直连和显式代理不一定哪个通。
    所以两种启动方式都试一遍，谁先打开页面用谁。
    返回 (browser, page, strategy_name)。
    """
    last = None
    for name, kw in STRATEGIES:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=not headed, **kw)
        except Exception as e:                      # noqa: BLE001
            last = e
            continue
        ctx_kw = {"viewport": {"width": viewport[0], "height": viewport[1]},
                  "device_scale_factor": 1}
        if record_dir:
            ctx_kw["record_video_dir"] = str(record_dir)
            ctx_kw["record_video_size"] = {"width": viewport[0], "height": viewport[1]}
        page = browser.new_context(**ctx_kw).new_page()
        for _ in range(tries_per_strategy):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=25_000)
                return browser, page, name
            except Exception as e:                  # noqa: BLE001
                last = e
                time.sleep(1.0)
        browser.close()
    raise RuntimeError(f"两种网络方式都打不开 {url}：{last}")
