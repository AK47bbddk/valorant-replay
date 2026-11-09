"""終わったとき・失敗したときにスマホへ通知する（ntfy / Discord Webhook）。

送り先は config の `notify.url`、無ければ環境変数 `VREPLAY_NOTIFY_URL`。
どちらも無ければ何もしない。通信の失敗で本体を止めない。
"""
from __future__ import annotations

import json
import os
import urllib.request
from email.header import Header


def _url(cfg: dict) -> str:
    return str((((cfg or {}).get("notify") or {}).get("url"))
               or os.environ.get("VREPLAY_NOTIFY_URL") or "").strip()


def _title_header(title: str) -> str:
    try:
        title.encode("ascii")
        return title
    except UnicodeEncodeError:
        return Header(title, "utf-8", maxlinelen=10000).encode()


# Python 既定の User-Agent（Python-urllib）は Discord に 403 で弾かれる
USER_AGENT = "DiscordBot (https://github.com/AK47bbddk/valorant-replay, 1.0)"


def _post(url: str, data: bytes, headers: dict[str, str]) -> bool:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return True


def send(cfg: dict, title: str, body: str, level: str = "info") -> bool:
    url = _url(cfg)
    if not url:
        return False
    try:
        if "discord.com/api/webhooks/" in url:
            content = f"**{title}**\n{body}"[:2000]
            data = json.dumps({"content": content}, ensure_ascii=False).encode("utf-8")
            return _post(url, data, {"Content-Type": "application/json"})
        return _post(url, str(body).encode("utf-8"),
                     {"Title": _title_header(title), "Priority": "4" if level == "error" else "3"})
    except Exception as e:
        print(f"通知できませんでした（{type(e).__name__}: {e}）")
        return False
