"""（任意）集計した数字と解説の字幕から、プレイブックの「ATK summary / DEF summary」の文章を Claude に書かせる。

環境変数 ANTHROPIC_API_KEY があるときだけ使う（Colab のシークレットに登録）。無ければ何もしない。
渡すのは集計済みの数字・型の名前・解説の一節だけで、動画や画像は送らない。
数字にない推測はしないよう指示し、「データから言えること」と「推測」を分けて書かせる。
同じ入力の結果は output.dir/_cache/llm にためて、2 回目からは API を呼ばない。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

API = "https://api.anthropic.com/v1/messages"

SYSTEM = (
    "あなたは VALORANT のプロチームのアナリスト兼コーチです。渡されるのは、試合映像を自動解析して集計した"
    "1 チーム・1 マップ・1 構成のデータ（ラウンドの勝率、初動の配置の型、設置サイトと時刻、先制キル、"
    "リテイク・ポストプラントの結果、ULT、解説者の発言から拾った意図のキーワードと一節）です。"
    "これをもとに、チーム内のプレイブックに貼る日本語の要約を書いてください。\n"
    "- 見出しは「ATK summary」「DEF summary」の 2 つ。それぞれ箇条書き 4〜7 行\n"
    "- 1 行目に全体の方針（例：デフォルトで MID を取り、2 巡目のアクションで最終形に持っていく）\n"
    "- 型の名前・回数・勝率など、データにある具体的な根拠を括弧で添える\n"
    "- データから言えないこと（スキルの使い方、コールの中身）は「推測：」と頭につけて最大 2 行まで\n"
    "- 回数が少ないもの（5 回未満）は「参考」と書く\n"
    "- 最後に「対策メモ」として、この相手と戦うときに狙えそうな弱点を 2 行"
)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def summarize(cfg: dict, data: dict, log=print) -> str | None:
    """data（JSON にできる dict）から要約文。キーが無い・失敗したら None。"""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    lcfg = cfg.get("llm") or {}
    model = lcfg.get("model") or "claude-sonnet-5-5"
    body = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha1((model + SYSTEM + body).encode("utf-8")).hexdigest()[:16]
    cache = Path(cfg["output"]["dir"]) / "_cache" / "llm" / f"{h}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        r = requests.post(API, timeout=120, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": int(lcfg.get("max_tokens") or 1500), "system": SYSTEM,
                  "messages": [{"role": "user", "content": "データ（JSON）:\n" + body}]})
        if r.status_code >= 400:
            log(f"  Claude API エラー {r.status_code}: {r.text[:200]}")
            return None
        text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
    except Exception as e:
        log(f"  Claude API に接続できませんでした（{e}）")
        return None
    if text:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return text or None
