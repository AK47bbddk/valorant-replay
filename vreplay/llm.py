"""（任意）集計した数字と解説の字幕から、プレイブックの「ATK summary / DEF summary」の文章を LLM に書かせる。

無料で使える OpenAI 互換の API を、キーがあるものから順に試す（Colab のシークレットに登録）：
  1. OPENROUTER_API_KEY … OpenRouter の DeepSeek V4 Flash 無料版（deepseek/deepseek-v4-flash:free）
  2. GEMINI_API_KEY     … Google AI Studio の Gemini Flash（無料枠）
  3. DEEPSEEK_API_KEY   … DeepSeek 公式（有料・安価）
  Claude（ANTHROPIC_API_KEY）は llm.provider: anthropic と書いたときだけ使う。
どれも無ければ何もしない（要約のスライドが省かれるだけ）。
渡すのは集計済みの数字・型の名前・解説の一節だけで、動画や画像は送らない。
数字にない推測はしないよう指示し、「データから言えること」と「推測」を分けて書かせる。
同じ入力の結果は output.dir/_cache/llm にためて、2 回目からは API を呼ばない。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

# provider → (環境変数, base_url, 既定のモデル)
PROVIDERS = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash:free"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-flash-latest"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-sonnet-5-5"),
}
AUTO_ORDER = ("openrouter", "gemini", "deepseek")

SYSTEM = (
    "あなたは VALORANT のプロチームのアナリスト兼コーチです。渡されるのは、試合映像を自動解析して集計した"
    "1 チーム・1 マップ・1 構成のデータ（ラウンドの勝率、初動の配置の型と各エージェントの立ち位置、設置サイトと時刻、"
    "先制キル、リテイク・ポストプラントの結果、ULT、解説者の発言から拾った意図のキーワードと一節）です。"
    "これをもとに、チーム内のプレイブックに貼る日本語の要約を書いてください。\n"
    "- 見出しは「ATK summary」「DEF summary」の 2 つ。それぞれ箇条書き 4〜7 行\n"
    "- 1 行目に全体の方針（例：デフォルトで MID を取り、2 巡目のアクションで最終形に持っていく）\n"
    "- 型の名前・回数・勝率など、データにある具体的な根拠を括弧で添える\n"
    "- データから言えないこと（スキルの使い方、コールの中身）は「推測：」と頭につけて最大 2 行まで\n"
    "- 回数が少ないもの（5 回未満）は「参考」と書く\n"
    "- 最後に「対策メモ」として、この相手と戦うときに狙えそうな弱点を 2 行\n"
    "- 前置きや考えた過程は書かず、見出しから始める"
)


def _chain(cfg: dict) -> list[tuple[str, str, str, str]]:
    """使える (provider, key, base_url, model) を試す順に。"""
    lcfg = cfg.get("llm") or {}
    want = (lcfg.get("provider") or "auto").lower()
    names = AUTO_ORDER if want == "auto" else (want,)
    out = []
    for name in names:
        if name not in PROVIDERS:
            continue
        env, base, model = PROVIDERS[name]
        key = os.environ.get(env)
        if not key:
            continue
        models = lcfg.get("models") if isinstance(lcfg.get("models"), dict) else {}
        model = models.get(name) or model
        if want != "auto":       # 1 つに決めたときだけ、model・base_url の上書きを効かせる
            base = lcfg.get("base_url") or base
            model = lcfg.get("model") or model
        out.append((name, key, base.rstrip("/"), model))
    return out


def available(cfg: dict | None = None) -> bool:
    return bool(_chain(cfg or {}))


def _clean(text: str) -> str:
    # 推論モデルが考えた過程を本文に混ぜて返すことがあるので落とす
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _call(name: str, key: str, base: str, model: str, body: str, max_tokens: int, log) -> str | None:
    if name == "anthropic":
        r = requests.post(base + "/messages", timeout=180, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "system": SYSTEM,
                  "messages": [{"role": "user", "content": "データ（JSON）:\n" + body}]})
        if r.status_code >= 400:
            log(f"  LLM（{name}）エラー {r.status_code}: {r.text[:200]}")
            return None
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if name == "openrouter":
        headers.update({"HTTP-Referer": "https://github.com/AK47bbddk/valorant-replay", "X-Title": "valorant-replay"})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.3,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "データ（JSON）:\n" + body}]}
    for attempt in range(3):
        r = requests.post(base + "/chat/completions", timeout=180, headers=headers, json=payload)
        if r.status_code == 429 and attempt < 2:      # 無料枠の混雑。少し待ってやり直す
            time.sleep(8 * (attempt + 1))
            continue
        if r.status_code >= 400:
            log(f"  LLM（{name} / {model}）エラー {r.status_code}: {r.text[:200]}")
            return None
        ch = (r.json().get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or ""
    return None


def summarize(cfg: dict, data: dict, log=print) -> str | None:
    """data（JSON にできる dict）から要約文。キーが無い・すべて失敗したら None。"""
    chain = _chain(cfg)
    if not chain:
        return None
    lcfg = cfg.get("llm") or {}
    max_tokens = int(lcfg.get("max_tokens") or 1500)
    body = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha1((SYSTEM + body).encode("utf-8")).hexdigest()[:16]
    cache = Path(cfg["output"]["dir"]) / "_cache" / "llm" / f"{h}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    for name, key, base, model in chain:
        try:
            text = _clean(_call(name, key, base, model, body, max_tokens, log) or "")
        except Exception as e:
            log(f"  LLM（{name}）に接続できませんでした（{e}）")
            continue
        if text:
            log(f"  要約：{name} / {model}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return text
    return None
