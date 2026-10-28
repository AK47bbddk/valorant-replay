"""（任意）集計した数字と解説の字幕から、プレイブックの「ATK summary / DEF summary」の文章を LLM に書かせる。

無料で使える OpenAI 互換の API を、キーがあるものから順に試す（Colab のシークレットに登録）：
  1. GEMINI_API_KEY     … Google AI Studio の Gemini 3.8 Flash（無料枠。日本語の文章がいちばん安定）。
     混雑（503）・回数制限（429）は数十秒待って何回かやり直してから次へ進む（できるだけ 3.8 Flash で書く）
  2. OPENROUTER_API_KEY … OpenRouter の無料モデル（名前の末尾が :free）。無料のモデルはよく入れ替わる
     （DeepSeek V4 Flash の無料版は 2026 年 10 月に終了）ので、実行のたびに OpenRouter のモデル一覧から
     いま無料の文章向けモデルを探して使う（一覧は output.dir/_cache に 12 時間ためる）
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
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "auto"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "claude-sonnet-5-5"),
}
AUTO_ORDER = ("gemini", "openrouter", "deepseek")
RETRY_STATUS = (429, 500, 502, 503, 504)       # 混雑・回数制限・一時的な不調。待てば通ることが多い
# OpenRouter の無料モデルを選ぶ順（id に含まれる語）。文章の要約に向かないもの（音声・画像生成・安全判定）は除く
FREE_PREFER = ("deepseek", "qwen", "gemma-4-31b", "nemotron-3-super", "nemotron-3-ultra", "llama", "gemma",
               "mistral", "nemotron")
FREE_SKIP = ("lyria", "voice", "tts", "audio", "safety", "guard", "embed", "image", "nano", "lightning", "2-6b")

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


def openrouter_free(cfg: dict | None = None, limit: int = 3) -> list[str]:
    """OpenRouter で今いる無料の文章向けモデルの id（選ぶ順）。一覧が取れなければ []。"""
    cache = Path((cfg or {}).get("output", {}).get("dir") or ".") / "_cache" / "openrouter_models.json"
    data = None
    try:
        if cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600:
            data = json.loads(cache.read_text("utf-8"))
    except Exception:
        data = None
    if data is None:
        try:
            r = requests.get("https://openrouter.ai/api/v1/models", timeout=20)
            data = r.json().get("data") or []
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), "utf-8")
        except Exception:
            return []
    free = []
    for m in data:
        mid = str(m.get("id") or "")
        pr = m.get("pricing") or {}
        out_mod = ((m.get("architecture") or {}).get("output_modalities")) or ["text"]
        if not (mid.endswith(":free") or (str(pr.get("prompt")) == "0" and str(pr.get("completion")) == "0")):
            continue
        if "text" not in out_mod or any(w in mid.lower() for w in FREE_SKIP):
            continue
        free.append((mid, int(m.get("context_length") or 0)))

    def rank(x):
        low = x[0].lower()
        k = next((i for i, w in enumerate(FREE_PREFER) if w in low), len(FREE_PREFER))
        return (k, -x[1])
    return [mid for mid, _ in sorted(free, key=rank)[:limit]]


def _chain(cfg: dict) -> list[tuple[str, str, str, str]]:
    """使える (provider, key, base_url, model) を試す順に。OpenRouter は無料モデルを何個か続けて並べる。"""
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
        if name == "openrouter":
            ms = [model] if model and model != "auto" else []
            ms += [m for m in openrouter_free(cfg) if m not in ms]
            out += [(name, key, base.rstrip("/"), m) for m in ms]
            continue
        if name == "gemini" and model != "gemini-flash-latest":
            # 3.8 Flash の名前が使えなくなったとき用に、最新の Flash（別名）も後ろに並べる
            out += [(name, key, base.rstrip("/"), model), (name, key, base.rstrip("/"), "gemini-flash-latest")]
            continue
        out.append((name, key, base.rstrip("/"), model))
    return out


def available(cfg: dict | None = None) -> bool:
    return bool(_chain(cfg or {}))


def _clean(text: str) -> str:
    # 推論モデルが考えた過程を本文に混ぜて返すことがあるので落とす
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _wait_seconds(r, attempt: int) -> float:
    """やり直すまでの秒数。サーバーが Retry-After を返していればそれ、無ければ 10・20・40・60 秒。"""
    try:
        ra = float(r.headers.get("Retry-After"))
        if 0 < ra <= 120:
            return ra
    except Exception:
        pass
    return float(min(60, 10 * 2 ** attempt))


def _call(name: str, key: str, base: str, model: str, body: str, max_tokens: int, log,
          retries: int = 1, reasoning: str | None = None) -> str | None:
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
    if reasoning:
        payload["reasoning_effort"] = reasoning        # Gemini の考える量（low・medium・high）
    for attempt in range(retries + 1):
        try:
            r = requests.post(base + "/chat/completions", timeout=180, headers=headers, json=payload)
        except requests.RequestException as e:        # 接続の切断・タイムアウトもやり直す
            if attempt < retries:
                log(f"  LLM（{name} / {model}）に接続できず（{type(e).__name__}）。10 秒待ってやり直します")
                time.sleep(10)
                continue
            raise
        if r.status_code in RETRY_STATUS and attempt < retries:
            w = _wait_seconds(r, attempt)
            why = "混雑" if r.status_code in (500, 502, 503, 504) else "回数制限"
            log(f"  LLM（{name} / {model}）{why}（{r.status_code}）。{w:.0f} 秒待ってやり直します（{attempt + 1}/{retries}）")
            time.sleep(w)
            continue
        if r.status_code >= 400:
            if name == "openrouter" and r.status_code in (404, 400) and any(
                    w in r.text for w in ("unavailable", "No endpoints", "not a valid model")):
                log(f"  LLM（openrouter / {model}）は使えなくなっていたので次を試します")
            else:
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
    retries = int(lcfg.get("retries") if lcfg.get("retries") is not None else 4)
    reasoning = lcfg.get("reasoning_effort") or "low"
    for name, key, base, model in chain:
        # Gemini は待ってでも使う（できるだけ 3.8 Flash で書く）。OpenRouter の無料モデルは次の候補があるので早めに切り替える
        rt = retries if (name == "gemini" and model != "gemini-flash-latest") else 1
        mt = max(max_tokens, 4096) if name == "gemini" else max_tokens   # 考える分のトークンも上限に含まれる
        try:
            text = _clean(_call(name, key, base, model, body, mt, log, retries=rt,
                                reasoning=reasoning if name == "gemini" else None) or "")
        except Exception as e:
            log(f"  LLM（{name}）に接続できませんでした（{e}）")
            continue
        if text:
            log(f"  要約：{name} / {model}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return text
    return None
