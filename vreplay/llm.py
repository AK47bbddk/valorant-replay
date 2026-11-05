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
_NO_CREDIT: set[str] = set()                    # 前払いのクレジットが切れた（402）プロバイダー。この実行ではもう呼ばない
_BUSY: dict[str, int] = {}                      # やり直しても書けなかった回数（続けて 2 冊だめなら、この実行では後回し）
TIMEOUT = 90                                    # 1 回の要求の待ち時間（秒）。混雑時は返事が来ないまま待ち続けるので短めに
# OpenRouter の無料モデルを選ぶ順（id に含まれる語）。文章の要約に向かないもの（音声・画像生成・安全判定）は除く
FREE_PREFER = ("deepseek", "qwen", "gemma-4-31b", "nemotron-3-super", "nemotron-3-ultra", "llama", "gemma",
               "mistral", "nemotron")
FREE_SKIP = ("lyria", "voice", "tts", "audio", "safety", "guard", "embed", "image", "nano", "lightning", "2-6b")

SYSTEM = (
    "あなたは VALORANT のプロチームのアナリスト兼コーチです。渡されるのは、試合映像を自動解析して集計した"
    "1 チーム・1 マップ・1 構成のデータ（ラウンドの勝率、初動の型（「B Main テイクデフォ」「守りデフォ（2-1-2）」など）"
    "と各エージェントの立ち位置、2 巡目のアクション（execution・ラッシュ・捨て〜スプリット・レイト）、設置サイトと時刻、"
    "先制キル、リテイク・ポストプラント・カウンターの結果、ULT、解説者の発言から拾った意図のキーワードと一節）です。"
    "これをもとに、チーム内のプレイブックに貼る日本語の要約を、コーチが手で書くときの短い箇条書きで書いてください。\n"
    "- 見出しは「ATK summary」「DEF summary」の 2 つ。どちらも必ず書く。それぞれ箇条書き 4〜6 行・1 行 60 文字以内\n"
    "- ATK は「デフォルト → 2 巡目のアクション → 最終形」の順に並べる"
    "（例：デフォして2巡目のアクションして最終形に持っていく）\n"
    "- DEF は「デフォルトの形 → 時間での寄せ方 → リテイク」の順。強いところも 1 行入れる\n"
    "- 型の名前・回数・勝率は、データにある書き方のまま括弧で添える（名前を言い換えない）\n"
    "- 最後に「対策メモ」として、相手の攻めへの対策を「ドライに対して→…」「セットに対して→…」の形で 2 行\n"
    "- データから言えないこと（スキルの使い方、コールの中身）は「推測：」と頭につけて最大 2 行\n"
    "- 回数が少ないもの（5 回未満）は「参考」と書く\n"
    "- 前置きや考えた過程は書かず、見出しから始める。文は必ず言い切って終わる"
)
SHORTER = (
    "前の回答は長すぎて途中で切れました。同じ見出し（ATK summary・DEF summary・対策メモ）のまま、"
    "各 3 行・1 行 40 文字以内に短くまとめ直してください。必ず最後まで書き切ってください。"
)
HEADS = {"atk summary": "ATK", "atk": "ATK", "def summary": "DEF", "def": "DEF",
         "対策メモ": "memo", "対策": "memo"}


def looks_truncated(text: str | None) -> bool:
    """出力が途中で切れている（最後の行が言い切りで終わっていない）。"""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return True
    return not lines[-1].rstrip().rstrip("*`").endswith(("。", "）", ")", "」", "！", "？", "．", "."))


def sections(text: str | None) -> dict[str, list[str]]:
    """見出しごとに分ける。{"ATK": [行], "DEF": [行], "memo": [行]}。切れている最後の行は落とす。"""
    out: dict[str, list[str]] = {"ATK": [], "DEF": [], "memo": []}
    cur, last = None, None
    for line in (text or "").splitlines():
        t = line.strip().strip("#").strip().strip("*").strip()
        if not t:
            continue
        low = t.lower().rstrip("：:").strip()
        key = next((v for k, v in HEADS.items() if low.startswith(k)), None)
        if key and len(low) <= len(key) + 12:
            cur = key
            continue
        if cur:
            body = re.sub(r"^\d+[.)、]\s*", "", re.sub(r"^[-・*\s]+", "", t))
            out[cur].append(body.strip())
            last = cur
    if last and out[last] and looks_truncated(text):
        out[last].pop()
    return out


def complete(text: str | None) -> bool:
    """ATK／DEF の両方があって、途中で切れていない。"""
    sec = sections(text)
    return len(sec["ATK"]) >= 2 and len(sec["DEF"]) >= 2 and not looks_truncated(text)


def worth_retry(text: str | None) -> bool:
    """言い直させる価値があるか。見出しは書けていて、長くなって途中で切れた（片側が無い）ときだけ。
    そもそも見出しが無い短い返事は、言い直しても同じなので next のモデルへ進む。"""
    if not text or complete(text):
        return False
    sec = sections(text)
    return bool(sec["ATK"] or sec["DEF"])


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
    order = lcfg.get("order") or os.environ.get("VREPLAY_LLM_ORDER") or ""
    if isinstance(order, str):
        order = [x.strip() for x in order.split(",") if x.strip()]
    names = (tuple(order) or AUTO_ORDER) if want == "auto" else (want,)
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
          retries: int = 1, reasoning: str | None = None, note: str | None = None) -> str | None:
    user = "データ（JSON）:\n" + body + (f"\n\n{note}" if note else "")
    if name == "anthropic":
        r = requests.post(base + "/messages", timeout=180, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "system": SYSTEM,
                  "messages": [{"role": "user", "content": user}]})
        if r.status_code >= 400:
            log(f"  LLM（{name}）エラー {r.status_code}: {r.text[:200]}")
            return None
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if name == "openrouter":
        headers.update({"HTTP-Referer": "https://github.com/AK47bbddk/valorant-replay", "X-Title": "valorant-replay"})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.3,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    if reasoning:
        payload["reasoning_effort"] = reasoning        # Gemini の考える量（low・medium・high）
    for attempt in range(retries + 1):
        try:
            r = requests.post(base + "/chat/completions", timeout=TIMEOUT, headers=headers, json=payload)
        except requests.RequestException as e:        # 接続の切断・タイムアウトもやり直す
            if attempt < retries:
                log(f"  LLM（{name} / {model}）に接続できず（{type(e).__name__}）。10 秒待ってやり直します")
                time.sleep(10)
                continue
            raise
        low = r.text.lower()
        if r.status_code == 429 and any(w in low for w in ("perday", "per day", "per_day", "daily")):
            _NO_CREDIT.add(name)                       # 1 日の上限：待っても今日は通らない
            log(f"  LLM（{name}）は 1 日の上限に達しています。この実行では使わずに次へ進みます")
            return None
        if r.status_code in RETRY_STATUS and attempt < retries:
            w = _wait_seconds(r, attempt)
            why = "混雑" if r.status_code in (500, 502, 503, 504) else "回数制限"
            log(f"  LLM（{name} / {model}）{why}（{r.status_code}）。{w:.0f} 秒待ってやり直します（{attempt + 1}/{retries}）")
            time.sleep(w)
            continue
        if r.status_code == 402 or "credits are depleted" in r.text:
            _NO_CREDIT.add(name)
            log(f"  LLM（{name}）は前払いのクレジットが切れています（402）。待っても通らないので、この実行では使わずに次へ進みます"
                + ("（AI Studio の［お支払い］でクレジットを追加すると戻ります）" if name == "gemini" else ""))
            return None
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
    max_tokens = int(lcfg.get("max_tokens") or 2400)
    body = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    h = hashlib.sha1((SYSTEM + body).encode("utf-8")).hexdigest()[:16]
    cache = Path(cfg["output"]["dir"]) / "_cache" / "llm" / f"{h}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    retries = int(lcfg.get("retries") if lcfg.get("retries") is not None else 4)
    reasoning = lcfg.get("reasoning_effort") or "low"
    retried = False                                 # 言い直しは 1 冊につき 1 回だけ
    if _BUSY.get("gemini", 0) >= 2:
        # 混雑が続いている：この実行の残りは、ほかのモデルを先に試し、だめなときだけ Gemini に戻る
        chain = [c for c in chain if c[0] != "gemini"] + [c for c in chain if c[0] == "gemini"]
    for name, key, base, model in chain:
        if name in _NO_CREDIT:
            continue
        # Gemini は待ってでも使う（できるだけ 3.8 Flash で書く）。OpenRouter の無料モデルは次の候補があるので早めに切り替える
        rt = retries if (name == "gemini" and model != "gemini-flash-latest") else 1
        mt = max(max_tokens, 8192) if name == "gemini" else max_tokens   # 考える分のトークンも上限に含まれる
        try:
            text = _clean(_call(name, key, base, model, body, mt, log, retries=rt,
                                reasoning=reasoning if name == "gemini" else None) or "")
            if not retried and worth_retry(text):
                # 途中で切れている・片方の見出しが無い。短く書くよう言い直して 1 回だけやり直す
                retried = True
                log(f"  要約（{name} / {model}）が途中で切れているので、短く書き直させます")
                again = _clean(_call(name, key, base, model, body, mt, log, retries=0,
                                     reasoning=reasoning if name == "gemini" else None,
                                     note=SHORTER) or "")
                if again and (complete(again) or len(again) > len(text)):
                    text = again
        except Exception as e:
            log(f"  LLM（{name}）に接続できませんでした（{e}）")
            continue
        if name == "gemini" and model != "gemini-flash-latest":
            if text:
                _BUSY["gemini"] = 0
            else:
                _BUSY["gemini"] = _BUSY.get("gemini", 0) + 1
                if _BUSY["gemini"] == 2:
                    log("  Gemini の混雑が続いているので、この実行の残りはほかのモデルを先に使います")
        if text:
            log(f"  要約：{name} / {model}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
            return text
    return None
