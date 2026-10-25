"""放送の字幕（YouTube の自動字幕）から、解説者が口にした「意図」の手がかりをラウンドごとに拾う。

映像からは「何をしたか」（配置・設置・キル）は分かっても「なぜそうしたか」は分からない。
公式放送の解説はそれをその場で言葉にしている（"they're going for the B split", "fake A, rotate B" など）。
自動字幕は動画本体よりずっと小さく、yt-dlp で数秒で取れる。

- fetch(video_id) … 自動字幕（英語優先、なければ日本語）を json3 で取り、[(開始秒, 終了秒, 文)] にする
- for_round(...)  … ラウンドの時間帯（購入フェーズの少し前〜終了直後）の字幕
- intents(text)   … キーワードから意図のタグ（ラッシュ・スプリット・フェイク・ラーク・リテイク…）と、
                    その根拠になった短い一節（100 文字以内）を返す

字幕は解説の発言なので、外れることも、別のチームの話をしていることもある。「解説の言及」として扱い、
勝因やプレイブックの数字とは分けて表示する。
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

# (タグ, 正規表現)。英語の放送と日本語の放送の両方
RULES = [
    ("ラッシュ", r"\b(rush(ing|es)?|run(ning)? it down|fast (hit|exec\w*)|full sprint|speed(ing)? (in|onto))\b|ラッシュ|速攻"),
    ("スプリット", r"\bsplit(ting)?\b|スプリット|挟(み|む)"),
    ("フェイク", r"\bfak(e|es|ed|ing)\b|フェイク"),
    ("ラーク", r"\blurk(s|ing|er)?\b|ラーク|裏取り"),
    ("リテイク", r"\bre-?tak(e|es|ing)\b|リテイク"),
    ("スタック", r"\bstack(s|ed|ing)?\b|スタック|寄せ"),
    ("ローテ", r"\brotat(e|es|ed|ing|ion)\b|ローテ"),
    ("デフォルト", r"\bdefault\b|デフォルト"),
    ("ミッドコントロール", r"\b(mid control|control (of )?mid|take mid|taking mid)\b|ミッド(の)?コントロール|中央"),
    ("セーブ・エコ", r"\b(eco|save|saving|saved)\b|エコ|セーブ"),
    ("フォース", r"\bforce(d)?( buy)?\b|フォース"),
    ("ボーナス", r"\bbonus\b|ボーナス"),
    ("オペレーター", r"\b(op|operator|awp)\b|オペ(レーター)?"),
    ("ポストプラント", r"\bpost[- ]?plant\b|ポストプラント|設置後"),
    ("リテイク阻止（遅延）", r"\b(delay(ing)?|stall(ing)?)\b|遅延"),
    ("エントリー", r"\b(entry|entr(y|ies)|first blood|opening (pick|kill))\b|エントリー|ファーストブラッド"),
    ("クラッチ", r"\bclutch(es|ed|ing)?\b|クラッチ"),
    ("ULT", r"\b(ult|ultimate)s?\b|アルティメット|ウルト"),
    ("タイムアウト", r"\b(tactical )?timeout\b|タイムアウト"),
    ("リテイク準備（寄り）", r"\b(play(ing)? (for )?(the )?retake|give (up )?(the )?site)\b|サイトを捨て"),
]
SITE_RE = re.compile(r"\b([abc])[ -]?(site|main|long|short|lobby|link|heaven|hall|garden|elbow)\b|\bmid\b|([ABC])サイト|([ABC])メイン|ミッド",
                     re.IGNORECASE)


def _parse_json3(raw: dict) -> list[tuple[float, float, str]]:
    out = []
    for ev in raw.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if not text:
            continue
        t0 = float(ev.get("tStartMs", 0)) / 1000.0
        t1 = t0 + float(ev.get("dDurationMs", 0)) / 1000.0
        out.append((t0, t1, text))
    return out


def fetch(video_id: str, cache_dir: Path | None = None, cookies: str = "", env=None, log=None) -> list[tuple[float, float, str]]:
    """自動字幕（英語 → 日本語の順）を取る。取れなければ []。cache_dir に json で残す。"""
    if cache_dir is not None:
        cp = Path(cache_dir) / f"{video_id}.captions.json"
        if cp.exists():
            try:
                return [tuple(x) for x in json.loads(cp.read_text(encoding="utf-8"))]
            except Exception:
                pass
    try:
        import yt_dlp
        from .ytrange import ydl_options
    except Exception:
        return []
    tmp = Path(tempfile.mkdtemp(prefix="vreplay_cc_"))
    out: list[tuple[float, float, str]] = []
    try:
        deno = shutil.which("deno", path=(env or {}).get("PATH")) if env else shutil.which("deno")
        opts = ydl_options("bv*", cookies, None, deno)

        class _Quiet:
            def debug(self, msg):
                pass
            warning = info = error = debug

        opts.update({"skip_download": True, "writeautomaticsub": True, "writesubtitles": True,
                     "subtitleslangs": ["en", "en-orig", "en.*", "ja"], "subtitlesformat": "json3",
                     "outtmpl": str(tmp / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
                     "ignoreerrors": True, "logger": _Quiet()})
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        files = sorted(tmp.glob("*.json3"), key=lambda p: (0 if ".en" in p.name else 1, p.name))
        if files:
            out = _parse_json3(json.loads(files[0].read_text(encoding="utf-8")))
    except Exception as e:
        if log:
            log(f"    字幕を取れませんでした（{str(e)[:120]}）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if cache_dir is not None and out:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{video_id}.captions.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def for_round(caps: list[tuple[float, float, str]], t0: float, t1: float) -> str:
    """[t0, t1]（動画の秒）にかかる字幕をつなげた文。"""
    return " ".join(tx for a, b, tx in caps if b >= t0 and a <= t1)


def intents(text: str, teams: list[str] | None = None, max_quotes: int = 2) -> dict:
    """字幕の文から意図のタグと根拠の一節。{"tags": [...], "sites": [...], "quotes": [...]}"""
    if not text:
        return {"tags": [], "sites": [], "quotes": []}
    tags, quotes = [], []
    for tag, pat in RULES:
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        tags.append(tag)
        if len(quotes) < max_quotes:
            a = max(0, m.start() - 45)
            b = min(len(text), m.end() + 45)
            q = text[a:b].strip()
            quotes.append(("…" if a > 0 else "") + q[:100] + ("…" if b < len(text) else ""))
    sites = []
    for m in SITE_RE.finditer(text):
        g = (m.group(1) or m.group(3) or m.group(4) or "").upper()
        name = f"{g}" if g else "Mid"
        if name not in sites:
            sites.append(name)
    out = {"tags": tags, "sites": sites, "quotes": quotes}
    if teams:
        out["teams"] = [t for t in teams if t and re.search(re.escape(t), text, re.IGNORECASE)]
    return out
