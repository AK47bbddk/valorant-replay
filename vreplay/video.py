"""YouTube の該当区間だけをダウンロードし、ffmpeg でフレームを流す。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np


COOKIE_HELP = (
    "YouTube にボット判定されました（Colab などクラウドの IP で起きます）。対処はどちらか："
    "① 自分の YouTube の cookies.txt を Drive の valorant-replay/cookies.txt に置く"
    "（または Colab の 🔑 に YT_COOKIES として中身を登録）"
    "② 自宅 PC で `vreplay fetch` を実行して動画を Drive に置いてから Colab で run する。"
    "手順は README の「YouTube に弾かれるとき」")


def resolve_cookies(cfg: dict) -> str:
    """cookies.txt の場所：config の video.cookies → 環境変数 YT_COOKIES（中身）→ 出力フォルダの cookies.txt。"""
    p = cfg["video"].get("cookies") or ""
    if p and Path(p).exists():
        return p
    text = os.environ.get("YT_COOKIES", "").strip()
    if text:
        f = Path(tempfile.gettempdir()) / "vreplay_cookies.txt"
        if not text.startswith("#"):
            text = "# Netscape HTTP Cookie File\n" + text
        f.write_text(text.replace("\r\n", "\n") + "\n", encoding="utf-8")
        return str(f)
    default = Path(cfg["output"]["dir"]) / "cookies.txt"
    return str(default) if default.exists() else ""


def _ytdlp_env() -> dict:
    """~/.deno/bin（deno の既定インストール先）を PATH に足した環境変数。"""
    env = dict(os.environ)
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.exists():
        env["PATH"] = f"{deno_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def find_cached(video_id: str, start: int, end: int, dirs) -> Path | None:
    stem = f"{video_id}_{start}_{end}"
    for d in dirs or []:
        d = Path(d)
        if d.exists():
            hit = [p for p in d.glob(stem + ".*") if p.suffix in (".mp4", ".webm", ".mkv") and p.stat().st_size > 0]
            if hit:
                return hit[0]
    return None


def download_segment(video_id: str, start: int, end: int, out_dir: Path, max_height: int = 1080,
                     cookies: str = "", cache_dirs=(), local: bool = False) -> Path:
    hit = find_cached(video_id, start, end, [out_dir, *cache_dirs])
    if hit:
        return hit
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video_id}_{start}_{end}"
    cmd = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--no-part", "--sleep-requests", "1",
           "-f", f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}]/b[height<={max_height}]",
           "--download-sections", f"*{start}-{end}",
           "-o", str(out_dir / (stem + ".%(ext)s")),
           f"https://www.youtube.com/watch?v={video_id}"]
    if cookies and Path(cookies).exists():
        cmd[3:3] = ["--cookies", cookies]
    # yt-dlp は YouTube の n チャレンジを解くのに JS ランタイム（deno 推奨）と解読スクリプト（yt-dlp-ejs）を使う
    env = _ytdlp_env()
    if not shutil.which("deno", path=env["PATH"]) and shutil.which("node"):
        cmd[3:3] = ["--js-runtimes", "node"]
    cmd[3:3] = ["--remote-components", "ejs:github"]   # yt-dlp-ejs が無くても解読スクリプトを取得できるように
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    files = find_cached(video_id, start, end, [out_dir])
    if r.returncode != 0:
        for p in out_dir.glob(stem + ".*"):     # 途中まで書かれた壊れたファイルをキャッシュと誤認しないよう消す
            p.unlink(missing_ok=True)
        files = None
    if r.returncode != 0 and "ffmpeg exited" in r.stderr and not local:
        # ネットワークドライブ（Colab の Drive など）への直接書き込みで ffmpeg が失敗したらローカル経由でやり直す
        tmp = Path(tempfile.mkdtemp(prefix="vreplay_"))
        got = download_segment(video_id, start, end, tmp, max_height, cookies, local=True)
        dst = out_dir / got.name
        shutil.move(str(got), dst)
        return dst
    if r.returncode != 0 or not files:
        if "Sign in to confirm" in r.stderr:
            extra = ("（cookies は読み込めています。それでも弾かれる場合は、cookies が無効になっているか、"
                     "Colab の IP 自体がブロックされています。② の自宅 PC で取得する方法が確実です）") if cookies else ""
            raise RuntimeError(COOKIE_HELP + extra + f"\n  yt-dlp: {r.stderr.strip()[-300:]}")
        hint = ""
        if "challenge solving failed" in r.stderr or "Only images are available" in r.stderr:
            hint = ("（YouTube の暗号解読に失敗：JS ランタイムがありません。Colab ではセル1で deno が入ります。"
                    "自宅 PC では `winget install DenoLand.Deno` を実行してください）")
        raise RuntimeError(f"yt-dlp failed{hint}: {r.stderr.strip()[-400:]}")
    return files


def probe(path: Path) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j.get("format", {}).get("duration") or 0)


@lru_cache(maxsize=1)
def _has_cuda() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    r = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True)
    return "cuda" in r.stdout


def iter_frames(path: Path, fps: float, start: float = 0.0, duration: float | None = None):
    """(t秒, BGR フレーム) を順に返す。t は動画ファイル先頭からの秒。"""
    w, h, _ = probe(path)
    base = ["ffmpeg", "-v", "error", "-nostdin"]
    tail = ["-ss", str(start)] + (["-t", str(duration)] if duration else []) + [
        "-i", str(path), "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    for hw in ([["-hwaccel", "cuda"]] if _has_cuda() else []) + [[]]:
        p = subprocess.Popen(base + hw + tail, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
        n, size = 0, w * h * 3
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield start + n / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
        p.wait()
        if n > 0 or not hw:
            return
        # CUDA デコードで 1 枚も出なければ CPU でやり直す


def grab(path: Path, t: float):
    """1 枚だけ取り出す（キャリブレーション用）。"""
    for _, f in iter_frames(path, fps=1, start=t, duration=1.0):
        return f
    return None
