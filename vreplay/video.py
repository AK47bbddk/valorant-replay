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


# YouTube の取得方法（yt-dlp の player_client）。None は yt-dlp の既定。
# 一部の形式は数 MB 進むと 403 で切られる（ffmpeg exited with code 8）ため、順に試す
CLIENTS = [None, "web_safari", "tv", "mweb", "web_embedded"]


def _err_lines(stderr: str, n: int = 6) -> str:
    keys = ("ERROR", "error", "Error", "403", "failed", "Invalid", "denied")
    lines = [l.strip() for l in stderr.replace("\r", "\n").splitlines() if any(k in l for k in keys)]
    return " / ".join(lines[-n:]) or stderr.strip()[-300:]


def download_segment(video_id: str, start: int, end: int, out_dir: Path, max_height: int = 1080,
                     cookies: str = "", cache_dirs=(), log=None) -> Path:
    hit = find_cached(video_id, start, end, [out_dir, *cache_dirs])
    if hit:
        return hit
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video_id}_{start}_{end}"
    env = _ytdlp_env()
    # いったんローカルの一時フォルダに落としてから移す（Drive などに ffmpeg が直接書くと失敗することがある）
    tmp = Path(tempfile.mkdtemp(prefix="vreplay_"))
    errors = []
    for client in CLIENTS:
        cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part", "--no-progress",
               "--sleep-requests", "1", "--remote-components", "ejs:github",
               "-f", f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}]/b[height<={max_height}]",
               "--download-sections", f"*{start}-{end}",
               "-o", str(tmp / (stem + ".%(ext)s")),
               f"https://www.youtube.com/watch?v={video_id}"]
        if cookies and Path(cookies).exists():
            cmd[3:3] = ["--cookies", cookies]
        if client:
            cmd[3:3] = ["--extractor-args", f"youtube:player_client={client}"]
        if not shutil.which("deno", path=env["PATH"]) and shutil.which("node"):
            cmd[3:3] = ["--js-runtimes", "node"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        got = find_cached(video_id, start, end, [tmp])
        if r.returncode == 0 and got:
            dst = out_dir / got.name
            shutil.move(str(got), dst)
            shutil.rmtree(tmp, ignore_errors=True)
            if errors and log:
                log(f"    取得方法 {client} で成功（既定の方法は失敗）")
            return dst
        for p in tmp.glob(stem + ".*"):         # 途中まで書かれた壊れたファイルは消す
            p.unlink(missing_ok=True)
        if "Sign in to confirm" in r.stderr:
            shutil.rmtree(tmp, ignore_errors=True)
            extra = ("（cookies は読み込めています。それでも弾かれる場合は、cookies が無効になっているか、"
                     "この IP 自体がブロックされています。② の自宅 PC で取得する方法が確実です）") if cookies else ""
            raise RuntimeError(COOKIE_HELP + extra + f"\n  yt-dlp: {_err_lines(r.stderr)}")
        errors.append(f"[{client or '既定'}] {_err_lines(r.stderr)}")
    shutil.rmtree(tmp, ignore_errors=True)
    hint = ""
    joined = " ".join(errors)
    if "challenge solving failed" in joined or "Only images are available" in joined:
        hint = ("（YouTube の暗号解読に失敗：JS ランタイムがありません。Colab ではセル1で deno が入ります。"
                "自宅 PC では `winget install DenoLand.Deno` を実行してください）")
    raise RuntimeError(f"yt-dlp failed{hint}（{len(CLIENTS)} 通りの取得方法をすべて試しました）:\n  "
                       + "\n  ".join(errors))


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


@lru_cache(maxsize=8)
def gpu_decode_ok(path: str) -> bool:
    """GPU（NVDEC）でデコードできるか 1 秒だけ試す。"""
    if not _has_cuda():
        return False
    for _ in iter_frames(Path(path), fps=1, start=0, duration=1.0, hwaccel=True, fallback=False):
        return True
    return False


def iter_frames(path: Path, fps: float, start: float = 0.0, duration: float | None = None,
                hwaccel: bool | None = None, fallback: bool = True):
    """(t秒, BGR フレーム) を順に返す。t は動画ファイル先頭からの秒。

    非参照フレーム（B フレーム）はデコードしない（-skip_frame noref）。2fps で間引くので影響はなく、デコードが速くなる。
    """
    w, h, _ = probe(path)
    if hwaccel is None:
        hwaccel = gpu_decode_ok(str(path))
    base = ["ffmpeg", "-v", "error", "-nostdin", "-skip_frame", "noref"]
    tail = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else []) + [
        "-i", str(path), "-an", "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    tries = ([["-hwaccel", "cuda"]] if hwaccel else []) + ([[]] if (fallback or not hwaccel) else [])
    for hw in tries:
        p = subprocess.Popen(base + hw + tail, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
        n, size = 0, w * h * 3
        try:
            while True:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                yield start + n / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                n += 1
        finally:
            p.stdout.close()
            p.kill()
            p.wait()
        if n > 0:
            return
        # GPU デコードで 1 枚も出なければ CPU でやり直す


def grab(path: Path, t: float):
    """1 枚だけ取り出す（キャリブレーション用）。"""
    for _, f in iter_frames(path, fps=1, start=t, duration=1.0, hwaccel=False):
        return f
    return None
