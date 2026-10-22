"""YouTube の該当区間だけをダウンロードし、ffmpeg でフレームを流す。"""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


def download_segment(video_id: str, start: int, end: int, out_dir: Path, max_height: int = 1080,
                     cookies: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{video_id}_{start}_{end}"
    done = list(out_dir.glob(stem + ".*"))
    if done:
        return done[0]
    cmd = ["yt-dlp", "--quiet", "--no-warnings", "--no-part",
           "-f", f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}]/b[height<={max_height}]",
           "--download-sections", f"*{start}-{end}",
           "-o", str(out_dir / (stem + ".%(ext)s")),
           f"https://www.youtube.com/watch?v={video_id}"]
    if cookies and Path(cookies).exists():
        cmd[1:1] = ["--cookies", cookies]
    r = subprocess.run(cmd, capture_output=True, text=True)
    files = list(out_dir.glob(stem + ".*"))
    if r.returncode != 0 or not files:
        hint = ""
        if "Sign in to confirm" in r.stderr or "bot" in r.stderr.lower():
            hint = "（YouTube にボット判定されました。config の video.cookies に cookies.txt を指定してください）"
        raise RuntimeError(f"yt-dlp failed{hint}: {r.stderr.strip()[-400:]}")
    return files[0]


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
