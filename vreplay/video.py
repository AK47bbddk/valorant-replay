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


PARTS = ".parts.json"


def parts_of(path) -> list[tuple[Path, float, float]] | None:
    """分割ダウンロードした動画（*.parts.json）なら [(ファイル, 区間の先頭からの秒, 長さ)]。普通の動画なら None。"""
    p = Path(path)
    if not p.name.endswith(PARTS):
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    return [(p.parent / x["file"], float(x["offset"]), float(x["length"])) for x in j["parts"]]


def video_size(path) -> int:
    ps = parts_of(path)
    return sum(f.stat().st_size for f, _, _ in ps if f.exists()) if ps else Path(path).stat().st_size


def remove_video(path):
    ps = parts_of(path)
    for f, _, _ in ps or []:
        f.unlink(missing_ok=True)
    Path(path).unlink(missing_ok=True)


def find_cached(video_id: str, start: int, end: int, dirs) -> Path | None:
    stem = f"{video_id}_{start}_{end}"
    for d in dirs or []:
        d = Path(d)
        if not d.exists():
            continue
        man = d / (stem + PARTS)
        if man.exists():
            try:
                if all(f.exists() and f.stat().st_size > 0 for f, _, _ in parts_of(man)):
                    return man
            except Exception:
                pass
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
                     cookies: str = "", cache_dirs=(), log=None, fragments: int = 4, parts: int = 1,
                     overlap: int = 12) -> Path:
    """[start, end] 秒の区間をダウンロードする。

    YouTube は 1 本の接続を実時間程度の速さ（1 MB/s 前後）に絞るので、parts > 1 なら区間を parts 個に分けて
    同時にダウンロードする（それぞれ overlap 秒だけ次の部分と重ねる）。そのときは各部分のファイルと、
    つなぎ方を書いた {video_id}_{start}_{end}.parts.json を返す。デコードは parts.json をそのまま渡せばよい。
    """
    hit = find_cached(video_id, start, end, [out_dir, *cache_dirs])
    if hit:
        if log:
            log(f"    キャッシュ済みの動画を使います（{hit.name}）")
        return hit
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = max(1, int(parts or 1))
    if parts == 1 or end - start < 300 * parts:
        return _download_one(video_id, start, end, out_dir, max_height, cookies, cache_dirs, log, fragments)
    from concurrent.futures import ThreadPoolExecutor
    bounds = [int(round(start + (end - start) * k / parts)) for k in range(parts + 1)]
    pieces = [(bounds[k], min(end, bounds[k + 1] + overlap)) for k in range(parts)]
    with ThreadPoolExecutor(max_workers=parts) as ex:
        files = list(ex.map(lambda se: _download_one(video_id, se[0], se[1], out_dir, max_height, cookies,
                                                     cache_dirs, None, fragments), pieces))
    man = out_dir / f"{video_id}_{start}_{end}{PARTS}"
    entries = []
    for k, (f, (a, _)) in enumerate(zip(files, pieces)):
        if Path(f).parent != out_dir:                  # キャッシュ側にあった部分はこちらへコピー
            dst = out_dir / Path(f).name
            shutil.copy(f, dst)
            f = dst
        entries.append({"file": Path(f).name, "offset": a - start, "length": bounds[k + 1] - bounds[k]})
    man.write_text(json.dumps({"video_id": video_id, "start": start, "end": end, "parts": entries},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    if log:
        log(f"    {parts} 分割で同時に取得しました")
    return man


def _download_one(video_id: str, start: int, end: int, out_dir: Path, max_height: int, cookies: str,
                  cache_dirs, log, fragments: int) -> Path:
    hit = find_cached(video_id, start, end, [out_dir, *cache_dirs])
    if hit and not hit.name.endswith(PARTS):
        return hit
    stem = f"{video_id}_{start}_{end}"
    env = _ytdlp_env()
    # いったんローカルの一時フォルダに落としてから移す（Drive などに ffmpeg が直接書くと失敗することがある）
    tmp = Path(tempfile.mkdtemp(prefix="vreplay_"))
    errors = []
    for client in CLIENTS:
        cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-part", "--no-progress",
               "--sleep-requests", "1", "--remote-components", "ejs:github",
               "--concurrent-fragments", str(max(1, int(fragments))),
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
    ps = parts_of(path)
    if ps:
        w, h, _ = probe(ps[0][0])
        return w, h, ps[-1][1] + ps[-1][2]
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
    ps = parts_of(path)
    if ps:
        path = str(ps[0][0])
    for _ in iter_frames(Path(path), fps=1, start=0, duration=1.0, hwaccel=True, fallback=False):
        return True
    return False


def iter_frames(path: Path, fps: float, start: float = 0.0, duration: float | None = None,
                hwaccel: bool | None = None, fallback: bool = True):
    """(t秒, BGR フレーム) を順に返す。t は動画ファイル先頭からの秒。

    非参照フレーム（B フレーム）はデコードしない（-skip_frame noref）。2fps で間引くので影響はなく、デコードが速くなる。
    """
    ps = parts_of(path)
    if ps:
        # 分割した動画：各部分の受け持ち（重なりを除いた範囲）だけを順に読む。時刻は start からの fps の格子にそろえる
        import math
        end = start + duration if duration else ps[-1][1] + ps[-1][2]
        for f, off, ln in ps:
            a, b = max(start, off), min(end, off + ln)
            if b - a <= 1e-6:
                continue
            a = start + math.ceil((a - start) * fps - 1e-6) / fps
            if a >= b:
                continue
            for t, fr in iter_frames(f, fps, a - off, b - a, hwaccel=hwaccel, fallback=fallback):
                yield t + off, fr
        return
    w, h, _ = probe(path)
    if hwaccel is None:
        hwaccel = gpu_decode_ok(str(path))
    base = ["ffmpeg", "-v", "error", "-nostdin", "-skip_frame", "noref"]
    seek = ["-ss", f"{start:.3f}"] + (["-t", f"{duration:.3f}"] if duration else [])
    out = ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    # GPU デコードでは、フレームを GPU 上に置いたまま fps フィルタで間引き、残った 2fps ぶんだけ CPU へ移す。
    # （-hwaccel cuda だけだと 60fps の全フレームを毎回 CPU へコピーしていて、CPU を大きく使っていた）
    cmds = []
    if hwaccel:
        cmds.append(base + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] + seek
                    + ["-i", str(path), "-an", "-vf", f"fps={fps},hwdownload,format=nv12"] + out)
        cmds.append(base + ["-hwaccel", "cuda"] + seek + ["-i", str(path), "-an", "-vf", f"fps={fps}"] + out)
    if fallback or not hwaccel:
        cmds.append(base + seek + ["-i", str(path), "-an", "-vf", f"fps={fps}"] + out)
    for cmd in cmds:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4)
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


def iter_keyframes(path: Path, start: float = 0.0, duration: float | None = None):
    """キーフレームだけを (t秒, BGR) で返す。それ以外のフレームはデコードしないので、全フレームを
    デコードするより 10 倍以上速い（YouTube の動画はおおむね 1〜5 秒ごとにキーフレームがある）。
    時刻は ffmpeg の showinfo が出す pts から求める。"""
    import queue
    import re
    import threading
    ps = parts_of(path)
    if ps:
        end = start + duration if duration else ps[-1][1] + ps[-1][2]
        for f, off, ln in ps:
            a, b = max(start, off), min(end, off + ln)
            if b - a > 1e-6:
                for t, fr in iter_keyframes(f, a - off, b - a):
                    if t + off < b:
                        yield t + off, fr
        return
    w, h, _ = probe(path)
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-skip_frame", "nokey", "-ss", f"{start:.3f}"]
    cmd += (["-t", f"{duration:.3f}"] if duration else []) + [
        "-i", str(path), "-an", "-vsync", "0", "-vf", "showinfo", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 2)
    q: queue.Queue = queue.Queue()
    pat = re.compile(rb"pts_time:\s*(-?[\d.]+)")

    def read_err():
        for line in p.stderr:
            m = pat.search(line)
            if m:
                q.put(float(m.group(1)))
        q.put(None)

    th = threading.Thread(target=read_err, daemon=True)
    th.start()
    size = w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            pt = q.get(timeout=30)
            if pt is None:
                break
            yield start + pt, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def keyframe_interval(path: Path, span: float = 180.0) -> float | None:
    """先頭 span 秒のキーフレーム間隔（中央値・秒）。読めなければ None。"""
    ps = parts_of(path)
    if ps:
        path = ps[0][0]
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+{span:.0f}",
                              "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return None
    ts = []
    for line in out.splitlines():
        a, _, f = line.partition(",")
        if "K" in f:
            try:
                ts.append(float(a))
            except ValueError:
                pass
    ts.sort()
    if len(ts) < 3:
        return None
    d = sorted(b - a for a, b in zip(ts, ts[1:]))
    return d[len(d) // 2]


def grab(path: Path, t: float):
    """1 枚だけ取り出す（キャリブレーション用）。"""
    for _, f in iter_frames(path, fps=1, start=t, duration=1.0, hwaccel=False):
        return f
    return None


def grab_key(path: Path, t: float, span: float = 10.0):
    """t 秒以降で最初のキーフレームを 1 枚（ほかのフレームをデコードしないので速い）。無ければ grab と同じ。"""
    for _, f in iter_keyframes(path, t, span):
        return f
    return grab(path, t)
