from datetime import datetime
from pathlib import Path

from vreplay import config, pipeline, video, vlr


def test_resolve_cookies(tmp_path, monkeypatch):
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    monkeypatch.delenv("YT_COOKIES", raising=False)
    assert video.resolve_cookies(cfg) == ""
    (tmp_path / "cookies.txt").write_text("# Netscape HTTP Cookie File\n")
    assert video.resolve_cookies(cfg) == str(tmp_path / "cookies.txt")
    monkeypatch.setenv("YT_COOKIES", ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx")
    p = video.resolve_cookies(cfg)
    assert open(p).read().startswith("# Netscape HTTP Cookie File\n.youtube.com")


def test_cache_hit_skips_download(tmp_path, monkeypatch):
    cache = tmp_path / "videos"
    cache.mkdir()
    (cache / "vid_10_20.mp4").write_bytes(b"x")
    monkeypatch.setattr(video.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DL した")))
    assert video.download_segment("vid", 10, 20, tmp_path / "work", cache_dirs=[cache]) == cache / "vid_10_20.mp4"


def test_fetch_saves_to_cache(tmp_path, monkeypatch):
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    m = vlr.Match(id="1", url="https://www.vlr.gg/1/x", team1="A", team2="B",
                  dt_jst=datetime(2026, 10, 21, 20, tzinfo=vlr.JST))
    m.maps = [vlr.MapData(order=1, name="Bind", game_id="1", score1=13, score2=5, vod_id="v", vod_t=100),
              vlr.MapData(order=2, name="Haven", game_id="2", score1=13, score2=9, vod_id="v", vod_t=3000)]
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    calls = []

    def fake_dl(vid, s, e, out, h, cookies, log=None, **kwargs):
        calls.append((vid, s, e, out))
        p = out / f"{vid}_{s}_{e}.mp4"
        out.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p
    monkeypatch.setattr(video, "download_segment", fake_dl)
    got = pipeline.fetch(cfg, match_urls=[m.url], log=lambda *a: None)
    assert [c[1:3] for c in calls] == [(80, 3030), (2980, 2980 + 20 + 22 * 140 + 900)]
    assert all(p.parent == tmp_path / "videos" for p in got)


def test_split_video_reads_like_one(tmp_path):
    """分割ダウンロード（区切りがキーフレームと合わない）を parts.json でつなぐと、1 本の動画と同じフレームが同じ時刻に出る。"""
    import json
    import shutil
    import subprocess

    import cv2
    import pytest

    from vreplay import video as V
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg が必要")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=s=160x120:r=30,geq=lum='mod(N\\,200)+20':cb=128:cr=128", "-t", "40",
                    "-c:v", "libx264", "-preset", "veryfast", "-g", "60", "-sc_threshold", "0", "-qp", "0",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    parts = [(0, 22, 13), (13, 22, 14), (27, 13, 13)]        # (開始, 取得する長さ, 受け持つ長さ)
    for a, d, _ in parts:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-t", str(d), "-i", str(src), "-c", "copy",
                        str(tmp_path / f"p{a}.mp4")], check=True)
    man = tmp_path / f"v_0_40{V.PARTS}"
    man.write_text(json.dumps({"parts": [{"file": f"p{a}.mp4", "offset": a, "length": n} for a, _, n in parts]}))
    lum = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
    ref = {round(t, 2): lum(f) for t, f in V.iter_frames(src, 2, 0, 40, hwaccel=False)}
    got = {round(t, 2): lum(f) for t, f in V.iter_frames(man, 2, 0, 40, hwaccel=False)}
    assert sorted(ref) == sorted(got)
    assert all(abs(ref[t] - got[t]) < 3 for t in ref)
    assert V.probe(man)[2] == 40 and V.find_cached("v", 0, 40, [tmp_path]) == man
    V.remove_video(man)
    assert not any(tmp_path.glob("p*.mp4")) and not man.exists()


def test_range_download_matches_source(tmp_path, monkeypatch):
    """YouTube の DASH と同じ作り（ftyp・moov・sidx・moof/mdat）の動画から、必要な時間帯だけをバイト範囲で取ると、
    元の動画と同じフレームが同じ時刻（区間の先頭から）に出る。"""
    import re
    import shutil
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import cv2
    import pytest

    from vreplay import video as V
    from vreplay import ytrange
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg が必要")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=s=160x120:r=30,geq=lum='mod(N\\,200)+20':cb=128:cr=128", "-t", "90",
                    "-c:v", "libx264", "-preset", "veryfast", "-g", "150", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
                    "-movflags", "+frag_keyframe+empty_moov+global_sidx+default_base_moof", "-f", "mp4", str(src)],
                   check=True)
    data = src.read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            m = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
            a, b = (int(m.group(1)), min(int(m.group(2)), len(data) - 1)) if m else (0, len(data) - 1)
            body = data[a:b + 1]
            self.send_response(206 if m else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(ytrange, "CHUNK", 30000)
    monkeypatch.setattr(ytrange, "format_info", lambda *a, **k: {"url": f"http://127.0.0.1:{srv.server_port}/v",
                                                                  "headers": {}, "format_id": "137"})
    try:
        man = V.download_segment("vid", 32, 70, tmp_path / "out", method="range", range_threads=4)
    finally:
        srv.shutdown()
    assert man.name.endswith(V.PARTS)
    lum = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
    ref = {round(t - 32, 2): lum(f) for t, f in V.iter_frames(src, 2, 32, 36, hwaccel=False)}
    got = {round(t, 2): lum(f) for t, f in V.iter_frames(man, 2, 0, 36, hwaccel=False)}
    common = sorted(set(ref) & set(got))
    assert len(common) >= 70
    assert all(abs(ref[t] - got[t]) < 3 for t in common)


def test_range_getter_falls_back_to_query_and_uses_ytdlp(tmp_path):
    """Range ヘッダーを拒否するサーバーでも &range=a-b で取れる。yt-dlp の接続（プロセス内）でも取れる。"""
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    from vreplay import ytrange
    data = bytes(range(256)) * 64

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = parse_qs(urlparse(self.path).query).get("range")
            if self.headers.get("Range") or not q:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            a, b = map(int, re.match(r"(\d+)-(\d+)", q[0]).groups())
            body = data[a:b + 1]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v?id=1"
    try:
        import importlib.util
        variants = [None] + ([ytrange.ydl_options("bv*")] if importlib.util.find_spec("yt_dlp") else [])
        for opts in variants:
            g = ytrange.Getter(url, {}, opts)
            assert g.probe() == "query"
            assert g.get(100, 4195) == data[100:4196]
    finally:
        srv.shutdown()


def test_range_download_hls_segments(tmp_path, monkeypatch):
    """DASH が使えないクライアントでも、HLS（m3u8）の必要な断片だけを並列に取ってつなげる。"""
    import shutil
    import subprocess
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    import cv2
    import pytest

    from vreplay import video as V
    from vreplay import ytrange
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg が必要")
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=s=160x120:r=30,geq=lum='mod(N\\,200)+20':cb=128:cr=128", "-t", "90",
                    "-c:v", "libx264", "-preset", "veryfast", "-g", "150", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
                    str(src)], check=True)
    hls = tmp_path / "hls"
    hls.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy", "-f", "hls", "-hls_time", "5",
                    "-hls_list_size", "0", "-hls_segment_filename", str(hls / "s%03d.ts"), str(hls / "v.m3u8")],
                   check=True)

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(hls)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(V, "_RANGE_FAILS", 0)
    monkeypatch.setattr(ytrange, "format_info", lambda *a, **k: {
        "url": f"http://127.0.0.1:{srv.server_port}/v.m3u8", "headers": {}, "format_id": "96", "kind": "hls"})
    try:
        man = V.download_segment("vid", 32, 70, tmp_path / "out", method="range", range_threads=4)
    finally:
        srv.shutdown()
    j = __import__("json").loads(man.read_text())
    assert j["mode"] == "hls" and len(j["parts"]) == 1
    lum = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()
    ref = {round(t - 32, 2): lum(f) for t, f in V.iter_frames(src, 2, 32, 36, hwaccel=False)}
    got = {round(t, 2): lum(f) for t, f in V.iter_frames(man, 2, 0, 36, hwaccel=False)}
    common = sorted(set(ref) & set(got))
    assert len(common) >= 70
    assert all(abs(ref[t] - got[t]) < 3 for t in common)


def test_mixed_resolution_parts_are_resized(tmp_path, monkeypatch):
    import json

    import numpy as np

    from vreplay import video as V
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (tmp_path / name).write_bytes(b"x")
    man = tmp_path / ("v_0_300" + V.PARTS)
    man.write_text(json.dumps({"video_id": "v", "start": 0, "end": 300, "parts": [
        {"file": "a.mp4", "offset": 0, "length": 100}, {"file": "b.mp4", "offset": 100, "length": 100},
        {"file": "c.mp4", "offset": 200, "length": 100}]}), encoding="utf-8")
    sizes = {"a.mp4": (1920, 1080), "b.mp4": (1280, 720), "c.mp4": (1920, 1080)}
    real_probe = V.probe

    def fake_probe(p):
        n = getattr(p, "name", str(p).split("/")[-1])
        if n in sizes:
            return (*sizes[n], 112.0)
        return real_probe(p)
    monkeypatch.setattr(V, "probe", fake_probe)
    V._parts_size.cache_clear()
    assert V._parts_size(man) == (1920, 1080)
    fr = np.zeros((720, 1280, 3), np.uint8)
    assert V._fit(fr, (1920, 1080)).shape == (1080, 1920, 3)


def test_broken_part_is_skipped(tmp_path, monkeypatch):
    import json

    import numpy as np

    from vreplay import video as V
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"x")
    man = tmp_path / ("w_0_200" + V.PARTS)
    man.write_text(json.dumps({"video_id": "w", "start": 0, "end": 200, "parts": [
        {"file": "a.mp4", "offset": 0, "length": 100}, {"file": "b.mp4", "offset": 100, "length": 100}]}),
        encoding="utf-8")
    monkeypatch.setattr(V, "has_video", lambda p: str(p).endswith("a.mp4"))
    real = V.iter_keyframes

    def fake_kf(path, start=0.0, duration=None):
        if str(path).endswith(".mp4"):
            assert str(path).endswith("a.mp4"), "壊れた部分を読もうとした"
            yield start, np.zeros((4, 4, 3), np.uint8)
            return
        yield from real(path, start, duration)
    monkeypatch.setattr(V, "iter_keyframes", fake_kf)
    monkeypatch.setattr(V, "_parts_size", lambda p: (4, 4))
    got = list(V.iter_keyframes(man, 0, 200))
    assert len(got) == 1 and got[0][0] == 0


def _fake_ytdlp(monkeypatch, V, avail):
    """avail: {クライアント: 取れる高さ}。-f の [height=h] / [height<=h] を見てファイルを作る偽の yt-dlp。"""
    import re
    import types
    calls = []

    def run(cmd, **kw):
        client = None
        if "--extractor-args" in cmd:
            client = cmd[cmd.index("--extractor-args") + 1].split("=")[1]
        fmt = cmd[cmd.index("-f") + 1]
        calls.append((client, fmt))
        h = avail.get(client)
        exact = re.search(r"\[height=(\d+)\]", fmt)
        if h is None or (exact and int(exact.group(1)) != h):
            return types.SimpleNamespace(returncode=1, stderr="ERROR: Requested format is not available", stdout="")
        out = cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4")
        Path(out).write_text(str(h))
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")
    monkeypatch.setattr(V.subprocess, "run", run)
    monkeypatch.setattr(V, "_part_height", lambda f: int(Path(f).read_text() or 0) if Path(f).exists() else 0)
    monkeypatch.setattr(V, "has_video", lambda f: Path(f).exists())
    V._GOOD_CLIENT.clear()
    return calls


def test_download_prefers_exact_1080_over_other_clients(tmp_path, monkeypatch):
    """既定のクライアントが 360p しか出さなくても、1080p を出せるクライアントを先に探す。"""
    from vreplay import video as V
    calls = _fake_ytdlp(monkeypatch, V, {None: 360, "web_safari": 1080})
    f = V._download_one("vid", 0, 100, tmp_path, 1080, "", (), None, 4)
    assert f.read_text() == "1080"
    assert all("[height=1080]" in fmt for _, fmt in calls)          # 低い解像度は試していない
    assert V._GOOD_CLIENT["vid"] == "web_safari"
    # 次の部分は 1080p が取れたクライアントから試す
    calls.clear()
    V._download_one("vid", 100, 200, tmp_path, 1080, "", (), None, 4)
    assert calls[0][0] == "web_safari"


def test_download_falls_back_when_no_1080(tmp_path, monkeypatch):
    from vreplay import video as V
    _fake_ytdlp(monkeypatch, V, {None: 720})
    f = V._download_one("vid", 0, 100, tmp_path, 1080, "", (), None, 4)
    assert f.read_text() == "720"
    import pytest
    with pytest.raises(RuntimeError):
        V._download_one("vid", 100, 200, tmp_path, 1080, "", (), None, 4, strict=True)


def test_split_download_redoes_low_resolution_parts(tmp_path, monkeypatch):
    """前の実行で 360p になった部分が残っていても、1080p で取り直してそろえる。"""
    from vreplay import video as V
    _fake_ytdlp(monkeypatch, V, {"web_safari": 1080})
    (tmp_path / "vid_0_252.mp4").write_text("360")                   # 1 つ目の部分だけ 360p のキャッシュ
    man = V.download_segment("vid", 0, 480, tmp_path, 1080, method="sections", parts=2, overlap=12)
    hs = [f.read_text() for f, _, _ in V.parts_of(man)]
    assert hs == ["1080", "1080"]
