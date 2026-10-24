from datetime import datetime

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
