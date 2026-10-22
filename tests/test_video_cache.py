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

    def fake_dl(vid, s, e, out, h, cookies, log=None):
        calls.append((vid, s, e, out))
        p = out / f"{vid}_{s}_{e}.mp4"
        out.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p
    monkeypatch.setattr(video, "download_segment", fake_dl)
    got = pipeline.fetch(cfg, match_urls=[m.url], log=lambda *a: None)
    assert [c[1:3] for c in calls] == [(80, 3030), (2980, 2980 + 20 + 22 * 140 + 900)]
    assert all(p.parent == tmp_path / "videos" for p in got)
