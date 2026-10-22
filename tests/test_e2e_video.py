"""合成動画で analyze_video を通しで動かす（ダウンロードと OCR だけ差し替え）。
HUD の数字は ROI の明るさで埋め込み、FakeOCR が明るさから文字列に戻す。"""
import shutil
import subprocess

import numpy as np
import pytest

from vreplay import config, pipeline, vlr
from vreplay.config import roi_px

from .test_rounds import _timeline

W, H = 640, 360


class BrightnessOCR:
    def line(self, img, allow=None):
        v = float(np.mean(img))
        if allow and ":" in allow:                    # タイマー
            if v < 6:
                return ""
            s = int(round(v / 2)) - 5
            return f"{s // 60}:{s % 60:02d}"
        return str(int(round(v / 10)) - 1)          # スコア

    def detect(self, img, allow=None, mag=1.5):
        return []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg が必要")
def test_analyze_video_synthetic(tmp_path, monkeypatch):
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    cfg["hud"]["auto_calibrate"] = False
    roi = cfg["hud"]["roi"]
    samples = _timeline()
    vid = tmp_path / "synthetic.mp4"
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", "2", "-i", "-", "-c:v", "libx264rgb", "-crf", "0", str(vid)], stdin=subprocess.PIPE)
    base = (np.random.default_rng(1).random((H, W, 3)) * 60).astype(np.uint8)
    for k, s in enumerate(samples):
        f = np.roll(base, k % 7, axis=1)
        def fill(name, val):
            x0, y0, x1, y1 = roi_px(roi[name], W, H)
            f[y0:y1, x0:x1] = val
        fill("timer", 0 if s.timer is None else (int(s.timer) + 5) * 2)
        fill("score_left", (s.sl + 1) * 10)
        fill("score_right", (s.sr + 1) * 10)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()

    import vreplay.video as V
    monkeypatch.setattr(V, "download_segment", lambda *a, **k: vid)
    m = vlr.Match(id="1", url="https://www.vlr.gg/1/x", team1="A", team2="B")
    md = vlr.MapData(order=1, name="Bind", game_id="1", score1=1, score2=2,
                     rounds=[vlr.Round(1, 1), vlr.Round(2, 2), vlr.Round(3, 2)], vod_id="vid", vod_t=100)
    rep = pipeline.base_report(m, md, cfg)
    pipeline.analyze_video(cfg, m, md, rep, tmp_path / "out", lambda: BrightnessOCR(), log=lambda *a: None)
    rs = rep["rounds"]
    assert [r["n"] for r in rs] == [1, 2, 3]
    assert all(r["assets"].get("minimap_gif") for r in rs)
    assert all(r["assets"].get("buy_panels") for r in rs[1:])
    assert rs[0]["yt_url"].startswith("https://youtu.be/vid?t=")
    assert "画面左 = A" in rep["analysis_note"], rep["analysis_note"]
