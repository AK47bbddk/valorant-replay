"""実際の放送から切り出した HUD（上部バンド 88 枚＋カード）で 1080p の動画を作り、
キャリブレーション → 並列スキャン → ライブ判定 → ラウンド分割 → 素材づくりを通しで動かす（ダウンロードだけ省略）。"""
import shutil
import subprocess

import numpy as np
import pytest

from vreplay import config, pipeline, vlr

from .fx import FX, full_frame_from_panels, tile_to_frame, topband

T0 = 27940.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg が必要")
def test_analyze_video_on_real_hud(tmp_path):
    tiles, times = topband("half")
    panels = full_frame_from_panels("cn")
    vid = tmp_path / "v.mp4"
    dur = int(times[-1] - T0) + 6
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                          "-r", "1", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                          "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    k = 0
    for s in range(dur):
        while k + 1 < len(times) and times[k + 1] - T0 <= s:
            k += 1
        f = tile_to_frame(tiles[k])
        f[480:1080] = np.maximum(f[480:1080], panels[480:1080])
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()

    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["video"]["fps"] = 1
    cfg["video"]["workers"] = 2
    (tmp_path / "out" / "_cache").mkdir(parents=True)
    shutil.copy(FX / "weapons.npz", tmp_path / "out" / "_cache" / "weapons.npz")
    m = vlr.Match(id="777943", url="https://www.vlr.gg/777943/x", team1="Dragon Ranger Gaming", team2="Rare Atom",
                  tag1="DRG", tag2="RA")
    players = [vlr.Player(n, a, 1) for n, a in (("vo0kashu", "Cypher"), ("Life", "Yoru"), ("Nicc", "Fade"),
                                                ("SpiritZ1", "Waylay"), ("Flex1n", "Harbor"))]
    players += [vlr.Player(n, a, 2) for n, a in (("kklin", "Chamber"), ("Z1Yan", "Omen"), ("potter", "Neon"),
                                                 ("nnono", "Viper"), ("midnight", "Fade"))]
    md = vlr.MapData(order=1, name="Summit", game_id="1", score1=13, score2=8,
                     rounds=[vlr.Round(n, 1, "防衛", "elim") for n in range(1, 6)], players=players,
                     vod_id="yJbW1DqEJ7c", vod_t=20)
    m.maps = [md]
    rep = pipeline.base_report(m, md, cfg)
    pipeline.analyze_video(cfg, m, md, rep, tmp_path / "out" / "map", vid, log=print)

    assert rep["video_ok"], rep["analysis_note"]
    rs = {r["n"]: r for r in rep["rounds"]}
    expect = {1: 27959.2, 2: 28088.4, 3: 28207.7, 4: 28324.6, 5: 28403.3}
    for n, t in expect.items():
        assert "yt_url" in rs[n], (n, rs[n]["notes"])
        got = int(rs[n]["yt_url"].split("t=")[1]) + 5          # yt の秒 = 開始 - 5
        # 合成動画は 5〜8 秒ごとの静止画なので、実際の動画より開始時刻の推定が粗い
        assert abs(got - (t - T0)) <= 4, (n, got, t - T0)
        assert rs[n]["assets"].get("minimap_gif")
    # 静止させたカード（第 1 ラウンドの画面）：DRG の生存 3 人はゴースト
    w = rs[1]["weapons"]["1"]
    assert [x["weapon"] for x in w] == ["Ghost", "Ghost", "Ghost"]
    assert {x["player"] for x in w} == {"vo0kashu", "Life", "Nicc"}
    assert "VLR との勝敗一致 100%" in rep["analysis_note"], rep["analysis_note"]
