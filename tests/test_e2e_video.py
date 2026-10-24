"""実際の放送から切り出した HUD（上部バンド 88 枚＋カード）で 1080p の動画を作り、
キャリブレーション → 並列スキャン → ライブ判定 → ラウンド分割 → 素材づくりを通しで動かす（ダウンロードだけ省略）。"""
import shutil
import subprocess

import numpy as np
import pytest

from vreplay import config, pipeline, vlr

from .fx import FX, full_frame_from_panels, table_frame, tile_to_frame, topband

T0 = 27940.0
EXPECT = {1: 27959.2, 2: 28088.4, 3: 28207.7, 4: 28324.6, 5: 28403.3}
# 購入フェーズの装備表（実データの武器シルエット）。第 3 ラウンドは開始 1 秒前までリプレイが流れている想定
TABLE = {1: "t27955", 2: "t28085", 3: "t28400", 4: "t28400", 5: "t28400"}
REPLAY_UNTIL = {3: -1.0}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg が必要")
def test_analyze_video_on_real_hud(tmp_path):
    tiles, times = topband("half")
    panels = full_frame_from_panels("cn")
    vid = tmp_path / "v.mp4"
    dur = int(times[-1] - T0) + 6
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                          "-r", "1", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                          "-g", "2", "-pix_fmt", "yuv420p", str(vid)], stdin=subprocess.PIPE)
    k = 0
    drops = {n: t - T0 for n, t in EXPECT.items()}
    for s in range(dur):
        while k + 1 < len(times) and times[k + 1] - T0 <= s:
            k += 1
        f = tile_to_frame(tiles[k])
        n = next((n for n, d in drops.items() if d - 8 <= s <= d + 1), None)
        if n is not None and s < drops[n] + REPLAY_UNTIL.get(n, -99):
            f = np.full_like(f, 70)                                   # リプレイ・選手カメラ（HUD なし）
        elif n is not None:
            f = table_frame(TABLE[n], frame=f)                        # 購入フェーズ〜開始直後：装備表
        else:
            f[480:1080] = np.maximum(f[480:1080], panels[480:1080])   # 試合中：プレイヤーカード
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
    for n, t in EXPECT.items():
        assert "yt_url" in rs[n], (n, rs[n]["notes"])
        got = int(rs[n]["yt_url"].split("t=")[1]) + 5          # yt の秒 = 開始 - 5
        # 合成動画は 5〜8 秒ごとの静止画なので、実際の動画より開始時刻の推定が粗い
        assert abs(got - (t - T0)) <= 4, (n, got, t - T0)
        assert rs[n]["assets"].get("minimap_gif")
    # 武器は装備表から（第 3 ラウンドは開始直前までリプレイでも、残りの 1〜2 秒で読む）
    def weapons(n, no):
        return {x["player"]: (x["weapon"], x.get("sidearm")) for x in rs[n]["weapons"][no]}
    for n in (1, 2, 3):
        assert rs[n].get("weapons_source") == "装備表", (n, rs[n].get("notes"))
    assert weapons(1, "1") == {"Life": ("Ghost", "Ghost"), "Nicc": ("Ghost", "Ghost"), "vo0kashu": ("Ghost", "Ghost"),
                               "Flex1n": ("Sheriff", "Sheriff"), "SpiritZ1": ("Ghost", "Ghost")}
    assert weapons(2, "1")["vo0kashu"] == ("Outlaw", "Ghost") and weapons(2, "1")["Nicc"] == ("Spectre", None)
    assert {w for w, _ in weapons(2, "2").values()} == {"Classic"}
    assert weapons(3, "2")["kklin"] == ("Operator", "Classic") and weapons(3, "1")["Nicc"] == ("Vandal", "Shorty")
    assert rs[3]["assets"].get("buy_panels")
    assert "VLR との勝敗一致 100%" in rep["analysis_note"], rep["analysis_note"]
