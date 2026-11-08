import json

from vreplay import quality as Q


def _rep(h, timer):
    rep = {"match_id": "1", "map": "Sunset", "order": 1, "tag1": "A", "tag2": "B", "video_ok": True,
           "analysis_note": f"レイアウト riot／タイマー読取 {timer}%・スコア読取 80%／検出ラウンド 17/18／点 8.2/枚",
           "tactics": {"aligned": True, "confidence": 0.8,
                       "rounds": {"1": {"snaps": {"0": {"1": {"src": "agents"}, "2": {"src": "dots"}}}}}},
           "rounds": [{"kill_sources": {"feed": 8, "card": 2}}]}
    if h:
        rep["video_size"] = [h * 16 // 9, h]
    return rep


def test_row_and_summary(tmp_path):
    for i, (h, t) in enumerate([(1080, 90), (720, 60), (None, 70)]):
        d = tmp_path / f"d{i}"
        d.mkdir()
        rep = _rep(h, t)
        rep["match_id"] = str(i)
        (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    rows = Q.collect(tmp_path)
    assert len(rows) == 3
    r = next(r for r in rows if r["height"] == 720)
    assert r["timer_ok"] == 0.6 and r["agents_snaps"] == 0.5 and r["feed_kills"] == 0.8
    assert abs(r["rounds_found"] - 17 / 18) < 1e-9 and r["dots_per_frame"] == 8.2
    sm = Q.summary(rows)
    assert [s["height"] for s in sm] == ["1080p", "720p", "不明"]
    out = Q.run({"output": {"dir": str(tmp_path)}}, log=lambda *a: None)
    assert out.exists()
