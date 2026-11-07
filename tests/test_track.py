from vreplay import track as TR


def test_outlier_is_dropped():
    # 0.03/秒で右へ歩く。t=10 だけ反対側（スキルの印など）を拾った
    dets = [(t, 0.1 + 0.03 * t, 0.5, 0.7) for t in range(0, 31, 5)]
    dets.append((10.0, 0.9, 0.1, 0.65))
    ch = TR.best_chain(dets, scale=0.8)
    assert all(abs(y - 0.5) < 1e-6 for _, _, y, _ in ch)
    assert len(ch) == 7


def test_resample_interpolates_and_respects_gap():
    ch = [(0.0, 0.1, 0.1, 0.8), (10.0, 0.3, 0.1, 0.8), (40.0, 0.5, 0.1, 0.8)]
    pts = TR.resample(ch, [0.0, 5.0, 10.0, 20.0, 40.0])
    by = {t: (x, src) for t, x, _, _, src in pts}
    assert by[5.0][1] == "interp" and abs(by[5.0][0] - 0.2) < 1e-6
    assert 20.0 not in by                    # 30 秒の空きは補間しない
    assert by[40.0][1] == "det"


def test_build_cuts_at_death_and_teleport_agents():
    frames = [(100.0 + t, [{"agent": "Omen", "team": 1, "x": 0.1 if t < 15 else 0.8, "y": 0.5, "score": 0.7},
                           {"agent": "Jett", "team": 2, "x": 0.5, "y": 0.2 + 0.01 * t, "score": 0.7}])
              for t in range(0, 41, 5)]
    out = TR.build(frames, 100.0, 140.0, scale=0.8, deaths={(2, "Jett"): 120.0})
    # Omen はテレポートできるので、遠くへ跳んだ後もつながる
    assert max(p[1] for p in out[(1, "Omen")]) > 0.7 and min(p[1] for p in out[(1, "Omen")]) < 0.2
    assert max(p[0] for p in out[(2, "Jett")]) <= 20.0
    assert TR.at(out[(2, "Jett")], 10.0)[0] == 10.0
