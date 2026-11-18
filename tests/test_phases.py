"""段階A（イベント列の抽出と型の分類）の固定。合成 tracks だけで、ネットワークも実データも使わない。"""
from vreplay import phases as P
from vreplay import tactics as TC


def S(e, zone, callout=None):
    return [e, 0.0, 0.0, 0.9, "det", zone, callout if callout is not None else zone]


def tracks(team1, team2):
    """team1/team2: {エージェント: [(経過秒, ゾーン)] または [(経過秒, ゾーン, コールアウト)]}"""
    def build(sched):
        out = {}
        for ag, pts in sched.items():
            out[ag] = [S(*p) if len(p) == 3 else S(p[0], p[1]) for p in pts]
        return out
    return {"1": build(team1), "2": build(team2)}


def rnd(tr, **kw):
    r = {"n": 1, "side1": "攻撃", "duration": "1:40", "t_drop": 100.0,
         "winner": "Alpha", "side": "攻撃", "winner_no": 1, "kills": [],
         "tactics": {"tracks": tr, "plant_site": None}}
    r.update(kw)
    return r


def still(places, es=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90)):
    """5 人を固定のエリアに置く（[{エージェント: [(e, ゾーン)]}]）。"""
    return tracks({ag: [(e, p) for e in es] for ag, p in zip(("Jett", "Omen", "Sova", "Viper", "Cypher"), places)},
                  {ag: [(e, "防衛側") for e in es] for ag in ("Raze", "Sage", "Fade", "Killjoy", "Skye")})


def kinds(ev, kind):
    return [e for e in ev if e.kind == kind]


# ---------------------------------------------------------------- イベント
def test_take_event():
    tr = tracks({"Jett": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A")],
                 "Omen": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A")],
                 "Sova": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A")],
                 "Viper": [(0, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側")],
                 "Cypher": [(0, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側")]},
                {"Raze": [(0, "防衛側"), (10, "防衛側"), (15, "防衛側"), (20, "防衛側")]})
    ev = P.round_events(rnd(tr))
    takes = kinds(ev, "テイク")
    assert any(e.zone == "A" and e.side == "攻撃" and e.t == 10 for e in takes), ev


def test_retake_event():
    tr = tracks({"Jett": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A"), (25, "Mid"), (30, "Mid")],
                 "Omen": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A"), (25, "Mid"), (30, "Mid")],
                 "Sova": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A"), (25, "Mid"), (30, "Mid")],
                 "Viper": [(0, "攻撃側"), (10, "攻撃側"), (20, "Mid"), (25, "Mid"), (30, "Mid")],
                 "Cypher": [(0, "攻撃側"), (10, "攻撃側"), (20, "Mid"), (25, "Mid"), (30, "Mid")]},
                {"Raze": [(0, "防衛側"), (10, "防衛側"), (20, "防衛側"), (25, "A"), (30, "A")],
                 "Sage": [(0, "防衛側"), (10, "防衛側"), (20, "防衛側"), (25, "A"), (30, "A")],
                 "Fade": [(0, "防衛側"), (10, "防衛側"), (20, "防衛側"), (25, "A"), (30, "A")],
                 "Killjoy": [(0, "防衛側"), (10, "防衛側"), (20, "防衛側"), (25, "防衛側"), (30, "防衛側")],
                 "Skye": [(0, "防衛側"), (10, "防衛側"), (20, "防衛側"), (25, "防衛側"), (30, "防衛側")]})
    ev = P.round_events(rnd(tr))
    assert any(e.kind == "テイク" and e.zone == "A" and e.side == "攻撃" and e.t == 10 for e in ev), ev
    assert any(e.kind == "取り返し" and e.zone == "A" and e.side == "防衛" and e.t == 25 for e in ev), ev


def test_initial_config_is_not_dominance():
    """守りが開始時に B に 2 人いても初期配置は支配に数えない。攻めが B を取るのはテイク。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (15, "B"), (20, "B")],
        "Omen": [(0, "攻撃側"), (15, "B"), (20, "B")],
        "Sova": [(0, "攻撃側"), (15, "B"), (20, "B")],
        "Viper": [(0, "攻撃側"), (15, "攻撃側"), (20, "攻撃側")],
        "Cypher": [(0, "攻撃側"), (15, "攻撃側"), (20, "攻撃側")],
    }, {
        "Raze": [(0, "B"), (15, "B"), (20, "B")],
        "Sage": [(0, "B"), (15, "B"), (20, "B")],
        "Fade": [(0, "防衛側"), (15, "防衛側"), (20, "防衛側")],
    })
    ev = P.round_events(rnd(tr))
    bs = [e for e in ev if e.zone == "B"]
    assert any(e.kind == "テイク" and e.side == "攻撃" for e in bs), ev
    assert not any(e.kind == "取り返し" for e in bs), ev


def test_rotate_event():
    es = (0, 10, 20, 30, 40)
    tr = tracks({ag: [(0, "攻撃側"), (10, "A"), (20, "A"), (30, "B"), (40, "B")] for ag in ("Jett", "Omen", "Sova")}
                | {ag: [(0, "攻撃側"), (10, "攻撃側"), (30, "B"), (40, "B")] for ag in ("Viper", "Cypher")},
                {ag: [(e, "防衛側") for e in es] for ag in ("Raze", "Sage")})
    ev = P.round_events(rnd(tr))
    assert any(e.kind == "ローテート" and e.zone == "B" and e.side == "攻撃" and e.t == 30 for e in ev), ev


def test_rotate_needs_two_samples_and_cancels_revert():
    def with_schedule(sched):
        return tracks({ag: sched for ag in ("Jett", "Omen", "Sova", "Viper", "Cypher")},
                      {ag: [(0, "防衛側")] for ag in ("Raze", "Sage")})

    # 1 サンプルだけ B に寄ってもローテートにしない
    one = with_schedule([(0, "攻撃側"), (10, "A"), (15, "B"), (20, "A")])
    assert not [e for e in P.round_events(rnd(one)) if e.kind == "ローテート" and e.zone == "B"]

    # 2 サンプル続いても、10 秒以内に元の A へ戻ったら取り消す
    back = with_schedule([(0, "攻撃側"), (10, "A"), (15, "B"), (20, "B"), (25, "A"), (30, "A")])
    rots = [e for e in P.round_events(rnd(back)) if e.kind == "ローテート"]
    assert not any(e.zone == "B" for e in rots), rots


def test_consecutive_same_zone_events_merged():
    """同じゾーンのテイクが連続したら 1 つにまとめる。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A"), (25, "A"), (30, "A")],
        "Omen": [(0, "攻撃側"), (10, "A"), (15, "A"), (20, "A"), (25, "A"), (30, "A")],
        "Sova": [(0, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側"), (25, "攻撃側"), (30, "攻撃側")],
        "Viper": [(0, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側"), (25, "攻撃側"), (30, "攻撃側")],
        "Cypher": [(0, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側"), (25, "攻撃側"), (30, "攻撃側")],
    }, {
        "Raze": [(0, "防衛側"), (10, "防衛側"), (15, "防衛側"), (20, "A"), (25, "防衛側"), (30, "防衛側")],
        "Sage": [(0, "防衛側"), (10, "防衛側"), (15, "防衛側"), (20, "A"), (25, "防衛側"), (30, "防衛側")],
    })
    ev = P.round_events(rnd(tr))
    takes = [e for e in ev if e.kind == "テイク" and e.zone == "A"]
    assert len(takes) == 1 and takes[0].t == 10, ev


def test_entry_event():
    tr = tracks({"Jett": [(0, "攻撃側"), (10, "A", "A Site"), (15, "A", "A Site")],
                 "Omen": [(0, "攻撃側"), (10, "A", "A Site"), (15, "A", "A Site")],
                 "Sova": [(0, "攻撃側"), (10, "A", "A Site"), (15, "A", "A Site")],
                 "Viper": [(0, "攻撃側"), (10, "A Main"), (15, "A Main")],
                 "Cypher": [(0, "攻撃側"), (10, "A Main"), (15, "A Main")]},
                {"Raze": [(0, "防衛側"), (10, "防衛側"), (15, "防衛側")]})
    ev = P.round_events(rnd(tr))
    assert any(e.kind == "エントリー" and e.zone == "A" and e.t == 10 for e in ev), ev


def test_freeze_boundary_10s():
    # ピックの後、次イベント（支配交代）まで 10 秒 → フリーズ。9 秒 → フリーズでない
    def with_gap(gap):
        t = 20 + gap
        tr = tracks({
            "Jett": [(0, "攻撃側"), (20, "A"), (t, "B"), (t + 5, "B")],
            "Omen": [(0, "攻撃側"), (20, "A"), (t, "B"), (t + 5, "B")],
            "Sova": [(0, "攻撃側"), (20, "A"), (t, "B"), (t + 5, "B")],
            "Viper": [(0, "攻撃側"), (20, "A"), (t, "A"), (t + 5, "A")],
            "Cypher": [(0, "攻撃側"), (20, "A"), (t, "A"), (t + 5, "A")],
        }, {"Raze": [(0, "防衛側")]})
        return P.classify_flow(rnd(tr, kills=[{"t": 120.0, "killer_team": "Alpha"}]))

    assert with_gap(10.0)["phases"]["mid"]["type"] == "フリーズ"
    assert with_gap(9.0)["phases"]["mid"]["type"] != "フリーズ"


def test_rush_boundary_15s():
    def entry_at(t):
        tr = tracks({"Jett": [(0, "攻撃側"), (t, "A", "A Site")],
                     "Omen": [(0, "攻撃側"), (t, "A", "A Site")],
                     "Sova": [(0, "攻撃側"), (t, "A", "A Site")]},
                    {"Raze": [(0, "防衛側")]})
        return P.classify_flow(rnd(tr))

    assert entry_at(15.0)["phases"]["early"]["type"] == "ラッシュ"
    assert entry_at(15.0)["phases"]["early"]["zone"] == "A"
    assert entry_at(16.0)["phases"]["early"]["type"] != "ラッシュ"


def test_fast_group_without_entry_is_default():
    """4 人が速くサイト側のゾーンへ寄っても、サイトへエントリーしなければラッシュにしない（寄った先のデフォルト）。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (15, "A", "A Main"), (20, "A", "A Main")],
        "Omen": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (15, "A", "A Main"), (20, "A", "A Main")],
        "Sova": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (15, "A", "A Main"), (20, "A", "A Main")],
        "Viper": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (15, "A", "A Main"), (20, "A", "A Main")],
        "Cypher": [(0, "攻撃側"), (5, "攻撃側"), (10, "攻撃側"), (15, "攻撃側"), (20, "攻撃側")],
    }, {ag: [(0, "防衛側")] for ag in ("Raze", "Sage", "Fade", "Killjoy", "Skye")})
    res = P.classify_flow(rnd(tr))
    assert res["phases"]["early"]["type"] == "Aデフォルト", res["phases"]["early"]
    assert not any(e["kind"] == "ラッシュ" for e in res["events"]), res["events"]


def test_slow_default_is_not_rush():
    """開幕にゾーンへ散っただけ（サイトエントリーも 4 人集結もない）のラウンドはラッシュにしない。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (5, "A"), (10, "A"), (15, "A"), (20, "A")],
        "Omen": [(0, "攻撃側"), (5, "A"), (10, "A"), (15, "A"), (20, "A")],
        "Sova": [(0, "攻撃側"), (5, "Mid"), (10, "Mid"), (15, "Mid"), (20, "Mid")],
        "Viper": [(0, "攻撃側"), (5, "Mid"), (10, "Mid"), (15, "Mid"), (20, "Mid")],
        "Cypher": [(0, "攻撃側"), (5, "B"), (10, "B"), (15, "B"), (20, "B")],
    }, {ag: [(0, "防衛側")] for ag in ("Raze", "Sage", "Fade", "Killjoy", "Skye")})
    res = P.classify_flow(rnd(tr))
    early = res["phases"]["early"]["type"]
    assert early == "Aデフォルト", res["phases"]["early"]
    assert early != "ラッシュ"


def test_fake_vs_regroup_priority():
    """フェイクとリグループが両方当てはまるときは、リグループセットを主にする。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (10, "A"), (15, "Mid"), (20, "B"), (25, "B"), (35, "A", "A Site")],
        "Omen": [(0, "攻撃側"), (10, "Mid"), (15, "Mid"), (20, "B"), (25, "B"), (35, "A", "A Site")],
        "Sova": [(0, "攻撃側"), (10, "Mid"), (15, "Mid"), (20, "B"), (25, "B"), (35, "A", "A Site")],
        "Viper": [(0, "攻撃側"), (10, "A"), (15, "Mid"), (20, "B"), (25, "B"), (35, "A", "A Site")],
        "Cypher": [(0, "攻撃側"), (10, "Mid"), (15, "Mid"), (20, "Mid"), (25, "Mid"), (35, "攻撃側")],
    }, {
        "Raze": [(0, "A"), (10, "A"), (20, "A"), (25, "B"), (30, "B"), (35, "B")],
        "Sage": [(0, "A"), (10, "A"), (20, "A"), (25, "B"), (30, "B"), (35, "B")],
        "Fade": [(0, "A"), (10, "A"), (20, "A"), (25, "B"), (30, "B"), (35, "B")],
        "Killjoy": [(0, "防衛側"), (10, "A"), (20, "A"), (25, "A"), (30, "A"), (35, "A")],
    })
    res = P.classify_flow(rnd(tr))
    assert res["phases"]["mid"]["type"] == "リグループセット", res["phases"]
    assert "フェイクセット" in res["phases"]["mid"]["candidates"], res["phases"]


def _ranks():
    calls = [
        {"name": "Site", "super": "A", "x": 0.30, "y": 0.25},
        {"name": "Heaven", "super": "A", "x": 0.36, "y": 0.10},
        {"name": "Main", "super": "A", "x": 0.20, "y": 0.70},
        {"name": "Spawn", "super": "Attacker Side", "x": 0.50, "y": 0.95},
        {"name": "Spawn", "super": "Defender Side", "x": 0.50, "y": 0.05},
    ]
    return TC.area_ranks(calls)


def test_late_deep_push():
    """サイトより奥のエリアへ進んだら、終盤の型はプッシュ。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (10, "A", "A Site"), (20, "A", "A Site"), (30, "A", "A Site"),
                 (40, "A", "Heaven"), (50, "A", "Heaven")],
        "Omen": [(0, "攻撃側"), (10, "A", "A Site"), (20, "A", "A Site"), (30, "A", "A Site"),
                 (40, "A", "Heaven"), (50, "A", "Heaven")],
        "Sova": [(0, "攻撃側"), (10, "A", "A Site"), (20, "A", "A Site"), (30, "A", "A Site"),
                 (40, "A", "A Site"), (50, "A", "A Site")],
        "Viper": [(0, "攻撃側"), (10, "A Main"), (20, "A Main"), (30, "A Main"), (40, "A Main"), (50, "A Main")],
    }, {"Raze": [(0, "防衛側"), (40, "防衛側")]})
    r = rnd(tr, plant_t=130.0, plant_site="A")
    res = P.classify_flow(r, ranks=_ranks())
    assert res["phases"]["late"]["type"] == "プッシュ", res["phases"]["late"]
    assert any(e["kind"] == "プッシュ" and e["side"] == "攻撃" for e in res["events"]), res["events"]
    flows = res["flow"]
    assert isinstance(flows, dict) and set(flows) == {"early", "mid", "late"}, flows
    assert "エントリー" in flows["early"], flows
    assert "エントリー" not in flows["late"], flows


def test_no_tracks_returns_none():
    r = {"n": 1, "side1": "攻撃", "tactics": {}}
    assert P.classify_flow(r) is None
    assert P.round_events(r) == []
    assert P.skip_reason(r) == "no_tracks"


def test_flow_text_when_classified():
    tr = still(["攻撃側", "A", "A", "Mid", "Mid"])
    res = P.classify_flow(rnd(tr))
    assert res is not None
    assert isinstance(res["flow"], dict) and set(res["flow"]) == {"early", "mid", "late"}
    assert all(isinstance(v, str) for v in res["flow"].values())
    assert res["segments"]["early"]["label"] == "序盤"


# ---------------------------------------------------------------- 段階D: 視点修正・攻め分類・暫定バリアフィルタ
def test_defense_perspective_early_mid_late():
    """D0: 防衛ラウンドは防衛チームの分布・イベントから型が決まる（攻撃の分布に引っ張られない）。"""
    # 攻撃(team1)は A に集まっているが、防衛(team2)は B に3人置いている → 防衛序盤は B寄せ
    tr = tracks(
        {
            "Jett": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (20, "A", "A Main"), (35, "A", "A Site")],
            "Omen": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (20, "A", "A Main"), (35, "A", "A Site")],
            "Sova": [(0, "攻撃側"), (5, "A", "A Main"), (10, "A", "A Main"), (20, "A", "A Main"), (35, "A", "A Main")],
            "Viper": [(0, "攻撃側"), (5, "攻撃側"), (10, "攻撃側"), (20, "攻撃側"), (35, "攻撃側")],
            "Cypher": [(0, "攻撃側"), (5, "攻撃側"), (10, "攻撃側"), (20, "攻撃側"), (35, "攻撃側")],
        },
        {
            "Raze": [(0, "B", "B Site"), (5, "B", "B Site"), (10, "B", "B Site"), (20, "A", "A Site"), (25, "A", "A Site"), (35, "A", "A Site"), (40, "A", "A Site")],
            "Sage": [(0, "B", "B Site"), (5, "B", "B Site"), (10, "B", "B Site"), (20, "A", "A Site"), (25, "A", "A Site"), (35, "A", "A Site"), (40, "A", "A Site")],
            "Fade": [(0, "B", "B Site"), (5, "B", "B Site"), (10, "B", "B Site"), (20, "A", "A Site"), (25, "A", "A Site"), (35, "A", "A Site"), (40, "A", "A Site")],
            "Killjoy": [(0, "A", "A Site"), (5, "A", "A Site"), (10, "A", "A Site"), (20, "A", "A Site"), (25, "A", "A Site"), (35, "A", "A Site"), (40, "A", "A Site")],
            "Skye": [(0, "Mid", "Mid Top"), (5, "Mid", "Mid Top"), (10, "Mid", "Mid Top"), (20, "A", "A Site"), (25, "A", "A Site"), (35, "A", "A Site"), (40, "A", "A Site")],
        },
    )
    r = rnd(tr, plant_t=130.0, plant_site="A", kills=[{"t": 112.0, "killer_team": "Alpha"}])
    res = P.classify_flow(r, side=P.DEFENSE)
    assert res["phases"]["early"]["type"] == "B寄せ", res["phases"]["early"]
    assert res["phases"]["mid"]["type"] in ("ローテート", "取り返し", "上がり"), res["phases"]["mid"]
    assert res["phases"]["late"]["type"] in ("リテイク", "寄せ", "寄せ／リテイク"), res["phases"]["late"]


def test_attack_entry_set_vs_split():
    """D1: 進入路エリアの種類数で セット(1種) / スプリット(2種以上) を判定する。"""
    tr_set = tracks(
        {
            "Jett": [(0, "攻撃側"), (15, "A", "A Main"), (20, "A", "A Site")],
            "Omen": [(0, "攻撃側"), (15, "A", "A Main"), (20, "A", "A Site")],
            "Sova": [(0, "攻撃側"), (15, "A", "A Main"), (20, "A", "A Main")],
        },
        {"Raze": [(0, "防衛側")]},
    )
    res_set = P.classify_flow(rnd(tr_set), side=P.ATTACK)
    assert res_set["phases"]["entry"]["type"] == "Aセット", res_set["phases"]["entry"]

    tr_split = tracks(
        {
            "Jett": [(0, "攻撃側"), (15, "A", "A Main"), (20, "A", "A Site")],
            "Omen": [(0, "攻撃側"), (15, "A", "A Ramps"), (20, "A", "A Site")],
            "Sova": [(0, "攻撃側"), (15, "Mid", "Mid Vent"), (20, "A", "A Site")],
        },
        {"Raze": [(0, "防衛側")]},
    )
    res_split = P.classify_flow(rnd(tr_split), side=P.ATTACK)
    assert res_split["phases"]["entry"]["type"] == "Aスプリット", res_split["phases"]["entry"]


def test_attack_early_5_vocab():
    """E1: 攻め開幕の型は取りに行くエリアに基づく Split 語彙（A/ランプ/ミッド/Bメインデフォルト＋ラッシュ/詰め待ち）に分類される。"""
    tr_mid = still(["攻撃側", "Mid", "Mid", "A", "B"])
    assert P.classify_flow(rnd(tr_mid))["phases"]["early"]["type"] == "ミッドデフォルト"
    tr_wait = still(["攻撃側", "攻撃側", "攻撃側", "攻撃側", "攻撃側"])
    assert P.classify_flow(rnd(tr_wait))["phases"]["early"]["type"] == "詰め待ち"

    tr_ramp = tracks({
        "Jett": [(0, "攻撃側"), (5, "A", "A Ramps"), (10, "A", "A Ramps")],
        "Omen": [(0, "攻撃側"), (5, "A", "A Ramps"), (10, "A", "A Ramps")],
        "Sova": [(0, "攻撃側"), (5, "攻撃側"), (10, "攻撃側")],
    }, {"Raze": [(0, "防衛側")]})
    assert P.classify_flow(rnd(tr_ramp))["phases"]["early"]["type"] == "ランプデフォルト"


def test_early_b_main_take_then_regroup_a_is_b_main_default():
    """E1: 合成データで『開幕はBメインを取り、最終的にAへ攻めたラウンド』が Bメインデフォルト に分類される。"""
    tr = tracks({
        "Jett": [(0, "攻撃側"), (5, "B", "B Garage"), (10, "B", "B Garage"), (20, "Mid", "Mid Bottom"), (35, "A", "A Main"), (40, "A", "A Site")],
        "Omen": [(0, "攻撃側"), (5, "B", "B Garage"), (10, "B", "B Garage"), (20, "Mid", "Mid Bottom"), (35, "A", "A Main"), (40, "A", "A Site")],
        "Sova": [(0, "攻撃側"), (5, "B", "B Garage"), (10, "B", "B Garage"), (20, "A", "A Main"), (35, "A", "A Main"), (40, "A", "A Site")],
        "Viper": [(0, "攻撃側"), (5, "B", "B Lobby"), (10, "B", "B Lobby"), (20, "A", "A Lobby"), (35, "A", "A Main"), (40, "A", "A Main")],
        "Cypher": [(0, "攻撃側"), (5, "B", "B Lobby"), (10, "B", "B Lobby"), (20, "A", "A Lobby"), (35, "A", "A Main"), (40, "A", "A Main")],
    }, {ag: [(0, "防衛側")] for ag in ("Raze", "Sage", "Fade", "Killjoy", "Skye")})
    res = P.classify_flow(rnd(tr, plant_t=145.0, plant_site="A"))
    assert res["phases"]["early"]["type"] == "Bメインデフォルト", res["phases"]["early"]
    assert res["phases"]["early"]["zone"] == "B"
    assert res["phases"]["entry"]["type"] == "Aセット", res["phases"]["entry"]


def test_barrier_filter_drops_outside_home_and_jumps():
    """D2: 最初のサンプルが自陣外にある場合と、開幕ウィンドウ内のジャンプ外れ値が落ちる。"""
    calls = [
        {"name": "Site", "super": "A", "x": 0.30, "y": 0.25},
        {"name": "Main", "super": "A", "x": 0.25, "y": 0.70},
        {"name": "Site", "super": "B", "x": 0.70, "y": 0.25},
        {"name": "Spawn", "super": "Attacker Side", "x": 0.50, "y": 0.95},
        {"name": "Spawn", "super": "Defender Side", "x": 0.50, "y": 0.05},
    ]
    ranks = TC.area_ranks(calls)
    # Jett: t=0 が A Site (自陣外) → 落ちる。t=5 が A Main (自陣内) → 開幕位置。t=10 が B Site (ジャンプ>0.40) → 落ちる。
    tr = {
        "1": {
            "Jett": [
                [0.0, 0.30, 0.25, 0.9, "det", "A", "A Site"],
                [5.0, 0.25, 0.70, 0.9, "det", "A", "A Main"],
                [10.0, 0.70, 0.25, 0.9, "det", "B", "B Site"],
                [15.0, 0.24, 0.68, 0.9, "det", "A", "A Main"],
            ]
        },
        "2": {"Raze": [[0.0, 0.50, 0.05, 0.9, "det", "防衛側", "Spawn"]]},
    }
    stats = {}
    norm = P._norm_tracks(tr, sides={1: P.ATTACK, 2: P.DEFENSE}, ranks=ranks, filter_stats=stats)
    pts = norm[1]["Jett"]
    assert [p[0] for p in pts] == [5.0, 15.0]
    assert stats["outside_home"] == 1
    assert stats["jump"] == 1
    assert stats["dropped"] == 2

    # 定義のないマップ（ranks=None）ではフィルタは無効
    stats_none = {}
    norm_none = P._norm_tracks(tr, sides={1: P.ATTACK, 2: P.DEFENSE}, ranks=None, filter_stats=stats_none)
    assert [p[0] for p in norm_none[1]["Jett"]] == [0.0, 5.0, 10.0, 15.0]
    assert stats_none == {}
