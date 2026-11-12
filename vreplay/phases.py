"""ラウンドを「エリア支配のイベント列」として読み、序盤・中盤・終盤の区間と型を機械で付ける。

プレイブック v5 の説明を、静止した配置の型から「アクション→リアクションの応酬の流れ」に
作り替えるための土台（段階A）。組み込み先は段階Bで、ここは純粋な判定だけ。

入力（report.json のラウンド要素）:
- `tactics.tracks`: {チーム: {エージェント: [[経過秒, x, y, conf, src, zone, callout], ...]}}。
  経過秒はラウンド開始（バリア解除）からの秒。位置→エリアの変換は生成時に終わっている。
- `kills`: 時刻つきのキル（`t` は絶対秒、`killer_team` はチーム名）。
- `plant_t`・`t_drop`・`side1`・`winner`・`side` など。

tracks を持たないラウンドは対象外とし、`classify_flow` は None を返す。理由は
`skip_reason(round_dict)` で区別できる（落とさず・混ぜない）。

数値（しきい値・秒数）はモジュール先頭の定数にまとめる。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from . import tactics as TC

ATTACK, DEFENSE = "攻撃", "防衛"
SITES = ("A", "B", "C")
ZONES = ("A", "Mid", "B", "C")

SAMPLE_SECONDS = 5.0          # 軌跡のサンプル間隔（tracks はこの間隔で作られる。時間の欠損時の補いに使う）
DOMINANCE_MIN = 2             # ゾーンを支配というのに必要な人数（相手より多く、2 人以上）
ENTRY_MIN = 2                 # サイトへ入った（エントリー）とみなす人数
PUSH_MIN = 2                  # 前進（プッシュ）とみなす人数
REGROUP_MIN = 3               # リグループ（1 ゾーンに集まり直し）とみなす人数
RUSH_SECONDS = 15.0           # 開幕からこの秒以内のエントリーはラッシュ
FREEZE_SECONDS = 10.0         # ピック後に動きが無いままこの秒数でフリーズ区間

# 流れ要約に出すイベント（ラッシュ・フリーズは型として出すので、要約には入れない）
FLOW_KINDS = ("プッシュ", "テイク", "取り返し", "ローテート", "エントリー", "ピック")
# 同じ時刻の並び順（支配の変化 → 動き → エントリー → ピック）
KIND_ORDER = {"プッシュ": 0, "テイク": 1, "取り返し": 1, "ローテート": 2,
              "エントリー": 3, "ピック": 4, "ラッシュ": 5, "フリーズ": 6}


@dataclass(frozen=True)
class Event:
    """時刻・種別・ゾーン・側を持つイベント。t はラウンド開始からの秒。"""
    t: float
    kind: str
    zone: str | None = None
    side: str | None = None

    def to_dict(self) -> dict:
        return {"t": round(float(self.t), 1), "kind": self.kind, "zone": self.zone, "side": self.side}


# ---------------------------------------------------------------- 小道具
def _canon(side: str | None) -> str | None:
    if not side:
        return None
    if "攻" in side:
        return ATTACK
    if "防" in side or "衛" in side or "衞" in side:
        return DEFENSE
    return None


def _opp(side: str | None) -> str | None:
    s = _canon(side)
    return DEFENSE if s == ATTACK else ATTACK if s == DEFENSE else None


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _clock(text: str | None) -> float | None:
    if not text or ":" not in str(text):
        return None
    try:
        m, s = str(text).split(":")[:2]
        return int(m) * 60 + float(s)
    except ValueError:
        return None


def _snap(e: float) -> float:
    """同じフレームのサンプルをまとめるため、経過秒を 0.1 秒に丸める。"""
    return round(float(e), 1)


def _team_no(key) -> int | None:
    if key in (1, "1"):
        return 1
    if key in (2, "2"):
        return 2
    return None


def _zone_of_sample(p) -> str:
    z = p[5] if len(p) > 5 else None
    return str(z) if z else "?"


def _callout_of_sample(p) -> str | None:
    c = p[6] if len(p) > 6 else None
    return str(c) if c else None


# ---------------------------------------------------------------- 入力の正規化
def tracks_of(round_dict: dict) -> dict:
    """ラウンド要素から tracks を取り出す（`tactics.tracks` 優先、無ければ `tracks`）。"""
    if not isinstance(round_dict, dict):
        return {}
    tact = round_dict.get("tactics")
    if isinstance(tact, dict) and tact.get("tracks"):
        return tact["tracks"] or {}
    return round_dict.get("tracks") or {}


def _norm_tracks(tracks: dict) -> dict:
    """tracks を {チーム番号: {エージェント: [サンプル]}} にそろえる。

    実データは `{チーム: {エージェント: [...]}}`。エージェント直下にサンプルが並ぶ形も受ける
    （そのときはチーム不明の 0 に入れる）。
    """
    if not isinstance(tracks, dict) or not tracks:
        return {}
    nested = any(isinstance(v, dict) for v in tracks.values()) and \
        all(isinstance(v, dict) for v in tracks.values())
    out: dict[int, dict] = defaultdict(dict)
    if nested:
        for key, agents in tracks.items():
            team = _team_no(key)
            if team is None:
                continue
            for agent, pts in (agents or {}).items():
                if pts:
                    out[team][agent] = list(pts)
    else:
        for agent, pts in tracks.items():
            if pts:
                out[0][agent] = list(pts)
    return dict(out)


def _team_sides(round_dict: dict) -> dict[int, str]:
    """チーム番号 → 側。`side1`（チーム1 の側）を優先し、無ければ winner と side から補う。"""
    s1 = _canon(round_dict.get("side1"))
    if not s1:
        wside = _canon(round_dict.get("side"))
        wno = round_dict.get("winner_no")
        if wside and wno in (1, 2):
            s1 = wside if wno == 1 else _opp(wside)
    if not s1:
        return {}
    return {1: s1, 2: _opp(s1)}


def _attack_team(sides: dict[int, str]) -> int | None:
    return next((t for t, s in sides.items() if s == ATTACK), None)


def _area_of(zone: str, p) -> str:
    callout = _callout_of_sample(p)
    if callout:
        return TC.area_name(zone, callout) or zone
    return zone


def _frames(norm: dict) -> list[dict]:
    """サンプルを 5 秒の格子ごとのフレームにまとめる。

    各フレームは {"e": 経過秒, "zones": {チーム: Counter(ゾーン)}, "areas": {チーム: Counter(エリア)}}。
    """
    zb: dict[float, dict] = defaultdict(lambda: {1: Counter(), 2: Counter(), 0: Counter()})
    ab: dict[float, dict] = defaultdict(lambda: {1: Counter(), 2: Counter(), 0: Counter()})
    for team, agents in norm.items():
        for _agent, pts in agents.items():
            for p in pts:
                if not isinstance(p, (list, tuple)) or not p:
                    continue
                e = _snap(_f(p[0]))
                zone = _zone_of_sample(p)
                zb[e][team][zone] += 1
                ab[e][team][_area_of(zone, p)] += 1
    out = []
    for e in sorted(zb):
        out.append({"e": e,
                    "zones": {t: Counter(zb[e][t]) for t in (1, 2)},
                    "areas": {t: Counter(ab[e][t]) for t in (1, 2)}})
    return out


def _duration(round_dict: dict, frames: list[dict]) -> float:
    sec = _clock(round_dict.get("duration"))
    if sec:
        return float(sec)
    mx = 0.0
    for f in frames:
        mx = max(mx, float(f["e"]))
    return mx + SAMPLE_SECONDS if mx else 0.0


def _plant_elapsed(round_dict: dict) -> float | None:
    tact = round_dict.get("tactics") or {}
    plant = tact.get("plant") if isinstance(tact, dict) else None
    plant = round_dict.get("plant_t") if plant is None else plant
    if plant is None:
        return None
    t0 = round_dict.get("t_drop")
    return _f(plant) - _f(t0) if t0 is not None else _f(plant)


def skip_reason(round_dict: dict) -> str | None:
    """分類できなかった理由。分類できる（tracks がある）ときは None。"""
    if not _norm_tracks(tracks_of(round_dict)):
        return "no_tracks"
    return None


# ---------------------------------------------------------------- 支配・イベント
def _dominator(frame: dict, zone: str) -> int | None:
    d1 = frame["zones"][1].get(zone, 0)
    d2 = frame["zones"][2].get(zone, 0)
    if d1 >= DOMINANCE_MIN and d1 > d2:
        return 1
    if d2 >= DOMINANCE_MIN and d2 > d1:
        return 2
    return None


def _dominance_events(frames: list[dict], sides: dict[int, str]) -> list[Event]:
    """支配の変化から テイク（無→自チーム）と 取り返し（相手→自チーム）を拾う。

    初期配置（最初のフレームで入っているエリア）は支配として数えない。同じチームが初期配置を
    保っている間は変化として出さず、そこから別チームが支配に変わったら テイク として拾う。
    """
    out: list[Event] = []
    if not frames:
        return out
    for z in ZONES:
        init = _dominator(frames[0], z)          # 初期配置で入っているチーム（支配としては数えない）
        prev: int | None = None
        still_init = init is not None
        for i in range(1, len(frames)):
            cur = _dominator(frames[i], z)
            if cur is None:
                prev, still_init = None, False
                continue
            if still_init and cur == init:       # 初期配置のまま（イベントにしない）
                continue
            still_init = False
            if cur == prev:
                continue
            nxt = _dominator(frames[i + 1], z) if i + 1 < len(frames) else None
            if nxt == cur:                       # 次のサンプルまで続くものだけ
                kind = "テイク" if prev is None else "取り返し"
                out.append(Event(float(frames[i]["e"]), kind, z, sides.get(cur)))
            prev = cur
    return out


def _centroid_zone(counter) -> str | None:
    zc = {z: int(n) for z, n in (counter or {}).items() if z in ZONES and n}
    if not zc:
        return None
    return min(zc, key=lambda z: (-zc[z], ZONES.index(z)))


def _rotate_events(frames: list[dict], sides: dict[int, str]) -> list[Event]:
    """ローテート：チームの人数の重心ゾーンが変わり、次のサンプルも維持される。"""
    out: list[Event] = []
    if not frames:
        return out
    for team in (1, 2):
        prev = _centroid_zone(frames[0]["zones"].get(team))
        for i in range(1, len(frames)):
            cur = _centroid_zone(frames[i]["zones"].get(team))
            if cur and prev and cur != prev:
                nxt = _centroid_zone(frames[i + 1]["zones"].get(team)) if i + 1 < len(frames) else None
                if nxt == cur:
                    out.append(Event(float(frames[i]["e"]), "ローテート", cur, sides.get(team)))
            prev = cur or prev
    return out


def _is_site_area(area: str | None) -> bool:
    return bool(area) and "site" in str(area).lower()


def _area_zone(area: str | None) -> str | None:
    if not area:
        return None
    t = str(area).split(" ")[0]
    return t if t in SITES else None


def _entry_events(frames: list[dict], atk: int | None) -> list[Event]:
    """エントリー：攻撃がサイトのコールアウトに 2 人以上入る（サイトごとに最初の 1 回）。"""
    out: list[Event] = []
    if atk is None:
        return out
    seen: set[str] = set()
    for f in frames:
        for area, n in f["areas"].get(atk, {}).items():
            z = _area_zone(area)
            if z and _is_site_area(area) and n >= ENTRY_MIN and z not in seen:
                seen.add(z)
                out.append(Event(float(f["e"]), "エントリー", z, ATTACK))
    return out


def _crossings(frames: list[dict], team: int, key: str, min_n: int):
    """あるチームのエリア人数が min_n に届いた瞬間（下から上へ）を順に返す。"""
    prev: Counter = Counter()
    for f in frames:
        cur = f[key].get(team, Counter())
        for area, n in cur.items():
            if n >= min_n and prev.get(area, 0) < min_n:
                yield float(f["e"]), area, int(n)
        prev = cur


def _opponent_half(area: str, side: str, ranks: dict | None) -> bool:
    """そのエリアが、side から見た相手側の陣地か。"""
    if not ranks or area not in ranks:
        return False
    half_atk = TC.attacker_half(area, ranks)
    return half_atk if side == DEFENSE else (not half_atk)


def _deeper(area: str, site_area: str, side: str, ranks: dict | None) -> bool:
    """サイトより奥（自分のスポーンから遠い）エリアか。"""
    if not ranks:
        return False
    ra, rs = ranks.get(area), ranks.get(site_area)
    if not ra or not rs:
        return False
    idx = 0 if side == ATTACK else 1          # ranks は (攻撃スポーンからの距離, 防衛スポーンからの距離)
    return ra[idx] > rs[idx]


def _push_events(frames: list[dict], sides: dict[int, str], atk: int | None, ranks: dict | None) -> list[Event]:
    """プッシュ：相手側のエリアへ 2 人以上が進む動きと、サイトより奥へ進む動き。"""
    out: list[Event] = []
    # 序盤・中盤の前進：相手側の陣地（サイト以外）に 2 人以上が新たに入る
    if ranks:
        for team in (1, 2):
            side = sides.get(team)
            if not side:
                continue
            for e, area, _n in _crossings(frames, team, "areas", PUSH_MIN):
                if _is_site_area(area):
                    continue
                if _opponent_half(area, side, ranks):
                    out.append(Event(e, "プッシュ", _area_zone(area), side))
    # サイトエントリー後：サイトより奥のエリアへ 2 人以上が進んだらプッシュ
    if ranks and atk is not None:
        for ent in _entry_events(frames, atk):
            site_area = f"{ent.zone} Site"
            for f in frames:
                if f["e"] <= ent.t + 1e-9:
                    continue
                deep = [a for a, n in f["areas"].get(atk, {}).items()
                        if n >= PUSH_MIN and _deeper(a, site_area, ATTACK, ranks)]
                if deep:
                    out.append(Event(float(f["e"]), "プッシュ", ent.zone, ATTACK))
                    break
    return out


def _kill_side(round_dict: dict, team_name) -> str | None:
    winner = round_dict.get("winner")
    wside = _canon(round_dict.get("side"))
    if not winner or not wside:
        return None
    return wside if team_name == winner else _opp(wside)


def _pick_events(round_dict: dict) -> list[Event]:
    """ピック：キル（時刻つき）。時刻はラウンド開始からの秒に直す。"""
    out: list[Event] = []
    t0 = round_dict.get("t_drop")
    base = _f(t0) if t0 is not None else 0.0
    for k in round_dict.get("kills") or []:
        if not isinstance(k, dict) or k.get("t") is None:
            continue
        out.append(Event(_f(k["t"]) - base, "ピック", None, _kill_side(round_dict, k.get("killer_team"))))
    return out


def _rush_events(frames: list[dict], atk: int | None) -> list[Event]:
    """ラッシュ：開幕 RUSH_SECONDS 以内のサイトエントリー（または重心の急な寄り）。"""
    if atk is None:
        return []
    out: list[Event] = []
    for ent in _entry_events(frames, atk):
        if ent.t <= RUSH_SECONDS + 1e-9:
            out.append(Event(ent.t, "ラッシュ", ent.zone, ATTACK))
    if not out:
        for f in frames:
            if f["e"] > RUSH_SECONDS + 1e-9:
                break
            z = _centroid_zone(f["zones"].get(atk))
            if z in SITES:
                out.append(Event(float(f["e"]), "ラッシュ", z, ATTACK))
                break
    return out


def round_events(round_dict: dict, ranks: dict | None = None) -> list[Event]:
    """ラウンドのイベント列（時刻・種別・ゾーン・側）。tracks が無ければ空リスト。

    ranks: エリア名 → （攻撃スポーンからの距離, 防衛スポーンからの距離）。あれば
    プッシュ（相手側への前進・サイトより奥へ）を拾う（`tactics.area_ranks` で作れる）。
    """
    norm = _norm_tracks(tracks_of(round_dict))
    if not norm:
        return []
    frames = _frames(norm)
    sides = _team_sides(round_dict)
    atk = _attack_team(sides)
    ev: list[Event] = []
    ev += _dominance_events(frames, sides)
    ev += _rotate_events(frames, sides)
    ev += _entry_events(frames, atk)
    ev += _push_events(frames, sides, atk, ranks)
    ev += _pick_events(round_dict)
    ev += _rush_events(frames, atk)
    ev.sort(key=lambda x: (round(float(x.t), 3), KIND_ORDER.get(x.kind, 99), x.zone or ""))
    return ev


# ---------------------------------------------------------------- 区間
def segment_phases(events: list[Event], round_dict: dict) -> dict:
    """序盤・中盤・終盤の区間（開始からの秒）。

    序盤: 開始〜最初のピックか最初の支配交代の早い方。中盤: 序盤の終わり〜プラント。
    終盤: プラント〜終了。プラントが無いラウンドは中盤が終了までで、終盤は空。
    """
    frames = _frames(_norm_tracks(tracks_of(round_dict)))
    dur = _duration(round_dict, frames)
    picks = [e.t for e in events if e.kind == "ピック"]
    doms = [e.t for e in events if e.kind in ("テイク", "取り返し")]
    cand = [t for t in ((min(picks) if picks else None), (min(doms) if doms else None)) if t is not None]
    early_end = min(cand) if cand else dur
    plant = _plant_elapsed(round_dict)
    mid_end = plant if plant is not None else dur
    mid_end = max(mid_end, early_end)
    late_start = mid_end if plant is not None else None
    return {
        "plant": round(plant, 1) if plant is not None else None,
        "early": {"label": "序盤", "start": 0.0, "end": round(early_end, 1)},
        "mid": {"label": "中盤", "start": round(early_end, 1), "end": round(mid_end, 1)},
        "late": {"label": "終盤", "start": round(late_start, 1) if late_start is not None else None,
                 "end": round(dur, 1) if plant is not None else None},
    }


# ---------------------------------------------------------------- 型
def _main_zone(counts: dict) -> str:
    zc = {z: int(n) for z, n in counts.items() if z in ZONES and n}
    if not zc:
        return "Mid"
    return min(zc, key=lambda z: (-zc[z], ZONES.index(z)))


def _early_type(frames: list[dict], events: list[Event], early_end: float, atk: int | None) -> dict:
    # ラッシュは開幕 15 秒以内の動きで決まる（序盤の終わりより後でも拾う）
    rush = [e for e in events if e.kind == "ラッシュ" and e.t <= RUSH_SECONDS + 1e-9]
    if rush:
        return {"type": f"{rush[0].zone} ラッシュ", "zone": rush[0].zone, "candidates": []}
    if atk is None:
        return {"type": None, "zone": None, "candidates": []}
    used: Counter = Counter()
    for f in frames:
        if f["e"] > early_end + 1e-9:
            break
        for z, n in f["zones"].get(atk, {}).items():
            if z in ZONES:
                used[z] += int(n)
    used = {z: n for z, n in used.items() if n}
    if len(used) >= 2:
        main = _main_zone(used)
        return {"type": f"{main} デフォルト", "zone": main, "candidates": sorted(used)}
    return {"type": "詰め待ち", "zone": _main_zone(used) if used else None, "candidates": sorted(used)}


def _freeze_at(events: list[Event], mid_end: float) -> float | None:
    changes = sorted(e.t for e in events if e.kind in ("ピック", "テイク", "取り返し"))
    for p in sorted(e.t for e in events if e.kind == "ピック"):
        nxt = next((t for t in changes if t > p + 1e-9), None)
        end = min(nxt, mid_end) if nxt is not None else mid_end
        if end - p >= FREEZE_SECONDS - 1e-9:
            return p
    return None


def _regroup_at(frames: list[dict], entries: list[Event], atk: int | None) -> float | None:
    if atk is None:
        return None
    for ent in entries:
        spread = False
        for f in frames:
            if f["e"] > ent.t + 1e-9:
                break
            zc = {z: int(n) for z, n in f["zones"].get(atk, {}).items() if z in ZONES and n}
            if len(zc) >= 2:
                spread = True
            elif spread and len(zc) == 1 and max(zc.values()) >= REGROUP_MIN:
                return float(f["e"])
    return None


def _fake_at(events: list[Event]) -> float | None:
    doms = [e for e in events if e.kind in ("テイク", "取り返し")]
    rots = [e for e in events if e.kind == "ローテート"]
    ents = [e for e in events if e.kind == "エントリー"]
    for d in doms:
        for r in rots:
            if r.t > d.t + 1e-9 and r.zone == d.zone and r.side != d.side:
                for ent in ents:
                    if ent.t > r.t + 1e-9 and ent.zone != d.zone:
                        return float(d.t)
    return None


def _mid_type(frames: list[dict], events: list[Event], mid0: float, mid1: float, atk: int | None) -> dict:
    entries = [e for e in events if e.kind == "エントリー" and mid0 - 1e-9 <= e.t <= mid1 + 1e-9]
    got: dict[str, float] = {}
    fz = _freeze_at(events, mid1)
    if fz is not None:
        got["フリーズ"] = fz
    rg = _regroup_at(frames, entries, atk)
    if rg is not None:
        got["リグループセット"] = rg
    fk = _fake_at(events)
    if fk is not None:
        got["フェイクセット"] = fk
    if not got:
        return {"type": None, "candidates": []}
    # リグループセットとフェイクセットが両方ならリグループを優先。その他は最初に成立した型を主とする
    if "リグループセット" in got and "フェイクセット" in got:
        main = "リグループセット"
    else:
        main = min(got, key=lambda k: got[k])
    return {"type": main, "candidates": sorted(k for k in got if k != main)}


def _attack_site(events: list[Event], round_dict: dict) -> str | None:
    for e in events:
        if e.kind == "エントリー" and e.zone in SITES:
            return e.zone
    tact = round_dict.get("tactics") or {}
    site = tact.get("plant_site") if isinstance(tact, dict) else None
    return site if site in SITES else None


def _late_type(frames: list[dict], plant: float | None, events: list[Event], round_dict: dict,
               atk: int | None, ranks: dict | None) -> dict:
    if plant is None or atk is None:
        return {"type": None, "zone": None, "detail": None}
    site = _attack_site(events, round_dict)
    if site is None:
        return {"type": None, "zone": None, "detail": None}
    site_area = f"{site} Site"
    if ranks:
        for f in frames:
            if f["e"] < plant - 1e-9:
                continue
            deep = [a for a, n in f["areas"].get(atk, {}).items()
                    if n >= PUSH_MIN and _deeper(a, site_area, ATTACK, ranks)]
            if deep:
                return {"type": "プッシュ", "zone": site, "detail": deep[0]}
    for f in frames:
        if f["e"] < plant - 1e-9:
            continue
        on = sum(int(n) for a, n in f["areas"].get(atk, {}).items()
                 if _is_site_area(a) and _area_zone(a) == site)
        if on >= PUSH_MIN:
            return {"type": "サイト耐え", "zone": site, "detail": None}
    return {"type": "下がる", "zone": site, "detail": None}


# ---------------------------------------------------------------- 流れ
def flow_text(events: list[Event]) -> str:
    """1 行の流れ要約（例: 「防衛 A プッシュ→攻撃 Mid テイク→防衛 ローテート→攻撃 A エントリー」）。"""
    steps: list[str] = []
    for e in events:
        if e.kind not in FLOW_KINDS:
            continue
        zone = f" {e.zone}" if e.zone else ""
        step = f"{e.side or '?'}{zone} {e.kind}"
        if not steps or steps[-1] != step:
            steps.append(step)
    return "→".join(steps)


def classify_flow(round_dict: dict, ranks: dict | None = None) -> dict | None:
    """1 ラウンドの型（区間ごと）・イベント列・流れ要約。tracks が無ければ None。

    ranks は `tactics.area_ranks(callouts_placed)` の形（あればプッシュ判定に使う）。
    """
    norm = _norm_tracks(tracks_of(round_dict))
    if not norm:
        return None
    frames = _frames(norm)
    events = round_events(round_dict, ranks)
    segs = segment_phases(events, round_dict)
    sides = _team_sides(round_dict)
    atk = _attack_team(sides)
    phases = {
        "early": _early_type(frames, events, segs["early"]["end"], atk),
        "mid": _mid_type(frames, events, segs["mid"]["start"], segs["mid"]["end"], atk),
        "late": _late_type(frames, segs["plant"], events, round_dict, atk, ranks),
    }
    return {"events": [e.to_dict() for e in events], "segments": segs, "phases": phases,
            "flow": flow_text(events)}
