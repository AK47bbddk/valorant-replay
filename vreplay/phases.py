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

import math
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
RUSH_SECONDS = 15.0           # 開幕からこの秒以内にサイトへエントリーしたらラッシュ
EARLY_TAKE_WINDOW = 20.0      # 開幕のエリア取り合い判定ウィンドウの下限秒
FREEZE_SECONDS = 10.0         # ピック後に動きが無いままこの秒数でフリーズ区間
ROTATE_REVERT_SECONDS = 10.0  # ローテート後にこの秒以内に元のゾーンへ戻ったら取り消す
EARLY_JUMP_PER_5S = 0.40      # 開幕ウィンドウ内で 5 秒あたりこれ以上の移動は外れ値として落とす
HOME_ATTACK_RATIO_MAX = 0.51  # 攻撃側の自陣側コールアウト判定（da / (da + dd) の上限）
HOME_DEFENSE_RATIO_MIN = 0.49 # 防衛側の自陣側コールアウト判定（da / (da + dd) の下限）
EARLY_BACK_RATIO_MAX = 0.38   # 開幕デフォルト判定で自陣後方エリアとして除外する da/(da+dd) 上限
POSTPLANT_WAIT_SECONDS = 5.0  # プラント後この秒以降の最初のサンプルでポストプラント配置を見る
_EARLY_BACK_KEYWORDS = ("spawn", "lobby", "sewer", "bottom", "b link")

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


def home_callouts(ranks: dict | None) -> dict[str, set[str]]:
    """マップの area_ranks から、サイド別の自陣側コールアウト集合を作る。"""
    if not ranks:
        return {}
    atk: set[str] = set()
    dfn: set[str] = set()
    for area, pair in ranks.items():
        if not pair or len(pair) < 2:
            continue
        da, dd = pair[0], pair[1]
        if da is None or dd is None or da + dd <= 0:
            continue
        ratio = float(da) / float(da + dd)
        if ratio <= HOME_ATTACK_RATIO_MAX and not _is_site_area(area):
            atk.add(area)
        if ratio >= HOME_DEFENSE_RATIO_MIN:
            dfn.add(area)
    if not atk and not dfn:
        return {}
    return {ATTACK: atk, DEFENSE: dfn}


def _is_home_sample(p, side: str, homes: dict[str, set[str]]) -> bool:
    """サンプル p が所属サイドの自陣側コールアウト集合に含まれるか。"""
    zone = _zone_of_sample(p)
    callout = _callout_of_sample(p) or ""
    low_c = callout.lower()
    if side == ATTACK:
        if zone == "攻撃側" or ("attacker" in low_c and "spawn" in low_c):
            return True
        if zone == "防衛側" or ("defender" in low_c and "spawn" in low_c):
            return False
    elif side == DEFENSE:
        if zone == "防衛側" or ("defender" in low_c and "spawn" in low_c):
            return True
        if zone == "攻撃側" or ("attacker" in low_c and "spawn" in low_c):
            return False
    area = _area_of(zone, p)
    return area in homes.get(side, set())


def _filter_agent_pts(pts: list, side: str | None, homes: dict[str, set[str]],
                      stats: dict | None = None) -> list:
    """最初期サンプルの自陣外誤検出と、開幕ウィンドウ内のジャンプ外れ値を落とす。"""
    ordered = sorted((p for p in pts if isinstance(p, (list, tuple)) and p), key=lambda p: _f(p[0]))
    if not ordered or not homes:
        return list(ordered)
    if stats is not None:
        stats.setdefault("outside_home", 0)
        stats.setdefault("jump", 0)
        stats.setdefault("dropped", 0)

    # 1. 最初のサンプルが開幕ウィンドウ内で自陣外なら誤検出として落とし、次の有効サンプルを探す
    start_idx = 0
    if side in (ATTACK, DEFENSE):
        while start_idx < len(ordered):
            p = ordered[start_idx]
            if _f(p[0]) <= RUSH_SECONDS + 1e-9 and not _is_home_sample(p, side, homes):
                if stats is not None:
                    stats["outside_home"] += 1
                    stats["dropped"] += 1
                start_idx += 1
            else:
                break
    rem = ordered[start_idx:]
    if not rem:
        return []

    # 2. 連続性チェック（開幕ウィンドウ内で 5 秒間隔にあり得ない距離のジャンプを落とす）
    kept = [rem[0]]
    prev = rem[0]
    for cur in rem[1:]:
        e_cur = _f(cur[0])
        if e_cur <= RUSH_SECONDS + 1e-9 and len(cur) >= 3 and len(prev) >= 3:
            x0, y0 = _f(prev[1]), _f(prev[2])
            x1, y1 = _f(cur[1]), _f(cur[2])
            dt = e_cur - _f(prev[0])
            if dt > 0 and (x0 != 0.0 or y0 != 0.0) and (x1 != 0.0 or y1 != 0.0):
                dist = math.hypot(x1 - x0, y1 - y0)
                limit = EARLY_JUMP_PER_5S * max(1.0, dt / SAMPLE_SECONDS)
                if dist > limit:
                    if stats is not None:
                        stats["jump"] += 1
                        stats["dropped"] += 1
                    continue
        kept.append(cur)
        prev = cur
    return kept


def _norm_tracks(tracks: dict, sides: dict[int, str] | None = None,
                 ranks: dict | None = None, filter_stats: dict | None = None) -> dict:
    """tracks を {チーム番号: {エージェント: [サンプル]}} にそろえる。

    ranks が渡された場合は暫定バリアフィルタ（自陣外・連続性チェック）を適用する。
    """
    if not isinstance(tracks, dict) or not tracks:
        return {}
    homes = home_callouts(ranks)
    nested = any(isinstance(v, dict) for v in tracks.values()) and \
        all(isinstance(v, dict) for v in tracks.values())
    out: dict[int, dict] = defaultdict(dict)
    if nested:
        for key, agents in tracks.items():
            team = _team_no(key)
            if team is None:
                continue
            side = sides.get(team) if sides else None
            for agent, pts in (agents or {}).items():
                if pts:
                    filtered = _filter_agent_pts(list(pts), side, homes, filter_stats)
                    if filtered:
                        out[team][agent] = filtered
    else:
        for agent, pts in tracks.items():
            if pts:
                filtered = _filter_agent_pts(list(pts), None, homes, filter_stats)
                if filtered:
                    out[0][agent] = filtered
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
    """ローテート：チームの人数の重心ゾーンが変わり、2 サンプル続いた場合だけ拾う。

    変化のあと ROTATE_REVERT_SECONDS 以内に元のゾーンへ戻ったら、そのローテートは取り消す。
    """
    out: list[Event] = []
    if not frames:
        return out
    for team in (1, 2):
        series = [(_centroid_zone(f["zones"].get(team)), float(f["e"])) for f in frames]
        prev = series[0][0]
        for i in range(1, len(series)):
            cur, t = series[i]
            if cur and prev and cur != prev:
                nxt = series[i + 1][0] if i + 1 < len(series) else None
                if nxt == cur:                   # 次のサンプルも同じゾーン（2 サンプル続く）
                    reverted = False
                    for z2, t2 in series[i + 1:]:
                        if t2 - t > ROTATE_REVERT_SECONDS + 1e-9:
                            break
                        if z2 == prev:           # 10 秒以内に元のゾーンへ逆戻り
                            reverted = True
                            break
                    if not reverted:
                        out.append(Event(t, "ローテート", cur, sides.get(team)))
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
    """ラッシュ：開幕 RUSH_SECONDS 以内にサイトへエントリー（サイトのコールアウトに 2 人以上）した場合のみ。

    人数の重心（最頻ゾーン）が変わっただけ、サイト側のゾーンに 4 人以上入っただけでは
    ラッシュにしない（エントリーしなければ、開幕の配分に応じたデフォルトになる）。
    """
    if atk is None:
        return []
    out: list[Event] = []
    for ent in _entry_events(frames, atk):
        if ent.t <= RUSH_SECONDS + 1e-9:
            out.append(Event(ent.t, "ラッシュ", ent.zone, ATTACK))
    return out


def _merge_events(events: list[Event]) -> list[Event]:
    """同じ側・同じゾーン・同じ種別の連続イベントを 1 つにまとめる（ゾーン無しのピックはまとめない）。"""
    out: list[Event] = []
    for e in events:
        if (out and e.zone is not None and out[-1].zone == e.zone
                and out[-1].kind == e.kind and out[-1].side == e.side):
            continue
        out.append(e)
    return out


def round_events(round_dict: dict, ranks: dict | None = None) -> list[Event]:
    """ラウンドのイベント列（時刻・種別・ゾーン・側）。tracks が無ければ空リスト。

    ranks: エリア名 → （攻撃スポーンからの距離, 防衛スポーンからの距離）。あれば
    プッシュ（相手側への前進・サイトより奥へ）と暫定バリアフィルタに使う。
    """
    sides = _team_sides(round_dict)
    norm = _norm_tracks(tracks_of(round_dict), sides=sides, ranks=ranks)
    if not norm:
        return []
    frames = _frames(norm)
    atk = _attack_team(sides)
    ev: list[Event] = []
    ev += _dominance_events(frames, sides)
    ev += _rotate_events(frames, sides)
    ev += _entry_events(frames, atk)
    ev += _push_events(frames, sides, atk, ranks)
    ev += _pick_events(round_dict)
    ev += _rush_events(frames, atk)
    ev.sort(key=lambda x: (round(float(x.t), 3), KIND_ORDER.get(x.kind, 99), x.zone or ""))
    return _merge_events(ev)


# ---------------------------------------------------------------- 区間
def segment_phases(events: list[Event], round_dict: dict, ranks: dict | None = None) -> dict:
    """序盤・中盤・終盤の区間（開始からの秒）。

    序盤: 開始〜最初のピックか最初の支配交代の早い方。中盤: 序盤の終わり〜プラント。
    終盤: プラント〜終了。プラントが無いラウンドは中盤が終了までで、終盤は空。
    """
    sides = _team_sides(round_dict)
    frames = _frames(_norm_tracks(tracks_of(round_dict), sides=sides, ranks=ranks))
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
def _defense_team(sides: dict[int, str]) -> int | None:
    return next((t for t, s in sides.items() if s == DEFENSE), None)


def _zone_label(z: str) -> str:
    return "MID" if z == "Mid" else z


def _main_zone(counts: dict) -> str:
    zc = {z: int(n) for z, n in counts.items() if z in ZONES and n}
    if not zc:
        return "Mid"
    return min(zc, key=lambda z: (-zc[z], ZONES.index(z)))


def _is_early_back_area(area: str, ranks: dict | None = None) -> bool:
    """開幕の待機・自陣後方エリア（スポーン・ロビー等）なら True。取りに行くエリアから除外する。"""
    if not area or area in ("?", "攻撃側", "防衛側"):
        return True
    low = area.lower()
    if any(w in low for w in _EARLY_BACK_KEYWORDS):
        return True
    if ranks and area in ranks:
        da, dd = ranks[area]
        if da is not None and dd is not None and da + dd > 0 and da / (da + dd) < EARLY_BACK_RATIO_MAX:
            return True
    return False


def _early_default_label(area: str) -> tuple[str, str]:
    """取りに行ったエリア名から（開幕デフォルト型名, ゾーン）を返す。"""
    low = area.lower()
    z = _area_zone(area)
    if "ramp" in low:
        return "ランプデフォルト", z or "A"
    if area == "Mid" or "mid" in low:
        return "ミッドデフォルト", "Mid"
    if area == "B" or low.startswith("b ") or z == "B":
        return "Bメインデフォルト", "B"
    if area == "A" or low.startswith("a ") or z == "A":
        return "Aデフォルト", "A"
    if area == "C" or low.startswith("c ") or z == "C":
        return "Cメインデフォルト", "C"
    return f"{area}デフォルト", (z or area)


def _first_early_area(frames: list[dict], early_end: float, atk: int,
                      def_team: int | None = None, ranks: dict | None = None) -> tuple[str | None, list[str]]:
    """序盤の最初のテイク（エリア支配の成立）またはそれに相当する開幕の前進エリアを返す。"""
    limit = max(float(early_end), EARLY_TAKE_WINDOW)
    seen: set[str] = set()
    for f in frames:
        if f["e"] > limit + 1e-9:
            break
        for a, n in f["areas"].get(atk, {}).items():
            if n >= 1 and not _is_early_back_area(a, ranks):
                seen.add(a)

    # 1. 最初のテイク（2人以上かつ防衛より多い状態が2サンプル連続で成立）
    for i in range(len(frames) - 1):
        if frames[i]["e"] > limit + 1e-9:
            break
        cands = []
        for a, n in frames[i]["areas"].get(atk, {}).items():
            if _is_early_back_area(a, ranks):
                continue
            dn = frames[i]["areas"].get(def_team, {}).get(a, 0) if def_team else 0
            n2 = frames[i + 1]["areas"].get(atk, {}).get(a, 0)
            dn2 = frames[i + 1]["areas"].get(def_team, {}).get(a, 0) if def_team else 0
            if n >= DOMINANCE_MIN and n > dn and n2 >= DOMINANCE_MIN and n2 > dn2:
                cands.append((n, a))
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            return cands[0][1], sorted(seen)

    # 2. 開幕の前進（2人以上が最初に入った取り合いエリア）
    for f in frames:
        if f["e"] > limit + 1e-9:
            break
        cands = [(n, a) for a, n in f["areas"].get(atk, {}).items()
                 if not _is_early_back_area(a, ranks) and n >= DOMINANCE_MIN]
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            return cands[0][1], sorted(seen)

    # 3. 開幕の前進（1人以上が最初に入った取り合いエリア）
    for f in frames:
        if f["e"] > limit + 1e-9:
            break
        cands = [(n, a) for a, n in f["areas"].get(atk, {}).items()
                 if not _is_early_back_area(a, ranks) and n >= 1]
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            return cands[0][1], sorted(seen)

    return None, []


def _early_type(frames: list[dict], events: list[Event], early_end: float, atk: int | None,
                def_team: int | None = None, ranks: dict | None = None) -> dict:
    # ラッシュは開幕 15 秒以内の動きで決まる（最優先）
    rush = [e for e in events if e.kind == "ラッシュ" and e.t <= RUSH_SECONDS + 1e-9]
    if rush:
        return {"type": "ラッシュ", "zone": rush[0].zone, "area": None, "candidates": []}
    if atk is None:
        return {"type": None, "zone": None, "area": None, "candidates": []}
    area, cands = _first_early_area(frames, early_end, atk, def_team=def_team, ranks=ranks)
    if area:
        typ, zone = _early_default_label(area)
        return {"type": typ, "zone": zone, "area": area, "candidates": cands}
    return {"type": "詰め待ち", "zone": None, "area": None, "candidates": []}


def _entry_type(norm: dict, events: list[Event], atk: int | None) -> dict:
    """攻めエントリーの型: 進入メンバーの直前の進入路エリア種類数（1種=セット, 2種以上=スプリット）。"""
    entries = [e for e in events if e.kind == "エントリー" and e.zone in SITES]
    if not entries or atk is None:
        return {"type": None, "zone": None, "routes": []}
    ent = entries[0]
    site = ent.zone
    ent_t = float(ent.t)
    ag_map = norm.get(atk) or {}
    prev_areas: list[str] = []
    for _ag, pts in ag_map.items():
        # サイトへの進入ウィンドウ内で最初にサイトへ入った時刻を探す
        t_in = None
        for p in pts:
            e = _f(p[0])
            if ent_t - SAMPLE_SECONDS - 1e-9 <= e <= ent_t + SAMPLE_SECONDS + 1e-9:
                z = _zone_of_sample(p)
                a = _area_of(z, p)
                if _is_site_area(a) and _area_zone(a) == site:
                    t_in = e
                    break
        if t_in is None:
            continue
        # 進入直前の進入路エリア（サイト外・スポーン外の直近サンプル）
        prev_p = None
        for p in pts:
            e = _f(p[0])
            if e >= t_in - 1e-9:
                break
            if e < t_in - 15.0 - 1e-9:
                continue
            z = _zone_of_sample(p)
            c = _callout_of_sample(p)
            a = _area_of(z, p)
            if not _is_site_area(a) and not TC.is_spawn(z, c) and z != "?":
                prev_p = a
        if prev_p:
            prev_areas.append(prev_p)
    routes = sorted(set(prev_areas))
    typ = f"{site}スプリット" if len(routes) >= 2 else f"{site}セット"
    return {"type": typ, "zone": site, "routes": routes}


def _early_type_defense(frames: list[dict], early_end: float, def_team: int | None) -> dict:
    """防衛の序盤の型: 防衛チームのゾーン分布から主ゾーンの寄せを判定する。"""
    if def_team is None:
        return {"type": None, "zone": None, "candidates": [], "dist": {}}
    used: Counter = Counter()
    for f in frames:
        if f["e"] > early_end + 1e-9:
            break
        for z, n in f["zones"].get(def_team, {}).items():
            if z in ZONES:
                used[z] += int(n)
    used_d = {z: int(n) for z, n in used.items() if n}
    if used_d:
        main = _main_zone(used_d)
        return {"type": f"{_zone_label(main)}寄せ", "zone": main, "candidates": sorted(used_d), "dist": used_d}
    return {"type": "自陣待機", "zone": None, "candidates": [], "dist": {}}


def _mid_type_defense(events: list[Event], mid0: float, mid1: float) -> dict:
    """防衛の中盤の型: 防衛側のイベント（上がり／ローテート／取り返し）から判定する。"""
    evs = [e for e in events if e.side == DEFENSE and mid0 - 1e-9 <= e.t <= mid1 + 1e-9]
    got: dict[str, float] = {}
    kind_map = {"プッシュ": "上がり", "ローテート": "ローテート", "取り返し": "取り返し", "テイク": "取り返し"}
    for e in evs:
        lbl = kind_map.get(e.kind)
        if lbl and lbl not in got:
            got[lbl] = float(e.t)
    if not got:
        return {"type": None, "candidates": []}
    main = min(got, key=lambda k: got[k])
    return {"type": main, "candidates": sorted(k for k in got if k != main)}


def _late_type_defense(events: list[Event], plant: float | None, round_dict: dict) -> dict:
    """防衛の終盤の型: プラント後の防衛側イベントからリテイク／寄せを判定する。"""
    if plant is None:
        return {"type": None, "zone": None, "detail": None}
    site = _attack_site(events, round_dict)
    evs = [e for e in events if e.side == DEFENSE and e.t >= plant - 1e-9]
    has_retake = any(e.kind in ("取り返し", "テイク") for e in evs)
    has_rotate = any(e.kind == "ローテート" for e in evs)
    if has_retake:
        return {"type": "リテイク", "zone": site, "detail": None}
    if has_rotate:
        return {"type": "寄せ", "zone": site, "detail": None}
    return {"type": "寄せ／リテイク", "zone": site, "detail": None}


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


def _segment_flow(events: list[Event], segs: dict) -> dict:
    """区間ごとの流れ要約。各イベントはその時刻が入る区間の要約にだけ出す。"""
    early_end = float(segs["early"]["end"])
    mid = segs["mid"]
    late = segs["late"]
    groups: dict[str, list[Event]] = {"early": [], "mid": [], "late": []}
    for e in events:
        if e.t <= early_end + 1e-9:
            groups["early"].append(e)
        if float(mid["start"]) - 1e-9 <= e.t <= float(mid["end"]) + 1e-9:
            groups["mid"].append(e)
        if late["start"] is not None and late["end"] is not None:
            if float(late["start"]) - 1e-9 <= e.t <= float(late["end"]) + 1e-9:
                groups["late"].append(e)
    return {seg: flow_text(evs) for seg, evs in groups.items()}


def postplant_sample(round_dict: dict, events: list[Event] | None = None,
                     ranks: dict | None = None) -> dict | None:
    """プラント後5秒以降の最初のサンプルで、生存している攻撃側の配置を返す。"""
    plant = _plant_elapsed(round_dict)
    if plant is None:
        return None
    sides = _team_sides(round_dict)
    atk = _attack_team(sides)
    if atk is None:
        return None
    norm = _norm_tracks(tracks_of(round_dict), sides=sides, ranks=ranks)
    ag_map = norm.get(atk) or {}
    if not ag_map:
        return None
    events = events if events is not None else round_events(round_dict, ranks)
    site = _attack_site(events, round_dict)
    if site is None:
        return None
    target_t = plant + POSTPLANT_WAIT_SECONDS
    cand_times = sorted(
        _snap(_f(p[0]))
        for pts in ag_map.values()
        for p in pts
        if isinstance(p, (list, tuple)) and p and _f(p[0]) >= target_t - 1e-9
    )
    if not cand_times:
        return None
    t0 = cand_times[0]
    positions: dict[str, tuple[float, float, str]] = {}
    for ag, pts in sorted(ag_map.items()):
        near = [p for p in pts if isinstance(p, (list, tuple)) and len(p) >= 3
                and abs(_snap(_f(p[0])) - t0) <= 1e-9]
        if not near:
            continue
        p = near[0]
        u, v = _f(p[1]), _f(p[2])
        z = _zone_of_sample(p)
        area = _area_of(z, p)
        positions[str(ag)] = (u, v, area)
    if not positions:
        return None
    return {"site": site, "t": t0, "positions": positions}


def classify_flow(round_dict: dict, ranks: dict | None = None, side: str = ATTACK) -> dict | None:
    """1 ラウンドの型（区間ごと）・イベント列・区間ごとの流れ要約。tracks が無ければ None。

    side: どちらの視点で型付けするか（ATTACK / DEFENSE）。
    ranks: `tactics.area_ranks(callouts_placed)` の形（あればプッシュ判定・暫定バリアフィルタに使う）。
    `flow` は {"early": 序盤, "mid": 中盤, "late": 終盤} の文字列で、その区間のイベントだけを含む。
    """
    sides = _team_sides(round_dict)
    filter_stats = {"outside_home": 0, "jump": 0, "dropped": 0}
    norm = _norm_tracks(tracks_of(round_dict), sides=sides, ranks=ranks, filter_stats=filter_stats)
    if not norm:
        return None
    frames = _frames(norm)
    events = round_events(round_dict, ranks)
    segs = segment_phases(events, round_dict, ranks=ranks)
    atk = _attack_team(sides)
    def_team = _defense_team(sides)
    view_side = _canon(side) or ATTACK
    if view_side == DEFENSE:
        phases = {
            "early": _early_type_defense(frames, segs["early"]["end"], def_team),
            "mid": _mid_type_defense(events, segs["mid"]["start"], segs["mid"]["end"]),
            "late": _late_type_defense(events, segs["plant"], round_dict),
        }
    else:
        phases = {
            "early": _early_type(frames, events, segs["early"]["end"], atk, def_team=def_team, ranks=ranks),
            "entry": _entry_type(norm, events, atk),
            "mid": _mid_type(frames, events, segs["mid"]["start"], segs["mid"]["end"], atk),
            "late": _late_type(frames, segs["plant"], events, round_dict, atk, ranks),
            "postplant": postplant_sample(round_dict, events=events, ranks=ranks),
        }
    return {"side": view_side, "events": [e.to_dict() for e in events], "segments": segs, "phases": phases,
            "flow": _segment_flow(events, segs), "filter_stats": filter_stats}
