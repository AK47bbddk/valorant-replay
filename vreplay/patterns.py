"""ラウンドのミニマップ軌跡から、プレイブック用の型（初期配置・ローテ・リテイク・
デフォルト・中盤・最終セット・ポストプラント・1ピック）を推定する。

座標は tactics.analyze_video が保存した positions を使う。
positions の 1 要素:
  [round, t, x, y, team, side(1=attack/2=defense), site, callout]
players の個別 ID は放送 HUD から取れないため、チーム単位の人数分布と時間変化を見る。
数値は「その時間帯の 1 サンプルあたりの平均人数」で、合計が 5 未満でも欠損として扱う。
"""
from __future__ import annotations

from collections import Counter

ATTACK, DEFENSE = "攻撃", "防衛"
CORE = ("A", "B", "C")
ZONES = ("A", "B", "C", "Mid", "Spawn", "Other")


def _zone(p) -> str:
    s = (p[6] if len(p) > 6 else "") or ""
    if s in CORE:
        return s
    low = str(s).lower()
    if "mid" in low:
        return "Mid"
    if "spawn" in low or "side" in low or s in ("攻撃側", "防衛側"):
        return "Spawn"
    return "Other"


def _counts(ps: list, t0: float, t1: float) -> dict[str, float]:
    """時間帯ごとの 1 サンプル平均人数。同時刻の点を 1 サンプルとして数える。"""
    by_t: dict[float, list] = {}
    for p in ps:
        t = float(p[1])
        if t0 <= t <= t1:
            by_t.setdefault(t, []).append(p)
    if not by_t:
        return {}
    counts: Counter = Counter()
    n = 0
    for arr in by_t.values():
        if len(arr) < 2:
            continue
        n += 1
        counts.update(_zone(p) for p in arr)
    if not n:
        return {}
    return {z: round(counts[z] / n, 1) for z in ZONES if counts[z]}


def _dominant(c: dict[str, float], core_min: float = 2.0) -> str | None:
    core = {z: v for z, v in c.items() if z in CORE}
    if core:
        z, v = max(core.items(), key=lambda kv: kv[1])
        if v >= core_min:
            return z
    mid = float(c.get("Mid") or 0)
    if mid >= core_min:
        return "Mid"
    return None


def _fmt(c: dict[str, float]) -> str:
    if not c:
        return "データなし"
    parts = []
    for z in ZONES:
        v = c.get(z)
        if v:
            parts.append(f"{z}{v:g}")
    return " ".join(parts) or "データなし"


def _style(c: dict[str, float]) -> str:
    core = {z: v for z, v in c.items() if z in CORE}
    if core:
        z, v = max(core.items(), key=lambda kv: kv[1])
        if v >= 3:
            return f"{z}寄せ"
        if v >= 2:
            return "バランス"
    if float(c.get("Mid") or 0) >= 2:
        return "Mid主導"
    return "分散"


def _trajectory(zones: list[str | None]) -> str:
    seq = []
    for z in zones:
        if z and (not seq or seq[-1] != z):
            seq.append(z)
    if not seq:
        return "不明"
    return "→".join(seq) if len(seq) > 1 else f"{seq[0]}維持"


def _first_time_at(ps: list, zone: str, need: int = 3) -> float | None:
    by_t: dict[float, list] = {}
    for p in ps:
        by_t.setdefault(float(p[1]), []).append(p)
    for t in sorted(by_t):
        if sum(_zone(p) == zone for p in by_t[t]) >= need:
            return t
    return None


def _top_callout(ps: list, zone: str | None = None) -> str | None:
    c = Counter()
    for p in ps:
        if zone and _zone(p) != zone:
            continue
        name = p[7] if len(p) > 7 else None
        if name:
            c[name] += 1
    return c.most_common(1)[0][0] if c else None


def analyze_round(rt: dict, positions: list, *, team_no: int, side1: str | None,
                  winner_no: int | None, how: str | None,
                  first_kill_team: int | None = None, first_kill_t: float | None = None) -> dict:
    """1 ラウンド・1 チームぶんの型を返す。

    rt: tactics の rounds[n]（start/end/plant/site）
    positions: そのラウンドの positions（全チーム。team_no で絞る）
    """
    ps = [p for p in positions
          if isinstance(p, (list, tuple)) and len(p) >= 8 and int(p[4]) == int(team_no)]
    raw_start, raw_end = rt.get("start"), rt.get("end")
    start = float(raw_start) if raw_start is not None else min((float(p[1]) for p in ps), default=0.0)
    end = float(raw_end) if raw_end is not None else max((float(p[1]) for p in ps), default=start + 100)
    plant = rt.get("plant")
    plant = float(plant) if plant is not None else None
    if plant is not None:
        plant = max(start, min(end, plant))
    if side1 not in (ATTACK, DEFENSE):
        side = None
    elif team_no == 1:
        side = side1
    else:
        side = ATTACK if side1 == DEFENSE else DEFENSE
    if not ps:
        return {"confidence": 0.0, "side": side}

    early = _counts(ps, start + 5, start + 18)
    if not early:
        early = _counts(ps, start, min(end, start + 20))
    mid = _counts(ps, start + 18, min(end, start + 50))
    pre = [p for p in ps if plant is None or float(p[1]) < plant]
    pre_end = plant if plant is not None else end + 1
    late = _counts(ps, max(start + 45, pre_end - 35), pre_end)
    if not late:
        late = _counts(pre, max(start + 30, pre_end - 20), pre_end)
    e_zone, m_zone, l_zone = _dominant(early), _dominant(mid), _dominant(late)
    final_zone = l_zone or m_zone
    if final_zone not in CORE and rt.get("site") in CORE:
        final_zone = rt["site"]

    out = {
        "confidence": round(min(1.0, len(ps) / 25.0), 2),
        "side": side,
        "setup": _fmt(early),
        "setup_style": _style(early),
        "mid": _fmt(mid),
        "late": _fmt(late),
    }

    if side == DEFENSE:
        out["rotation"] = _trajectory([e_zone, m_zone, l_zone])
        out["setup_zone"] = e_zone
        out["late_zone"] = l_zone
        if plant is not None:
            pp = [p for p in ps if float(p[1]) >= plant]
            zc = _counts(pp, plant, end + 1)
            site, cnt = max(((z, v) for z, v in zc.items() if z in CORE), key=lambda kv: kv[1],
                            default=(None, 0.0))
            total = sum(v for z, v in zc.items() if z != "Spawn")
            off = max(0.0, total - cnt)
            if site and cnt >= 3:
                retake = f"{site}一斉"
            elif site and cnt >= 2:
                retake = f"{site}+外角" if off >= 1 else f"{site}重視"
            elif site:
                retake = "少人数/遅延"
            else:
                retake = "不明"
            out["retake"] = {
                "label": retake,
                "site": site,
                "group": round(cnt, 1),
                "route": _top_callout(pp, site),
                "success": winner_no == team_no,
                "rounds": 1,
            }
    elif side == ATTACK:
        out["default"] = _style(early)
        out["default_zone"] = e_zone
        out["mid_change"] = _trajectory([e_zone, m_zone, l_zone])
        out["mid_zone"] = m_zone
        out["final_zone"] = final_zone
        if final_zone:
            set_t = _first_time_at(pre, final_zone)
            out["set_time"] = round(set_t - start, 1) if set_t is not None else None
            out["rush"] = bool(set_t is not None and set_t - start <= 20)
        else:
            out["set_time"] = None
            out["rush"] = False
        if plant is not None:
            pp = [p for p in ps if float(p[1]) >= plant]
            zc = _counts(pp, plant, end + 1)
            site, cnt = max(((z, v) for z, v in zc.items() if z in CORE), key=lambda kv: kv[1],
                            default=(None, 0.0))
            total = sum(v for z, v in zc.items() if z != "Spawn")
            off = max(0.0, total - cnt)
            if site and cnt >= 2 and off >= 2:
                label = "サイト+外角"
            elif site and cnt >= 2:
                label = "サイト内保持"
            elif total >= 2:
                label = "サイト外"
            else:
                label = "不明"
            out["post_plant"] = {
                "label": label, "site": site, "on_site": round(cnt, 1),
                "off_site": round(off, 1), "route": _top_callout(pp, site),
                "success": winner_no == team_no,
            }

    if first_kill_team == team_no and first_kill_t is not None:
        pick = _counts(ps, float(first_kill_t) - 8, float(first_kill_t) + 8)
        pz = _dominant(pick) or "?"
        if side == ATTACK:
            if final_zone and final_zone != pz:
                label = f"{pz}で1ピック→{final_zone}"
            elif final_zone:
                label = f"{pz}で1ピック→{pz}維持"
            else:
                label = f"{pz}で1ピック"
        elif side == DEFENSE:
            label = f"{pz}で1ピック→{'リテイク' if plant is not None else '維持'}"
        else:
            label = None
        if label:
            out["one_pick"] = {"label": label, "zone": pz,
                               "time": round(float(first_kill_t) - start, 1)}
    return out
