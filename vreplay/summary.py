"""ラウンド表・詳細の表示用の文字列（Notion と HTML で共通）。"""
from __future__ import annotations

from collections import Counter

HOW_ICON = {"elim": "💀", "boom": "💥", "defuse": "🔧", "time": "⏱"}
# 値段の高い順（表で目立たせたい武器を先に）
ORDER = ["Operator", "Outlaw", "Marshal", "Odin", "Ares", "Vandal", "Phantom", "Guardian", "Bulldog", "Judge",
         "Bucky", "Spectre", "Stinger", "Warden", "Sheriff", "Bandit", "Ghost", "Frenzy", "Shorty", "Classic"]


def weapon_summary(players: list[dict] | None) -> str:
    """[{weapon, weapon_ja}] → 'ヴァンダル×3・オペレーター'"""
    if not players:
        return ""
    c = Counter(p.get("weapon") for p in players if p.get("weapon"))
    ja = {p.get("weapon"): p.get("weapon_ja") for p in players if p.get("weapon")}
    parts = [f"{ja[w]}×{n}" if n > 1 else ja[w] for w, n in sorted(c.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99)]
    unknown = sum(1 for p in players if not p.get("weapon"))
    if unknown:
        parts.append(f"不明×{unknown}")
    return "・".join(parts)


def tag(rep: dict, team: str | None) -> str:
    if team == rep["team1"]:
        return rep["tag1"]
    if team == rep["team2"]:
        return rep["tag2"]
    return team or "?"


def first_kill(rep: dict, r: dict) -> str:
    ks = r.get("kills") or []
    if not ks:
        return "-"
    k = ks[0]
    return f"{k.get('killer') or '?'}→{k.get('victim') or '?'}（{tag(rep, k.get('killer_team'))}）"


def kill_count(rep: dict, r: dict) -> str:
    ks = r.get("kills") or []
    if not ks:
        return "-"
    a = sum(1 for k in ks if k.get("killer_team") == rep["team1"])
    b = sum(1 for k in ks if k.get("killer_team") == rep["team2"])
    return f"{a}-{b}"


def ult_used(r: dict) -> str:
    us = r.get("ults") or []
    return "、".join(f"{u.get('player')}" + (f"（{u['agent']}）" if u.get("agent") else "") for u in us) or "-"


def buy_cell(r: dict, no: str) -> str:
    """'フル 22.5k\nヴァンダル×3・オペレーター'"""
    econ = r.get(f"buy{no}") or "-"
    w = weapon_summary((r.get("weapons") or {}).get(no))
    return f"{econ}\n{w}" if w else econ


def how(r: dict) -> str:
    h = r.get("how_ja") or "-"
    icon = HOW_ICON.get(r.get("how") or "", "")
    side = r.get("side")
    return f"{icon}{h}" + (f"（{side}）" if side else "")


def player_weapons(r: dict, no: str) -> str:
    ps = (r.get("weapons") or {}).get(no) or []
    return " / ".join(f"{p['player']} {p.get('weapon_ja') or '?'}" for p in ps)
