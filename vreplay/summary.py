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


def ult_used(r: dict, rep: dict | None = None) -> str:
    us = r.get("ults") or []
    out = []
    for u in us:
        who = u.get("player") or (f"枠{u['slot']}" if u.get("slot") else "?")
        s = who + (f"（{u['agent']}）" if u.get("agent") else "")
        if rep and u.get("team"):
            s += f"［{u['team']}］"
        out.append(s)
    return "、".join(out) or "-"


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
    """'Life ヴァンダル＋ゴースト / Nicc スペクター'（サブ武器はメインと違うときだけ）"""
    ps = (r.get("weapons") or {}).get(no) or []
    out = []
    for p in ps:
        s = f"{p['player']} {p.get('weapon_ja') or '?'}"
        if p.get("sidearm") and p.get("sidearm") != p.get("weapon"):
            s += f"＋{p.get('sidearm_ja') or p['sidearm']}"
        out.append(s)
    return " / ".join(out)


REASON_ICON = {"エース": "🔥", "クラッチ": "🧊", "リテイク": "🔁", "ポストプラント成功": "💣", "ラッシュ": "⚡",
               "カウンター": "🛡️", "数的不利から逆転": "↩️", "エコ勝ち": "💰", "1ピック": "🎯", "先制されて逆転": "↩️",
               "時間切れまで守り切り": "⏱", "設置前に全滅": "💀", "設置させずに守り勝ち": "🛡️", "アンチエコ": "💰",
               "ピストル": "🔫", "初動ファイト勝ち": "🎯", "初動ファイト負けから逆転": "↩️", "エントリー成功": "🚪",
               "オペレーターで1ピック": "🔭", "トレード勝ち": "🤝", "マルチキル": "💥", "ULTリテイク": "🌀",
               "ULTで攻め切り": "🌀", "フォースバイ勝ち": "💰"}


def _icon(tag: str) -> str:
    if tag in REASON_ICON:
        return REASON_ICON[tag]
    if tag.startswith("初動") and tag.endswith("ファイト勝ち"):
        return REASON_ICON["初動ファイト勝ち"]
    if tag.startswith("初動"):
        return REASON_ICON["初動ファイト負けから逆転"]
    if tag.endswith("エントリー成功"):
        return REASON_ICON["エントリー成功"]
    return ""


def reason_cell(r: dict) -> str:
    """表の「勝因」欄：主な勝因＋次の 1 つ（'💣ポストプラント成功・1ピック'）。"""
    tags = (r.get("reason") or {}).get("tags") or []
    if not tags:
        return "-"
    return _icon(tags[0]) + "・".join(tags[:2])


def reason_icon(r: dict) -> str:
    main = (r.get("reason") or {}).get("main")
    return _icon(main or "") or "🏁"


def setup_line(rep: dict, r: dict) -> str:
    """初動の配置：'G2（守り）バランス A2 Mid1 B2 ／ 100T（攻め）MID 3 デフォルト Mid3 A1 B1'"""
    sus = ((r.get("tactics") or {}).get("setups")) or {}
    out = []
    for no, tg in (("1", rep.get("tag1")), ("2", rep.get("tag2"))):
        su = sus.get(no)
        if not su:
            continue
        side = "攻め" if su.get("side") == "攻撃" else "守り" if su.get("side") == "防衛" else ""
        out.append(f"{tg}（{side}）{su.get('name')} {su.get('shape') or ''}".strip())
    return " ／ ".join(out)
