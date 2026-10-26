"""前半・後半 12 ラウンドの組み立て（ラウンドの種類の並び）と、同じ初動の形からの分岐。

ラウンドの種類（チームの視点）：
  ピストル → アンチエコ／エコ → ボーナス／フォース → フルバイ … のように、自分と相手の購入と前のラウンドの結果から決める。
  付け足し：ULT を使った数（2 個以上は「ULTラウンド」）、自分がオペレーターを持っている「オペ」、
  相手がオペレーターを持っている「対オペ」。
  例：「フルバイ＋ULT2（対オペ）」

分岐（攻撃側）：
  1:15 前後の初動の配置（tactics の setups）でいちばん人数の多いゾーンと、実際に設置した（設置が無ければ 0:40 に
  いちばん人数の多い）サイトを比べる。違えば「フェイク A→B」、MID から入れば「MID→B」、同じなら「A 直行」。
  同じ形のラウンドを並べると、前半に見せた形から後半でフェイクに変えた、といった組み立てが見える。
"""
from __future__ import annotations

from collections import Counter, defaultdict

SITES = ("A", "B", "C")
ECO_LOW = {"eco", "semi-eco"}
ECO_MID = {"half", "semi-buy"}


def _half(n: int) -> str:
    return "前半" if n <= 12 else "後半" if n <= 24 else "OT"


def _half_start(n: int) -> int:
    return 1 if n <= 12 else 13 if n <= 24 else n


def _side_of(rr: dict, no: int) -> str | None:
    s1 = rr.get("side1")
    if s1 not in ("攻撃", "防衛"):
        return None
    if no == 1:
        return s1
    return "防衛" if s1 == "攻撃" else "攻撃"


def _has_op(rr: dict, no: int) -> bool:
    for p in (rr.get("weapons") or {}).get(str(no)) or []:
        if (p.get("weapon") or "").lower() == "operator":
            return True
    return False


def round_type(rep: dict, rr: dict, no: int, prev: dict[int, dict] | None = None) -> dict:
    """1 ラウンド・1 チームの種類。prev は同じマップの {n: round}（前のラウンドの勝敗を見る）。"""
    n = int(rr.get("n") or 0)
    me = rep.get(f"team{no}")
    my = rr.get(f"eco{no}")
    op = rr.get(f"eco{3 - no}")
    hs = _half_start(n)
    idx = n - hs                                   # 半分の中で何番目か（0 始まり）
    won_prev = []
    for k in range(hs, n):
        p = (prev or {}).get(k)
        if p is not None:
            won_prev.append(p.get("winner_no") == no)
    if n in (1, 13):
        typ = "ピストル"
    elif n > 24:
        typ = "OT"
    elif idx == 2 and won_prev[:2] == [True, True] and my not in ("full",):
        typ = "ボーナス"
    elif my in ECO_LOW:
        typ = "エコ"
    elif my in ECO_MID and op in ECO_LOW:
        typ = "アンチエコ"
    elif my in ECO_MID:
        typ = "フォース" if won_prev and not won_prev[-1] else "ハーフ"
    elif my == "full":
        typ = "アンチエコ" if op in ECO_LOW or op == "pistol" else "対フォース" if op in ECO_MID else "フルバイ"
    elif my == "pistol":
        typ = "ピストル"
    else:
        typ = "?"
    ults = [u.get("agent") or u.get("player") or "?" for u in rr.get("ults") or [] if u.get("team") == me]
    mods = []
    if len(ults) >= 2:
        mods.append(f"ULT{len(ults)}")
    elif ults:
        mods.append("ULT1")
    if _has_op(rr, no):
        mods.append("オペ")
    if _has_op(rr, 3 - no):
        mods.append("対オペ")
    label = typ + (f"＋{mods[0]}" if mods and mods[0].startswith("ULT") else "")
    extra = [m for m in mods if not m.startswith("ULT")]
    if extra:
        label += f"（{'・'.join(extra)}）"
    return {"type": typ, "mods": mods, "label": label, "ults": ults, "ult_round": len(ults) >= 2,
            "half": _half(n), "idx": idx + 1, "side": _side_of(rr, no), "win": rr.get("winner_no") == no}


def all_types(rep: dict) -> dict[tuple[int, int], dict]:
    by_n = {int(r.get("n") or 0): r for r in rep.get("rounds") or []}
    return {(n, no): round_type(rep, rr, no, by_n) for n, rr in sorted(by_n.items()) for no in (1, 2)}


def flow(rep: dict) -> list[dict]:
    """チーム × 半分ごとの並び：[{"no", "tag", "half", "side", "items": [(n, label, win)]}]"""
    types = all_types(rep)
    out = []
    for no in (1, 2):
        tag = rep.get(f"tag{no}") or rep.get(f"team{no}")
        by_half = defaultdict(list)
        side_of = {}
        for (n, k), t in sorted(types.items()):
            if k != no:
                continue
            by_half[t["half"]].append((n, t["label"], t["win"]))
            side_of.setdefault(t["half"], t["side"])
        for h in ("前半", "後半", "OT"):
            if by_half.get(h):
                out.append({"no": no, "tag": tag, "half": h, "side": side_of.get(h), "items": by_half[h]})
    return out


def flow_text(item: dict) -> str:
    """'前半（攻撃）R1 ピストル✗ → R2 エコ✗ → R3 フォース○ …'"""
    seq = " → ".join(f"R{n} {lab}{'○' if w else '✗'}" for n, lab, w in item["items"])
    w = sum(1 for _, _, x in item["items"] if x)
    return f"{item['half']}（{item['side'] or '?'}・{w}-{len(item['items']) - w}）{seq}"


# ---------------------------------------------------------------- 分岐
def _top_zone(zones: dict | None, prefer=SITES + ("Mid",)) -> str | None:
    z = {k: v for k, v in (zones or {}).items() if k in prefer and v}
    if not z:
        return None
    best = max(z.values())
    tops = [k for k in prefer if z.get(k) == best]
    return tops[0] if len(tops) == 1 else "Mid" if "Mid" in tops else None


def branch(rep: dict, rr: dict, no: int) -> dict | None:
    """攻撃側の 1 ラウンドの「見せた形 → 実際の攻め先」。"""
    if _side_of(rr, no) != "攻撃":
        return None
    rt = rr.get("tactics") or {}
    su = (rt.get("setups") or {}).get(str(no)) or {}
    if not su:
        return None
    start = _top_zone(su.get("zones"))
    final = rt.get("plant_site") if rt.get("plant_site") in SITES else None
    if final is None:
        tl = (rt.get("timeline") or {})
        for mark in ("0:40", "0:20", "1:00"):
            final = _top_zone((tl.get(mark) or {}).get(str(no)), prefer=SITES)
            if final:
                break
    if not final:
        tag = None
    elif not start:
        tag = f"分散→{final}"
    elif start == final:
        tag = f"{final}直行"
    elif start == "Mid":
        tag = f"MID→{final}"
    else:
        tag = f"フェイク {start}→{final}"
    return {"setup": su.get("name"), "shape": su.get("shape"), "start": start, "final": final, "tag": tag,
            "planted": rt.get("plant_site") in SITES}


def branches(rep: dict) -> list[dict]:
    """攻撃側の同じ初動の形ごとに、ラウンドの並びと攻め先。[{"no", "tag", "setup", "rounds": [...] , "note"}]"""
    out = []
    for no in (1, 2):
        groups = defaultdict(list)
        for rr in sorted(rep.get("rounds") or [], key=lambda r: int(r.get("n") or 0)):
            b = branch(rep, rr, no)
            if not b or not b["setup"]:
                continue
            n = int(rr.get("n") or 0)
            groups[b["setup"]].append({"n": n, "half": _half(n), "tag": b["tag"], "final": b["final"],
                                       "win": rr.get("winner_no") == no, "shape": b["shape"]})
        for setup, rs in groups.items():
            finals = Counter(r["final"] for r in rs if r["final"])
            note = ""
            if len(finals) >= 2:
                first = next((r for r in rs if r["final"]), None)
                later = [r for r in rs if r["final"] and first and r["final"] != first["final"]]
                if first and later:
                    note = (f"R{first['n']} で {first['final']} に行った形から、"
                            + "・".join(f"R{r['n']} は {r['final']}" for r in later[:3]) + " に分岐")
            out.append({"no": no, "tag": rep.get(f"tag{no}") or rep.get(f"team{no}"), "setup": setup,
                        "rounds": rs, "finals": dict(finals), "note": note})
    out.sort(key=lambda g: (g["no"], -len(g["rounds"])))
    return out


def branch_text(g: dict) -> str:
    seq = "、".join(f"R{r['n']} {r['tag'] or '?'}{'○' if r['win'] else '✗'}" for r in g["rounds"])
    return f"{g['tag']}「{g['setup']}」{len(g['rounds'])} 回：{seq}" + (f"（{g['note']}）" if g["note"] else "")


# ---------------------------------------------------------------- 複数マップの集計（プレイブック用）
def plan_summary(rows: list[dict]) -> dict:
    """deck.records の行（1 チームの視点）から、半分の中の何ラウンド目に何をしがちか。
    返り値 {"by_idx": {idx: Counter(type)}, "after_pistol_loss": Counter, "after_pistol_win": Counter,
            "ult_rounds": Counter(idx), "ult_rate": {idx: 平均 ULT 数}}"""
    by_idx = defaultdict(Counter)
    after_loss, after_win = Counter(), Counter()
    ult_rounds, ult_sum, cnt = Counter(), Counter(), Counter()
    by_map = defaultdict(dict)
    for r in rows:
        if r.get("rtype"):
            by_map[(r["match_id"], r["order"], r["no"])][r["n"]] = r
    for rs in by_map.values():
        for n, r in rs.items():
            t = r["rtype"]
            i = t["idx"]
            if t["half"] == "OT":
                continue
            by_idx[i][t["type"]] += 1
            cnt[i] += 1
            ult_sum[i] += len(t.get("ults") or [])
            if t.get("ult_round"):
                ult_rounds[i] += 1
            if i == 2:
                p = rs.get(n - 1)
                if p is not None:
                    (after_win if p["win"] else after_loss)[t["type"]] += 1
    return {"by_idx": dict(by_idx), "after_pistol_loss": after_loss, "after_pistol_win": after_win,
            "ult_rounds": ult_rounds, "ult_rate": {i: ult_sum[i] / cnt[i] for i in cnt if cnt[i]}}
