"""構成の特徴：何が強い・何が弱い、今のメタにどう刺さるか、アンチピック。

人が書いたメモ（kb の comps.yaml）と、エージェントの特徴（agents.yaml の traits）から決まる一般的な見立て、
DB のラウンドから数えた相手の型ごとの勝率を並べる。
"""
from __future__ import annotations

from collections import Counter, defaultdict

ROLE_JA = {"duelist": "デュエリスト", "initiator": "イニシエーター", "controller": "コントローラー", "sentinel": "センチネル"}


def profile(comp, kb) -> dict:
    roles, traits, tiers = Counter(), Counter(), {}
    who = defaultdict(list)
    for a in comp or []:
        d = kb.agent(a)
        if d.get("role"):
            roles[d["role"]] += 1
        for t in d.get("traits") or []:
            traits[t] += 1
            who[t].append(a)
        if d.get("tier"):
            tiers[a] = d["tier"]
    return {"comp": list(comp or []), "roles": dict(roles), "traits": dict(traits), "who": dict(who),
            "tiers": tiers, "archetypes": archetypes(comp, kb)}


def archetypes(comp, kb) -> list[str]:
    """構成の型の名前（相手の構成を分けるのにも使う）。"""
    roles = Counter(kb.role(a) for a in comp or [])
    names = {kb.agent(a).get("name") or a for a in comp or []}
    out = []
    if roles["duelist"] >= 2:
        out.append("2 デュエ")
    if roles["controller"] >= 2:
        out.append("2 コン")
    if roles["initiator"] >= 2:
        out.append("2 イニシ")
    if roles["sentinel"] >= 2:
        out.append("2 センチネル")
    if "KAY/O" in names:
        out.append("KAY/O 入り")
    if names & {"Chamber"} or (names & {"Jett"} and roles["sentinel"] >= 1):
        out.append("オペ持ち")
    if not any("flash" in (kb.agent(a).get("traits") or []) for a in comp or []):
        out.append("フラッシュなし")
    if roles["duelist"] == 0 and comp:
        out.append("デュエなし")
    return out or (["標準（1-1-1-2 など）"] if comp else [])


def rules(p: dict) -> dict:
    """特徴（traits）からの一般的な見立て。{"attack": {"strong": [], "weak": []}, "defense": {...}}"""
    t, who = p["traits"], p["who"]
    W = lambda k: "・".join(who.get(k) or [])
    out = {"attack": {"strong": [], "weak": []}, "defense": {"strong": [], "weak": []}}
    a, d = out["attack"], out["defense"]
    if t.get("flash", 0) == 0:
        a["weak"].append("フラッシュが無い → チョークポイントを抜けるのが難しい（索敵と撃ち合いで抜ける形を決めておく）")
    elif t["flash"] >= 2:
        a["strong"].append(f"フラッシュが多い（{W('flash')}）→ フラッシュ＋索敵＋デュエリストの形でチョークを抜けやすい")
    if t.get("recon", 0) == 0:
        d["weak"].append("索敵が無い → ドライで詰められると情報が遅れる（足音・ジャンピで取る）")
        a["weak"].append("索敵が無い → オフアングル・定点待ちを先に割れない")
    if t.get("smoke", 0) >= 2:
        a["strong"].append(f"スモークが 2 枚以上（{W('smoke')}）→ 視界を割ってエリアを取り、スプリットしやすい")
    if t.get("dry", 0) or t.get("op", 0):
        a["strong"].append(f"ドライ・オペが強い（{W('dry') or W('op')}）→ 相手の動きに後出しじゃんけんができる")
        d["strong"].append(f"1 ピックが取りやすい（{W('dry') or W('op')}）")
    if t.get("anti_rush", 0):
        d["strong"].append(f"ラッシュ止め（{W('anti_rush')}）→ 2 デュエの速いエントリーに強い")
    if t.get("postplant", 0) >= 2:
        a["strong"].append(f"設置後のスキルが多い（{W('postplant')}）→ ポストプラントが強い")
    if t.get("trap", 0) >= 2:
        d["strong"].append(f"置き情報が多い（{W('trap')}）→ 少ない人数で 1 か所を見られ、人数を寄せやすい")
    if t.get("suppress", 0):
        a["strong"].append(f"スキル封じ（{W('suppress')}）→ 罠・壁の多いセンチネル構成を崩せる")
    if p["roles"].get("duelist", 0) >= 2:
        a["strong"].append("デュエが 2 人 → 速いエントリーで人数差を作れる")
        d["weak"].append("サイトを長く守るスキルが少ない")
    if p["roles"].get("duelist", 0) == 0:
        a["weak"].append("デュエがいない → 最初に入る人を決めておく")
    if t.get("mobility", 0) == 0:
        a["weak"].append("移動スキルが無い → 入った後の展開が遅くなりやすい")
    return out


def _rate(w, n, base=0.5, k=3.0):
    return (w + base * k) / (n + k) if n + k else None


def data(rows: list[dict], kb) -> dict:
    """相手の型ごとの勝率・相手のエージェントがいるときの勝率（アンチピックの候補）。rows は 1 つの構成の行。"""
    by_arch = defaultdict(lambda: [0, 0])
    by_agent = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    seen_maps = defaultdict(set)
    for r in rows:
        if r.get("pistol"):
            continue
        oc = r.get("opp_comp") or ()
        tot[0] += bool(r["win"])
        tot[1] += 1
        for a in archetypes(oc, kb):
            by_arch[a][0] += bool(r["win"])
            by_arch[a][1] += 1
            seen_maps[("arch", a)].add((r["match_id"], r["order"]))
        for a in oc:
            by_agent[a][0] += bool(r["win"])
            by_agent[a][1] += 1
            seen_maps[("agent", a)].add((r["match_id"], r["order"]))
    base = tot[0] / tot[1] if tot[1] else 0.5
    arch = [{"name": a, "w": w, "n": n, "rate": w / n if n else None, "adj": _rate(w, n, base),
             "maps": len(seen_maps[("arch", a)])} for a, (w, n) in by_arch.items()]
    arch.sort(key=lambda x: x["adj"] or 0)
    ag = [{"name": a, "w": w, "n": n, "rate": w / n if n else None, "adj": _rate(w, n, base),
           "maps": len(seen_maps[("agent", a)])} for a, (w, n) in by_agent.items() if n >= 6]
    ag.sort(key=lambda x: x["adj"] or 0)
    anti = [x for x in arch if x["n"] >= 6 and (x["adj"] or 0) <= base - 0.06][:3]
    anti_agents = [x for x in ag if (x["adj"] or 0) <= base - 0.06][:3]
    good = [x for x in reversed(arch) if x["n"] >= 6 and (x["adj"] or 0) >= base + 0.06][:3]
    return {"base": base, "rounds": tot[1], "archetypes": arch, "agents": ag, "anti": anti,
            "anti_agents": anti_agents, "good": good}


def report(comp, rows: list[dict], kb, map_name: str | None = None) -> dict:
    p = profile(comp, kb)
    return {"profile": p, "rules": rules(p), "notes": kb.comp_notes(comp, map_name), "data": data(rows, kb)}
