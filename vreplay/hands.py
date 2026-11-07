"""ラウンドの「手」（攻め・守りの型）、ターン（アクション → リアクションの応酬）、次のラウンドの読み替え、
手 × 手の相性表（じゃんけん）。プレイブックの行（deck.records）から計算する。

手の判定条件（kb の maps/<map>.yaml の hands.<side>[].when）
------------------------------------------------------------
攻め（attack）
  site: A                      攻め先（設置したサイト。設置が無ければ 3 人以上がそのサイトに入ったところ）。リストも可
  commit_by: 30                開始から 30 秒以内にサイトに入った（設置または 3 人以上）
  commit_after: 50             50 秒以降に入った
  via_any: [Mid Top, B Market] 入る前 25 秒（入らなければ 0〜60 秒）に、どれかに 1 人以上いた
  via_all: [[..], [..]]        グループごとに 1 人以上（2 方向から割る＝スプリット）
  via_min: {areas: [..], n: 3} その場所の合計が n 人以上
  shown: Mid                   1:15 の初動で人数が多かったゾーン（見せた場所）。site と違えばフェイク
  no_plant: true               設置していない
守り（defense）。既定は開始 5〜25 秒の、点がいちばん多く見えた時刻で判定（window: [5, 25] で変えられる）
  zone_min: {A: 3}             そのゾーンに 3 人以上
  zone_max: {A: 1}             そのゾーンに 1 人以下
  area_min: {Mid Top: 2}       その場所に 2 人以上
  push: Mid                    開始 30 秒までに、攻め側の陣地にあるそのゾーンの場所へ出た（push_n 人以上、既定 1）
  flat: true                   どのサイトにも 1 人以上いて、3 人以上のゾーンが無い
共通
  buy: [フル, ハーフ]           購入（deck の buy）。省けば問わない
エリア名は valorant-api のコールアウト（「A Main」「Mid Top」）。前方一致・大文字小文字は無視。
"""
from __future__ import annotations

from collections import Counter, defaultdict

ATTACK, DEFENSE = "攻撃", "防衛"
SITES = ("A", "B", "C")
ZONES = ("A", "Mid", "B", "C")
SIDE_KEY = {ATTACK: "attack", DEFENSE: "defense"}


def _low(s) -> str:
    return str(s or "").strip().lower()


def _area_in(area: str | None, names) -> bool:
    a = _low(area)
    if not a:
        return False
    for nm in names or []:
        n = _low(nm)
        if a == n or a.startswith(n + " ") or a.startswith(n):
            return True
    return False


LANES = ("main", "lobby", "long", "short")


def _lane(area: str) -> bool:
    """サイトの手前の通路（A Main・B Lobby など）。ここに 3 人いてもまだ「サイトに入った」とはしない。"""
    return any(w in _low(area).split() for w in LANES)


# ---------------------------------------------------------------- 時刻ごとの人数
def samples(r: dict) -> dict[float, dict]:
    """開始からの秒 → {"zones": Counter, "areas": Counter}。10 秒ごとの snaps（追跡）があればそれを優先し、
    その間はミニマップの点（pos）で埋める。スポーンは数えない。"""
    out: dict[float, dict] = {}
    by = defaultdict(lambda: {"zones": Counter(), "areas": Counter()})
    for e, zone, area in r.get("pos") or []:
        k = round(float(e))
        by[k]["zones"][zone] += 1
        by[k]["areas"][area] += 1
    for k, v in by.items():
        out[float(k)] = v
    for e, v in (r.get("snaps") or {}).items():
        if not v:
            continue
        zs = Counter({z: int(c) for z, c in (v.get("zones") or {}).items() if z in ZONES})
        ar = Counter({a: int(c) for a, c in (v.get("areas") or {}).items()
                      if a and not str(a).startswith(("攻撃側", "防衛側")) and "spawn" not in _low(a)})
        if zs or ar:
            out[float(e)] = {"zones": zs, "areas": ar, "src": v.get("src")}
    return dict(sorted(out.items()))


def at(sm: dict, e: float, tol: float = 3.5) -> dict | None:
    if not sm:
        return None
    k = min(sm, key=lambda x: abs(x - e))
    return sm[k] if abs(k - e) <= tol else None


def _rep(sm: dict, lo: float, hi: float, prefer: float) -> dict | None:
    """lo〜hi 秒で点がいちばん多く見えた時刻（同じなら prefer に近い方）。"""
    cand = [(min(5, sum(v["zones"].values())), -abs(e - prefer), e) for e, v in sm.items() if lo <= e <= hi]
    if not cand:
        return None
    return sm[max(cand)[2]]


# ---------------------------------------------------------------- 特徴
def features(r: dict) -> dict:
    sm = samples(r)
    f: dict = {"aligned": bool(sm)}
    if r["side"] == ATTACK:
        commit_e, commit_site = None, None
        from . import tactics as TC
        for e, v in sm.items():
            # サイトに入った＝そのゾーンの「入口の通路（Main・Lobby・攻め側の陣地）」より先に 3 人
            inside = Counter()
            for a, c in v["areas"].items():
                z = a.split(" ")[0]
                if z in SITES and not _lane(a) and not TC.attacker_half(a, r.get("ranks"), margin=0.5):
                    inside[z] += c
            if not v["areas"]:
                inside = Counter({z: c for z, c in v["zones"].items() if z in SITES})
            s = next((z for z in SITES if inside.get(z, 0) >= 3), None)
            if s:
                commit_e, commit_site = e, s
                break
        site = r.get("plant_site") or commit_site
        pe = r.get("plant_e")
        if pe is not None and (commit_e is None or pe < commit_e):
            commit_e = pe
        f.update(site=site, commit_e=commit_e, planted=bool(r.get("planted")))
        lo, hi = ((commit_e - 25, commit_e) if commit_e is not None else (0, 60))
        via = Counter()
        for e, v in sm.items():
            if lo <= e <= hi:
                for a, c in v["areas"].items():
                    via[a] = max(via[a], c)
        f["via"] = dict(via)
        su = r.get("setup") or {}
        shown = su.get("area_zone")
        if not shown:
            v = _rep(sm, 15, 30, 25)
            if v and v["zones"]:
                shown = v["zones"].most_common(1)[0][0]
        f["shown"] = shown
        f["key_area"] = su.get("area")
    else:
        v = _rep(sm, 5, 25, 20)
        f["zones"] = dict(v["zones"]) if v else {}
        f["areas"] = dict(v["areas"]) if v else {}
        push = Counter()
        from . import tactics as TC
        for e, v2 in sm.items():
            if e > 30:
                continue
            for a, c in v2["areas"].items():
                if TC.attacker_half(a, r.get("ranks")):
                    z = a.split(" ")[0] if a.split(" ")[0] in ZONES else None
                    if z:
                        push[z] = max(push[z], c)
        f["push"] = dict(push)
    return f


def _cmp(have: dict, need: dict, ge: bool) -> bool:
    for k, n in (need or {}).items():
        c = sum(v for kk, v in have.items() if _low(kk) == _low(k) or _low(kk).startswith(_low(k) + " "))
        if ge and c < int(n):
            return False
        if not ge and c > int(n):
            return False
    return True


def matches(when: dict, f: dict, r: dict) -> bool:
    if not when:
        return False
    if when.get("buy"):
        b = when["buy"] if isinstance(when["buy"], list) else [when["buy"]]
        if r.get("buy") not in b:
            return False
    if r["side"] == ATTACK:
        if "site" in when:
            s = when["site"] if isinstance(when["site"], list) else [when["site"]]
            if f.get("site") not in s:
                return False
        if "commit_by" in when and (f.get("commit_e") is None or f["commit_e"] > float(when["commit_by"])):
            return False
        if "commit_after" in when and (f.get("commit_e") is None or f["commit_e"] < float(when["commit_after"])):
            return False
        via = f.get("via") or {}
        if "via_any" in when and not any(_area_in(a, when["via_any"]) for a in via):
            return False
        for grp in when.get("via_all") or []:
            if not any(_area_in(a, grp) for a in via):
                return False
        vm = when.get("via_min")
        if vm and sum(c for a, c in via.items() if _area_in(a, vm.get("areas"))) < int(vm.get("n", 1)):
            return False
        if "shown" in when and _low(f.get("shown")) != _low(when["shown"]):
            return False
        if when.get("no_plant") and f.get("planted"):
            return False
        return True
    zones, areas = f.get("zones") or {}, f.get("areas") or {}
    if not zones and not f.get("push"):
        return False
    if "zone_min" in when and not _cmp(zones, when["zone_min"], True):
        return False
    if "zone_max" in when and not _cmp(zones, when["zone_max"], False):
        return False
    if "area_min" in when and not _cmp(areas, when["area_min"], True):
        return False
    if "push" in when:
        ps = when["push"] if isinstance(when["push"], list) else [when["push"]]
        n = int(when.get("push_n", 1))
        if not any((f.get("push") or {}).get(z, 0) >= n for z in ps):
            return False
    if when.get("flat"):
        if not zones or any(zones.get(z, 0) < 1 for z in ("A", "B")) or max(zones.values()) >= 3:
            return False
    return True


def generic(f: dict, r: dict) -> str:
    """知識ベースに当てはまる手が無いときの名前。"""
    if r["side"] == ATTACK:
        site = f.get("site")
        if not site:
            ka = f.get("key_area")
            return f"設置なし（{ka} まで）" if ka else "設置なし"
        ce = f.get("commit_e")
        zs = {a.split(" ")[0] for a, c in (f.get("via") or {}).items() if c}
        if ce is not None and ce <= 30:
            return f"{site} ラッシュ"
        if "Mid" in zs and site in zs:
            return f"MID 経由 {site} スプリット"
        if f.get("shown") and f["shown"] != site:
            return f"{'MID' if f['shown'] == 'Mid' else f['shown']} 見せ → {site}"
        if ce is not None and ce >= 60:
            return f"{site} レイト"
        return f"{site} execute"
    push = {z: c for z, c in (f.get("push") or {}).items() if c}
    if push:
        z = max(push, key=push.get)
        return f"{'MID' if z == 'Mid' else z} プッシュ"
    zs = f.get("zones") or {}
    if not zs:
        return "不明"
    top = max(zs.items(), key=lambda kv: (kv[1], kv[0] != "Mid"))
    if top[0] in SITES and top[1] >= 3:
        return f"{top[0]}{top[1]}"
    if zs.get("Mid", 0) >= 2:
        return "mid focus"
    if all(zs.get(z, 0) >= 1 for z in ("A", "B")) and top[1] <= 2:
        return "平たく守る"
    return "守りデフォ"


def classify(r: dict, kbmap: dict | None) -> dict:
    """1 行（1 チーム × 1 ラウンド）の手。{"name", "kb", "style", "detail", "feat"}"""
    f = features(r)
    if not f.get("aligned") and not r.get("plant_site"):
        return {"name": "不明", "kb": False, "feat": f}
    for h in ((kbmap or {}).get("hands") or {}).get(SIDE_KEY[r["side"]]) or []:
        if matches(h.get("when") or {}, f, r):
            return {"name": h["name"], "kb": True, "style": h.get("style"), "detail": h.get("detail"), "feat": f}
    return {"name": generic(f, r), "kb": False, "feat": f}


def annotate(rows: list[dict], kb=None) -> list[dict]:
    """rows のすべてに r["hand"] と、相手の手 r["opp_hand"] を付ける。"""
    from . import kb as K
    kb = kb or K.load()
    cache = {}
    for r in rows:
        mp = r.get("map")
        if mp not in cache:
            cache[mp] = kb.map(mp)
        r["hand"] = classify(r, cache[mp])
    by = {(r["match_id"], r["order"], r["n"], r["no"]): r for r in rows}
    for r in rows:
        o = by.get((r["match_id"], r["order"], r["n"], 3 - r["no"]))
        r["opp_hand"] = (o or {}).get("hand")
        r["_opp_row"] = o
    return rows


# ---------------------------------------------------------------- 相性表（じゃんけん）
def pairs(rows: list[dict]) -> list[dict]:
    """1 ラウンド＝1 組（攻めの行・守りの行）。"""
    out, seen = [], set()
    for r in rows:
        if r["side"] != ATTACK or not r.get("hand"):
            continue
        k = (r["match_id"], r["order"], r["n"])
        if k in seen:
            continue
        seen.add(k)
        d = r.get("_opp_row")
        if not d or not d.get("hand"):
            continue
        out.append({"atk": r, "def": d, "a": r["hand"]["name"], "d": d["hand"]["name"], "atk_win": bool(r["win"])})
    return out


def matrix(rows: list[dict], skip=("不明",), eco: bool = False) -> dict:
    """攻めの手 × 守りの手 → 攻めの勝ち/回数。ピストル・エコは既定で外す。"""
    m = defaultdict(lambda: [0, 0])
    for p in pairs(rows):
        a, d = p["atk"], p["def"]
        if not eco and (a.get("pistol") or a.get("buy") in ("エコ", "ピストル") or d.get("buy") in ("エコ", "ピストル")):
            continue
        if p["a"] in skip or p["d"] in skip:
            continue
        c = m[(p["a"], p["d"])]
        c[0] += p["atk_win"]
        c[1] += 1
    atk = Counter()
    dfn = Counter()
    for (a, d), (w, n) in m.items():
        atk[a] += n
        dfn[d] += n
    return {"cells": {k: tuple(v) for k, v in m.items()}, "atk": [a for a, _ in atk.most_common()],
            "def": [d for d, _ in dfn.most_common()]}


def kb_edge(kbmap: dict, a: str, d: str) -> str | None:
    """知識ベースの相性：攻めの手 a と守りの手 d のどちらが有利か（'攻め'／'守り'／None）。"""
    hs = (kbmap or {}).get("hands") or {}
    ah = next((h for h in hs.get("attack") or [] if h["name"] == a), None)
    dh = next((h for h in hs.get("defense") or [] if h["name"] == d), None)
    if (ah and d in (ah.get("beats") or [])) or (dh and a in (dh.get("loses_to") or [])):
        return "攻め"
    if (ah and d in (ah.get("loses_to") or [])) or (dh and a in (dh.get("beats") or [])):
        return "守り"
    return None


# ---------------------------------------------------------------- ターン（1 ラウンドの応酬）
def _clock(e: float) -> str:
    left = max(0, 100 - int(round(e)))
    return f"{left // 60}:{left % 60:02d}"


def turns(atk: dict, dfn: dict, step: float = 10.0) -> list[dict]:
    """攻め・守りの行から、時間順の出来事。[{"e", "side", "kind", "text", "reaction"}]。
    10 秒ごとの配置の変化（テイク・前に出る・ローテ・転換）と、キル・設置。
    相手の出来事から 15 秒以内の出来事は、その「リアクション」として印を付ける。"""
    from . import tactics as TC
    ev = []
    sa, sd = samples(atk), samples(dfn)
    taken: set = set()
    prev_major = None
    prev_def: Counter = Counter()
    pushed = False
    e = 0.0
    end = max([0.0] + list(sa) + list(sd))
    plant = atk.get("plant_e")
    while e <= min(end, 100.0):
        va, vd = at(sa, e, 3.5), at(sd, e, 3.5)
        if va:
            for a, c in va["areas"].most_common():
                if c >= 2 and a not in taken and not TC.is_spawn(a.split(" ")[0], a):
                    taken.add(a)
                    ev.append({"e": e, "side": ATTACK, "kind": "take", "text": f"{a} に {c} 人"})
                    break
            core = {z: c for z, c in va["zones"].items() if z in ZONES}
            if core:
                major = max(core, key=core.get)
                if core[major] >= 3 and prev_major and major != prev_major and (plant is None or e < plant):
                    ev.append({"e": e, "side": ATTACK, "kind": "shift",
                               "text": f"{'MID' if prev_major == 'Mid' else prev_major} → "
                                       f"{'MID' if major == 'Mid' else major} へ転換"})
                if core[major] >= 3:
                    prev_major = major
        if vd:
            if not pushed and e <= 30:
                fw = [a for a, c in vd["areas"].items() if c and TC.attacker_half(a, dfn.get("ranks"))]
                if fw:
                    pushed = True
                    ev.append({"e": e, "side": DEFENSE, "kind": "push", "text": f"{fw[0]} へ前に出る"})
            cur = Counter({z: c for z, c in vd["zones"].items() if z in SITES})
            if prev_def:
                for z in SITES:
                    if cur.get(z, 0) - prev_def.get(z, 0) >= 2:
                        frm = [x for x in SITES + ("Mid",) if prev_def.get(x, 0) > cur.get(x, 0) and x != z]
                        ev.append({"e": e, "side": DEFENSE, "kind": "rotate",
                                   "text": f"{('MID' if frm[0] == 'Mid' else frm[0]) + ' → ' if frm else ''}{z} へ "
                                           f"+{cur[z] - prev_def.get(z, 0)}（ローテ）"})
            if cur:
                prev_def = cur
        e += step
    for k, (e_, own) in enumerate(atk.get("feed") or []):
        if k == 0:
            ev.append({"e": e_, "side": ATTACK if own else DEFENSE, "kind": "first",
                       "text": f"先制キル（{'攻め' if own else '守り'}）"})
    # 5 秒以内に同じ側が 2 キル以上 → 集団戦
    fd = atk.get("feed") or []
    i = 0
    while i < len(fd):
        j = i
        while j + 1 < len(fd) and fd[j + 1][0] - fd[j][0] <= 5:
            j += 1
        if j > i:
            a_ = sum(1 for x in fd[i:j + 1] if x[1])
            d_ = j - i + 1 - a_
            ev.append({"e": fd[i][0], "side": ATTACK if a_ >= d_ else DEFENSE, "kind": "fight",
                       "text": f"集団戦 攻{a_}-守{d_}"})
        i = j + 1
    if plant is not None:
        ev.append({"e": plant, "side": ATTACK, "kind": "plant", "text": f"{atk.get('plant_site') or ''} 設置".strip()})
    ev.sort(key=lambda x: (x["e"], x["kind"] != "take"))
    last = {ATTACK: None, DEFENSE: None}
    for x in ev:
        other = DEFENSE if x["side"] == ATTACK else ATTACK
        x["reaction"] = last[other] is not None and 0 <= x["e"] - last[other] <= 15 and x["kind"] not in ("first", "fight")
        last[x["side"]] = x["e"]
        x["clock"] = _clock(x["e"])
    return ev


def turn_line(ev: list[dict], k: int = 7) -> str:
    """「攻: B Main に 3 人(1:28) → 守: A → B へ +2（ローテ）(1:15) → …」"""
    out = []
    for x in ev[:k]:
        s = "攻" if x["side"] == ATTACK else "守"
        out.append(f"{'↪ ' if x['reaction'] else ''}{s}: {x['text']}（{x['clock']}）")
    return " → ".join(out)


# ---------------------------------------------------------------- 次のラウンドの読み替え
def reads(rows: list[dict], skip=("不明",)) -> list[dict]:
    """同じ試合・同じ側の続くラウンドで、負けた側が手をどう変えたか。
    [{"side", "lost_with", "lost_to", "next", "n", "w"}]（負けた手 → 次の手 の回数と、その次のラウンドの勝ち）"""
    seq = defaultdict(list)
    for r in rows:
        if r.get("hand") and r.get("opp_hand") and not r.get("pistol"):
            seq[(r["match_id"], r["order"], r["team"], r["side"])].append(r)
    agg = defaultdict(lambda: [0, 0])
    for key, rs in seq.items():
        rs.sort(key=lambda r: r["n"])
        for a, b in zip(rs, rs[1:]):
            if b["n"] - a["n"] > 2 or a["win"]:
                continue
            h, o, nx = a["hand"]["name"], a["opp_hand"]["name"], b["hand"]["name"]
            if h in skip or nx in skip:
                continue
            c = agg[(a["side"], h, o, nx)]
            c[0] += 1
            c[1] += bool(b["win"])
    out = [{"side": s, "lost_with": h, "lost_to": o, "next": nx, "n": n, "w": w, "changed": nx != h}
           for (s, h, o, nx), (n, w) in agg.items()]
    out.sort(key=lambda x: (-x["n"], -x["w"]))
    return out


def cycles_seen(rows: list[dict], kbmap: dict, k: int = 3) -> list[dict]:
    """知識ベースの「じゃんけんの回り方」（cycles）の、隣り合う 2 手が実際に続いたラウンド（勝った手）。"""
    out = []
    for cyc in (kbmap or {}).get("cycles") or []:
        steps = []
        for x, y in zip(cyc, cyc[1:] + cyc[:1]):
            # x に対して y が出て、y の側が勝った
            hit = []
            for p in pairs(rows):
                if (p["a"] == y and p["d"] == x and p["atk_win"]) or (p["d"] == y and p["a"] == x and not p["atk_win"]):
                    hit.append(p)
            steps.append({"from": x, "to": y, "n": len(hit), "ex": hit[:k]})
        out.append({"cycle": cyc, "steps": steps})
    return out


# ---------------------------------------------------------------- 1 ピック・トレード
def trade_rate(rows: list[dict], window: float = 4.0) -> tuple[int, int]:
    """味方が倒されてから window 秒以内に倒し返した回数 / 味方が倒された回数。"""
    t = d = 0
    for r in rows:
        fd = r.get("feed") or []
        for i, (e, own) in enumerate(fd):
            if own:
                continue
            d += 1
            if any(own2 and 0 < e2 - e <= window for e2, own2 in fd[i + 1:]):
                t += 1
    return t, d


def first_pick_shapes(rows: list[dict], k: int = 5) -> list[dict]:
    """先制キル（1 ピック）を取ったときの形：場所 × 取った選手のエージェント → 回数・勝ち。"""
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r.get("fk_own") or r.get("fk_e") is None:
            continue
        who = r.get("fk_player")
        ag = (r.get("agent_of") or {}).get(who) or "?"
        where = r.get("first_fight") or "?"
        tm = "20 秒以内" if r["fk_e"] <= 20 else "20〜45 秒" if r["fk_e"] <= 45 else "45 秒以降"
        c = agg[(where, ag, tm)]
        c[0] += 1
        c[1] += bool(r["win"])
    out = [{"where": w, "agent": a, "time": t, "n": n, "w": wn} for (w, a, t), (n, wn) in agg.items()]
    out.sort(key=lambda x: -x["n"])
    return out[:k]


def summary(rows: list[dict]) -> dict:
    """手ごとの回数・勝ち（側ごと）。"""
    out = {ATTACK: Counter(), DEFENSE: Counter()}
    wins = {ATTACK: Counter(), DEFENSE: Counter()}
    for r in rows:
        h = (r.get("hand") or {}).get("name")
        if not h:
            continue
        out[r["side"]][h] += 1
        wins[r["side"]][h] += bool(r["win"])
    return {s: [{"name": h, "n": n, "w": wins[s][h]} for h, n in out[s].most_common()] for s in out}
