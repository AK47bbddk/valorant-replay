"""プレイブックを「初心者がそのまま真似できる手順」に変える。

deck の型の行・指標から、
  - この構成の形（役割の内訳と、そこから言えること）
  - 5 人の担当（誰がどのエリアに立つか）
  - 型ごとの「やること」の 1〜2 文（指示形。数字は根拠として括弧に残す）
  - 覚える順番（①ピストル → ②攻めのデフォルト → ③2 巡目 → ④守りのデフォルト → ⑤リテイク）
  - 用語の説明（テイクデフォ・後取り・前目・タイミングプッシュ…）
を作る。プレミアやフルパで 5 人そろえたときに、上から順に読めば同じことができるのを目標にする。

役割（デュエリスト／イニシエーター／コントローラー／センチネル）は valorant-api から取り、
取れないときだけ内蔵の表を使う（新しいエージェントは「?」になる）。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

API = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
CACHE_DAYS = 30

ROLE_JA = {"duelist": "デュエリスト", "initiator": "イニシエーター",
           "controller": "コントローラー", "sentinel": "センチネル"}
# valorant-api が使えないとき用。あとから出たエージェントは「?」になる（取れたときは API の値が勝つ）
FALLBACK_ROLES = {
    "jett": "duelist", "phoenix": "duelist", "raze": "duelist", "reyna": "duelist", "yoru": "duelist",
    "neon": "duelist", "iso": "duelist", "waylay": "duelist",
    "sova": "initiator", "breach": "initiator", "skye": "initiator", "kayo": "initiator",
    "fade": "initiator", "gekko": "initiator", "tejo": "initiator",
    "brimstone": "controller", "omen": "controller", "viper": "controller", "astra": "controller",
    "harbor": "controller", "clove": "controller",
    "cypher": "sentinel", "killjoy": "sentinel", "sage": "sentinel", "chamber": "sentinel",
    "deadlock": "sentinel", "vyse": "sentinel",
}
GLOSSARY = {
    "テイクデフォ": "そのエリアを取りに行くデフォルト（1 巡目の立ち位置）。2 人以上で入って情報とエリアを取る",
    "デフォルト": "ラウンドの始めの立ち位置と、そこからの決まった動き。いきなり攻めずに情報を取る形",
    "1 巡目": "ラウンド開始〜1:15 ごろ。エリアを取って相手の人数を見るまで",
    "2 巡目": "1:05 ごろ〜設置まで。1:00 に開幕のスキルが戻るので、ここで取り返し（後取り）や仕掛けが起きる",
    "後取り": "1 巡目では取らなかった手前のエリアを、2 巡目で取り返してから入ること",
    "捨て": "片方のサイトを見せて（人数を寄せさせて）、そこを捨てて別のサイトへ行くこと",
    "スプリット": "2 方向から挟んでサイトに入ること",
    "ラッシュ": "スキルを使って 5 人でまっすぐ速く入ること（0:30 以内に設置）",
    "レイト": "1:00 以降まで時間を使ってから入ること",
    "execution": "デフォルトで取ったエリアから、スキルを合わせてサイトに入ること（エグゼキュート）",
    "ポストプラント": "設置したあとの立ち位置。解除を止めるために散って時間を使う",
    "前目": "守りで、攻め側の陣地の近くまで出て待つこと",
    "3人守り": "そのサイトに 3 人置いて強気に守ること（「<サイト>強気」とも言う）",
    "mid focus": "MID に 2 人以上置いて、真ん中から相手の動きを止める守り",
    "MID 寄せ": "MID に 2 人以上置いて、真ん中から相手の動きを止める守り",
    "広がる配置": "どこにも 2 人固めず、1 人ずつ広く見る守り",
    "タイミングプッシュ": "守りがデフォルトのあと（1:15〜1:00）に、攻め側のエリアへ出て当たりに行くこと",
    "カウンター": "設置前の早い時間に当たって先制キルを取る守り",
    "リテイク": "設置されたサイトを取り返すこと",
    "ファストリテイク": "設置されてすぐ（12 秒以内）に人数をそろえて入り、解除の時間を作ること",
    "Anti eco": "相手がお金のないラウンド。こちらはフルバイで、固まらず撃ち合いで取り切る",
    "Eco": "こちらがお金を節約するラウンド。固まって 1 ピックを狙うか、次のために貯める",
    "Pistol": "1 ラウンド目と 13 ラウンド目。勝つと次の 2 ラウンドも取りやすい",
    "ドライ": "スキルを使わずに人数と撃ち合いだけで入ること",
}
# 役割の内訳から言える、構成の形の話（桑田さんの記事の「何が強くて何が弱いのか」）
COMP_NOTES = {
    "controller2": ("コントローラーが 2 人：スモークを広く焚けるので、本命のサイトがばれにくく、エリアを安く広く取れる",
                    "スモークに頼る分、入るときのフラッシュやスタンが足りなくなりやすい"),
    "initiator2": ("イニシエーターが 2 人：索敵とフラッシュが多く、入るときの事故が少ない",
                   "スモークが 1 枚なので、サイトを隠しきれずに本命がばれやすい"),
    "duelist0": ("デュエリスト無し：スキルで固めてから入るので、無理な撃ち合いが減る",
                 "先に入って切り開く役がいないので、時間を使う形になりやすい（ラッシュが弱い）"),
    "duelist2": ("デュエリスト 2 人：速い攻めとエントリーが強い",
                 "情報とスモークが減るので、相手の形を読めないまま入ることが増える"),
    "sentinel2": ("センチネルが 2 人：守りで足止めが多く、少人数でもサイトを守れる",
                  "攻めで使えるスキルが減るので、エリアの取り合いで押し負けやすい"),
    "sentinel0": ("センチネル無し：全員が前に出られるので、エリアを広く取れる",
                  "裏取りや速い攻めへの足止めが無いので、守りで事故が増える"),
}


def _norm(s: str) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def agent_roles(cfg: dict | None, names: list[str], log=None) -> dict[str, str]:
    """エージェント名 → 役割（英語の小文字。分からなければ "?"）。valorant-api を 30 日キャッシュ。"""
    meta = None
    p = None
    if cfg and (cfg.get("output") or {}).get("dir"):
        p = Path(cfg["output"]["dir"]) / "_cache" / "agent_roles.json"
        if p.exists() and time.time() - p.stat().st_mtime < CACHE_DAYS * 86400:
            try:
                meta = json.loads(p.read_text("utf-8"))
            except Exception:
                meta = None
    if meta is None:
        try:
            with urlopen(API, timeout=20) as r:
                meta = {_norm(a["displayName"]): _norm((a.get("role") or {}).get("displayName"))
                        for a in json.load(r)["data"]}
            if p:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        except Exception as e:
            if log:
                log(f"    エージェントの役割を取れませんでした（{e}）。内蔵の表を使います")
            meta = {}
    out = {}
    for n in dict.fromkeys(names):
        out[n] = meta.get(_norm(n)) or FALLBACK_ROLES.get(_norm(n)) or "?"
    return out


def comp_notes(roles: dict[str, str]) -> dict:
    """役割の内訳と、そこから言える強み・弱み。"""
    c = Counter(r for r in roles.values() if r and r != "?")
    order = ("duelist", "initiator", "controller", "sentinel")
    line = "・".join(f"{ROLE_JA[r]} {c[r]}" for r in order if c.get(r))
    if c.get("?") or any(v == "?" for v in roles.values()):
        line += "（役割の分からないエージェントあり）"
    keys = []
    for r in order:
        n = c.get(r, 0)
        if n >= 2:
            keys.append(f"{r}2")
        elif n == 0:
            keys.append(f"{r}0")
    # 食い違う指摘は落とす（2 コントローラーなら「スモークが 1 枚」とは言わない、など）
    skip = set()
    if c.get("controller", 0) >= 2:
        skip.add(("initiator2", "weak"))
    if c.get("initiator", 0) >= 2:
        skip.add(("controller2", "weak"))
    if c.get("duelist", 0) >= 1:
        skip.add(("duelist0", "weak"))
    strong = [COMP_NOTES[k][0] for k in keys if k in COMP_NOTES and (k, "strong") not in skip]
    weak = [COMP_NOTES[k][1] for k in keys if k in COMP_NOTES and (k, "weak") not in skip]
    return {"line": line or "役割が分かりません", "counts": dict(c), "strong": strong, "weak": weak,
            "roles": dict(roles),
            "agents": [{"agent": a, "role": ROLE_JA.get(r, "?"), "role_en": r}
                       for a, r in sorted(roles.items())]}


def duties(rows: list[dict], top: int = 1) -> list[dict]:
    """5 人の担当：エージェント → よく立っていたエリア（アイコン照合できたラウンドだけ）。

    返り値 [{"agent", "areas": [(エリア, 回数)], "n"}]。照合できていなければ空。
    """
    by: dict[str, Counter] = {}
    for r in rows:
        su = r.get("setup") or {}
        who = su.get("who_area") or {}
        for area, ags in who.items():
            for a in ags:
                by.setdefault(a, Counter())[area] += 1
    out = []
    for a, c in by.items():
        out.append({"agent": a, "areas": c.most_common(max(1, top)), "n": sum(c.values())})
    out.sort(key=lambda d: -d["n"])
    return out


def _n_text(g: dict) -> str:
    from .deck import _pct, _rate
    return f"{g['n']} 回・勝ち {g['w']}（{_pct(_rate(g['w'], g['n']))}）"


def _spread_text(g: dict, k: int = 5) -> str:
    sp = g.get("spread") or {}
    items = sorted(sp.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return "・".join(f"{a} {v:g}人" for a, v in items)


def row_todo(g: dict, side: str) -> str:
    """型の行の「やること」。指示形の 1〜2 文。"""
    from .deck import ATTACK
    kind, name = g["kind"], g["name"]
    spread = _spread_text(g)
    sites = g.get("sites") or Counter()
    site = g.get("site")
    if kind in ("Default", "Push") and side == ATTACK:
        t = f"この立ち位置で始める（{spread}）。" if spread else "この立ち位置で始める。"
        if g.get("area"):
            t += f"{g['area']} を 2 人で取って、1:15 に相手の人数を見る。"
        if sites:
            t += "そのあとの攻め先：" + "、".join(f"{s} {c}" for s, c in sites.most_common()) + "。"
        return t + f"（{_n_text(g)}）"
    if kind in ("Default", "Push"):
        t = f"この立ち位置で守る（{spread}）。" if spread else "この立ち位置で守る。"
        if name.endswith("前目"):
            t += f"{g.get('area')} まで出て、攻めが来る前に当たる。来なければデフォルトの位置に戻る。"
        elif "3人守り" in name:
            t += "そのサイトに 3 人置いて強気に守り、反対側はリテイク前提にする。"
        elif name.startswith("mid focus"):
            t += "MID に 2 人以上置いて、真ん中を通られないようにする。"
        else:
            t += "固めずに広く見て、音とスキルで相手の本命を判断してから寄る。"
        return t + f"（{_n_text(g)}）"
    if kind == "Rush":
        return (f"スキルを全部使って {site} に 5 人でまっすぐ入る。設置まで 30 秒以内。"
                f"迷ったら止まらずに入り切る。（{_n_text(g)}）")
    if kind == "Split":
        return (f"{g.get('shown')} に人数を見せてから、そこを捨てて {site} に入る。"
                f"見せる側は音とスキルだけ使って、キルは取らなくてよい。（{_n_text(g)}）")
    if kind == "Execute" and name.endswith("後取り"):
        area = name.replace(" 後取り", "")
        return (f"1 巡目で取らなかった {area} を、1:05 以降（相手のスキルが戻るころ）に取り返してから {site} に入る。"
                f"（{_n_text(g)}）")
    if kind == "Execute" and "レイト" in name:
        return (f"1:00 以降まで時間を使い、相手のスキルを吐かせてから {site} に入る。"
                f"残り 0:45 までには設置する。（{_n_text(g)}）")
    if kind == "Execute":
        return (f"デフォルトで取ったエリアから、スキルを合わせて {site} に入る。"
                f"人数の少ない側に行くのが基本。（{_n_text(g)}）")
    if kind == "PostPlant":
        return (f"{site} に設置したら散って、解除の音に合わせて撃つ。固まらない。（{_n_text(g)}）")
    if kind == "Retake":
        if "ファスト" in name:
            return (f"{site} を設置されたら、12 秒以内に人数をそろえて一気に入る。（{_n_text(g)}）")
        if "遅い" in name:
            return (f"{site} を設置されたら、すぐ入らずに人数と ULT をそろえてから入る。"
                    f"解除の時間（7 秒）は残しておく。（{_n_text(g)}）")
        return f"{site} を設置されたら取り返す。入る前に人数をそろえる。（{_n_text(g)}）"
    if kind == "Counter":
        place = name.replace(" カウンター", "")
        return (f"開始すぐ（20 秒以内）に {place} で当たって先制キルを取り、取ったら下がる。"
                f"（{_n_text(g)}）")
    if kind == "Pistol":
        return f"ピストルはこの形で入る。勝つと次の 2 ラウンドも取りやすい。（{_n_text(g)}）"
    if kind == "AntiEco":
        return f"相手がお金のないラウンド。固まらず距離を取って撃ち合う。（{_n_text(g)}）"
    if kind == "Eco":
        return f"お金を節約するラウンド。固まって 1 ピックを狙うか、次に備える。（{_n_text(g)}）"
    return _n_text(g)


def steps(prs_atk: list[dict], prs_def: list[dict], comp: dict | None = None,
          duty: list[dict] | None = None) -> list[dict]:
    """覚える順番。[{"no", "title", "items": [{"name", "todo"}], "rows"}]"""
    out = []
    if comp:
        items = [{"name": "構成", "todo": comp["line"]}]
        for d in (duty or [])[:5]:
            items.append({"name": d["agent"], "todo": f"{d['areas'][0][0]} に立つことが多い（{d['areas'][0][1]} 回）"})
        if len(items) == 1:
            items.append({"name": "担当", "todo": "下の Default の立ち位置を 5 人で割り振る"})
        out.append({"no": 1, "title": "5 人の役割と担当を決める", "items": items, "rows": []})

    def pick(prs, kinds, k=1, key=None):
        part = [g for g in prs if g["kind"] in kinds]
        part.sort(key=key or (lambda g: -g["n"]))
        return part[:k]

    no = len(out) + 1
    for title, part in (
        ("攻めのデフォルト（1 巡目）", pick(prs_atk, ("Default", "Push"), 2)),
        ("攻めの 2 巡目（よく勝っている順）", pick(
            prs_atk, ("Rush", "Split", "Execute"), 2,
            key=lambda g: (-(g["w"] / g["n"] if g["n"] else 0), -g["n"]))),
        ("設置したあと", pick(prs_atk, ("PostPlant",), 2)),
        ("守りのデフォルト（1 巡目）", pick(prs_def, ("Default", "Push"), 2)),
        ("取り返し（リテイク・カウンター）", pick(prs_def, ("Retake", "Counter"), 2)),
        ("ピストルとお金のないラウンド", pick(prs_atk, ("Pistol", "AntiEco", "Eco"), 2)),
    ):
        if not part:
            continue
        out.append({"no": no, "title": title, "rows": part,
                    "items": [{"name": g["name"], "todo": g.get("todo") or ""} for g in part]})
        no += 1
    return out


def glossary_for(texts: list[str], k: int = 12) -> list[tuple[str, str]]:
    """出てきた言葉だけの用語集。"""
    joined = "　".join(str(t or "") for t in texts)
    out = [(w, d) for w, d in GLOSSARY.items() if w in joined]
    for w in ("1 巡目", "2 巡目", "デフォルト"):          # 基本の言葉は必ず入れる
        if w not in [x for x, _ in out] and w in GLOSSARY:
            out.insert(0, (w, GLOSSARY[w]))
    return out[:k]


# ---------------------------------------------------------------- 難しさ（ランク帯・フルパ向けの但し書き）
# 型の種類ごとの、そろえるものと難しさ。プロの型をそのまま持ってくると、
# スキルの精度やコールが要るものがあるので、それを先に書いておく。
DIFFICULTY = {
    "Rush": ("やさしい", "決めごとが少ないので、野良でもそろえやすい。合図（誰が先に入るか）だけ決める"),
    "Pistol": ("やさしい", "1 ラウンド目だけなので、形を 1 つ覚えれば使える"),
    "AntiEco": ("やさしい", "固まらず距離を取るだけ。スキルはほぼ要らない"),
    "Eco": ("やさしい", "固まって 1 ピックを狙うだけ。無理に勝ちに行かない"),
    "Default": ("ふつう", "立ち位置を 5 人で割り振れば真似できる。誰がどこを見るかだけ決めておく"),
    "Execute": ("ふつう", "入る合図とスキルの順番を決めておく。バラバラに入らない"),
    "PostPlant": ("ふつう", "設置後の立ち位置を覚えるだけ。固まらないのがいちばん大事"),
    "Retake": ("ふつう", "入る前に人数をそろえる。ばらばらに入ると数的不利になる"),
    "Split": ("むずかしい", "見せる側と入る側で時間を合わせる必要がある。声を出せる 5 人向け"),
    "Counter": ("むずかしい", "出る時間と引く時間を全員で合わせる。外すと数的不利になる"),
    "Push": ("むずかしい", "出る時間（1:15〜1:00）を全員で合わせる。相手に読まれると弱い"),
}


def difficulty(g: dict, side: str | None = None) -> dict:
    """その型を真似するときの難しさと、そろえておくこと。"""
    kind, name = g.get("kind") or "", g.get("name") or ""
    level, note = DIFFICULTY.get(kind, ("ふつう", "立ち位置を 5 人で割り振れば真似できる"))
    if "後取り" in name or "レイト" in name:
        level, note = "むずかしい", "1:05・残り 0:45 といった時間を全員が覚えてから使う"
    elif name.endswith("前目"):
        level, note = "むずかしい", "出る時間と引く時間を合わせる。来なければ必ずデフォルトの位置へ戻る"
    elif "ファストリテイク" in name:
        level, note = "むずかしい", "設置から 12 秒以内に 4 人そろえる。声が出せないと間に合わない"
    elif "3人守り" in name:
        level, note = "やさしい", "3 人をそのサイトに置くだけ。反対側はリテイク前提と決めておく"
    rows = g.get("rows") or []
    ults = sum(1 for r in rows if len(r.get("ults") or []) >= 2)
    if rows and ults / len(rows) >= 0.5:
        note += "。ULT を 2 つ以上使うラウンドが多い（" + f"{ults}/{len(rows)}）ので、溜まっていないときは別の型にする"
    return {"level": level, "note": note}


# ---------------------------------------------------------------- 構成の代替案
# 同じ役割の中でも、できることが違うグループ（分かっているものだけ）
SUBGROUP = {
    "viper": "壁型スモーク", "harbor": "壁型スモーク",
    "omen": "置くスモーク", "brimstone": "置くスモーク", "astra": "置くスモーク", "clove": "置くスモーク",
    "killjoy": "設置型センチネル", "cypher": "設置型センチネル", "vyse": "設置型センチネル",
    "deadlock": "設置型センチネル",
    "sage": "足止め・蘇生", "chamber": "狙撃型センチネル",
    "sova": "索敵イニシエーター", "fade": "索敵イニシエーター", "gekko": "索敵イニシエーター",
    "tejo": "索敵イニシエーター",
    "breach": "フラッシュ・スタン", "skye": "フラッシュ・スタン", "kayo": "フラッシュ・スタン",
    "jett": "入り込むデュエリスト", "raze": "入り込むデュエリスト", "neon": "入り込むデュエリスト",
    "yoru": "だまし型デュエリスト", "phoenix": "自分で入るデュエリスト", "reyna": "撃ち合いデュエリスト",
    "iso": "撃ち合いデュエリスト",
}


# 内蔵の表で名前を出すとき、そのままでは読めないもの
FALLBACK_NAMES = {"kayo": "KAY/O"}


def all_roles(cfg: dict | None = None, log=None) -> dict[str, str]:
    """全エージェントの表示名 → 役割。API が使えないときは内蔵の表。"""
    names = None
    try:
        meta = None
        if cfg and (cfg.get("output") or {}).get("dir"):
            p = Path(cfg["output"]["dir"]) / "_cache" / "agent_names.json"
            if p.exists() and time.time() - p.stat().st_mtime < CACHE_DAYS * 86400:
                meta = json.loads(p.read_text("utf-8"))
        if meta is None:
            with urlopen(API, timeout=20) as r:
                meta = {a["displayName"]: _norm((a.get("role") or {}).get("displayName"))
                        for a in json.load(r)["data"]}
            if cfg and (cfg.get("output") or {}).get("dir"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
        names = {k: v for k, v in (meta or {}).items() if v}
    except Exception as e:
        if log:
            log(f"    エージェントの一覧を取れませんでした（{e}）")
    return names or {FALLBACK_NAMES.get(n, n.title()): r for n, r in FALLBACK_ROLES.items()}


def alternatives(roles: dict[str, str], cfg: dict | None = None, k: int = 3) -> list[dict]:
    """「このエージェントがいないとき、誰で代われるか」。同じ役割の中から出す。

    スキルの中身までは見ていないので、できることが近いグループ（壁型スモーク・設置型センチネルなど）が
    分かるものはそれも添える。
    """
    pool = all_roles(cfg)
    have = {_norm(a) for a in roles}
    out = []
    for agent, role in sorted(roles.items()):
        if role in (None, "", "?"):
            out.append({"agent": agent, "role": "?", "subgroup": None, "picks": [], "near": []})
            continue
        same = [n for n, r in sorted(pool.items()) if r == role and _norm(n) not in have]
        sub = SUBGROUP.get(_norm(agent))
        near = [a for a in same if SUBGROUP.get(_norm(a)) == sub] if sub else []
        picks = (near + [a for a in same if a not in near])[:k]
        out.append({"agent": agent, "role": ROLE_JA.get(role, role), "subgroup": sub,
                    "picks": picks, "near": near[:k]})
    return out
