"""プレイブックを Notion にも書く（チーム内でよく使う形にそろえる）。

ページの形（例：「delight split vyse comp」）
  # 概要
    ## ATK summary   … 方針の箇条書き（Claude があればその文章、なければ strong / weak point から）
    [ATK のデータベース] … 1 行 = 1 つの型（初動の配置の型・セット・ポストプラント）。名前は「1-3-1 MID デフォルト」など
    ## DEF summary
    [DEF のデータベース] … 初動の配置の型・リテイク
  ## メモ            … 解説の言及・weak point・データの量

データベースの各行のページは、左に説明（回数・勝率・形・コールアウト・その後の展開）、
右にその型のときの位置（マップ画像に重ねたもの）を並べ、その下に例のラウンドの動画と、
動画のすぐ下にトグルで畳んだそのラウンドの分析（勝因・ミニマップ GIF・開始時の配置図・購入と武器・キル・ULT）を置く。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from . import deck as D
from .notion import Notion, _bullet, _columns, _h, _img, _para, _rt, round_blocks

ATTACK, DEFENSE = D.ATTACK, D.DEFENSE


def page_id(s: str) -> str | None:
    """Notion のページ URL か ID から 32 桁の ID。"""
    if not s:
        return None
    m = re.findall(r"[0-9a-f]{32}", s.replace("-", "").lower())
    return m[-1] if m else None


def _video(url: str) -> dict:
    return {"type": "video", "video": {"type": "external", "external": {"url": url}}}


def _db(nt: Notion, parent: str, title: str, kind_options: list[str]) -> str:
    r = nt.req("POST", "/databases", {
        "parent": {"type": "page_id", "page_id": parent}, "is_inline": True,
        "title": [_rt(title)],
        "initial_data_source": {"properties": {
            "名前": {"title": {}},
            "種類": {"select": {"options": [{"name": n} for n in kind_options]}},
            "タグ": {"multi_select": {}},
            "回数": {"number": {}},
            "勝率": {"number": {"format": "percent"}},
            "購入": {"multi_select": {}},
        }}})
    return r["data_sources"][0]["id"]


class _Uploader:
    """1 冊を書くあいだ、同じ画像は 1 回だけアップロードする（例のラウンドは複数の型に出てくる）。"""

    def __init__(self, nt: Notion):
        self.nt, self.ids = nt, {}

    def __call__(self, path) -> str | None:
        if not path or not Path(path).exists():
            return None
        key = str(path)
        if key not in self.ids:
            try:
                self.ids[key] = self.nt.upload(key)
            except Exception:
                self.ids[key] = None
        return self.ids[key]


def _round_toggle(r: dict, rep: dict | None, up) -> dict | None:
    """例のラウンドの動画の下に置く、畳んだ「このラウンドの分析」。"""
    if not rep:
        return None
    rr = next((x for x in rep.get("rounds") or [] if int(x.get("n") or 0) == int(r["n"])), None)
    if rr is None:
        return None
    a = rr.get("assets") or {}
    uploads = {p: up(p) for p in a.values() if p}
    kids = []
    bits = []
    if (r.get("rtype") or {}).get("label"):
        bits.append(f"組み立て：{r['rtype']['label']}")
    if r.get("setup"):
        bits.append(f"初動の形：{r['setup'].get('name')}" + (f"（{r['setup']['shape']}）" if r["setup"].get("shape") else ""))
    if (r.get("branch") or {}).get("tag"):
        bits.append(f"攻め先：{r['branch']['tag']}")
    if r.get("opp_setup"):
        bits.append(f"相手の形：{r['opp_setup'].get('name')}")
    if bits:
        kids.append(_para(_rt("　".join(bits), bold=True)))
    kids += round_blocks(rep, rr, uploads, flat=True)
    title = (f"このラウンドの分析（{r.get('tag')} vs {r.get('opp_tag')} {r.get('map')} R{r['n']}"
             f"・ミニマップ GIF・配置図・キル・武器・ULT）")
    return {"type": "toggle", "toggle": {"rich_text": [_rt(title)], "children": kids[:100]}}


def _row(nt: Notion, ds: str, name: str, kind: str, rows: list[dict], tags: list[str], body_left: list,
         img_path: Path | None, examples: list[dict], reps: dict | None = None, up=None):
    n = len(rows)
    w = sum(r["win"] for r in rows)
    buys = sorted({r["buy"] for r in rows if r.get("buy") and r["buy"] != "?"})
    pg = nt.create_page(ds, {
        "名前": {"title": [_rt(name)]}, "種類": {"select": {"name": kind}},
        "タグ": {"multi_select": [{"name": t[:100]} for t in dict.fromkeys(tags) if t][:8]},
        "回数": {"number": n}, "勝率": {"number": round(w / n, 3) if n else None},
        "購入": {"multi_select": [{"name": b} for b in buys]}})
    left = list(body_left)
    up = up or _Uploader(nt)
    fid = up(img_path) if img_path else None
    right = [_img(fid, "この型のときの位置（全ラウンドの重ね合わせ）")] if fid else []
    cl = _columns(left, right)
    nt.append(pg["id"], [cl] if cl else left)
    # 例のラウンド：動画のすぐ下に、そのラウンドの分析をトグルで畳んで置く
    # （トグルの中に横並びは入れられない＝入れ子の段数の制限。画像は縦に並べる）
    ex = []
    for r in examples:
        ex.append(_para(_rt(f"vs {r['opp_tag']} R{r['n']}（{'勝ち' if r['win'] else '負け'}"
                            + (f"・{r['reason_main']}" if r.get("reason_main") else "") + "）", bold=True)))
        ex.append(_video(r["yt"]))
        tg = _round_toggle(r, (reps or {}).get((str(r["match_id"]), r["order"])), up)
        if tg:
            ex.append(tg)
    for b in ex:                                        # トグルの中身が多いので 1 つずつ送る
        nt.append(pg["id"], [b])


def _plan_blocks(rows: list[dict], side: str, label: str) -> list[dict]:
    """12R の組み立て（何ラウンド目に何をしがちか・試合ごとの並び）と、攻撃なら同じ初動の形からの分岐。"""
    d = D.plan_data(rows).get(side)
    out = []
    if d:
        ps = d["summary"]
        out.append(_h(3, f"{label} 12R の組み立て"))
        for i in range(1, 13):
            c = ps["by_idx"].get(i)
            if not c:
                continue
            w, n = d["wins"].get(i, [0, 0])
            ult = ps["ult_rate"].get(i, 0.0)
            out.append(_bullet(_rt(f"{i}R 目　", bold=True),
                               _rt("・".join(f"{t} {v}" for t, v in c.most_common(3))
                                   + f"　勝ち {w}/{n}　ULT 平均 {ult:.1f}"
                                   + (f"（ULTラウンド {ps['ult_rounds'][i]}）" if ps["ult_rounds"].get(i) else ""))))
        for name, cnt in (("ピストル勝ちの 2R", ps["after_pistol_win"]), ("ピストル負けの 2R", ps["after_pistol_loss"])):
            if cnt:
                out.append(_para(_rt(f"{name}：" + "・".join(f"{t} {v}" for t, v in cnt.most_common()), color="gray")))
        for (dt, opp, mp, h), rs in list(d["seqs"].items())[-4:]:
            out.append(_bullet(_rt(f"vs {opp} {h}　", bold=True),
                               _rt(" → ".join(f"{r['rtype']['label']}{'○' if r['win'] else '✗'}" for r in rs))))
    if side == ATTACK:
        bg = [g for g in D.branch_groups(rows) if g["n"] >= 2]
        if bg:
            out.append(_h(3, "同じ初動の形からの分岐"))
            for g in bg[:8]:
                tags = "・".join(f"{t} {c}" for t, c in g["tags"].most_common())
                seq = "、".join(f"{r['opp_tag']} R{r['n']} {(r.get('branch') or {}).get('tag') or '?'}"
                               f"{'○' if r['win'] else '✗'}" for r in g["rows"][:8])
                out.append(_bullet(_rt(f"{g['name']}（{g['shape']}）　{g['w']}/{g['n']} 勝・フェイク {len(g['fakes'])}　",
                                       bold=True), _rt(f"{tags}｜{seq}")))
    return out


def _code(shape: str) -> str:
    """'A1 Mid3 B1' → '1-3-1'（A-MID-B の人数）。C があるマップは 'A1 B2 C1 Mid1' → '1-2-1・MID1'"""
    z = {m.group(1): int(m.group(2)) for m in re.finditer(r"(A|B|C|Mid)(\d)", shape or "")}
    if not z:
        return ""
    if "C" in z:
        return f"{z.get('A', 0)}-{z.get('B', 0)}-{z.get('C', 0)}" + (f"・MID{z['Mid']}" if z.get("Mid") else "")
    return f"{z.get('A', 0)}-{z.get('Mid', 0)}-{z.get('B', 0)}"


def _examples(rows: list[dict], k: int = 3) -> list[dict]:
    rs = [r for r in rows if r.get("yt")]
    rs.sort(key=lambda r: (not r["win"], str(r["date"]), r["n"]))
    return rs[:k]


def _caster_tags(rows: list[dict]) -> list[str]:
    c = Counter(t for r in rows for t in (r.get("caster") or {}).get("tags", []))
    return [t for t, _ in c.most_common(3)]


def _summary_lines(llm_text: str | None, side_label: str, strong: list[dict], weak: list[dict]) -> list[dict]:
    if llm_text:
        # Claude の文章から該当する見出しの下だけ取り出す
        part, take = [], False
        for line in llm_text.splitlines():
            t = line.strip().lstrip("#").strip()
            if not t:
                continue
            if t.lower().startswith(("atk summary", "def summary", "対策メモ")):
                take = t.lower().startswith(side_label.lower())
                continue
            if take:
                part.append(_bullet(_rt(t.lstrip("-・* "))))
        if part:
            return part
    out = [_bullet(_rt("強み：", bold=True), _rt(p["title"])) for p in strong]
    out += [_bullet(_rt("弱み：", bold=True), _rt(p["title"])) for p in weak]
    return out or [_para(_rt("まだデータが少なく、はっきりした傾向は出ていません", color="gray"))]


def export(nt: Notion, cfg: dict, key: dict, rows: list[dict], base: list[dict], parent: str, img_dir: Path,
           log=print, reps: dict | None = None) -> str:
    """1 冊ぶんを Notion に書く。同じ名前のページが親の下にあれば中身を作り直す。返り値はページ URL。"""
    title = key["title"]
    pid = None
    for b in nt.children(parent):
        if b.get("type") == "child_page" and (b.get("child_page") or {}).get("title") == title:
            pid = b["id"]
            break
    if pid:
        nt.clear(pid)
    else:
        pid = nt.req("POST", "/pages", {"parent": {"page_id": parent},
                                        "properties": {"title": {"title": [_rt(title)]}}})["id"]
    maps = sorted({(r["date"], r["opp"]) for r in rows}, key=lambda x: str(x[0]))
    n_al = sum(1 for r in rows if r["aligned"])
    nt.append(pid, [{"type": "callout", "callout": {"rich_text": [_rt(
        f"{key['map']}・{' / '.join(key.get('comp') or [])}｜{len(maps)} マップ（"
        + "、".join(f"{o} {d or ''}".strip() for d, o in maps[-6:]) + f"）｜位置データ {n_al}/{len(rows)} ラウンド｜"
        "動画の自動解析から作った叩き台です")], "icon": {"type": "emoji", "emoji": "📘"}}},
        _h(1, "概要")])
    meta = D.tactics_meta(cfg, key["map"])
    up = _Uploader(nt)
    for side, label in ((ATTACK, "ATK"), (DEFENSE, "DEF")):
        srows = [r for r in rows if r["side"] == side]
        ms = D.evaluate(rows, base, side)
        strong, weak = D.points(ms, True), D.points(ms, False, k=2)
        nt.append(pid, [_h(2, f"{label} summary")] + _summary_lines(key.get("llm"), label, strong, weak))
        if not srows:
            continue
        nt.append(pid, _plan_blocks(rows, side, label))
        ds = _db(nt, pid, label, ["Default", "Set", "PostPlant", "Retake"])
        for g in D.setup_clusters(srows, k=8):
            follow = []
            if side == ATTACK and g["sites"]:
                follow.append("その後の設置：" + "、".join(f"{s} {c}" for s, c in g["sites"].most_common()))
            if g["fights"]:
                follow.append("初動のファイト：" + "、".join(f"{z} {c}" for z, c in g["fights"].most_common(3)))
            ct = _caster_tags(g["rows"])
            left = [_para(_rt(f"{g['n']} 回・勝ち {g['w']}（{D._pct(D._rate(g['w'], g['n']))}）・形 {g['shape']}", bold=True)),
                    _bullet(_rt("よく映る場所：" + ("・".join(g["callouts"]) or "-")))]
            if g.get("roles"):
                left.append(_bullet(_rt("担当（アイコン照合）：" + g["roles"])))
            left += [_bullet(_rt(x)) for x in follow]
            if ct:
                left.append(_bullet(_rt("解説の言及：" + "・".join(ct))))
            # アイコン照合の配置図（Valotactics 風）があればそれ、無ければ点の位置の重ね描き
            img = next((Path(b) for b in g.get("boards") or [] if Path(b).exists()), None) or D.position_png(
                cfg, key["map"], g["rows"], "1:15", img_dir / f"np_{label}_{D._slug(g['name'])}.jpg", meta)
            _row(nt, ds, f"{_code(g['shape'])} {g['name']}".strip().replace("デフォルト", "default"), "Default", g["rows"],
                 [g["shape"]] + ct, left, img, _examples(g["rows"]), reps, up)
        if side == ATTACK:
            for g in D.sets(srows):
                buys = "・".join(f"{b} {c}" for b, c in g["buys"].most_common())
                left = [_para(_rt(f"{g['n']} 回・勝ち {g['w']}（{D._pct(D._rate(g['w'], g['n']))}）"
                                  + (f"・設置 残り {D._clock(100 - g['plant'])}" if g["plant"] is not None else ""), bold=True)),
                        _bullet(_rt("購入：" + buys))]
                pz = D._zones_line([r["plant_zones"] for r in g["rows"] if r["plant_zones"]])
                if pz:
                    left.append(_bullet(_rt("設置直後の位置（平均人数）：" + pz)))
                img = D.position_png(cfg, key["map"], g["rows"], None, img_dir / f"np_set_{D._slug(g['name'])}.jpg",
                                     meta, plant=True)
                _row(nt, ds, g["name"], "Set", g["rows"], _caster_tags(g["rows"]), left, img, _examples(g["rows"]), reps, up)
        else:
            for site in D.CORE:
                rr = [r for r in srows if r["plant_site"] == site]
                if len(rr) < 2:
                    continue
                w = sum(r["win"] for r in rr)
                pz = D._zones_line([r["plant_zones"] for r in rr if r["plant_zones"]])
                ul = D._ult_line([r for r in rr if any(u["post"] for u in r["ults"])])
                left = [_para(_rt(f"{site} を設置された {len(rr)} 回・取り返した {w}（{D._pct(D._rate(w, len(rr)))}）", bold=True))]
                if pz:
                    left.append(_bullet(_rt("設置された時点の位置（平均人数）：" + pz)))
                if ul:
                    left.append(_bullet(_rt("リテイクで使った ULT（勝ち/使用）：" + ul)))
                img = D.position_png(cfg, key["map"], rr, None, img_dir / f"np_retake_{site}.jpg", meta, plant=True)
                _row(nt, ds, f"{site} retake", "Retake", rr, _caster_tags(rr), left, img, _examples(rr), reps, up)
    # メモ
    memo = [_h(2, "メモ")]
    it = D.intent_summary(rows)
    for side, label in ((ATTACK, "ATK"), (DEFENSE, "DEF")):
        if it.get(side):
            memo.append(_bullet(_rt(f"{label} 解説の言及：", bold=True),
                                _rt("・".join(f"{t} {c}" for t, c in it[side]["tags"]))))
            for q in it[side]["quotes"][:3]:
                memo.append(_bullet(_rt(q, color="gray")))
    for side, label in ((ATTACK, "ATK"), (DEFENSE, "DEF")):
        for p in D.points(D.evaluate(rows, base, side), False, k=2):
            memo.append(_bullet(_rt(f"{label} 弱み：", bold=True), _rt(p["title"])))
    nt.append(pid, memo)
    return f"https://www.notion.so/{pid.replace('-', '')}"
