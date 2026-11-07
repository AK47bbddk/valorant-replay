"""プレイブックの「マップのメタ」「じゃんけん（手の相性）」「手のディテール」「ターン」「読み替え」「構成の特徴」のスライド。

知識ベース（vreplay.kb）に人が書いたことと、DB のラウンドから数えたこと（vreplay.hands / vreplay.comp）を並べる。
数字の無い主張は知識ベースのもの、数字のあるものはデータ。どちらか分かるように書き分ける。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import hands as H
from .deck import ATTACK, DEFENSE, _e, _icon, _pct, _rate, _slide, _slug, media_img, yt_plus

EVAL_JA = [("distance", "サイト距離"), ("take", "テイク難易度"), ("branch", "分岐"), ("long", "ロング"), ("hold", "ホールド")]
STRIP = (0, 10, 20, 30, 40, 50)


def _when_text(when: dict, side: str) -> str:
    """判定条件を人の言葉に。"""
    if not when:
        return "（判定条件なし：知識ベースの説明だけ）"
    out = []
    if "site" in when:
        out.append(f"{when['site'] if not isinstance(when['site'], list) else '・'.join(when['site'])} に行く")
    if "commit_by" in when:
        out.append(f"開始 {when['commit_by']} 秒以内にサイトへ")
    if "commit_after" in when:
        out.append(f"{when['commit_after']} 秒以降にサイトへ")
    if "via_any" in when:
        out.append(f"入る前に {'・'.join(when['via_any'])} のどれかにいる")
    for g in when.get("via_all") or []:
        out.append(f"{'・'.join(g)} のどれかにもいる")
    if "via_min" in when:
        out.append(f"{'・'.join(when['via_min'].get('areas') or [])} に {when['via_min'].get('n', 1)} 人以上")
    if "shown" in when:
        out.append(f"1:15 に {when['shown']} を見せる")
    for k, ja in (("zone_min", "以上"), ("zone_max", "以下"), ("area_min", "以上")):
        for z, n in (when.get(k) or {}).items():
            out.append(f"{z} に {n} 人{ja}")
    if "push" in when:
        out.append(f"開始 30 秒までに {when['push'] if not isinstance(when['push'], list) else '・'.join(when['push'])} の前（攻め側の陣地）へ出る")
    if when.get("flat"):
        out.append("A・B に 1 人以上ずつ、3 人以上の場所なし")
    return "・".join(out)


def strip_png(cfg: dict, map_name: str, rows: list[dict], path: Path, es=STRIP, size: int = 300) -> list[Path] | None:
    """開始（1:40）から 10 秒ごとの、その手のラウンドの位置の重ね合わせ（横に並べた 1 枚）。
    エージェントの軌跡（追跡）があればそれ、無ければミニマップの点。"""
    import cv2
    bg = None
    for r in rows:
        p_ = r.get("uv_base")
        if p_ and Path(p_).exists():
            bg = cv2.imread(p_)
            break
    if bg is None:
        bg = _icon(cfg, map_name, log=lambda *a: None)
    if bg is None:
        bg = np.full((512, 512, 3), 245, np.uint8)
    bg = cv2.resize(bg, (size, size))
    panels, total = [], 0
    for e in es:
        pts = []
        for r in rows:
            tr = r.get("tracks") or {}
            got = False
            for agent, lst in tr.items():
                p = min(lst, key=lambda q: abs(q[0] - e)) if lst else None
                if p is not None and abs(p[0] - e) <= 2.6:
                    pts.append((p[1], p[2]))
                    got = True
            if not got:
                pts += [(u, v) for e2, u, v, _ in r.get("uv") or [] if abs(e2 - e) <= 3]
        total += len(pts)
        heat = np.zeros((size, size), np.float32)
        for u, v in pts:
            x, y = int(u * size), int(v * size)
            if 0 <= x < size and 0 <= y < size:
                heat[y, x] += 1
        heat = cv2.GaussianBlur(heat, (0, 0), 5)
        heat /= max(1e-6, float(heat.max()))
        col = np.zeros_like(bg)
        col[:] = (193, 23, 0)
        a = (np.clip(heat * 1.5, 0, 1) * 0.8)[:, :, None]
        im = (bg.astype(np.float32) * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)
        left = 100 - e
        lab = f"{left // 60}:{left % 60:02d}"
        cv2.rectangle(im, (0, 0), (64, 24), (255, 255, 255), -1)
        cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (26, 26, 26), 2, cv2.LINE_AA)
        panels.append(im)
    if total < 5:
        return None
    # 1 枚ずつ保存する（埋め込みの画像は幅 360px に縮めるので、横に並べた 1 枚だと潰れる）
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for e, im in zip(es, panels):
        p = path.with_name(f"{path.stem}_{e:02d}.jpg")
        cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    return out


def _links(rows: list[dict], k: int = 3) -> str:
    out = []
    for r in sorted(rows, key=lambda r: (not r["win"], str(r.get("date"))))[:k]:
        u = yt_plus(r.get("yt"), -5)
        if u:
            out.append(f'<a href="{_e(u)}">{_e(r.get("opp_tag") or "")} R{r["n"]}{"〇" if r["win"] else "✕"}</a>')
    return "　".join(out)


# ---------------------------------------------------------------- マップのメタ
def map_meta_slides(cfg: dict, map_name: str, kbmap: dict, kb, base: list[dict], foot: str) -> list[str]:
    slides = []
    areas = kbmap.get("areas") or []
    # データ：初動（開始 40 秒まで）に攻めが 2 人以上入ったエリアと、そのラウンドの攻めの勝率／初動のファイトの場所
    atk = [r for r in base if r["side"] == ATTACK and not r.get("pistol")]
    taken = defaultdict(lambda: [0, 0])
    for r in atk:
        seen = set()
        for e, v in H.samples(r).items():
            if e > 40:
                continue
            for a, c in v["areas"].items():
                if c >= 2:
                    seen.add(a)
        for ar in areas:
            if any(H._area_in(a, ar.get("callouts") or [ar["name"]]) for a in seen):
                taken[ar["name"]][0] += bool(r["win"])
                taken[ar["name"]][1] += 1
    ff = Counter(r.get("first_fight") for r in atk if r.get("first_fight"))
    trs = []
    for ar in areas:
        ev = ar.get("eval") or {}
        w, n = taken.get(ar["name"], (0, 0))
        trs.append(
            f"<tr><td style='white-space:nowrap'><b>{_e(ar['name'])}</b>{'<span class=sub>（Obsidian）</span>' if ar.get('source') == 'obsidian' else ''}</td>"
            + "".join(f"<td style='white-space:nowrap'>{_e(ev.get(k) or '-')}</td>" for k, _ in EVAL_JA)
            + f"<td>{'ドライで進める' if ar.get('dry') else 'スキルが要る' if ar.get('dry') is False else '-'}"
              f"{'・チョーク' if ar.get('choke') else ''}</td>"
            + f"<td class='num'>{f'{n}（攻め {_pct(_rate(w, n))}）' if n else '-'}</td>"
            + f"<td class='sub'>{_e(ar.get('note') or '／'.join(ar.get('notes') or [])[:120])}</td></tr>")
    body = (f'<p class="sub" style="font-size:17px">{_e(kbmap.get("summary") or "")}</p>'
            + (f'<table><thead><tr><th>エリア</th>' + "".join(f"<th style='white-space:nowrap'>{ja}</th>" for _, ja in EVAL_JA)
               + '<th>進め方</th><th class="num">初動で攻めが 2 人以上（ラウンド）</th><th>考え方</th></tr></thead>'
               f'<tbody>{"".join(trs)}</tbody></table>' if trs else
               '<p class="empty">このマップのエリアは知識ベースにまだありません（config/kb/maps/ か Obsidian の maps/ に書く）</p>')
            + (f'<p class="sub">初動のファイトの場所（攻めの全ラウンド）：'
               f'{_e("・".join(f"{z} {c}" for z, c in ff.most_common(4)))}</p>' if ff else ""))
    slides.append(_slide("Map Meta", f"{map_name} のメタ：どのエリアをどう取るか", body, foot))

    ch = (kb.concepts or {}).get("choke") or {}
    chokes = [a for a in areas if a.get("choke")]
    if ch:
        steps = "".join(f"<li><b>{_e(s['name'])}</b>：{_e(s['text'])}</li>" for s in ch.get("steps") or [])
        per = "".join(f"<li><b>{_e(a['name'])}</b>：{_e(a.get('note') or '')}</li>" for a in chokes)
        fm = (kb.concepts or {}).get("formation") or {}
        fm_li = "".join(f"<li>{_e(x)}</li>" for x in fm.get("points") or [])
        slides.append(_slide("Choke Point", ch.get("title") or "チョークポイント",
                             f'<div class="cols"><div><h3>守り</h3><ol class="steps">{steps}</ol>'
                             f'<h3>攻めの答え</h3><p>{_e(ch.get("attack_answer") or "")}</p></div>'
                             f'<div><h3>{_e(map_name)} のチョーク</h3><ul class="plain">{per or "<li class=empty>未記入</li>"}</ul>'
                             f'<h3>{_e(fm.get("title") or "")}</h3><ul class="plain sub">{fm_li}</ul></div></div>', foot))
    return slides


def rps_slides(kbmap: dict, kb, base: list[dict], foot: str, map_name: str) -> list[str]:
    """手 × 手の相性表（同マップの全チーム）と、知識ベースのじゃんけんの回り方。"""
    m = H.matrix(base)
    A, D = m["atk"][:7], m["def"][:6]
    if not A or not D:
        return []
    head = "".join(f"<th>{_e(d)}</th>" for d in D)
    trs = []
    for a in A:
        tds = []
        for d in D:
            w, n = m["cells"].get((a, d), (0, 0))
            edge = H.kb_edge(kbmap, a, d)
            mark = {"攻め": " ▲", "守り": " ▼"}.get(edge, "")
            if not n:
                tds.append(f"<td class='num sub'>{mark.strip() or '-'}</td>")
                continue
            r = w / n
            bgc = "#e8f1fe" if r >= 0.6 else "#fdeeee" if r <= 0.4 else "transparent"
            tds.append(f"<td class='num' style='background:{bgc}'>{w}/{n}<span class=sub>（{_pct(r)}）</span>{mark}</td>")
        n_a = sum(m["cells"].get((a, d), (0, 0))[1] for d in m["def"])
        w_a = sum(m["cells"].get((a, d), (0, 0))[0] for d in m["def"])
        trs.append(f"<tr><th>{_e(a)}<span class=sub> {w_a}/{n_a}</span></th>{''.join(tds)}</tr>")
    rps = (kb.concepts or {}).get("rps") or {}
    acts = "".join(f"<li><b>{_e(x['name'])}</b>（{'攻' if x['side'] == 'attack' else '守'}）→ "
                   f"{_e(next((y['name'] for y in rps.get('actions') or [] if y['id'] == x['beats']), ''))} に強い</li>"
                   for x in rps.get("actions") or [])
    body = (f'<table><thead><tr><th>攻め＼守り</th>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table>'
            '<p class="sub">マスは「攻めの勝ち/回数（攻めの勝率）」。ピストル・エコは除く。青＝攻め有利・赤＝守り有利。'
            '▲▼ は知識ベースの見立て（▲＝攻めの手が有利・▼＝守りの手が有利）。数字と見立てが逆なら、その相手は見立てと違う戦い方をしている。</p>'
            + (f'<div class="cols"><div><h3>じゃんけんの 4 つの手</h3><ul class="plain sub">{acts}</ul></div>' if acts else "<div class=cols>"))
    cyc = H.cycles_seen(base, kbmap)
    cyc_html = ""
    for c in cyc:
        st = " → ".join(f"{_e(s['from'])}<span class=sub>（→ 次の手が勝った {s['n']} 回）</span>" for s in c["steps"])
        cyc_html += f"<p>{st} → {_e(c['cycle'][0])}</p>"
    body += (f'<div><h3>よくある読み合いの流れ（知識ベース）</h3>{cyc_html or "<p class=empty>未記入</p>"}</div></div>')
    return [_slide("Rock-Paper-Scissors", f"{map_name} のじゃんけん：攻めの手 × 守りの手", body, foot)]


def hand_slides(cfg: dict, map_name: str, kbmap: dict, rows: list[dict], base: list[dict], out_dir: Path,
                foot: str, slug: str, min_n: int = 2, k: int = 6) -> list[str]:
    """手のディテール（1 手 1 スライド）：説明・判定条件・回数と勝率・相性・10 秒ごとの位置・見本の動画。"""
    slides = []
    mbase = H.matrix(base)
    kb_hands = {SIDE: {h["name"]: h for h in ((kbmap.get("hands") or {}).get(H.SIDE_KEY[SIDE]) or [])}
                for SIDE in (ATTACK, DEFENSE)}
    for side in (ATTACK, DEFENSE):
        srows = [r for r in rows if r["side"] == side and r.get("hand") and r["hand"]["name"] != "不明"
                 and not r.get("pistol")]
        cnt = Counter(r["hand"]["name"] for r in srows)
        for name, n in cnt.most_common(k):
            if n < min_n:
                continue
            hr = [r for r in srows if r["hand"]["name"] == name]
            w = sum(r["win"] for r in hr)
            bw = [r for r in base if r["side"] == side and (r.get("hand") or {}).get("name") == name]
            h = kb_hands[side].get(name) or {}
            # 相手の手ごとの結果（この手を使った側から見た勝ち）
            vs = defaultdict(lambda: [0, 0])
            for r in hr:
                o = (r.get("opp_hand") or {}).get("name")
                if o:
                    vs[o][0] += bool(r["win"])
                    vs[o][1] += 1
            vs_li = "".join(f"<li>vs {_e(o)}：{ww}/{nn}（{_pct(_rate(ww, nn))}）</li>"
                            for o, (ww, nn) in sorted(vs.items(), key=lambda kv: -kv[1][1])[:5])
            # 同マップ全体で、この手にいちばん勝っている相手の手（＝この手の対策）
            best = None
            for (a, d), (ww, nn) in mbase["cells"].items():
                if nn < 2:
                    continue
                if side == ATTACK and a == name:
                    sc = (nn - ww) / nn
                    cand = d
                elif side == DEFENSE and d == name:
                    sc = ww / nn
                    cand = a
                else:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, cand, nn)
            img = strip_png(cfg, map_name, hr, out_dir / "img" / slug / f"hand_{_slug(name)}.jpg")
            how = "".join(f"<li>{_e(x)}</li>" for x in h.get("how") or [])
            kb_rel = ""
            if h.get("beats") or h.get("loses_to"):
                bits = ([f"{'・'.join(h['beats'])} に強い"] if h.get("beats") else []) + \
                       ([f"{'・'.join(h['loses_to'])} に弱い"] if h.get("loses_to") else [])
                kb_rel = f"<p class=sub>知識ベース：{_e('／'.join(bits))}</p>"
            body = (f'<div class="cols"><div>'
                    f'<p class="todo">{_e(h.get("detail") or r_detail(hr[0]))}</p>'
                    f'<p class="sub">判定：{_e(_when_text(h.get("when") or {}, side)) if h else "知識ベースに無い手（自動の名前）"}</p>'
                    f'<ul class="plain"><li>{n} 回・勝ち {w}（{_pct(_rate(w, n))}）'
                    + (f'・同マップ全チーム {sum(r["win"] for r in bw)}/{len(bw)}（{_pct(_rate(sum(r["win"] for r in bw), len(bw)))}）' if bw else "")
                    + "</li>"
                    + (f"<li>この手に一番勝っている相手の手（同マップ）：<b>{_e(best[1])}</b>"
                       f"（{_pct(best[0])}・{best[2]} 回）</li>" if best else "")
                    + f"</ul>{kb_rel}" + (f"<h3>やること</h3><ol class=steps>{how}</ol>" if how else "")
                    + f"</div><div><h3>相手の手ごと</h3><ul class='plain'>{vs_li or '<li class=empty>-</li>'}</ul>"
                    f"<p class='sub'>見本：{_links(hr)}</p></div></div>"
                    + ('<div class="strip">'
                       + "".join(media_img(q, "jpg", f"{name} の位置") for q in img) + '</div>'
                       '<p class="sub">左から 1:40・1:30・1:20・1:10・1:00・0:50 の位置（この手のラウンドの重ね合わせ。'
                       '追跡できたラウンドはエージェントの軌跡、それ以外はミニマップの点）</p>' if img else ""))
            slides.append(_slide(f"{'Attack' if side == ATTACK else 'Defense'} Hand", name, body, foot))
    return slides


def r_detail(r: dict) -> str:
    f = (r.get("hand") or {}).get("feat") or {}
    if r["side"] == ATTACK:
        bits = []
        if f.get("site"):
            bits.append(f"{f['site']} へ")
        if f.get("commit_e") is not None:
            bits.append(f"開始 {int(f['commit_e'])} 秒でサイトに入る")
        via = sorted((f.get("via") or {}).items(), key=lambda kv: -kv[1])[:3]
        if via:
            bits.append("経由：" + "・".join(a for a, _ in via))
        return "・".join(bits) or "攻めの手"
    zs = f.get("zones") or {}
    return "初動の人数：" + " ".join(f"{z}{c}" for z, c in sorted(zs.items())) if zs else "守りの手"


# ---------------------------------------------------------------- ターン・読み替え
def turn_slides(rows: list[dict], foot: str, k: int = 14, per: int = 7) -> list[str]:
    slides = []
    for side in (ATTACK, DEFENSE):
        rs = sorted([r for r in rows if r["side"] == side and r.get("_opp_row") and not r.get("pistol")],
                    key=lambda r: (str(r.get("date")), r["match_id"], r["n"]))[-k:]
        if not rs:
            continue
        trs = []
        for r in rs:
            a, d = (r, r["_opp_row"]) if side == ATTACK else (r["_opp_row"], r)
            ev = H.turns(a, d)
            trs.append(f"<tr><td>{_e(r.get('opp_tag') or '')} R{r['n']}</td>"
                       f"<td>{_e((r.get('hand') or {}).get('name') or '-')}</td>"
                       f"<td>{_e((r.get('opp_hand') or {}).get('name') or '-')}</td>"
                       f"<td class='sub'>{_e(H.turn_line(ev)) or '-'}</td>"
                       f"<td>{'〇' if r['win'] else '✕'}</td>"
                       f"<td>{('<a href=' + chr(34) + _e(r['yt']) + chr(34) + '>▶</a>') if r.get('yt') else ''}</td></tr>")
        for k0 in range(0, len(trs), per):
            slides.append(_slide("Turn", f"ターン：アクション → リアクションの応酬（{side}・直近）"
                                 + (f" {k0 // per + 1}" if len(trs) > per else ""),
                                 '<table><thead><tr><th>ラウンド</th><th>自分の手</th><th>相手の手</th>'
                                 '<th>流れ（↪＝相手への 15 秒以内のリアクション）</th>'
                                 f'<th>結果</th><th></th></tr></thead><tbody>{"".join(trs[k0:k0 + per])}</tbody></table>'
                                 '<p class="sub">10 秒ごとの配置の変化（テイク・前に出る・ローテ・転換）とキル・設置から並べています。'
                                 'スキルの中身はミニマップでは分からないので、動画で確かめてください。</p>', foot))
    rd = H.reads(rows)
    if rd:
        cols = []
        for side in (ATTACK, DEFENSE):
            lst = [x for x in rd if x["side"] == side][:6]
            if not lst:
                continue
            li = "".join(f"<li>{_e(x['lost_with'])}（vs {_e(x['lost_to'])}）で負け → 次は <b>{_e(x['next'])}</b>"
                         f"{'' if x['changed'] else '（同じ手）'}：{x['n']} 回・次のラウンド勝ち {x['w']}</li>" for x in lst)
            cols.append(f"<div><h3>{side}</h3><ul class='plain'>{li}</ul></div>")
        t, d = H.trade_rate(rows)
        fp = H.first_pick_shapes(rows)
        fp_li = "".join(f"<li>{_e(x['where'])}・{_e(x['agent'])}・{_e(x['time'])}：{x['n']} 回（勝ち {x['w']}）</li>" for x in fp)
        slides.append(_slide("Reads", "読み替え：負けた次のラウンドに何を出したか",
                             f'<div class="cols">{"".join(cols)}</div>'
                             f'<div class="cols"><div><h3>1 ピックの形（先制キルを取った場所・エージェント・時間）</h3>'
                             f'<ul class="plain sub">{fp_li or "<li class=empty>-</li>"}</ul></div>'
                             f'<div><h3>後衛のトレード</h3><p>味方が倒されて 4 秒以内に倒し返した：{t}/{d}'
                             f'（{_pct(_rate(t, d))}）</p></div></div>', foot))
    return slides


# ---------------------------------------------------------------- 構成の特徴
def comp_slides(rep: dict, foot: str, map_name: str) -> list[str]:
    p, ru, notes, dt = rep["profile"], rep["rules"], rep["notes"], rep["data"]
    roles = "・".join(f"{v} {k_}" for k_, v in p["roles"].items())
    from .comp import ROLE_JA
    roles = "・".join(f"{ROLE_JA.get(k_, k_)} {v}" for k_, v in p["roles"].items())

    def side_html(sd: str) -> str:
        strong = [x for nt in notes for x in ((nt.get(sd) or {}).get("strong") or [])]
        weak = [x for nt in notes for x in ((nt.get(sd) or {}).get("weak") or [])]
        rs, rw = ru[sd]["strong"], ru[sd]["weak"]
        li = lambda xs, cls="": "".join(f"<li class='{cls}'>{_e(x)}</li>" for x in xs)
        empty = '<li class="empty">（まだありません）</li>'
        return (f'<span class="label ok">strong</span><ul class="plain">{li(strong)}{li(rs, "sub") or ("" if strong else empty)}</ul>'
                f'<span class="label ng">weak</span><ul class="plain">{li(weak)}{li(rw, "sub") or ("" if weak else empty)}</ul>')

    plan = "".join(f"<p class='todo'>{_e(nt['plan'])}</p>" for nt in notes if nt.get("plan"))
    body = (f'<p class="sub">{_e(" / ".join(p["comp"]))}（{_e(roles)}）・型：{_e("・".join(p["archetypes"]))}'
            + (f'・Tier：{_e("・".join(f"{a} {t}" for a, t in p["tiers"].items()))}' if p.get("tiers") else "") + "</p>"
            + plan
            + f'<div class="cols"><div><h3>攻撃</h3>{side_html("attack")}</div><div><h3>防衛</h3>{side_html("defense")}</div></div>'
            + ('<p class="sub">黒い行はコーチのメモ（config/kb/comps.yaml）、小さい灰色の行はエージェントの特徴（agents.yaml）からの自動の見立て。</p>'
               if notes else '<p class="sub">この構成のメモはまだありません（config/kb/comps.yaml に must: [エージェント…] で書くと黒い行で出ます）。'
               '灰色の行はエージェントの特徴からの自動の見立て。</p>'))
    slides = [_slide("Composition", "この構成の特徴：何が強い・何が弱い", body, foot)]
    meta = "".join(f"<li>{_e(nt['meta'])}</li>" for nt in notes if nt.get("meta"))
    anti_kb = "".join(f"<li><b>{_e(a.get('who'))}</b>：{_e(a.get('why') or '')}"
                      + (f"<br><span class=sub>工夫：{_e(a.get('fix'))}</span>" if a.get("fix") else "") + "</li>"
                      for nt in notes for a in nt.get("anti") or [])
    anti_dt = "".join(f"<li>{_e(x['name'])}：{x['w']}/{x['n']}（{_pct(x['rate'])}・{x['maps']} マップ）</li>"
                      for x in dt["anti"] + dt["anti_agents"])
    good = "".join(f"<li>{_e(x['name'])}：{x['w']}/{x['n']}（{_pct(x['rate'])}）</li>" for x in dt["good"])
    arch = "".join(f"<tr><td>{_e(x['name'])}</td><td class='num'>{x['w']}/{x['n']}</td><td class='num'>{_pct(x['rate'])}</td>"
                   f"<td class='num'>{x['maps']}</td></tr>" for x in sorted(dt["archetypes"], key=lambda x: -x["n"])[:8])
    body2 = (f'<div class="cols"><div><h3>今のメタにどう刺さるか</h3><ul class="plain">{meta or "<li class=empty>未記入（comps.yaml の meta）</li>"}</ul>'
             f'<h3>相手の型ごとの勝率（ラウンド・ピストル除く・全体 {_pct(dt["base"])}）</h3>'
             f'<table><thead><tr><th>相手の型</th><th class="num">勝ち/回数</th><th class="num">勝率</th><th class="num">マップ</th></tr></thead>'
             f'<tbody>{arch}</tbody></table></div>'
             f'<div><h3>アンチピック（苦手な相手）</h3><ul class="plain">{anti_kb}</ul>'
             f'<p class="sub">データで勝率が低い相手：</p><ul class="plain sub">{anti_dt or "<li class=empty>（まだはっきりした差はありません）</li>"}</ul>'
             f'<p class="sub">データで勝率が高い相手：</p><ul class="plain sub">{good or "<li class=empty>-</li>"}</ul></div></div>')
    slides.append(_slide("Composition", f"{map_name}：今のメタとアンチピック", body2, foot))
    return slides
