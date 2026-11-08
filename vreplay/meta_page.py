"""マップのメタの冊子（`vreplay meta`）：チームを問わず、そのマップで使われている手・相性・読み替え。

プレイブック（チーム × 構成）の前に読む「じゃんけんの手の一覧」。知識ベース（config/kb/maps/）に書いた手が
実際にどのくらい出て、何に勝っているかを数字で確かめ、知識ベースを直す材料にする。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import db
from . import deck as D
from . import deck_meta as DM
from . import hands as H
from . import kb as K


def team_table(rows: list[dict], side: str, k: int = 12) -> str:
    by = defaultdict(Counter)
    wins = defaultdict(Counter)
    for r in rows:
        if r["side"] != side or r.get("pistol") or not r.get("hand"):
            continue
        by[r["tag"] or r["team"]][r["hand"]["name"]] += 1
        wins[r["tag"] or r["team"]][r["hand"]["name"]] += bool(r["win"])
    trs = []
    for t, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:k]:
        tot = sum(c.values())
        top = "・".join(f"{h} {n}（勝 {wins[t][h]}）" for h, n in c.most_common(3))
        trs.append(f"<tr><td>{D._e(t)}</td><td class='num'>{tot}</td><td class='sub'>{D._e(top)}</td></tr>")
    return ("<table><thead><tr><th>チーム</th><th class='num'>ラウンド</th><th>よく出す手（回数・勝ち）</th></tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>") if trs else ""


def build_map(cfg: dict, map_name: str, rows: list[dict], out_dir: Path, log=print) -> Path:
    kb = K.load(cfg)
    kbmap = kb.map(map_name)
    D.media_reset(cfg)
    n_maps = len({(r["match_id"], r["order"]) for r in rows})
    foot = f"{n_maps} マップ・{len({(r['match_id'], r['order'], r['n']) for r in rows})} ラウンド（全チーム）"
    slides = [D._slide(map_name, f"{map_name} のメタ", f'<p class="sub" style="font-size:20px">{D._e(kbmap.get("summary") or "")}</p>'
                       '<p class="sub">マップの考え方（知識ベース）→ 手 × 手の相性（データ）→ 手のディテール → 読み替え。'
                       'チームのプレイブックはこの上に「その構成でどの手を使うか」を足したものです。</p>', foot, h1=True)]
    slides += DM.map_meta_slides(cfg, map_name, kbmap, kb, rows, foot)
    slides += DM.rps_slides(kbmap, kb, rows, foot, map_name)
    tt = "".join(f"<div><h3>{s}</h3>{team_table(rows, s)}</div>" for s in (H.ATTACK, H.DEFENSE))
    slides.append(D._slide("Teams", "チームごとのよく出す手", f'<div class="cols">{tt}</div>', foot))
    slides += DM.hand_slides(cfg, map_name, kbmap, rows, rows, out_dir, foot, f"meta_{D._slug(map_name)}", k=8)
    slides += [s for s in DM.turn_slides(rows, foot) if "Reads" in s]
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{D._e(map_name)} のメタ</title>'
            f'<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">'
            f'<style>{D.CSS}</style></head><body><main class="deck"><div class="toolbar"><p><a href="index.html">← 一覧</a></p>'
            f'<p>印刷（PDF 保存）すると 1 スライド 1 ページになります</p></div>{"".join(slides)}</main>'
            f'{D.MOTION_JS}{D.media_js()}</body></html>')
    p = out_dir / f"meta_{D._slug(map_name)}.html"
    p.write_text(page, encoding="utf-8")
    m = H.matrix(rows)
    (out_dir / f"meta_{D._slug(map_name)}.json").write_text(json.dumps({
        "map": map_name, "maps": n_maps, "teams": len({r["team"] for r in rows}), "summary": H.summary(rows),
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "matrix": [{"atk": a, "def": d, "atk_win": w, "n": n} for (a, d), (w, n) in m["cells"].items()],
        "reads": H.reads(rows)[:30]}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  {map_name} のメタ: {p}")
    return p


def build(cfg: dict, out_dir: Path, map_name: str | None = None, source: str | None = None, log=print) -> list[Path]:
    conn = db.open_warehouse(cfg)
    rows = [r for rep in D.load_reports(conn) for r in D.records(rep)]
    if source:
        rows = [r for r in rows if (r.get("source") or "vlr") == source]
    H.annotate(rows, K.load(cfg))
    try:
        db.store_hands(conn, rows)
        conn.commit()
    except Exception as e:
        log(f"  round_hands への書き込みに失敗: {e}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in rows:
        if not map_name or map_name.lower() == str(r["map"]).lower():
            by[r["map"]].append(r)
    out = []
    for mp, rs in sorted(by.items()):
        try:
            out.append(build_map(cfg, mp, rs, out_dir, log=log))
        except Exception as e:
            log(f"  {mp} のメタを作れませんでした（{e}）")
    from . import site
    site.build(out_dir, log=log)
    return out
