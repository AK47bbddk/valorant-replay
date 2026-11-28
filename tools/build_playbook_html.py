"""match_summary.json と report.json からプレイブック用ラウンド分類 HTML を生成する。

使い方:
  python tools/build_playbook_html.py \
      --summary /tmp/match_754737_1_Summit_v2/match_summary.json \
      --report /tmp/match_754737_1_Summit/report.json \
      --team LOUD --side ATK \
      --out /tmp/match_754737_1_Summit_v2/playbook_loud_attack.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence


# デフォルトの初動配置パターン定義（LOUD 攻撃向け）
DEFAULT_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "sage_omen_a_main",
        "title": "セージ＋オーメンが A メイン",
        "short": "Sge+Ome Aメイン",
        "desc": "開幕 (t≤4s) に Sage と Omen が共に A 側（A メイン / A ロビー）に配置",
        "color": "#ef4444",
    },
    {
        "id": "fade_b",
        "title": "フェイドが B",
        "short": "Fade B",
        "desc": "開幕 (t≤4s) に Fade が B 側（B ロビー / B メイン）に配置",
        "color": "#3b82f6",
    },
    {
        "id": "fade_chamber_b_main",
        "title": "フェイド＋チェンバーが B メイン",
        "short": "Fade+Chm Bメイン",
        "desc": "開幕 (t≤4s) に Fade と Chamber が揃って B 側（B メイン / B ロビー）に配置",
        "color": "#8b5cf6",
    },
    {
        "id": "opening_4_1",
        "title": "初動で 4-1",
        "short": "初動 4-1",
        "desc": "開幕 (t≤4s) に片側ソロ 1 名・逆側主力（4-1 構成）で展開",
        "color": "#f59e0b",
    },
]


def _nearest_callout(x: float, y: float, callouts: list[dict[str, Any]]) -> tuple[str, str]:
    """(x, y) に最も近いコールアウトの (super, label) を返す。"""
    if not callouts:
        side = "A" if x >= 0.58 else ("B" if x <= 0.42 else "Mid")
        return side, side
    best_c = min(callouts, key=lambda c: math.hypot(float(c["x"]) - x, float(c["y"]) - y))
    sup = str(best_c.get("super") or "").strip()
    name = str(best_c.get("name") or "").strip()
    label = f"{sup} {name}".strip() if sup and name and sup != name else (name or sup or "?")
    return sup, label


def _macro_side(x: float, sup: str) -> str:
    """開幕の左右サイド（A / B）を判定する。"""
    s = sup.upper()
    if s == "A":
        return "A"
    if s == "B":
        return "B"
    return "A" if x >= 0.48 else "B"


def _is_a_main_zone(x: float, sup: str) -> bool:
    return sup.upper() == "A" or x >= 0.58


def _is_b_zone(x: float, sup: str) -> bool:
    return sup.upper() == "B" or x < 0.45


def classify_rounds(
    summary: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    team: str | None = None,
    side: str = "ATK",
    opening_window_sec: float = 4.0,
) -> dict[str, Any]:
    """指定チーム・攻守（例: LOUD / ATK）のラウンドを抽出し、初動配置パターンへ分類する。"""
    rep = report or {}
    callouts = (rep.get("tactics") or {}).get("callouts_placed") or []
    rep_rounds = {int(r["n"]): r for r in (rep.get("rounds") or []) if isinstance(r.get("n"), int)}
    team1 = str(rep.get("team1") or summary.get("team1") or "Team 1")

    side_u = side.upper()
    classified: list[dict[str, Any]] = []

    for r in summary.get("rounds") or []:
        r_team = r.get("atk_team") if side_u == "ATK" else r.get("def_team")
        if team and str(r_team or "").lower() != team.lower():
            continue

        rn = int(r.get("round_num") or 0)
        rr = rep_rounds.get(rn, {})
        winner = str(rr.get("winner") or "")
        won = bool(team and winner.lower() == team.lower()) if winner else None
        is_team1 = bool(r_team and str(r_team).lower() == team1.lower())
        buy_self = str(rr.get("buy1" if is_team1 else "buy2") or "")
        buy_opp = str(rr.get("buy2" if is_team1 else "buy1") or "")

        # 開幕窓 (t <= opening_window_sec) で各エージェントの初動位置を取得
        opening_agents: dict[str, dict[str, Any]] = {}
        for fr in r.get("timeline") or []:
            if float(fr.get("t", 0.0)) > opening_window_sec + 1e-6:
                break
            for a in fr.get("agents") or []:
                if a.get("side") != side_u:
                    continue
                ag = str(a.get("agent") or "").strip()
                if not ag or ag in opening_agents:
                    continue
                x, y = float(a["x"]), float(a["y"])
                sup, lbl = _nearest_callout(x, y, callouts)
                opening_agents[ag] = {
                    "agent": ag,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "super": sup,
                    "callout": lbl,
                    "macro": _macro_side(x, sup),
                    "in_a_main": _is_a_main_zone(x, sup),
                    "in_b": _is_b_zone(x, sup),
                    "t": float(fr.get("t", 0.0)),
                }

        a_agents = [v["agent"] for v in opening_agents.values() if v["macro"] == "A"]
        b_agents = [v["agent"] for v in opening_agents.values() if v["macro"] == "B"]
        n_a, n_b = len(a_agents), len(b_agents)

        # 初動配置フォーメーション表記（重なり等で4名検出の 3-1 は 4-1 相当として扱う）
        if n_a >= 3 and n_b == 1:
            formation = f"A{max(4, n_a)}-B1"
            is_4_1 = True
        elif n_b >= 3 and n_a == 1:
            formation = f"B{max(4, n_b)}-A1"
            is_4_1 = True
        else:
            formation = f"A{n_a}-B{n_b}"
            is_4_1 = False

        # 各パターン判定
        tags: list[str] = []
        sage = opening_agents.get("Sage")
        omen = opening_agents.get("Omen")
        fade = opening_agents.get("Fade")
        chamber = opening_agents.get("Chamber")

        if sage and omen and sage["in_a_main"] and omen["in_a_main"]:
            tags.append("sage_omen_a_main")
        if fade and fade["in_b"]:
            tags.append("fade_b")
        if fade and chamber and fade["in_b"] and chamber["in_b"]:
            tags.append("fade_chamber_b_main")
        if is_4_1:
            tags.append("opening_4_1")

        # 最初のスモーク位置・時刻
        first_smoke = None
        for fr in r.get("timeline") or []:
            sms = fr.get("smokes") or []
            if sms:
                s0 = sms[0]
                _, s_lbl = _nearest_callout(float(s0["x"]), float(s0["y"]), callouts)
                first_smoke = {"t": float(fr.get("t", 0.0)), "callout": s_lbl, "count": len(sms)}
                break

        # 中盤〜終盤 (t >= 10s) の重心から最終攻めサイト傾向を推定
        late_xs = [
            float(a["x"])
            for fr in (r.get("timeline") or [])
            if float(fr.get("t", 0.0)) >= 10.0
            for a in (fr.get("agents") or [])
            if a.get("side") == side_u
        ]
        late_x = round(sum(late_xs) / len(late_xs), 2) if late_xs else None
        if late_x is None:
            final_tendency = "不明"
        elif late_x >= 0.56:
            final_tendency = "A 攻め"
        elif late_x <= 0.42:
            final_tendency = "B 攻め"
        else:
            final_tendency = "Mid / 混戦"

        classified.append({
            "round": r.get("round"),
            "round_num": rn,
            "team": r_team,
            "side": side_u,
            "winner": winner,
            "won": won,
            "buy_self": buy_self,
            "buy_opp": buy_opp,
            "formation": formation,
            "a_agents": a_agents,
            "b_agents": b_agents,
            "opening_agents": list(opening_agents.values()),
            "tags": tags,
            "first_smoke": first_smoke,
            "final_tendency": final_tendency,
            "late_x": late_x,
            "raw_gif_file": r.get("raw_gif_file"),
            "gif_file": r.get("gif_file"),
            "compare_gif_file": r.get("compare_gif_file"),
            "n_frames": r.get("n_frames"),
        })

    # パターンごとの集計
    pattern_stats: list[dict[str, Any]] = []
    for pat in DEFAULT_PATTERNS:
        pid = pat["id"]
        matched = [c for c in classified if pid in c["tags"]]
        wins = sum(1 for c in matched if c["won"] is True)
        losses = sum(1 for c in matched if c["won"] is False)
        a_cnt = sum(1 for c in matched if c["final_tendency"] == "A 攻め")
        b_cnt = sum(1 for c in matched if c["final_tendency"] == "B 攻め")
        m_cnt = sum(1 for c in matched if c["final_tendency"] == "Mid / 混戦")
        pattern_stats.append({
            **pat,
            "count": len(matched),
            "rounds": [c["round"] for c in matched],
            "wins": wins,
            "losses": losses,
            "a_executes": a_cnt,
            "b_executes": b_cnt,
            "mid_executes": m_cnt,
        })

    return {
        "map_name": summary.get("map_name") or rep.get("map") or "Unknown",
        "team": team or "ALL",
        "side": side_u,
        "total_rounds": len(classified),
        "patterns": pattern_stats,
        "rounds": classified,
    }


def _gif_frame_data_uri(gif_path: Path, frame_idx: int = 0) -> str | None:
    """GIF の指定フレームを軽量 PNG の data URI に変換する（自己完結 HTML 用）。"""
    if not gif_path.is_file():
        return None
    try:
        with Image.open(gif_path) as im:
            n = getattr(im, "n_frames", 1)
            idx = min(max(0, frame_idx), max(0, n - 1))
            im.seek(idx)
            rgb = im.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def render_playbook_html(
    data: dict[str, Any],
    *,
    gifs_dir: Path | None = None,
    embed_snapshots: bool = False,
) -> str:
    """分類結果からインタラクティブなプレイブック用 HTML を生成する。"""
    pat_map = {p["id"]: p for p in data["patterns"]}
    total = data["total_rounds"]
    wins_all = sum(1 for r in data["rounds"] if r["won"] is True)
    losses_all = sum(1 for r in data["rounds"] if r["won"] is False)

    # KPI パターンカード
    kpi_cards: list[str] = []
    for p in data["patterns"]:
        wr = f"{p['wins']}勝{p['losses']}敗" if (p["wins"] + p["losses"]) else "-"
        r_list = ", ".join(p["rounds"]) if p["rounds"] else "該当なし"
        kpi_cards.append(
            f"<button class='kpi' data-filter='{p['id']}' style='--accent:{p['color']}'>"
            f"<div class='kpi-top'><span class='kpi-title'>{html.escape(p['title'])}</span>"
            f"<span class='kpi-cnt'>{p['count']}/{total}R</span></div>"
            f"<div class='kpi-desc'>{html.escape(p['desc'])}</div>"
            f"<div class='kpi-meta'>戦績: <b>{wr}</b> ｜ 最終傾向: A攻め {p['a_executes']} / B攻め {p['b_executes']} / Mid {p['mid_executes']}</div>"
            f"<div class='kpi-rounds'>{html.escape(r_list)}</div>"
            "</button>"
        )

    # マトリクス表の行
    table_rows: list[str] = []
    for r in data["rounds"]:
        tags_str = " ".join(r["tags"]) if r["tags"] else "other"
        res_cls = "win" if r["won"] is True else ("lose" if r["won"] is False else "")
        res_txt = "WIN" if r["won"] is True else ("LOSE" if r["won"] is False else "-")
        checks = "".join(
            f"<td class='chk'>{'<span class=\"dot\" style=\"background:' + p['color'] + '\">✓</span>' if p['id'] in r['tags'] else '·'}</td>"
            for p in data["patterns"]
        )
        smk_txt = f"t={r['first_smoke']['t']:.0f}s {r['first_smoke']['callout']}" if r["first_smoke"] else "-"
        a_txt = ", ".join(r["a_agents"]) or "-"
        b_txt = ", ".join(r["b_agents"]) or "-"
        table_rows.append(
            f"<tr data-tags='{tags_str}'>"
            f"<td><b>{html.escape(str(r['round']))}</b></td>"
            f"<td><span class='badge {res_cls}'>{res_txt}</span></td>"
            f"<td class='small'>{html.escape(r['buy_self'] or '-')}</td>"
            f"<td><span class='form'>{html.escape(r['formation'])}</span></td>"
            f"{checks}"
            f"<td class='small'><b>A:</b> {html.escape(a_txt)}<br><b>B:</b> {html.escape(b_txt)}</td>"
            f"<td class='small'>{html.escape(smk_txt)}</td>"
            f"<td><span class='tend'>{html.escape(r['final_tendency'])}</span></td>"
            "</tr>"
        )

    # ラウンド詳細カード
    round_cards: list[str] = []
    for r in data["rounds"]:
        tags_str = " ".join(r["tags"]) if r["tags"] else "other"
        res_cls = "win" if r["won"] is True else ("lose" if r["won"] is False else "")
        res_txt = "WIN" if r["won"] is True else ("LOSE" if r["won"] is False else "-")
        tag_pills = "".join(
            f"<span class='pill' style='background:{pat_map[t]['color']}22;border-color:{pat_map[t]['color']};color:{pat_map[t]['color']}'>{html.escape(pat_map[t]['short'])}</span>"
            for t in r["tags"] if t in pat_map
        ) or "<span class='pill muted'>その他（標準/全員A等）</span>"

        agent_chips = "".join(
            f"<span class='chip {'chip-a' if a['macro'] == 'A' else 'chip-b'}'>"
            f"<b>{html.escape(a['agent'])}</b> {html.escape(a['callout'])}</span>"
            for a in sorted(r["opening_agents"], key=lambda x: (0 if x["macro"] == "A" else 1, x["agent"]))
        )
        smk_txt = f"初回スモーク: t={r['first_smoke']['t']:.0f}s ({r['first_smoke']['callout']})" if r["first_smoke"] else "初回スモーク: 検出なし"

        media_html = ""
        if embed_snapshots and gifs_dir is not None and r.get("gif_file"):
            p_ann = gifs_dir / str(r["gif_file"])
            u0 = _gif_frame_data_uri(p_ann, 0)
            u2 = _gif_frame_data_uri(p_ann, 2)
            if u0 and u2:
                media_html = (
                    "<div class='media-pair'>"
                    f"<div><div class='cap'>開幕 (t=0s)</div><img src='{u0}' alt='{r['round']} t0'></div>"
                    f"<div><div class='cap'>初動展開 (t=4s)</div><img src='{u2}' alt='{r['round']} t4'></div>"
                    "</div>"
                )
        elif r.get("raw_gif_file") and r.get("gif_file"):
            media_html = (
                "<div class='media-pair'>"
                f"<div><div class='cap'>未解析 (Raw)</div><img src='{html.escape(str(r['raw_gif_file']))}' loading='lazy'></div>"
                f"<div><div class='cap'>解析あり (Annotated)</div><img src='{html.escape(str(r['gif_file']))}' loading='lazy'></div>"
                "</div>"
            )

        round_cards.append(
            f"<div class='rcard' data-tags='{tags_str}'>"
            f"<div class='rcard-head'><div><b>{html.escape(str(r['round']))}</b> "
            f"<span class='badge {res_cls}'>{res_txt}</span> "
            f"<span class='form'>{html.escape(r['formation'])}</span> "
            f"<span class='tend'>{html.escape(r['final_tendency'])}</span></div>"
            f"<div class='small muted'>{html.escape(r['buy_self'])} vs {html.escape(r['buy_opp'])}</div></div>"
            f"<div class='pills'>{tag_pills}</div>"
            f"{media_html}"
            f"<div class='chips'>{agent_chips}</div>"
            f"<div class='small muted' style='margin-top:6px'>{html.escape(smk_txt)}</div>"
            "</div>"
        )

    pat_headers = "".join(f"<th class='chk'>{html.escape(p['short'])}</th>" for p in data["patterns"])
    filter_btns = (
        "<button class='fbtn active' data-filter='all'>すべて ({total}R)</button>".format(total=total)
        + "".join(
            f"<button class='fbtn' data-filter='{p['id']}'>{html.escape(p['short'])} ({p['count']}R)</button>"
            for p in data["patterns"]
        )
        + "<button class='fbtn' data-filter='other'>その他</button>"
    )

    title = f"{data['team']} {data['side']} プレイブック — {str(data['map_name']).capitalize()} ({total}R)"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#0b1020; --card:#131b2e; --border:#24314f; --text:#e5ecf6; --muted:#93a4c3; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:18px; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,sans-serif; line-height:1.45; }}
  .wrap {{ max-width:1360px; margin:0 auto; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:12px; padding-bottom:14px; border-bottom:1px solid var(--border); margin-bottom:16px; }}
  h1 {{ margin:0; font-size:22px; letter-spacing:0.02em; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-bottom:16px; }}
  .kpi {{ text-align:left; background:var(--card); border:1px solid var(--border); border-left:4px solid var(--accent); border-radius:10px; padding:12px 14px; color:var(--text); cursor:pointer; transition:transform .12s,border-color .12s; }}
  .kpi:hover,.kpi.active {{ border-color:var(--accent); transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.35); }}
  .kpi-top {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
  .kpi-title {{ font-weight:700; font-size:14px; }}
  .kpi-cnt {{ font-weight:800; font-size:15px; color:var(--accent); }}
  .kpi-desc {{ font-size:11.5px; color:var(--muted); margin:4px 0 6px; }}
  .kpi-meta {{ font-size:12px; }}
  .kpi-rounds {{ font-size:11.5px; color:#cbd5e1; margin-top:4px; font-family:ui-monospace,monospace; }}
  .fbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; align-items:center; }}
  .fbtn {{ background:#182238; color:var(--text); border:1px solid var(--border); border-radius:999px; padding:6px 12px; font-size:12px; cursor:pointer; }}
  .fbtn.active {{ background:#2563eb; border-color:#3b82f6; font-weight:700; }}
  .panel {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:18px; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th,td {{ padding:8px 8px; border-bottom:1px solid #1e2942; text-align:left; vertical-align:middle; }}
  th {{ color:var(--muted); font-size:11.5px; font-weight:600; white-space:nowrap; }}
  .chk {{ text-align:center; width:74px; }}
  .dot {{ display:inline-block; width:20px; height:20px; line-height:20px; border-radius:50%; color:#fff; font-weight:800; font-size:12px; }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:5px; font-size:11px; font-weight:700; }}
  .badge.win {{ background:#14532d; color:#86efac; }}
  .badge.lose {{ background:#7f1d1d; color:#fca5a5; }}
  .form {{ display:inline-block; background:#1e293b; border:1px solid #334155; padding:2px 7px; border-radius:5px; font-family:ui-monospace,monospace; font-size:11.5px; font-weight:700; }}
  .tend {{ display:inline-block; background:#172554; color:#93c5fd; padding:2px 7px; border-radius:5px; font-size:11px; font-weight:600; }}
  .rgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:14px; }}
  .rcard {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; }}
  .rcard-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; }}
  .pills {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }}
  .pill {{ font-size:11px; font-weight:700; padding:2px 8px; border-radius:999px; border:1px solid #334155; }}
  .media-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px; }}
  .media-pair img {{ width:100%; height:auto; border-radius:6px; display:block; border:1px solid #24314f; }}
  .cap {{ font-size:11px; color:var(--muted); text-align:center; margin-bottom:3px; }}
  .chips {{ display:flex; gap:5px; flex-wrap:wrap; }}
  .chip {{ font-size:11px; padding:2px 7px; border-radius:5px; }}
  .chip-a {{ background:#7f1d1d55; border:1px solid #ef444466; color:#fecaca; }}
  .chip-b {{ background:#1e3a8a55; border:1px solid #3b82f666; color:#bfdbfe; }}
  .small {{ font-size:11.5px; }}
  .muted {{ color:var(--muted); }}
  .hidden {{ display:none !important; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="sub">戦績: <b>{wins_all}勝 {losses_all}敗</b> ｜ 初動窓: t≤4.0s ｜ 各パターンカードまたはフィルタボタンをクリックして絞り込み</div>
    </div>
  </header>

  <section class="kpi-grid">{''.join(kpi_cards)}</section>

  <div class="fbar">
    <span class="small muted">絞り込み:</span>
    {filter_btns}
  </div>

  <section class="panel">
    <table>
      <thead>
        <tr>
          <th>R</th><th>結果</th><th>バイ</th><th>初動形</th>
          {pat_headers}
          <th>開幕配置 (A / B)</th><th>初回スモーク</th><th>最終傾向</th>
        </tr>
      </thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </section>

  <section class="rgrid">{''.join(round_cards)}</section>
</div>
<script>
(function() {{
  const buttons = document.querySelectorAll('.fbtn, .kpi');
  const rows = document.querySelectorAll('tbody tr');
  const cards = document.querySelectorAll('.rcard');
  function applyFilter(f) {{
    document.querySelectorAll('.fbtn').forEach(b => b.classList.toggle('active', b.dataset.filter === f));
    document.querySelectorAll('.kpi').forEach(b => b.classList.toggle('active', b.dataset.filter === f));
    const match = el => {{
      if (f === 'all') return true;
      const tags = (el.dataset.tags || '').split(/\\s+/);
      return tags.includes(f);
    }};
    rows.forEach(r => r.classList.toggle('hidden', !match(r)));
    cards.forEach(c => c.classList.toggle('hidden', !match(c)));
  }}
  buttons.forEach(b => b.addEventListener('click', () => applyFilter(b.dataset.filter)));
}})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="プレイブック用ラウンド分類 HTML を生成")
    ap.add_argument("--summary", required=True, help="match_summary.json のパス")
    ap.add_argument("--report", default=None, help="report.json のパス（コールアウト・勝敗・バイ情報用）")
    ap.add_argument("--team", default=None, help="対象チーム名（例: LOUD）")
    ap.add_argument("--side", default="ATK", choices=["ATK", "DEF"], help="攻守フィルタ")
    ap.add_argument("--out", required=True, help="出力 HTML パス")
    ap.add_argument("--embed-snapshots", action="store_true", help="GIF の開幕/初動スナップショットを base64 埋め込み")
    args = ap.parse_args(argv)

    summary_p = Path(args.summary)
    summary = json.loads(summary_p.read_text("utf-8"))
    report = json.loads(Path(args.report).read_text("utf-8")) if args.report else None

    data = classify_rounds(summary, report, team=args.team, side=args.side)
    html_text = render_playbook_html(
        data,
        gifs_dir=summary_p.parent,
        embed_snapshots=bool(args.embed_snapshots),
    )
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html_text, "utf-8")

    # 分類結果 JSON も隣に出力
    json_p = out_p.with_suffix(".json")
    json_p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "team": data["team"],
        "side": data["side"],
        "total_rounds": data["total_rounds"],
        "patterns": [{k: p[k] for k in ("id", "title", "count", "rounds", "wins", "losses")} for p in data["patterns"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
