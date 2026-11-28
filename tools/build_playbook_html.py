"""複数試合の match_summary.json + report.json からプレイブック用ラウンドタグ付け＆検索 HTML を生成する。

使い方（複数試合）:
  python tools/build_playbook_html.py \
      --match /tmp/loud_summit_5_out/753450_2_Summit/match_summary.json:/tmp/loud_summit_5/753450_2_Summit/report.json:"vs EDG (9/26)" \
      --match /tmp/match_754737_1_Summit_v2/match_summary.json:/tmp/match_754737_1_Summit/report.json:"vs VIT (10/18)" \
      --team LOUD --side ATK --embed-snapshots \
      --out /tmp/loud_summit_5_out/playbook_loud_summit.html

単一試合互換:
  python tools/build_playbook_html.py --summary match_summary.json --report report.json --team LOUD --side ATK --out out.html
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

from PIL import Image


# 4 カテゴリ・17 タグのプレイブック分類定義
TAG_CATEGORIES: list[dict[str, Any]] = [
    {"id": "opening", "label": "初動の配置", "color": "#ef4444"},
    {"id": "smoke", "label": "スモークの位置", "color": "#06b6d4"},
    {"id": "mid", "label": "中盤の配置", "color": "#a855f7"},
    {"id": "late", "label": "終盤の配置", "color": "#10b981"},
]

TAG_DEFS: list[dict[str, Any]] = [
    # 1) 初動の配置
    {"id": "open_sage_omen_a", "cat": "opening", "label": "セージ＋オーメンがAメイン", "color": "#ef4444"},
    {"id": "open_fade_b", "cat": "opening", "label": "フェイドがB", "color": "#f97316"},
    {"id": "open_fade_chamber_b", "cat": "opening", "label": "フェイド＋チェンバーがBメイン", "color": "#ec4899"},
    {"id": "open_4_1", "cat": "opening", "label": "初動で4-1配置", "color": "#f59e0b"},
    {"id": "open_5_0", "cat": "opening", "label": "初動で5-0配置", "color": "#eab308"},
    # 2) スモークの位置
    {"id": "smk_a_main", "cat": "smoke", "label": "Aメインスモーク", "color": "#06b6d4"},
    {"id": "smk_a_back", "cat": "smoke", "label": "Aの奥スモーク", "color": "#0ea5e9"},
    {"id": "smk_a_link", "cat": "smoke", "label": "Aリンク膨らみスモーク", "color": "#38bdf8"},
    {"id": "smk_bottom", "cat": "smoke", "label": "ボトムスモーク", "color": "#22d3ee"},
    # 3) 中盤の配置
    {"id": "mid_a_main_set", "cat": "mid", "label": "Aメインセット", "color": "#a855f7"},
    {"id": "mid_b_main_set", "cat": "mid", "label": "Bメインセット", "color": "#8b5cf6"},
    {"id": "mid_a_split", "cat": "mid", "label": "Aスプリット", "color": "#c084fc"},
    {"id": "mid_b_split", "cat": "mid", "label": "Bスプリット", "color": "#7c3aed"},
    {"id": "mid_a_fake_b", "cat": "mid", "label": "AフェイクB", "color": "#d946ef"},
    {"id": "mid_b_fake_a", "cat": "mid", "label": "BフェイクA", "color": "#e879f9"},
    # 4) 終盤の配置
    {"id": "late_a_postplant", "cat": "late", "label": "Aポストプラント", "color": "#10b981"},
    {"id": "late_b_postplant", "cat": "late", "label": "Bポストプラント", "color": "#34d399"},
]

TAG_BY_ID: dict[str, dict[str, Any]] = {t["id"]: t for t in TAG_DEFS}


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


def _buy_category(buy_str: str) -> str:
    s = buy_str or ""
    if "ピストル" in s:
        return "ピストル"
    if "フル" in s:
        return "フル"
    if "ハーフ" in s:
        return "ハーフ"
    if "エコ" in s:
        return "エコ"
    return "その他"


def _classify_smoke_tags(smokes: list[dict[str, Any]], callouts: list[dict[str, Any]]) -> list[str]:
    """スモーク位置（Summit 基準＋コールアウト名）からスモークタグを抽出する。"""
    found: set[str] = set()
    for s in smokes:
        x, y = float(s["x"]), float(s["y"])
        sup, lbl = _nearest_callout(x, y, callouts)
        low = lbl.lower()

        # ボトムスモーク（Mid Bottom / Mid Window / Mid Fountain）
        if ("mid bottom" in low or "mid window" in low or "mid fountain" in low) or (
            0.45 <= x <= 0.60 and 0.30 <= y <= 0.55
        ):
            found.add("smk_bottom")

        # Aリンク膨らみスモーク（A Link / Mid Bend 付近）
        if ("a link" in low or "mid bend" in low) or (0.58 <= x <= 0.74 and 0.34 <= y <= 0.51):
            found.add("smk_a_link")

        # Aメインスモーク（A Main / A Lobby 入口〜A Garden 手前）
        if ("a main" in low) or (0.68 <= x <= 0.85 and 0.46 <= y <= 0.64):
            found.add("smk_a_main")

        # Aの奥スモーク（A Art / A Cave / A Hall / A Site 奥）
        if ("a art" in low or "a cave" in low or "a hall" in low) or (x >= 0.68 and y <= 0.39):
            found.add("smk_a_back")
    return sorted(found)


def classify_rounds(
    summary: dict[str, Any],
    report: dict[str, Any] | None = None,
    *,
    team: str | None = None,
    side: str = "ATK",
    match_id: str | None = None,
    match_label: str | None = None,
    gifs_dir: Path | None = None,
    opening_window_sec: float = 4.0,
) -> dict[str, Any]:
    """1 試合分の対象チーム・攻守ラウンドを抽出し、全 17 タグを付与する。"""
    rep = report or {}
    callouts = (rep.get("tactics") or {}).get("callouts_placed") or []
    rep_rounds = {int(r["n"]): r for r in (rep.get("rounds") or []) if isinstance(r.get("n"), int)}
    team1 = str(rep.get("team1") or summary.get("team1") or "Team 1")
    team2 = str(rep.get("team2") or summary.get("team2") or "Team 2")
    mid_str = match_id or Path(str(summary.get("match_dir") or "match")).name
    mlabel = match_label or f"{team1} vs {team2} ({mid_str})"

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
        timeline = r.get("timeline") or []
        max_t = max((float(fr.get("t", 0.0)) for fr in timeline), default=0.0)

        # 1) 初動の配置 (t <= opening_window_sec)
        opening_agents: dict[str, dict[str, Any]] = {}
        for fr in timeline:
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

        is_4_1 = False
        is_5_0 = False
        if n_a >= 3 and n_b == 0:
            formation = "A5-B0"
            is_5_0 = True
        elif n_b >= 3 and n_a == 0:
            formation = "B5-A0"
            is_5_0 = True
        elif n_a >= 3 and n_b == 1:
            formation = "A4-B1"
            is_4_1 = True
        elif n_b >= 3 and n_a == 1:
            formation = "B4-A1"
            is_4_1 = True
        else:
            formation = f"A{n_a}-B{n_b}"

        tags: list[str] = []
        sage = opening_agents.get("Sage")
        omen = opening_agents.get("Omen")
        fade = opening_agents.get("Fade")
        chamber = opening_agents.get("Chamber")

        if sage and omen and sage["in_a_main"] and omen["in_a_main"]:
            tags.append("open_sage_omen_a")
        if fade and fade["in_b"]:
            tags.append("open_fade_b")
        if fade and chamber and fade["in_b"] and chamber["in_b"]:
            tags.append("open_fade_chamber_b")
        if is_4_1:
            tags.append("open_4_1")
        if is_5_0:
            tags.append("open_5_0")

        # 2) スモークの位置（全フレーム）
        all_smokes: list[dict[str, Any]] = []
        first_smoke = None
        early_a_smk = False
        early_b_smk = False
        for fr in timeline:
            t_val = float(fr.get("t", 0.0))
            sms = fr.get("smokes") or []
            for s in sms:
                all_smokes.append(s)
                if t_val <= 14.0:
                    sx = float(s["x"])
                    if sx >= 0.65:
                        early_a_smk = True
                    elif sx <= 0.38:
                        early_b_smk = True
            if first_smoke is None and sms:
                s0 = sms[0]
                _, s_lbl = _nearest_callout(float(s0["x"]), float(s0["y"]), callouts)
                first_smoke = {"t": t_val, "callout": s_lbl, "count": len(sms)}

        tags.extend(_classify_smoke_tags(all_smokes, callouts))

        # 3) 中盤の配置 (t >= 6s)
        a_main_pushers: set[str] = set()
        b_main_pushers: set[str] = set()
        mid_a_pushers: set[str] = set()
        mid_b_pushers: set[str] = set()
        early_a_press: set[str] = set()
        early_b_press: set[str] = set()
        late_xs: list[float] = []
        final_xs: list[float] = []
        final_ys: list[float] = []

        for fr in timeline:
            t_val = float(fr.get("t", 0.0))
            for a in fr.get("agents") or []:
                if a.get("side") != side_u:
                    continue
                ag = str(a.get("agent") or f"p_{round(float(a['x']),1)}")
                ax, ay = float(a["x"]), float(a["y"])

                if 2.0 <= t_val <= 14.0:
                    if ax >= 0.66 and ay <= 0.66:
                        early_a_press.add(ag)
                    elif ax <= 0.36 and ay <= 0.62:
                        early_b_press.add(ag)

                if t_val >= 6.0:
                    if ax >= 0.68 and ay <= 0.58:
                        a_main_pushers.add(ag)
                    if ax <= 0.36 and ay <= 0.56:
                        b_main_pushers.add(ag)
                    if 0.44 <= ax <= 0.68 and ay <= 0.52:
                        mid_a_pushers.add(ag)
                    if 0.35 <= ax <= 0.58 and ay <= 0.52:
                        mid_b_pushers.add(ag)

                if t_val >= 10.0:
                    late_xs.append(ax)
                if t_val >= max(10.0, max_t - 10.0):
                    final_xs.append(ax)
                    final_ys.append(ay)

        late_x = round(sum(late_xs) / len(late_xs), 2) if late_xs else None
        final_x = round(sum(final_xs) / len(final_xs), 2) if final_xs else late_x
        final_y = round(sum(final_ys) / len(final_ys), 2) if final_ys else None

        exec_site = None
        if late_x is not None:
            if late_x >= 0.56:
                exec_site = "A"
            elif late_x <= 0.42:
                exec_site = "B"

        if exec_site == "A":
            if len(a_main_pushers) >= 1 and len(mid_a_pushers) >= 2:
                tags.append("mid_a_split")
            elif len(a_main_pushers) >= 2:
                tags.append("mid_a_main_set")
            if len(early_b_press) >= 2 and (early_b_smk or n_b >= 2):
                tags.append("mid_b_fake_a")
        elif exec_site == "B":
            if len(b_main_pushers) >= 1 and len(mid_b_pushers) >= 2:
                tags.append("mid_b_split")
            elif len(b_main_pushers) >= 2:
                tags.append("mid_b_main_set")
            if len(early_a_press) >= 2 and (early_a_smk or n_a >= 2):
                tags.append("mid_a_fake_b")

        # 4) 終盤の配置（ポストプラント）
        method_str = str(rr.get("method") or "").lower()
        plant_site = str(rr.get("plant_site") or rr.get("site") or "").upper()
        reached_postplant = (
            max_t >= 20.0
            or ("爆破" in method_str or "解除" in method_str or "spike" in method_str or "defuse" in method_str)
            or plant_site in ("A", "B")
        )
        if reached_postplant:
            if plant_site == "A" or (plant_site == "" and final_x is not None and final_x >= 0.58 and (final_y is None or final_y <= 0.58)):
                tags.append("late_a_postplant")
            elif plant_site == "B" or (plant_site == "" and final_x is not None and final_x <= 0.42 and (final_y is None or final_y <= 0.58)):
                tags.append("late_b_postplant")

        final_tendency = "A 攻め" if exec_site == "A" else ("B 攻め" if exec_site == "B" else "Mid / 混戦")

        classified.append({
            "uid": f"{mid_str}_{r.get('round')}",
            "match_id": mid_str,
            "match_label": mlabel,
            "round": r.get("round"),
            "round_num": rn,
            "team": r_team,
            "opponent": team2 if is_team1 else team1,
            "side": side_u,
            "winner": winner,
            "won": won,
            "buy_self": buy_self,
            "buy_opp": buy_opp,
            "buy_cat": _buy_category(buy_self),
            "formation": formation,
            "a_agents": a_agents,
            "b_agents": b_agents,
            "opening_agents": list(opening_agents.values()),
            "tags": list(dict.fromkeys(tags)),
            "first_smoke": first_smoke,
            "final_tendency": final_tendency,
            "late_x": late_x,
            "max_t": round(max_t, 1),
            "gifs_dir": str(gifs_dir) if gifs_dir else "",
            "raw_gif_file": r.get("raw_gif_file"),
            "gif_file": r.get("gif_file"),
            "compare_gif_file": r.get("compare_gif_file"),
            "n_frames": r.get("n_frames"),
        })

    return {
        "map_name": summary.get("map_name") or rep.get("map") or "Unknown",
        "team": team or "ALL",
        "side": side_u,
        "match_id": mid_str,
        "match_label": mlabel,
        "total_rounds": len(classified),
        "rounds": classified,
    }


def merge_classified_matches(match_results: list[dict[str, Any]], *, team: str, side: str) -> dict[str, Any]:
    """複数試合の分類結果を 1 つのプレイブックデータセットに統合する。"""
    all_rounds: list[dict[str, Any]] = []
    matches_meta: list[dict[str, Any]] = []
    map_name = "Unknown"
    for mr in match_results:
        if mr.get("map_name"):
            map_name = str(mr["map_name"])
        rs = mr.get("rounds") or []
        all_rounds.extend(rs)
        wins = sum(1 for r in rs if r.get("won") is True)
        losses = sum(1 for r in rs if r.get("won") is False)
        matches_meta.append({
            "match_id": mr["match_id"],
            "match_label": mr["match_label"],
            "rounds_count": len(rs),
            "wins": wins,
            "losses": losses,
        })

    tag_counts: dict[str, dict[str, Any]] = {}
    for td in TAG_DEFS:
        tid = td["id"]
        matched = [r for r in all_rounds if tid in (r.get("tags") or [])]
        tag_counts[tid] = {
            **td,
            "count": len(matched),
            "wins": sum(1 for r in matched if r.get("won") is True),
            "losses": sum(1 for r in matched if r.get("won") is False),
        }

    return {
        "map_name": map_name,
        "team": team,
        "side": side.upper(),
        "matches_count": len(matches_meta),
        "total_rounds": len(all_rounds),
        "matches": matches_meta,
        "tag_stats": tag_counts,
        "rounds": all_rounds,
    }


def _gif_frame_data_uri(gif_path: Path, frame_idx: int = 0) -> str | None:
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
    embed_snapshots: bool = False,
) -> str:
    """複数試合対応のタグ付け＆AND/ORマルチタグ検索プレイブック HTML を生成する。"""
    total = data["total_rounds"]
    wins_all = sum(1 for r in data["rounds"] if r.get("won") is True)
    losses_all = sum(1 for r in data["rounds"] if r.get("won") is False)
    tag_stats = data.get("tag_stats") or {}

    # 試合フィルタボタン
    match_btns = [f"<button class='mbtn active' data-match='all'>全{data.get('matches_count', 1)}試合 ({total}R)</button>"]
    for m in data.get("matches") or []:
        match_btns.append(
            f"<button class='mbtn' data-match='{html.escape(m['match_id'])}'>"
            f"{html.escape(m['match_label'])} ({m['rounds_count']}R / {m['wins']}W-{m['losses']}L)</button>"
        )

    # カテゴリ別タグピル群
    cat_sections: list[str] = []
    for cat in TAG_CATEGORIES:
        cid = cat["id"]
        pills: list[str] = []
        for td in TAG_DEFS:
            if td["cat"] != cid:
                continue
            st = tag_stats.get(td["id"], {"count": 0, "wins": 0, "losses": 0})
            cnt = st["count"]
            wr = f" {st['wins']}W-{st['losses']}L" if cnt > 0 else ""
            pills.append(
                f"<button class='tbtn' data-tag='{td['id']}' style='--c:{td['color']}'>"
                f"<span class='tlabel'>{html.escape(td['label'])}</span>"
                f"<span class='tcnt'>{cnt}R{wr}</span></button>"
            )
        cat_sections.append(
            f"<div class='cat-box'><div class='cat-head' style='border-left-color:{cat['color']}'>"
            f"{html.escape(cat['label'])}</div><div class='tag-row'>{''.join(pills)}</div></div>"
        )

    # ラウンドカード
    round_cards: list[str] = []
    for r in data["rounds"]:
        tags_list = r.get("tags") or []
        tags_attr = " ".join(tags_list)
        res_cls = "win" if r.get("won") is True else ("lose" if r.get("won") is False else "none")
        res_txt = "WIN" if r.get("won") is True else ("LOSE" if r.get("won") is False else "-")

        tag_pills = "".join(
            f"<span class='pill' style='background:{TAG_BY_ID[t]['color']}22;border-color:{TAG_BY_ID[t]['color']};color:{TAG_BY_ID[t]['color']}'>"
            f"{html.escape(TAG_BY_ID[t]['label'])}</span>"
            for t in tags_list if t in TAG_BY_ID
        ) or "<span class='pill muted'>タグ未該当</span>"

        agent_chips = "".join(
            f"<span class='chip {'chip-a' if a['macro'] == 'A' else 'chip-b'}'>"
            f"<b>{html.escape(a['agent'])}</b> {html.escape(a['callout'])}</span>"
            for a in sorted(r.get("opening_agents") or [], key=lambda x: (0 if x["macro"] == "A" else 1, x["agent"]))
        )
        smk_txt = (
            f"初回スモーク: t={r['first_smoke']['t']:.0f}s ({r['first_smoke']['callout']})"
            if r.get("first_smoke") else "初回スモーク: 検出なし"
        )

        media_html = ""
        gdir = Path(r["gifs_dir"]) if r.get("gifs_dir") else None
        if embed_snapshots and gdir is not None and r.get("gif_file"):
            p_ann = gdir / str(r["gif_file"])
            u0 = _gif_frame_data_uri(p_ann, 0)
            mid_idx = max(2, int((r.get("n_frames") or 6) // 3))
            u_mid = _gif_frame_data_uri(p_ann, mid_idx)
            if u0 and u_mid:
                media_html = (
                    "<div class='media-pair'>"
                    f"<div><div class='cap'>初動 (t=0s)</div><img src='{u0}' loading='lazy'></div>"
                    f"<div><div class='cap'>中盤展開 (#{mid_idx:02d})</div><img src='{u_mid}' loading='lazy'></div>"
                    "</div>"
                )
        elif r.get("raw_gif_file") and r.get("gif_file"):
            prefix = f"{r['match_id']}/" if data.get("matches_count", 1) > 1 else ""
            media_html = (
                "<div class='media-pair'>"
                f"<div><div class='cap'>未解析 (Raw)</div><img src='{html.escape(prefix + str(r['raw_gif_file']))}' loading='lazy'></div>"
                f"<div><div class='cap'>解析あり (Annotated)</div><img src='{html.escape(prefix + str(r['gif_file']))}' loading='lazy'></div>"
                "</div>"
            )

        round_cards.append(
            f"<div class='rcard' data-match='{html.escape(r['match_id'])}' data-tags='{html.escape(tags_attr)}' "
            f"data-res='{res_cls}' data-buy='{html.escape(r.get('buy_cat') or '')}' data-tend='{html.escape(r.get('final_tendency') or '')}'>"
            f"<div class='rcard-head'>"
            f"<div><span class='mtag'>{html.escape(r['match_label'])}</span> "
            f"<b>{html.escape(str(r['round']))}</b> "
            f"<span class='badge {res_cls}'>{res_txt}</span> "
            f"<span class='form'>{html.escape(r['formation'])}</span> "
            f"<span class='tend'>{html.escape(r['final_tendency'])}</span></div>"
            f"<div class='small muted'>{html.escape(r['buy_self'])} vs {html.escape(r['buy_opp'])}</div>"
            f"</div>"
            f"<div class='pills'>{tag_pills}</div>"
            f"{media_html}"
            f"<div class='chips'>{agent_chips}</div>"
            f"<div class='small muted' style='margin-top:6px'>{html.escape(smk_txt)} ｜ 長さ: {r.get('max_t', 0):.0f}s</div>"
            "</div>"
        )

    title = f"{data['team']} {data['side']} プレイブック検索 — {str(data['map_name']).capitalize()} ({data.get('matches_count', 1)}試合 / {total}R)"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --bg:#0a0f1d; --card:#121a2e; --card2:#17213a; --border:#253352; --text:#e6edf7; --muted:#93a5c6; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:18px; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,sans-serif; line-height:1.45; }}
  .wrap {{ max-width:1440px; margin:0 auto; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:10px; padding-bottom:12px; border-bottom:1px solid var(--border); margin-bottom:14px; }}
  h1 {{ margin:0; font-size:21px; }}
  .sub {{ color:var(--muted); font-size:12.5px; margin-top:3px; }}
  .controls {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; margin-bottom:16px; }}
  .bar-row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }}
  .mbtn,.sbtn {{ background:var(--card2); color:var(--text); border:1px solid var(--border); border-radius:999px; padding:5px 11px; font-size:12px; cursor:pointer; }}
  .mbtn.active,.sbtn.active {{ background:#2563eb; border-color:#3b82f6; font-weight:700; }}
  .cat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:10px; margin-top:10px; }}
  .cat-box {{ background:#0d1426; border:1px solid #1e2a45; border-radius:9px; padding:10px; }}
  .cat-head {{ font-size:12px; font-weight:700; padding-left:8px; border-left:3px solid #64748b; margin-bottom:8px; color:#cbd5e1; }}
  .tag-row {{ display:flex; flex-wrap:wrap; gap:6px; }}
  .tbtn {{ display:inline-flex; align-items:center; gap:6px; background:#151f36; color:var(--text); border:1px solid #2b3b5e; border-radius:8px; padding:5px 9px; font-size:12px; cursor:pointer; transition:all .12s; }}
  .tbtn:hover {{ border-color:var(--c); }}
  .tbtn.active {{ background:var(--c); border-color:var(--c); color:#08101e; font-weight:800; }}
  .tcnt {{ font-size:10.5px; opacity:0.85; font-family:ui-monospace,monospace; }}
  .stats-bar {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; background:var(--card2); border:1px solid var(--border); border-radius:9px; padding:10px 14px; margin-bottom:14px; font-size:13px; }}
  .reset-btn {{ background:#334155; color:#fff; border:none; border-radius:7px; padding:6px 12px; font-size:12px; cursor:pointer; font-weight:600; }}
  .reset-btn:hover {{ background:#475569; }}
  .rgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr)); gap:14px; }}
  .rcard {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; }}
  .rcard-head {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }}
  .mtag {{ display:inline-block; background:#1e293b; color:#cbd5e1; border-radius:5px; padding:2px 6px; font-size:11px; margin-right:4px; }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:5px; font-size:11px; font-weight:700; }}
  .badge.win {{ background:#14532d; color:#86efac; }}
  .badge.lose {{ background:#7f1d1d; color:#fca5a5; }}
  .form {{ display:inline-block; background:#1e293b; border:1px solid #334155; padding:2px 7px; border-radius:5px; font-family:ui-monospace,monospace; font-size:11px; font-weight:700; }}
  .tend {{ display:inline-block; background:#172554; color:#93c5fd; padding:2px 7px; border-radius:5px; font-size:11px; font-weight:600; }}
  .pills {{ display:flex; gap:5px; flex-wrap:wrap; margin-bottom:8px; }}
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
      <div class="sub">全試合通算: <b>{wins_all}勝 {losses_all}敗</b> ｜ タグを複数クリックして AND / OR 検索できます</div>
    </div>
  </header>

  <section class="controls">
    <div class="bar-row">
      <span class="small muted">試合:</span>
      {''.join(match_btns)}
    </div>
    <div class="bar-row">
      <span class="small muted">タグ結合:</span>
      <button class="sbtn active" data-mode="AND">すべて一致 (AND)</button>
      <button class="sbtn" data-mode="OR">いずれか一致 (OR)</button>
      <span class="small muted" style="margin-left:10px">勝敗:</span>
      <button class="sbtn active" data-resf="all">すべて</button>
      <button class="sbtn" data-resf="win">WIN</button>
      <button class="sbtn" data-resf="lose">LOSE</button>
      <span class="small muted" style="margin-left:10px">バイ:</span>
      <button class="sbtn active" data-buyf="all">すべて</button>
      <button class="sbtn" data-buyf="フル">フル</button>
      <button class="sbtn" data-buyf="ハーフ">ハーフ</button>
      <button class="sbtn" data-buyf="エコ">エコ</button>
      <button class="sbtn" data-buyf="ピストル">ピストル</button>
    </div>
    <div class="cat-grid">{''.join(cat_sections)}</div>
  </section>

  <div class="stats-bar">
    <div id="live-stats">表示中: <b>{total} / {total} ラウンド</b></div>
    <button class="reset-btn" id="reset-all">条件をすべてクリア</button>
  </div>

  <section class="rgrid">{''.join(round_cards)}</section>
</div>
<script>
(function() {{
  let selMatch = 'all';
  let selMode = 'AND';
  let selRes = 'all';
  let selBuy = 'all';
  const selTags = new Set();
  const cards = Array.from(document.querySelectorAll('.rcard'));
  const statsEl = document.getElementById('live-stats');

  function update() {{
    let shown = 0, wins = 0, losses = 0, aCnt = 0, bCnt = 0;
    const req = Array.from(selTags);
    for (const c of cards) {{
      let ok = true;
      if (selMatch !== 'all' && c.dataset.match !== selMatch) ok = false;
      if (ok && selRes !== 'all' && c.dataset.res !== selRes) ok = false;
      if (ok && selBuy !== 'all' && c.dataset.buy !== selBuy) ok = false;
      if (ok && req.length > 0) {{
        const ctags = (c.dataset.tags || '').split(/\\s+/).filter(Boolean);
        if (selMode === 'AND') ok = req.every(t => ctags.includes(t));
        else ok = req.some(t => ctags.includes(t));
      }}
      c.classList.toggle('hidden', !ok);
      if (ok) {{
        shown++;
        if (c.dataset.res === 'win') wins++;
        else if (c.dataset.res === 'lose') losses++;
        if (c.dataset.tend === 'A 攻め') aCnt++;
        else if (c.dataset.tend === 'B 攻め') bCnt++;
      }}
    }}
    const wr = (wins + losses) ? Math.round(wins * 100 / (wins + losses)) + '%' : '-';
    statsEl.innerHTML = `表示中: <b>${{shown}} / ${{cards.length}} ラウンド</b> ｜ 戦績: <b>${{wins}}勝 ${{losses}}敗 (${{wr}})</b> ｜ 最終傾向: A攻め ${{aCnt}} / B攻め ${{bCnt}}`;
  }}

  document.querySelectorAll('.mbtn').forEach(b => b.addEventListener('click', () => {{
    selMatch = b.dataset.match;
    document.querySelectorAll('.mbtn').forEach(x => x.classList.toggle('active', x === b));
    update();
  }}));
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => {{
    selMode = b.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(x => x.classList.toggle('active', x === b));
    update();
  }}));
  document.querySelectorAll('[data-resf]').forEach(b => b.addEventListener('click', () => {{
    selRes = b.dataset.resf;
    document.querySelectorAll('[data-resf]').forEach(x => x.classList.toggle('active', x === b));
    update();
  }}));
  document.querySelectorAll('[data-buyf]').forEach(b => b.addEventListener('click', () => {{
    selBuy = b.dataset.buyf;
    document.querySelectorAll('[data-buyf]').forEach(x => x.classList.toggle('active', x === b));
    update();
  }}));
  document.querySelectorAll('.tbtn').forEach(b => b.addEventListener('click', () => {{
    const t = b.dataset.tag;
    if (selTags.has(t)) selTags.delete(t);
    else selTags.add(t);
    b.classList.toggle('active', selTags.has(t));
    update();
  }}));
  document.getElementById('reset-all').addEventListener('click', () => {{
    selMatch = 'all'; selMode = 'AND'; selRes = 'all'; selBuy = 'all'; selTags.clear();
    document.querySelectorAll('.tbtn').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.mbtn').forEach(x => x.classList.toggle('active', x.dataset.match === 'all'));
    document.querySelectorAll('[data-mode]').forEach(x => x.classList.toggle('active', x.dataset.mode === 'AND'));
    document.querySelectorAll('[data-resf]').forEach(x => x.classList.toggle('active', x.dataset.resf === 'all'));
    document.querySelectorAll('[data-buyf]').forEach(x => x.classList.toggle('active', x.dataset.buyf === 'all'));
    update();
  }});
  update();
}})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="複数試合のプレイブック用ラウンドタグ付け＆検索 HTML を生成")
    ap.add_argument("--match", action="append", default=[], help="summary.json:report.json[:Label]（複数指定可）")
    ap.add_argument("--summary", default=None, help="単一試合用 match_summary.json")
    ap.add_argument("--report", default=None, help="単一試合用 report.json")
    ap.add_argument("--team", default="LOUD", help="対象チーム名（例: LOUD）")
    ap.add_argument("--side", default="ATK", choices=["ATK", "DEF"], help="攻守フィルタ")
    ap.add_argument("--out", required=True, help="出力 HTML パス")
    ap.add_argument("--embed-snapshots", action="store_true", help="GIF の初動・中盤スナップショットを base64 埋め込み")
    args = ap.parse_args(argv)

    specs: list[tuple[Path, Path | None, str | None]] = []
    for m_spec in args.match:
        parts = m_spec.split(":", 2)
        sp = Path(parts[0])
        rp = Path(parts[1]) if len(parts) >= 2 and parts[1] else None
        lbl = parts[2] if len(parts) >= 3 and parts[2] else None
        specs.append((sp, rp, lbl))
    if not specs and args.summary:
        specs.append((Path(args.summary), Path(args.report) if args.report else None, None))
    if not specs:
        raise SystemExit("--match または --summary を指定してください")

    per_match: list[dict[str, Any]] = []
    for sp, rp, lbl in specs:
        sm = json.loads(sp.read_text("utf-8"))
        rep = json.loads(rp.read_text("utf-8")) if (rp and rp.exists()) else None
        per_match.append(
            classify_rounds(
                sm,
                rep,
                team=args.team,
                side=args.side,
                match_id=sp.parent.name,
                match_label=lbl,
                gifs_dir=sp.parent,
            )
        )

    merged = merge_classified_matches(per_match, team=args.team, side=args.side)
    html_text = render_playbook_html(merged, embed_snapshots=bool(args.embed_snapshots))
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(html_text, "utf-8")
    out_p.with_suffix(".json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")

    print(json.dumps({
        "team": merged["team"],
        "side": merged["side"],
        "matches_count": merged["matches_count"],
        "total_rounds": merged["total_rounds"],
        "tag_counts": {k: v["count"] for k, v in merged["tag_stats"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
