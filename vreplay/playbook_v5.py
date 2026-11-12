"""プレイブック v5：マップ × 5人構成のページ。図（マップ図・動く図・VODカード）を主役にする。

v4（`playbook_v4.py`）はチーム単位で集計を文章で並べていた。v5 は単位を「マップ × 5人構成」に変え、
構成ごとに 1 ページを作る。判定は `gamemodel`、図は `mapfig`、動く図は `deck` の部品を再利用する。
仕様の正本は `docs/PLAYBOOK_v5.md`。

設定値（モジュール冒頭の定数）は docs/PLAYBOOK_v5.md と各ページ付録にも出る。
"""
from __future__ import annotations

import copy
import html
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import deck as D
from . import gamemodel as GM
from . import kb as K
from . import mapfig as MF
from . import tactics as TC

ATTACK, DEFENSE = GM.ATTACK, GM.DEFENSE

# ============================================================== 設定値
ENV_START = "2026-09-01"
ENV_LABEL = "Champions 2026"
PAGE_MIN_ROUNDS = 30
PAGE_MIN_FULL5 = 15
PAGE_MIN_SETUP_RATE = 0.60
PATTERN_MIN = 3
PATTERN_DIST = 0.09
HAND_MIN_N = 3
HAND_MIN_WINRATE = 0.5

SETTINGS = [
    ("ENV_START", ENV_START, "これ以降の日付のラウンドだけをページ集計に入れる"),
    ("ENV_LABEL", ENV_LABEL, "現在の環境の呼び名"),
    ("PAGE_MIN_ROUNDS", PAGE_MIN_ROUNDS, "環境内の構成ラウンド数の下限"),
    ("PAGE_MIN_FULL5", PAGE_MIN_FULL5, "1:40 で5人そろったラウンド数の下限"),
    ("PAGE_MIN_SETUP_RATE", PAGE_MIN_SETUP_RATE, "setup が取れたラウンドの割合の下限"),
    ("PATTERN_MIN", PATTERN_MIN, "型（配置クラスタ）の成立ラウンド数"),
    ("PATTERN_DIST", PATTERN_DIST, "型クラスタの距離しきい値（表示座標 0..1）"),
    ("HAND_MIN_N", HAND_MIN_N, "手札の成立ラウンド数"),
    ("HAND_MIN_WINRATE", HAND_MIN_WINRATE, "手札の勝率の下限"),
]

# 内部のノード名 → プレイヤーの言葉。表示は必ずここを通す（略号や設計用語を出さない）
GLOSSARY = {
    "fast_deep": "深くまで早く入って主導権を取る",
    "bu_reaction": "薄い側に圧をかけられたら逆側を取りに行く",
    "bu_counter": "前に出てきた相手に合わせて当たる",
    "hold": "構えて待つ",
    "freeze": "先に1人倒されたら止まって立て直す",
    "v_reaction": "空いたバイタルを取って近いサイトへ寄る",
    "v_counter": "バイタルの当たりを返り討ちにする",
    "v_hold": "バイタルを取ったら構えるか別へ回る",
    "v_occupy": "取ったバイタルに居座って圧をかける",
    "solo": "1人をサイトに残して回る",
    "reaction_def": "薄いサイトに来たら裏から挟む",
    "choke_def": "サイトの入口で止める",
    "back_def": "サイトの裏で時間を稼いで合流する",
    "retake_def": "取り返すと決めて時間を使う",
    "site_occupy": "デュエリストが奥に入り全員で続く",
    "direct": "バイタルでキルが出たらサイトへ飛び込む",
    "challenge": "入口で当たってそのまま入る",
    "joker": "1人で裏を取り別の場所を揺さぶる",
    "choke_counter": "入口に来た相手をカウンター",
    "boom": "1〜2人残して退き時間を稼ぐ",
    "fast": "早く集まってリテイク",
    "surround": "複数の入口から囲い込んでリテイク",
    "slow": "そろえてからゆっくりリテイク",
    "save": "寄せずに1人で時間を稼ぐ",
    "up2": "人数有利なら距離を詰める",
    "down1": "人数不利なら分かれて片方で当たる",
    "anti_eco": "マネー有利なら普段と違う手を打つ",
    "timing_push": "タイミングを見て前に出る",
    "one_man": "薄くなった側を1人で持ち、厚い側に人数を寄せる",
    "stack": "1か所に集めて守る",
}
ABBREV = ("DC", "PrP", "PoC", "BP", "BTP")
ROLE_JA = {"duelist": "デュエリスト", "controller": "コントローラー", "initiator": "イニシエーター",
           "sentinel": "センチネル"}

# 守り側で使う動作語。攻め専用の言い回しを守りの型名・手札名に残さない。
DEF_ACTION_WORD = {
    "fast_deep": "前に出て間合いを詰める",
}
# 攻め専用の言い回し（守りの表示では使わない）
ATTACK_ONLY_PHRASES = ("深くまで早く入って", "バイタルでキルが出たらサイトへ飛び込む",
                       "デュエリストが奥に入り全員で続く", "1人で裏を取り別の場所")
# 守りの局面判定が返しうるノード（守りの表示に攻め専用の表現が混ざらないことの検査対象）
DEFENSE_NODES = ("fast_deep", "bu_counter", "hold", "v_occupy", "v_reaction", "v_counter", "solo",
                 "timing_push", "one_man", "reaction_def", "stack", "choke_def", "back_def",
                 "fast", "surround", "slow", "save", "up2", "down1", "anti_eco")


def glossary_word(nid: str | None) -> str:
    """内部ノード ID → プレイヤーの言葉。知らない ID はそのまま返す（略号は含まれない）。"""
    if not nid:
        return ""
    return GLOSSARY.get(str(nid), str(nid))


def glossary_word_side(nid: str | None, side: str) -> str:
    """側に合わせた動作語。守りでは攻め専用の言い回しを使わない。"""
    if not nid:
        return ""
    if side == DEFENSE:
        if nid in DEF_ACTION_WORD:
            return DEF_ACTION_WORD[nid]
        word = glossary_word(nid)
        if any(ph in word for ph in ATTACK_ONLY_PHRASES):
            return "その場で構える"
        return word
    return glossary_word(nid)


def has_abbrev(text: str) -> bool:
    return any(a in str(text or "") for a in ABBREV)


# ============================================================== 小道具
def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _s(text: str) -> str:
    """本文の 1 行（検証はこの p.s を数える）。"""
    return f'<p class="s">{_e(text)}</p>'


def _ul(lines: list[str]) -> str:
    return f"<ul>{''.join(f'<li>{_e(x)}</li>' for x in lines)}</ul>" if lines else ""


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _wl(w: int, n: int) -> str:
    return f"{n} 回中 {w} 回勝ち"


_FS_CSS = re.compile(r"font-size:\s*([0-9]*\.?[0-9]+)px")
_FS_ATTR = re.compile(r'font-size="([0-9]*\.?[0-9]+)"')


def _bump_css(css: str) -> str:
    return _FS_CSS.sub(lambda m: f"font-size:{max(float(m.group(1)), 14):g}px", css)


def _bump_svg(svg: str) -> str:
    return _FS_ATTR.sub(lambda m: f'font-size="{max(float(m.group(1)), 14):g}"', svg)


def _bump_html(h: str) -> str:
    """HTML 内の font-size 指定（style 属性ふくむ）を 14px 以上へ引き上げる。"""
    return _FS_CSS.sub(lambda m: f"font-size:{max(float(m.group(1)), 14):g}px", h)


# ============================================================== 構成のグルーピング
def valid_comp_key(ck) -> bool:
    if not ck or not isinstance(ck, str) or "?" in ck:
        return False
    parts = [p for p in ck.split("|") if p.strip()]
    return bool(parts)


def group_by_comp(records: list[dict]) -> tuple[dict[str, list[dict]], int]:
    """comp_key（5エージェント完全一致）でまとめる。構成不明は除外して数を返す。"""
    by: dict[str, list[dict]] = defaultdict(list)
    unknown = 0
    for r in records:
        ck = r.get("comp_key")
        if valid_comp_key(ck):
            by[ck].append(r)
        else:
            unknown += 1
    return dict(by), unknown


def agents_of(comp_key: str) -> list[str]:
    return [a for a in str(comp_key).split("|") if a]


# ============================================================== ゲート
def gate_ok(stat: dict) -> tuple[bool, str]:
    """ページを作る条件。落ちたら理由（条件と実数）を返す。"""
    total = int(stat.get("total") or 0)
    full5 = int(stat.get("full5") or 0)
    used = int(stat.get("used") or 0)
    if total < PAGE_MIN_ROUNDS:
        return False, f"データ不足: ラウンド数 {total} < {PAGE_MIN_ROUNDS}"
    if full5 < PAGE_MIN_FULL5:
        return False, f"データ不足: 5人そろった配置が {full5} 件 < {PAGE_MIN_FULL5}"
    rate = used / total if total else 0.0
    if rate < PAGE_MIN_SETUP_RATE:
        return False, f"データ不足: setup が取れた割合 {rate:.0%} < {PAGE_MIN_SETUP_RATE:.0%}"
    return True, ""


# ============================================================== 型（配置クラスタ）
def round_distance(p: dict, q: dict) -> float:
    """2 ラウンド間の、同じエージェント同士の表示座標のユークリッド距離の平均。"""
    ks = set(p) & set(q)
    if not ks:
        return 9.0
    return sum(math.hypot(p[a][0] - q[a][0], p[a][1] - q[a][1]) for a in ks) / len(ks)


def leader_clusters(points: list[dict], th: float = PATTERN_DIST,
                    min_n: int = PATTERN_MIN) -> tuple[list[list[int]], list[int]]:
    """リーダークラスタリング。大きいまとまりから確定し、min_n 未満は単発に回す。"""
    remaining = set(range(len(points)))
    clusters: list[list[int]] = []
    while remaining:
        best = None
        for i in sorted(remaining):
            grp = [j for j in sorted(remaining) if round_distance(points[i], points[j]) <= th]
            if best is None or len(grp) > len(best[1]):
                best = (i, grp)
        clusters.append(best[1])
        remaining -= set(best[1])
    clusters.sort(key=len, reverse=True)
    patterns = [c for c in clusters if len(c) >= min_n]
    singles = [i for c in clusters if len(c) < min_n for i in c]
    return patterns, singles


def _zone_head(zones: Counter, side: str) -> str:
    """1:40 のゾーン分布から人数の置き方を言う。攻めは「集めて」、守りは「置いて」。"""
    a, b, mid = int(zones.get("A", 0)), int(zones.get("B", 0)), int(zones.get("Mid", 0))
    verb = "集めて" if side == ATTACK else "置いて"
    if a >= 3:
        return f"Aに{a}枚{verb}"
    if b >= 3:
        return f"Bに{b}枚{verb}"
    if a >= 2 and b >= 2:
        return "AとBに2枚ずつ分けて"
    if mid >= 2 and (a or b):
        return f"MIDに{mid}枚{verb}"
    if a >= 2:
        return f"A側に{a}枚{verb}"
    if b >= 2:
        return f"B側に{b}枚{verb}"
    if mid >= 2:
        return f"MIDに{mid}枚{verb}"
    return "広く散らして"


def pattern_name(zones: Counter, action: str | None = None, side: str = ATTACK) -> str:
    """1:40 のゾーン分布 + 多数派の動きから、側に合った動きの名前を組み立てる。"""
    return _zone_head(zones, side) + (action or "その場で構える")


# ============================================================== VOD カード
def yt_id(url: str | None) -> str | None:
    if not url:
        return None
    m = re.search(r"(?:youtu\.be/|v=|embed/)([A-Za-z0-9_-]{6,})", str(url))
    return m.group(1) if m else None


def thumbnail_url(vid: str) -> str:
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"


def vod_card(yt: str | None, label: str) -> str:
    """VOD カード。HTML は直接組み立てる（テキスト用エスケープを通さない）。属性値は 1 回だけ escape。"""
    vid = yt_id(yt)
    if not yt or not vid:
        return ""
    return (f'<figure class="vod"><a href="{html.escape(str(yt), quote=True)}" target="_blank" rel="noopener">'
            f'<img src="{html.escape(thumbnail_url(vid), quote=True)}" alt="{html.escape(label, quote=True)}" '
            f'loading="lazy"></a><figcaption>{_e(label)}</figcaption></figure>')


def sample_label(r: dict) -> str:
    """見本ラベル。「使ったチーム vs 相手」＋ラウンド番号・勝敗。"""
    team = r.get("tag") or r.get("team") or ""
    opp = r.get("opp_tag") or r.get("opp") or ""
    who = f"{team} vs {opp}" if opp else str(team)
    return f"{who} R{r.get('n')}{'○' if r.get('win') else '✕'}"


def sample_key(item_or_r: dict) -> str:
    """見本ラウンドの同一性（ページ内で重複しないことの判定に使う）。"""
    r = item_or_r.get("r") if isinstance(item_or_r, dict) and "r" in item_or_r else item_or_r
    return f'{r.get("match_id")}:{r.get("order")}:{r.get("n")}'


def is_full5(item: dict) -> bool:
    """1:40 に 5 人そろったラウンドか。"""
    su = item.get("su")
    return bool(su) and len(su) == 5


def sample_candidates(items: list[dict], used: set[str] | None = None) -> list[dict]:
    """見本の候補 = full5 かつ未使用。勝ち → 早いラウンドの順。"""
    used = used or set()
    pool = [it for it in items if is_full5(it) and sample_key(it) not in used]
    pool.sort(key=lambda it: (not it["r"].get("win"), sample_key(it)))
    return pool


def _vod_for(item: dict) -> str:
    if not item:
        return ""
    r = item["r"]
    return vod_card(r.get("yt"), sample_label(r))


# ============================================================== 動く図
RECORD_MARKS = [("1:40", 0.0), ("1:30", 10.0), ("1:20", 20.0)]


def _walk_from_view(view: np.ndarray, n: int = 160) -> np.ndarray:
    w = cv2.resize((view[:, :, 3] > 100).astype(np.uint8), (n, n), interpolation=cv2.INTER_AREA)
    w = cv2.morphologyEx(w, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return w.astype(bool)


def motion_frames(r: dict, opp: dict | None, model, k: int, walk) -> dict | None:
    """見本 1 ラウンドの動く図データ。各節目で 3 人以上取れたときだけ返す。"""
    marks = list(RECORD_MARKS)
    ev = GM.events(r, opp or {}, model)
    if ev and 20 < ev[0]["e"] <= 60:
        marks.append(("接敵", float(ev[0]["e"])))
    snaps = []
    for mk, e in marks:
        sn = GM.snap(r, model, e, tol=8)
        if len(sn) < 3:
            if mk == "接敵":
                continue
            return None
        snaps.append((mk, sn))
    agents: list[str] = []
    for _mk, sn in snaps:
        for a in sn:
            if a and a != "?" and a not in agents:
                agents.append(a)
    if len(agents) < 3:
        return None
    team = "def" if r["side"] == DEFENSE else "atk"
    # 節目ごとに、同じ場所で重なる駒を少しずらす（名札が隠れないように）
    xs_by_agent: dict[str, list] = {a: [None] * len(snaps) for a in agents}
    for mi, (_mk, sn) in enumerate(snaps):
        present = [a for a in agents if a in sn]
        raw = [MF.to_view(sn[a][0], sn[a][1], k) for a in present]
        if len(raw) > 1:
            raw = MF.spread(raw, 0.03, 0.025)
        for a, (x, y) in zip(present, raw):
            xs_by_agent[a][mi] = [round(x, 4), round(y, 4)]
    dots = []
    for a in agents:
        xs = list(xs_by_agent[a])
        fwd = None
        for i in range(len(xs)):
            if xs[i] is None:
                xs[i] = fwd
            else:
                fwd = xs[i]
        bwd = None
        for i in range(len(xs) - 1, -1, -1):
            if xs[i] is None:
                xs[i] = bwd
            else:
                bwd = xs[i]
        if any(x is None for x in xs):
            continue
        route = [D.route(walk, xs[i], xs[i + 1]) for i in range(len(xs) - 1)] if walk is not None else []
        dots.append({"label": a, "xy": xs, "route": route})
    if len(dots) < 3:
        return None
    return {"marks": [mk for mk, _ in snaps], "dots": dots, "team": team}


def _motion_html(title: str, fr: dict, bg: str) -> str:
    return D.motion_html({"name": title}, fr, bg)


def _pick_motion(items: list[dict], used: set[str], model, k: int, walk, bg: str, title: str,
                 *, allow_used: bool = False):
    """その中で動く図が作れる full5 の見本を 1 つ返す。`used` の見本は避ける。"""
    pool = sample_candidates(items, used)
    if not pool and allow_used:
        pool = sorted([it for it in items if is_full5(it)],
                      key=lambda it: (not it["r"].get("win"), sample_key(it)))
    for it in pool:
        fr = motion_frames(it["r"], it.get("opp"), model, k, walk)
        if fr:
            return it, _motion_html(title, fr, bg)
    return None, ""


# ============================================================== ラウンド操作
def _move_verb(side: str, a0, a1, model) -> str:
    if not a1 or a1 == a0:
        if side == DEFENSE:
            return "前に出て取る" if (a0 and model.half(a0)) else "その場で構える"
        return "様子を見る"
    if side == DEFENSE:
        h0, h1 = model.half(a0 or ""), model.half(a1)
        if h1 and not h0:
            return "前に出て取る"
        if h0 and not h1:
            return "下がって時間を稼ぐ"
        if GM.zone_of(a0) != GM.zone_of(a1):
            return "ローテする"
        return "その場で構える"
    if GM.in_site(a1):
        return "入る"
    if GM.zone_of(a1) != GM.zone_of(a0):
        return "切り替える"
    return "詰める"


def _agent_lines(item: dict, model) -> list[str]:
    """見本ラウンドの各エージェント：1:40 のエリア → 1:20 のエリア + 動詞（キル数は書かない）。"""
    r = item["r"]
    side = r["side"]
    s20 = GM.snap(r, model, 20, tol=8)
    out = []
    for ag, (_u, _v, a0) in sorted(item["su"].items()):
        if not ag or ag == "?":
            continue
        a1 = (s20.get(ag) or (0, 0, None))[2]
        verb = _move_verb(side, a0, a1, model)
        if a1 and a1 != a0:
            out.append(f"{ag}：1:40 は {a0 or '不明'}、1:20 に {a1}（{verb}）")
        else:
            out.append(f"{ag}：1:40 は {a0 or '不明'}、{verb}")
    return out[:5]


def _static_layers(item: dict, model, marks=("1:40",)) -> list[dict]:
    r = item["r"]
    team = "def" if r["side"] == DEFENSE else "atk"
    e_of = {"1:40": 0.0, "1:30": 10.0, "1:20": 20.0}
    layers = []
    for mk in marks:
        sn = item["su"] if mk == "1:40" else GM.snap(r, model, e_of[mk], tol=8)
        pieces = [{"agent": a, "u": u, "v": v, "team": team} for a, (u, v, _ar) in sn.items() if a and a != "?"]
        if pieces:
            layers.append({"mark": mk, "pieces": pieces})
    return layers


def _fig_svg(fig, item: dict, model, *, marks=("1:40",), alt="", legend=True, px=460) -> str:
    layers = _static_layers(item, model, marks)
    if not layers:
        return ""
    return _bump_svg(fig.svg(layers, alt=alt, legend=legend, px=px))


# ============================================================== データの組み立て
def _prepare(reports: list[dict], map_name: str, cfg: dict, log=print) -> dict:
    allr = [r for rep in reports if str(rep.get("map") or "").lower() == map_name.lower() for r in D.records(rep)]
    if not allr:
        raise ValueError(f"{map_name} のラウンドがありません")
    meta = TC.load_map(map_name, cfg, log=None) or {}
    callouts = list(meta.get("callouts") or [])
    fig = None
    for attempt in range(3):
        try:
            fig = MF.Figure(map_name, cfg, allr, log=log)
        except Exception as e:                                   # noqa: BLE001
            log(f"  マップ図の作成でエラー（{e}）")
            fig = None
        if fig is not None and fig.ok and D.map_data_uri(cfg, map_name, None, callouts):
            break
        fig = None
        log(f"  マップ画像を用意できませんでした。やり直します（{attempt + 1}/3）")
    if fig is None or not fig.ok:
        raise RuntimeError(f"{map_name} のマップ画像を用意できませんでした。生成を中止します")
    kbo = K.load(cfg)
    roles = {a: (kbo.role(a) or "") for r in allr for _p, a in r["players"]}
    model = GM.MapModel(fig.regions, fig.adj, fig.k, kbo.map(map_name), fig.spawns, allr, roles,
                        cent=None if fig.fillable else fig.label_xy)
    walk = _walk_from_view(fig.view) if fig.fillable else None
    return {"allr": allr, "fig": fig, "model": model, "callouts": callouts, "walk": walk,
            "kb": kbo, "k": fig.k, "vitals": model.vitals(), "bg": fig._uri}


def _items_for(rounds: list[dict], allr: list[dict], model) -> list[dict]:
    idx = {(r["match_id"], r["order"], r["n"], r["no"]): r for r in allr}
    items = []
    for r in rounds:
        opp = idx.get((r["match_id"], r["order"], r["n"], 3 - r["no"])) or {}
        su = GM.setup_at_140(r, model)
        cls = GM.classify(r, opp, model) if su is not None else {}
        items.append({"r": r, "opp": opp, "su": su, "cls": cls})
    return items


def comp_stats(rounds: list[dict], items: list[dict]) -> dict:
    used_def = sum(1 for it in items if it["su"] is not None and it["r"]["side"] == DEFENSE)
    used_atk = sum(1 for it in items if it["su"] is not None and it["r"]["side"] == ATTACK)
    full5 = sum(1 for it in items if it["su"] is not None and len(it["su"]) == 5)
    return {"total": len(rounds), "used": used_def + used_atk, "used_def": used_def, "used_atk": used_atk,
            "excluded": len(rounds) - used_def - used_atk, "full5": full5,
            "dates": sorted({str(r.get("date")) for r in rounds if r.get("date")}),
            "teams": sorted({str(r.get("tag") or r.get("team")) for r in rounds}),
            "wins": sum(1 for r in rounds if r.get("win"))}


# ============================================================== 型
def _pattern_suffix(medoid_it: dict) -> str:
    """名前がぶつかったときに付ける、内容で区別できる短い説明（ゾーン分布＋代表エリア）。"""
    zones = Counter(GM.zone_of(a) for _u, _v, a in medoid_it["su"].values() if GM.zone_of(a))
    z = "".join(f"{k}{v}" for k, v in sorted(zones.items()))
    areas = Counter(a for _u, _v, a in medoid_it["su"].values() if a)
    top = areas.most_common(1)
    area = top[0][0] if top else ""
    return " ".join(x for x in (z, area) if x)


def unique_name(base: str, suffix: str, used: set[str]) -> str:
    """同じ名前の別クラスタに、内容で区別できる名前を付ける（最後は連番で必ず一意にする）。"""
    if base not in used:
        return base
    cand = f"{base}（{suffix}）" if suffix else base
    i = 2
    while cand in used:
        cand = f"{base}（{suffix} {i}）" if suffix else f"{base}（{i}）"
        i += 1
    return cand


def build_patterns(items: list[dict], model, k: int, side: str,
                   used_names: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """側の型（配置クラスタ）。対象＝ピストルでない・1:40 に5人そろったラウンド。"""
    used_names = used_names if used_names is not None else set()
    pool = [it for it in items if it["r"]["side"] == side and not it["r"]["pistol"]
            and it["su"] is not None and len(it["su"]) == 5]
    points = [{a: MF.to_view(u, v, k) for a, (u, v, _ar) in it["su"].items()} for it in pool]
    pats_idx, singles_idx = leader_clusters(points)
    out = []
    for grp in pats_idx:
        members = [pool[i] for i in grp]
        # メドイド（勝ちを優先）
        winner = [i for i in grp if pool[i]["r"]["win"]]
        cand = winner or grp
        med = min(cand, key=lambda a: sum(round_distance(points[a], points[b]) for b in grp))
        med_it = pool[med]
        zones = Counter(GM.zone_of(a) for _u, _v, a in med_it["su"].values() if GM.zone_of(a))
        bu = Counter(it["cls"].get("buildup") for it in members if it["cls"].get("buildup"))
        action = glossary_word_side(bu.most_common(1)[0][0], side) if bu else None
        name = unique_name(pattern_name(zones, action, side), _pattern_suffix(med_it), used_names)
        used_names.add(name)
        out.append({"side": side, "name": name, "n": len(grp), "w": sum(it["r"]["win"] for it in members),
                    "members": members, "medoid": med_it, "zones": dict(zones)})
    out.sort(key=lambda p: (-p["n"], p["name"]))
    singles = []
    for i in singles_idx:
        it = pool[i]
        zones = Counter(GM.zone_of(a) for _u, _v, a in it["su"].values() if GM.zone_of(a))
        action = next((v for ph, v in it["cls"].items() if ph != "setup" and v), None)
        name = unique_name(pattern_name(zones, glossary_word_side(action, side), side),
                           _pattern_suffix(it), used_names)
        used_names.add(name)
        singles.append({"side": side, "name": name, "n": 1, "item": it})
    return out, singles


# ============================================================== 手札
def hand_nodes(items: list[dict], side: str) -> list[dict]:
    nodes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        if it["r"]["side"] != side or it["r"]["pistol"] or it["su"] is None:
            continue
        for ph, val in it["cls"].items():
            if ph == "setup" or not val:
                continue
            nodes[val][0] += 1
            nodes[val][1] += int(it["r"]["win"])
    out = []
    for nid, (n, w) in nodes.items():
        if n >= HAND_MIN_N and (w / n) >= HAND_MIN_WINRATE:
            out.append({"id": nid, "n": n, "w": w, "rate": w / n, "name": glossary_word_side(nid, side)})
    out.sort(key=lambda x: (-x["n"], x["name"]))
    return out[:5]


# ============================================================== なぜこの構成か
def _node_stats(items: list[dict]) -> dict[str, list[int]]:
    """局面ノードごとの (回数, 勝ち数)。ピストルと setup は数えない。"""
    nodes: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        if it["su"] is None or it["r"]["pistol"]:
            continue
        for ph, val in it["cls"].items():
            if ph == "setup" or not val:
                continue
            nodes[val][0] += 1
            nodes[val][1] += int(it["r"]["win"])
    return nodes


def _why_claims(comp_rounds: list[dict], items: list[dict], base_rounds: list[dict],
                base_items: list[dict], model) -> list[dict]:
    claims = []

    # 1. 設置サイトの偏り（15pt 以上）
    pa = [r for r in comp_rounds if r.get("plant_site") in ("A", "B")]
    ba = [r for r in base_rounds if r.get("plant_site") in ("A", "B")]
    if len(pa) >= 8 and len(ba) >= 20:
        sha = sum(1 for r in pa if r["plant_site"] == "A") / len(pa)
        bha = sum(1 for r in ba if r["plant_site"] == "A") / len(ba)
        if abs(sha - bha) >= 0.15:
            site = "A" if sha > bha else "B"
            share = sha if site == "A" else 1 - sha
            base_share = bha if site == "A" else 1 - bha
            ex = next((it for it in items if it["r"].get("plant_site") == site and it["su"]), None)
            claims.append({"text": f"{site} サイトへの設置が {share:.0%}（全体 {base_share:.0%}）。{site} を軸に組み立てる。",
                           "item": ex, "metric": "plant_site"})

    # 2. ファーストキル取得率（10pt 以上）
    fa = [r for r in comp_rounds if r.get("fk_own") is not None]
    fb = [r for r in base_rounds if r.get("fk_own") is not None]
    if len(fa) >= 8 and len(fb) >= 20:
        fr = sum(1 for r in fa if r["fk_own"]) / len(fa)
        br = sum(1 for r in fb if r["fk_own"]) / len(fb)
        if abs(fr - br) >= 0.10:
            ex = next((it for it in items if it["r"].get("fk_own") and it["su"]), None) or \
                 next((it for it in items if it["su"]), None)
            word = "取る" if fr >= br else "取られる"
            claims.append({"text": f"ファーストキルを {fr:.0%} のラウンドで{word}（全体 {br:.0%}）。序盤の当たりが持ち味。",
                           "item": ex, "metric": "fk"})

    # 3. 特定ノードの勝率（n>=5 かつ全体より 10pt 以上高い）
    nodes = _node_stats(items)
    base_nodes = _node_stats(base_items)
    best = None
    for nid, (n, w) in nodes.items():
        bn, bw = base_nodes.get(nid, (0, 0))
        if n < 5 or bn < 5 or has_abbrev(glossary_word(nid)):
            continue
        rate = w / n
        base_rate = bw / bn
        if rate - base_rate < 0.10:
            continue
        score = rate * n
        if best is None or score > best[0]:
            ex = next((it for it in items if it["cls"].get(_node_phase(it, nid)) == nid), None)
            best = (score, nid, n, w, rate, base_rate, ex)
    if best:
        _sc, nid, n, w, rate, base_rate, ex = best
        claims.append({"text": f"{glossary_word(nid)} の勝率 {rate:.0%}（全体 {base_rate:.0%}、{_wl(w, n)}）。はまると強い。",
                       "item": ex, "metric": "node"})

    # 4. 守りのバイタル占有（15pt 以上）
    vit = model.vitals()
    if vit:
        da = [it for it in items if it["r"]["side"] == DEFENSE and it["su"]]
        oc = 0
        for it in da:
            if any(a in vit for _u, _v, a in it["su"].values()):
                oc += 1
        if len(da) >= 8:
            orate = oc / len(da)
            base_oc = _base_vital_rate(base_rounds, model, vit)
            if base_oc is not None and abs(orate - base_oc) >= 0.15:
                ex = next((it for it in da if any(a in vit for _u, _v, a in it["su"].values())), None)
                claims.append({"text": f"守りで 1:40 にバイタル（{'・'.join(vit)}）へ人がいる割合 {orate:.0%}（全体 {base_oc:.0%}）。",
                               "item": ex, "metric": "vital"})
    # 重複 metric を除いて最大 3
    seen, out = set(), []
    for c in claims:
        if c["metric"] in seen or not c.get("item"):
            continue
        seen.add(c["metric"])
        out.append(c)
    return out[:3]


def _node_phase(it: dict, nid: str) -> str:
    for ph, val in it["cls"].items():
        if val == nid:
            return ph
    return ""


def _base_vital_rate(base_rounds: list[dict], model, vit: list[str]):
    """全体（同マップ全構成）の守り 1:40 バイタル占有。"""
    vals = []
    for r in base_rounds:
        if r["side"] != DEFENSE:
            continue
        su = GM.setup_at_140(r, model)
        if not su:
            continue
        vals.append(1 if any(a in vit for _u, _v, a in su.values()) else 0)
    return sum(vals) / len(vals) if vals else None


# ============================================================== スライド
def _slide(eyebrow: str, title: str, body: str, foot: str = "", h1: bool = False, cls: str = "") -> str:
    h = f"<h1>{_e(title)}</h1>" if h1 else f"<h2>{_e(title)}</h2>"
    klass = f"slide {cls}".strip()
    return (f'<section class="{klass}"><div class="eyebrow">{_e(eyebrow)}</div>{h}{body}'
            f'<div class="foot"><span>{_e(foot)}</span><span>vreplay プレイブック v5</span></div></section>')


def _cover_slide(ctx, comp, stat) -> str:
    agents = agents_of(comp)
    teams = "・".join(stat["teams"])
    period = f"{stat['dates'][0]} 〜 {stat['dates'][-1]}" if stat["dates"] else "-"
    body = (_s(f"使うチーム：{teams}") +
            _s(f"期間：{period}（{ENV_LABEL}）") +
            _s(f"{ctx['map']} {stat['total']} チームラウンド・1:40 の配置が取れたのは {stat['used']}。"))
    return _slide(f"{ctx['map']} / {ENV_LABEL}", "・".join(agents), body, ctx["map"], h1=True)


def _env_slide(ctx, comp, comp_rows) -> str:
    bars = []
    mx = max((s["total"] for _, s in comp_rows), default=1) or 1
    for key, s in comp_rows:
        w = int(round(100 * s["total"] / mx))
        wr = (s["wins"] / s["total"]) if s["total"] else 0
        cur = " cur" if key == comp else ""
        bars.append(f'<div class="barrow{cur}"><span class="bl">{"・".join(agents_of(key))}</span>'
                    f'<span class="bar" style="width:{w}%"></span>'
                    f'<span class="bv">{s["total"]}R・{wr:.0%}</span></div>')
    svg = _bump_svg(ctx["fig"].svg(vital=ctx["vitals"], labels=list(ctx["vitals"]),
                                   alt=f"{ctx['map']} のバイタル（{'・'.join(ctx['vitals'])}）"))
    body = (f'<div class="barchart">{"".join(bars)}</div>'
            f'<div class="vitwrap">{svg}</div>' +
            _s(f"{ENV_LABEL} の {ctx['map']} は {len(comp_rows)} 構成。採用が多い順に並べている。"))
    return _slide("環境とマップ", f"{ENV_LABEL} の構成とバイタル", body, ctx["map"])


def _why_slide(ctx, claims) -> str:
    if not claims:
        return _slide("なぜこの構成か", "なぜこの構成か",
                      _s("他の構成とはっきりした差は出ていない。"), ctx["map"])
    cols = []
    for c in claims:
        fig = _fig_svg(ctx["fig"], c["item"], ctx["model"], alt="根拠の見本",
                       marks=("1:40", "1:30", "1:20"), legend=False, px=320)
        cols.append(f'<div class="claim">{fig}{_vod_for(c["item"])}{_s(c["text"])}</div>')
    return _slide("なぜこの構成か", "なぜこの構成か", f'<div class="claims">{"".join(cols)}</div>', ctx["map"])


def _hand_slide(ctx, side_label, item, motion, hand, reason: str = "") -> str:
    eyebrow = f"{side_label} / 手札（{hand['n']} 回・{hand['rate']:.0%}）"
    if not item:
        body = _s(reason or f"{hand['name']}：1:40 に5人そろった見本ラウンドがありません。動く図は出せません。")
        return _slide(eyebrow, hand["name"], body, ctx["map"])
    eyebrow += f" 見本 {sample_label(item['r'])}"
    lines = _agent_lines(item, ctx["model"])
    body = (f'<div class="handfig"><div>{motion}{_vod_for(item)}</div>'
            f'<div class="handtxt">{_ul(lines)}</div></div>')
    return _slide(eyebrow, hand["name"], body, ctx["map"])


def _side_reason_slide(ctx, side, kind, reason) -> str:
    label = "守り" if side == DEFENSE else "攻め"
    return _slide(f"{label} / {kind}", f"{label}の{kind}", _s(reason), ctx["map"])


def side_avg_winrate(items: list[dict], side: str):
    """その側の平均勝率（型と同じ母数＝ピストルを除く・1:40 に5人そろったラウンド）。"""
    pool = [it for it in items if it["r"]["side"] == side and not it["r"]["pistol"] and is_full5(it)]
    if not pool:
        return None
    return sum(1 for it in pool if it["r"]["win"]) / len(pool)


def choose_placement(patterns: list[dict], avg):
    """基本配置 = 回数の多い順で、側の平均勝率以上の最初の型。下回る型は「避けるべき型」に回す。"""
    chosen, avoid = None, []
    for p in patterns:
        rate = (p["w"] / p["n"]) if p["n"] else 0.0
        if avg is None or rate >= avg:
            if chosen is None:
                chosen = p
        else:
            avoid.append(p)
    return chosen, avoid


def _placement_slide(ctx, side, patterns, items, used) -> tuple[str, dict]:
    label = "守り" if side == DEFENSE else "攻め"
    avg = side_avg_winrate(items, side)
    chosen, avoid = choose_placement(patterns, avg)
    pool_n = sum(1 for it in items if it["r"]["side"] == side and not it["r"]["pistol"] and is_full5(it))
    item, motion, reason = None, "", ""
    reason_kind = ""
    if chosen:
        item, motion = _pick_motion(chosen["members"], used, ctx["model"], ctx["k"], ctx["walk"], ctx["bg"],
                                    f"{label}の基本配置", allow_used=True)
        if item:
            used.add(sample_key(item))
    texts = []
    if chosen and motion:
        reason_kind = "rendered"
        rate = chosen["w"] / chosen["n"] if chosen["n"] else 0.0
        zones = "・".join(f"{z}{n}枚" for z, n in sorted(chosen["zones"].items(), key=lambda kv: -kv[1]))
        texts.append(_s(f"基本配置：{chosen['name']}（{_wl(chosen['w'], chosen['n'])}・勝率 {rate:.0%}、"
                        f"この側の平均 {avg:.0%}）。"))
        texts.append(_s(f"1:40 の人数は {zones or '-'}。1:30・1:20 の動きを同じ駒でつないでいる。"))
    elif chosen:
        reason_kind = "no_motion"
        reason = (f"{label}の基本配置は {chosen['name']}（{_wl(chosen['w'], chosen['n'])}）だが、"
                  f"動く図にできる見本ラウンドがない。")
        texts.append(_s(reason))
    elif not patterns and pool_n < PATTERN_MIN:
        reason_kind = "no_rounds"
        reason = (f"{label}の基本配置は出せません（1:40 に5人そろった非ピストルラウンドが {pool_n} 件で、"
                  f"型の成立に必要な {PATTERN_MIN} 件に届きません）。")
        texts.append(_s(reason))
    elif not patterns:
        reason_kind = "distance"
        reason = (f"{label}の基本配置は出せません（1:40 に5人そろった非ピストルラウンドは {pool_n} 件あるが、"
                  f"同じ形とみなせるまとまりが {PATTERN_MIN} 件に届きません）。")
        texts.append(_s(reason))
    else:
        reason_kind = "winrate"
        reason = f"{label}の基本配置は出せません（この側の平均勝率 {avg:.0%} を上回る型がない）。"
        texts.append(_s(reason))
    if avoid:
        a = "、".join(f"{p['name']}（{_wl(p['w'], p['n'])}）" for p in avoid)
        texts.append(_s(f"避けるべき型：{a}。"))
    if motion:
        body = f'<div class="handfig">{motion}<div class="handtxt">{"".join(texts)}</div></div>'
    else:
        body = f'<div class="handtxt">{"".join(texts)}</div>'
    meta = {"side": side, "name": chosen["name"] if chosen else None,
            "n": chosen["n"] if chosen else 0, "w": chosen["w"] if chosen else 0,
            "rate": round(chosen["w"] / chosen["n"], 4) if chosen and chosen["n"] else None,
            "avg": round(avg, 4) if avg is not None else None,
            "meets_winrate": bool(chosen) and (avg is None or chosen["w"] / chosen["n"] >= avg),
            "pool_n": pool_n, "reason_kind": reason_kind,
            "avoid": [{"name": p["name"], "n": p["n"], "w": p["w"]} for p in avoid],
            "rendered": bool(motion), "sample_round": sample_key(item) if item else None, "reason": reason}
    return _slide(f"{label} / 配置", f"{label}の基本配置", body, ctx["map"]), meta


def _pattern_slide(ctx, def_pats, atk_pats) -> str:
    rows = []
    for p in def_pats + atk_pats:
        side = "守り" if p["side"] == DEFENSE else "攻め"
        link = ""
        yt = p["medoid"]["r"].get("yt")
        if yt:
            link = f'<a href="{html.escape(str(yt), quote=True)}" target="_blank" rel="noopener">VOD</a>'
        rows.append(f"<tr><td>{side}</td><td>{_e(p['name'])}</td><td>{p['n']}</td>"
                    f"<td>{_wl(p['w'], p['n'])}</td><td>{link}</td></tr>")
    table = ('<table><thead><tr><th>側</th><th>型名</th><th>回数</th><th>勝敗</th><th>見本</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')
    return _slide("配置 / 型の一覧", "型の一覧", table, ctx["map"])


def _role_table(ctx, comp, items, side) -> str:
    side_items = [it for it in items if it["r"]["side"] == side]
    rows = []
    for ag in agents_of(comp):
        a0 = Counter(GM.zone_of(a) for it in side_items if it["su"] and ag in it["su"]
                     for _u, _v, a in [it["su"][ag]] if GM.zone_of(a))
        a1 = Counter()
        for it in side_items:
            sn = GM.snap(it["r"], ctx["model"], 20, tol=8)
            if ag in sn and GM.zone_of(sn[ag][2]):
                a1[GM.zone_of(sn[ag][2])] += 1
        start = a0.most_common(1)[0][0] if a0 else "-"
        mid = a1.most_common(1)[0][0] if a1 else "-"
        role = ROLE_JA.get(ctx["kb"].role(ag) or "", ctx["kb"].role(ag) or "-")
        rows.append(f"<tr><td>{_e(ag)}</td><td>{_e(start)}</td><td>{_e(mid)}</td><td>{_e(role)}</td></tr>")
    return ('<table><thead><tr><th>エージェント</th><th>序盤の持ち場</th><th>中盤の動き</th><th>役割</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _role_slide(ctx, comp, items) -> str:
    body = (f'<div class="rolewrap"><div><h3>守り</h3>{_role_table(ctx, comp, items, DEFENSE)}</div>'
            f'<div><h3>攻め</h3>{_role_table(ctx, comp, items, ATTACK)}</div></div>')
    return _slide("真似る手順", "役割表", body, ctx["map"])


def _appendix_slide(ctx, stat, singles, unknown, def_pats, atk_pats) -> str:
    tables = []
    tables.append('<table><thead><tr><th>数合わせ</th><th>値</th></tr></thead><tbody>'
                  f'<tr><td>総数</td><td>{stat["total"]}</td></tr>'
                  f'<tr><td>守り（使用）</td><td>{stat["used_def"]}</td></tr>'
                  f'<tr><td>攻め（使用）</td><td>{stat["used_atk"]}</td></tr>'
                  f'<tr><td>除外（1:40 が取れない）</td><td>{stat["excluded"]}</td></tr>'
                  f'<tr><td>ピストル</td><td>{stat.get("pistol", 0)}</td></tr></tbody></table>')
    tables.append('<table><thead><tr><th>設定値</th><th>値</th><th>意味</th></tr></thead><tbody>'
                  + "".join(f'<tr><td>{_e(k)}</td><td>{_e(v)}</td><td>{_e(d)}</td></tr>'
                            for k, v, d in SETTINGS) + "</tbody></table>")
    tables.append('<table><thead><tr><th>内部の呼び方</th><th>この冊子での言い方</th></tr></thead><tbody>'
                  + "".join(f'<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>' for k, v in GLOSSARY.items())
                  + "</tbody></table>")
    single_txt = "、".join(f"{s['name']}（{s['n']}）" for s in singles) or "なし"
    return _slide("付録", "付録",
                  _s(f"単発の型（1〜{PATTERN_MIN - 1} 回）：{single_txt}") +
                  _s(f"構成不明で外したラウンド（{ctx['map']} 全体）：{unknown}") +
                  f'<div class="appendix">{"".join(tables)}</div>', ctx["map"], cls="appendix")


def build_page(ctx, comp: str, rounds: list[dict], items: list[dict], stat: dict, comp_rows) -> tuple[str, dict]:
    ctx = {**ctx, "map": ctx["map"]}
    used_names: set[str] = set()
    def_pats, def_singles = build_patterns(items, ctx["model"], ctx["k"], DEFENSE, used_names)
    atk_pats, atk_singles = build_patterns(items, ctx["model"], ctx["k"], ATTACK, used_names)
    claims = _why_claims(rounds, items, ctx["base_rounds"], ctx["base_items"], ctx["model"])

    slides = [_cover_slide(ctx, comp, stat), _env_slide(ctx, comp, comp_rows), _why_slide(ctx, claims)]

    sample_keys = set()
    used_samples: set[str] = set()
    hands_meta: list[dict] = []
    # 手札（守り・攻めの両側。見本は full5 から選び、手札どうしで重ならないようにする）
    for side, label in ((DEFENSE, "守り"), (ATTACK, "攻め")):
        hands = hand_nodes(items, side)
        if not hands:
            side_n = sum(1 for it in items if it["r"]["side"] == side and not it["r"]["pistol"]
                         and it["su"] is not None)
            slides.append(_side_reason_slide(
                ctx, side, "手札",
                f"{label}の手札はありません（対象 {side_n} ラウンド。{HAND_MIN_N} 回以上かつ勝率 "
                f"{HAND_MIN_WINRATE:.0%} 以上の動きが出ませんでした）。"))
            continue
        for h in hands:
            src = [it for it in items if it["r"]["side"] == side and not it["r"]["pistol"]
                   and h["id"] in it["cls"].values() and is_full5(it)]
            item, motion = _pick_motion(src, used_samples, ctx["model"], ctx["k"], ctx["walk"], ctx["bg"],
                                        h["name"])
            if item:
                used_samples.add(sample_key(item))
                sample_keys.add(item["r"]["comp_key"])
                status, reason = "ok", ""
            elif not src:
                status = "no_full5"
                reason = f"{h['name']}：1:40 に5人そろった見本ラウンドがありません。動く図は出せません。"
            else:
                status = "no_motion"
                reason = (f"{h['name']}：5人そろった見本ラウンドは {len(src)} 件ありますが、"
                          f"動く図にできるラウンドがありませんでした。")
            slides.append(_hand_slide(ctx, label, item, motion, h, reason))
            r = item["r"] if item else None
            hands_meta.append({
                "side": side, "id": h["id"], "name": h["name"], "n": h["n"], "w": h["w"],
                "rate": round(h["rate"], 4), "sample_round": sample_key(item) if item else None,
                "sample_full5": bool(item), "sample_status": status,
                "sample_team": (r.get("tag") or r.get("team")) if r else None,
                "sample_opp": r.get("opp_tag") if r else None,
                "sample_label": sample_label(r) if r else "",
            })

    placement_meta: list[dict] = []
    for side, pats in ((DEFENSE, def_pats), (ATTACK, atk_pats)):
        html, meta = _placement_slide(ctx, side, pats, items, used_samples)
        slides.append(html)
        placement_meta.append(meta)
    slides.append(_pattern_slide(ctx, def_pats, atk_pats))
    slides.append(_role_slide(ctx, comp, items))
    slides.append(_appendix_slide(ctx, stat, def_singles + atk_singles, ctx["unknown"], def_pats, atk_pats))

    # 型の見本ラウンドの構成も照合対象に入れる
    for p in def_pats + atk_pats:
        sample_keys.add(p["medoid"]["r"]["comp_key"])

    body = "".join(s for s in slides if s)
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e("・".join(agents_of(comp)))} {_e(ctx["map"])} プレイブック v5</title>'
            f'<style>{ctx["css"]}</style></head><body>{ctx["fig"].defs()}'
            f'<main class="deck"><div class="toolbar"><p>{_e("・".join(agents_of(comp)))} の {_e(ctx["map"])}'
            f'</p><p>印刷すると 1 スライド 1 ページ</p></div>{body}</main>{ctx["js"]}</body></html>')
    manifest = {"key": comp, "agents": agents_of(comp), "total": stat["total"], "used_def": stat["used_def"],
                "used_atk": stat["used_atk"], "excluded": stat["excluded"], "full5": stat["full5"],
                "dates": stat["dates"], "teams": stat["teams"],
                "patterns": [{"side": p["side"], "name": p["name"], "n": p["n"], "w": p["w"]}
                             for p in def_pats + atk_pats],
                "hands": hands_meta,
                "placements": placement_meta,
                "sample_comp_keys": sorted(k for k in sample_keys if k)}
    return _bump_html(page), manifest


# ============================================================== index
def _index_page(ctx, pages: list[dict], skipped: list[dict], old: list[dict]) -> str:
    rows = []
    for p in sorted(pages, key=lambda x: -x["total"]):
        wr = p["wins"] / p["total"] if p["total"] else 0
        rows.append(f'<tr><td>{"・".join(p["agents"])}</td><td>{_e("・".join(p["teams"]))}</td><td>{p["total"]}</td>'
                    f'<td>{wr:.0%}</td><td>{p["dates"][-1] if p["dates"] else "-"}</td>'
                    f'<td><a href="{html.escape(p["file"], quote=True)}">ページ</a></td></tr>')
    main_table = ('<table><thead><tr><th>5人構成</th><th>チーム</th><th>ラウンド</th><th>勝率</th><th>直近</th>'
                  f'<th>リンク</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    sk = []
    for s in sorted(skipped, key=lambda x: -x["total"]):
        sk.append(f'<tr><td>{"・".join(s["agents"])}</td><td>{s["total"]}</td><td>{s["full5"]}</td>'
                  f'<td>{_e(s["reason"])}</td></tr>')
    sk_table = ('<table><thead><tr><th>5人構成</th><th>ラウンド</th><th>5人そろった配置</th><th>見送りの理由</th>'
                f'</tr></thead><tbody>{"".join(sk)}</tbody></table>') if sk else ""
    old_rows = []
    for o in sorted(old, key=lambda x: -x["total"]):
        old_rows.append(f'<tr><td>{"・".join(o["agents"])}</td><td>{o["total"]}</td>'
                        f'<td>{_e("・".join(o["teams"]))}</td></tr>')
    old_table = ('<table><thead><tr><th>5人構成</th><th>ラウンド</th><th>チーム</th></tr></thead><tbody>'
                 f'{"".join(old_rows)}</tbody></table>') if old_rows else ""
    lead = f'まずは採用ラウンド数が最も多い {("・".join(pages[0]["agents"]))} から見る。' if pages else \
        "ページを作れる構成がまだない。"
    body = (f'<p class="s">{_e(lead)}</p>'
            f'<h2>{_e(ENV_LABEL)} の構成（{ENV_START} 以降）</h2>{main_table}'
            f'<h2>ページを見送った構成</h2>{sk_table}'
            f'<h2>旧環境（{ENV_START} より前）</h2>{old_table}')
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(ctx["map"])} プレイブック v5</title><style>{ctx["css"]}</style></head>'
            f'<body><main class="deck"><div class="toolbar"><p>{_e(ctx["map"])} プレイブック v5 の入口</p>'
            f'<p>{_e(ENV_LABEL)}</p></div>{body}</main></body></html>')
    return _bump_html(page)


# ============================================================== 生成
def _seed_cache(cfg: dict, out: Path, log=print) -> None:
    """生成先の _work/_cache を、見つかった既存キャッシュで自己完結させる。"""
    dst = Path(out) / "_work" / "_cache"
    root = Path(__file__).resolve().parent.parent
    srcs = []
    env = os.environ.get("VREPLAY_OUTPUT_DIR")
    if env:
        srcs.append(Path(env) / "_cache")
    base = (cfg.get("output") or {}).get("dir")
    if base:
        srcs.append(Path(base) / "_cache")
    srcs += [root.parent / "output_work" / "_cache", root.parent / "output" / "_cache",
             root / "output_work" / "_cache", root / "output" / "_cache"]
    for s in srcs:
        try:
            if s.exists() and s.is_dir() and s.resolve() != dst.resolve():
                shutil.copytree(s, dst, dirs_exist_ok=True)
                log(f"  キャッシュを {s} から {dst} に写しました")
                return
        except Exception as e:                                   # noqa: BLE001
            log(f"  キャッシュの写しでエラー（{e}）")
    log("  既存キャッシュが見つかりませんでした（マップ画像の取得を試みます）")


def _motion_js() -> str:
    """deck の MOTION_JS を複製し、reduced-motion 対応を足す。三項演算子は検証の「?」禁止に合わせて書き換える。"""
    js = D.MOTION_JS
    js = js.replace("let t=0, playing=true;",
                    "var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;"
                    "let t=0, playing=!reduce;")
    js = js.replace("var t=L[k]>0?Math.min(1,r/L[k]):0;",
                    "var t=0;if(L[k]>0){t=Math.min(1,r/L[k]);}")
    js = js.replace("e.target.textContent=playing?'一時停止':'再生';",
                    "if(playing){e.target.textContent='一時停止';}else{e.target.textContent='再生';}")
    js = js.replace("el.querySelector('.mo-play').addEventListener('click',function(e){playing=!playing;",
                    "var pb=el.querySelector('.mo-play');"
                    "if(playing){pb.textContent='一時停止';}else{pb.textContent='再生';}"
                    "pb.addEventListener('click',function(e){playing=!playing;")
    return js


def _css() -> str:
    v5 = """
.deck{max-width:1180px}
.slide{aspect-ratio:auto;min-height:200px}
.barchart{margin:8px 0}
.barrow{display:grid;grid-template-columns:220px 1fr 140px;align-items:center;gap:8px;margin:4px 0}
.barrow .bl{font-size:14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.barrow .bar{display:block;height:14px;background:var(--key-weak);border:1px solid var(--key);border-radius:3px}
.barrow.cur .bar{background:var(--key)}
.barrow .bv{font-size:14px;color:var(--text-sub)}
.vitwrap{margin-top:8px}
.claims{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;align-items:start}
.claim{min-width:0}
.claim .s{font-size:14px;line-height:1.6;margin:6px 0 0}
.vod{margin:6px 0 0;border:1px solid var(--line-weak);border-radius:var(--radius);overflow:hidden;background:var(--bg-sub)}
.vod img{display:block;width:100%;height:auto}
.vod figcaption{font-size:14px;padding:3px 8px}
.handfig{display:grid;grid-template-columns:minmax(300px,440px) minmax(240px,1fr);gap:18px;align-items:start}
.mo-map{width:min(100%,440px)}
.claim .mo-map{width:min(100%,360px)}
.handtxt ul{margin:0;padding-left:1.2em}
.handtxt li{font-size:14px;line-height:1.7;margin:0 0 4px}
.handtxt .s{margin:0 0 6px}
.rolewrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.appendix{display:grid;gap:10px}
.appendix table{font-size:14px}
figure{margin:0}
@media (max-width:900px){.handfig,.rolewrap{grid-template-columns:1fr}}
"""
    return _bump_css(D.CSS + MF.CSS + v5)


def generate(reports: list[dict], map_name: str, out, cfg: dict, log=print) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _seed_cache(cfg, out, log)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("output", {})
    cfg["output"]["dir"] = str(work)

    ctx0 = _prepare(reports, map_name, cfg, log)
    allr = ctx0["allr"]
    by_comp, unknown = group_by_comp(allr)
    env_rounds = [r for r in allr if str(r.get("date") or "") >= ENV_START]
    old_rounds = [r for r in allr if str(r.get("date") or "") < ENV_START]
    env_by, _ = group_by_comp(env_rounds)
    old_by, _ = group_by_comp(old_rounds)
    base_rounds = env_rounds

    env_stats = {k: comp_stats(v, _items_for(v, allr, ctx0["model"])) for k, v in env_by.items()}
    comp_rows = sorted(env_stats.items(), key=lambda kv: -kv[1]["total"])

    base_items = _items_for(base_rounds, allr, ctx0["model"])
    ctx = {**ctx0, "map": map_name, "base_rounds": base_rounds, "base_items": base_items, "unknown": unknown,
           "css": _css(), "js": _motion_js()}

    pages, skipped, page_meta = [], [], []
    for comp in sorted(env_by, key=lambda k: -len(env_by[k])):
        rounds = env_by[comp]
        items = _items_for(rounds, allr, ctx0["model"])
        stat = comp_stats(rounds, items)
        stat["pistol"] = sum(1 for r in rounds if r.get("pistol"))
        ok, reason = gate_ok(stat)
        if not ok:
            skipped.append({"key": comp, "agents": agents_of(comp), "total": stat["total"],
                            "full5": stat["full5"], "reason": reason})
            continue
        html_page, manifest = build_page(ctx, comp, rounds, items, stat, comp_rows)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", comp).strip("-").lower() or "comp"
        fn = f"comp_{slug}.html"
        (out / fn).write_text(html_page, encoding="utf-8")
        manifest["file"] = fn
        manifest["wins"] = stat["wins"]
        page_meta.append(manifest)
        pages.append({"agents": agents_of(comp), "teams": stat["teams"], "total": stat["total"],
                      "wins": stat["wins"], "dates": stat["dates"], "file": fn})
        log(f"  ページを作成: {fn}（{stat['total']} ラウンド）")

    old_meta = [{"key": k, "agents": agents_of(k), "total": len(v),
                 "teams": sorted({str(r.get("tag") or r.get("team")) for r in v})} for k, v in old_by.items()]
    (out / "index.html").write_text(_index_page(ctx, pages, skipped, old_meta), encoding="utf-8")

    manifest = {"map": map_name, "env_start": ENV_START, "env_label": ENV_LABEL,
                "settings": [{"name": k, "value": v, "note": d} for k, v, d in SETTINGS],
                "unknown_rounds": unknown, "pages": page_meta, "skipped": skipped, "old_env": old_meta}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"プレイブック v5: {out / 'index.html'}")
    return out / "index.html"
