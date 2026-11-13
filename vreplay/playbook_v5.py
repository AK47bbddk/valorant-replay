"""プレイブック v5：マップ × 5人構成のページ。図（3枚TactiValボード・実プレイGIF・VODカード）を主役にする。

v4（`playbook_v4.py`）はチーム単位で集計を文章で並べていた。v5 は単位を「マップ × 5人構成」に変え、
構成ごとに 1 ページを作る。段階Cでは、基本配置・役割表を残したうえで、構成のラウンドを
攻撃／防衛のサイド別に分割し、各サイドで「序盤」「中盤」「終盤」の3部構成（上位3型まで）を作る。
図は1枚1時刻の TactiVal ボード3枚（序盤・中盤・終盤）をサイド別に用意し、文面は「アクション→リアクション」で組み立てる。
"""
from __future__ import annotations

import copy
import hashlib
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

from . import board_img as BI
from . import deck as D
from . import gamemodel as GM
from . import kb as K
from . import mapfig as MF
from . import phases as P
from . import tactics as TC
from . import tactival as TV

ATTACK, DEFENSE = GM.ATTACK, GM.DEFENSE
SEGS = (("early", "序盤"), ("mid", "中盤"), ("late", "終盤"))
SIDE_SEGS = {
    ATTACK: (("early", "開幕"), ("entry", "エントリー"), ("postplant", "ポストプラント")),
    DEFENSE: (("early", "序盤"), ("mid", "中盤"), ("late", "終盤")),
}

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
    ("PATTERN_MIN", PATTERN_MIN, "基本配置（配置クラスタ）の成立ラウンド数"),
    ("PATTERN_DIST", PATTERN_DIST, "基本配置クラスタの距離しきい値（表示座標 0..1）"),
    ("SAMPLE_SECONDS", P.SAMPLE_SECONDS, "軌跡サンプルの間隔（秒）"),
    ("RUSH_SECONDS", P.RUSH_SECONDS, "開幕からこの秒以内のサイトエントリーをラッシュと判定"),
    ("FREEZE_SECONDS", P.FREEZE_SECONDS, "ピック後に動きが無いままこの秒数以上でフリーズと判定"),
]

PHASE_RULES = [
    ("攻撃・開幕", "開始〜最初のピックまたは最初の支配交代の早い方", "Aデフォルト・MIDデフォルト・Bデフォルト・ラッシュ・詰め待ち"),
    ("攻撃・エントリー", "サイトへの進入ウィンドウ", "Aセット・Bセット・Aスプリット・Bスプリット"),
    ("攻撃・ポストプラント", "プラント後5秒以降の最初の配置（サイト別上位2種）", "コールアウト構成による配置クラスタ"),
    ("防衛・序盤", "開始〜最初のピックまたは最初の支配交代の早い方", "A寄せ・MID寄せ・B寄せ・自陣待機"),
    ("防衛・中盤", "序盤の終わり〜プラント", "上がり・ローテート・取り返し"),
    ("防衛・終盤", "プラント〜ラウンド終了", "リテイク・寄せ・寄せ／リテイク"),
]

# 旧ノード表（後方互換の単体テスト用に残すが、生成ページの文面組み立てには使わない）
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

# 守り側で使う動作語（後方互換テスト用）
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

# 生成ページに出してはいけない旧ノード表の文言（段階Bの検証対象）
OLD_NODE_PHRASES = tuple(dict.fromkeys(
    list(GLOSSARY.values()) + list(DEF_ACTION_WORD.values()) + ["はまると強い"]
))


def glossary_word(nid: str | None) -> str:
    """内部ノード ID → プレイヤーの言葉（後方互換テスト用）。生成ページの本文には使わない。"""
    if not nid:
        return ""
    return GLOSSARY.get(str(nid), str(nid))


def glossary_word_side(nid: str | None, side: str) -> str:
    """側に合わせた動作語（後方互換テスト用）。守りでは攻め専用の言い回しを使わない。"""
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


def _ul(lines: list[str], cls: str = "") -> str:
    if not lines:
        return ""
    attr = f' class="{cls}"' if cls else ""
    return f"<ul{attr}>{''.join(f'<li>{_e(x)}</li>' for x in lines)}</ul>"


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


def _slug(s: str) -> str:
    h = hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:8]
    ascii_part = re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()
    return f"{ascii_part}-{h}" if ascii_part else h


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


# ============================================================== 基本配置（1:40 配置クラスタ）
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
    """1:40 のゾーン分布 + 側に合った動きの名前を組み立てる。"""
    default_action = "展開する" if side == ATTACK else "エリアを押さえる"
    return _zone_head(zones, side) + (action or default_action)


def _placement_action(members: list[dict], side: str) -> str:
    """基本配置のクラスタ名をフェーズの流れから決める（旧ノード表の文言は使わない）。"""
    if side == DEFENSE:
        def_evs = Counter()
        for it in members:
            fl = it.get("flow") or {}
            early_end = float(((fl.get("segments") or {}).get("early") or {}).get("end") or 20.0)
            for ev in fl.get("events") or []:
                if ev.get("side") == DEFENSE and float(ev.get("t") or 0.0) <= early_end + 1e-9:
                    if ev.get("kind") in ("プッシュ", "ローテート", "取り返し"):
                        def_evs[ev["kind"]] += 1
        top = def_evs.most_common(1)[0][0] if def_evs else None
        if top == "プッシュ":
            return "前目で情報を取る"
        if top == "ローテート":
            return "寄せて守る"
        return "エリアを押さえる"
    early_types = Counter(
        ((it.get("flow") or {}).get("phases") or {}).get("early", {}).get("type")
        for it in members
        if ((it.get("flow") or {}).get("phases") or {}).get("early", {}).get("type")
    )
    if early_types:
        et = early_types.most_common(1)[0][0]
        return f"{et}へ展開する"
    return "展開する"


# ============================================================== VOD カード・実プレイ GIF
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
    r = item.get("r") if "r" in item else item
    return vod_card(r.get("yt"), sample_label(r))


# ============================================================== フェーズ分類と時刻・3枚図
def _round_dict_of(r: dict) -> dict:
    """phases.classify_flow に渡す report.json のラウンド要素を取り出す。"""
    if not isinstance(r, dict):
        return {}
    if isinstance(r.get("_round"), dict):
        return r["_round"]
    if isinstance(r.get("tactics"), dict):
        return r
    return {}


def classify_round_flow(r: dict, ranks: dict | None = None) -> tuple[dict | None, str | None]:
    """1 ラウンド記録 r を phases.classify_flow で分類し、(結果, スキップ理由) を返す。"""
    rr = _round_dict_of(r)
    reason = P.skip_reason(rr)
    if reason:
        return None, reason
    use_ranks = ranks if ranks is not None else (r.get("ranks") if isinstance(r, dict) else None)
    side = (r.get("side") if isinstance(r, dict) else None) or ATTACK
    res = P.classify_flow(rr, use_ranks, side=side)
    if res is None:
        return None, "no_tracks"
    return res, None


def _stable_early_time(frames: list[dict], start: float, end: float) -> float:
    """序盤区間の代表サンプル（開幕後最初の安定サンプル）の経過秒を選ぶ。"""
    in_seg = [f for f in frames if start - 1e-9 <= float(f["e"]) <= end + 1e-9]
    if not in_seg:
        return round(min(5.0, max(0.0, end)), 1)
    after_zero = [f for f in in_seg if float(f["e"]) > 1e-9]
    cand = after_zero or in_seg
    for idx, f in enumerate(cand):
        has_zone = any(sum(f["zones"][t].get(z, 0) for z in P.ZONES) > 0 for t in (1, 2))
        if not has_zone:
            continue
        if idx + 1 < len(cand) and cand[idx]["zones"] == cand[idx + 1]["zones"]:
            return round(float(f["e"]), 1)
    for f in cand:
        if any(sum(f["zones"][t].get(z, 0) for z in P.ZONES) > 0 for t in (1, 2)):
            return round(float(f["e"]), 1)
    return round(float(cand[0]["e"]), 1)


def phase_board_times(round_or_item: dict, flow: dict | None = None) -> list[tuple[str, str, float]]:
    """代表ラウンドの区間ごとの配置図時刻 [(seg_key, seg_label, elapsed_sec), ...] を返す。

    時刻の既定:
    - 序盤: 区間の代表サンプル（開幕後最初の安定サンプル）
    - 中盤: 区間中央（に最も近いサンプル）
    - 終盤: プラント直後（プラント以降の最初のサンプル）
    区間が無いラウンド（プラント無し等）はある区間だけ出す。
    """
    if isinstance(round_or_item, dict) and "r" in round_or_item:
        r_obj = round_or_item["r"]
        rr = _round_dict_of(r_obj)
        flow = flow or round_or_item.get("flow")
    else:
        r_obj = round_or_item if isinstance(round_or_item, dict) else {}
        rr = _round_dict_of(r_obj)
    if flow is None and rr:
        flow = P.classify_flow(rr, r_obj.get("ranks") if isinstance(r_obj, dict) else None)
    norm = P._norm_tracks(P.tracks_of(rr)) if rr else {}
    frames = P._frames(norm) if norm else []
    if not flow:
        pe = (r_obj or {}).get("plant_e") if isinstance(r_obj, dict) else None
        if pe is None and rr:
            pe = P._plant_elapsed(rr)
        out = [("early", "序盤", 5.0), ("mid", "中盤", 20.0)]
        if pe is not None:
            out.append(("late", "終盤", round(float(pe), 1)))
        return out

    segs = flow["segments"]
    out: list[tuple[str, str, float]] = []
    # 1. 序盤: 開幕後最初の安定サンプル
    e_start = float(segs["early"]["start"])
    e_end = float(segs["early"]["end"])
    out.append(("early", "序盤", _stable_early_time(frames, e_start, e_end)))

    # 2. 中盤: 区間中央（区間がある場合のみ）
    m_start = segs["mid"].get("start")
    m_end = segs["mid"].get("end")
    if m_start is not None and m_end is not None and float(m_end) > float(m_start) + 1e-9:
        center = (float(m_start) + float(m_end)) / 2.0
        in_mid = [f for f in frames if float(m_start) - 1e-9 <= float(f["e"]) <= float(m_end) + 1e-9]
        if in_mid:
            best_f = min(in_mid, key=lambda f: abs(float(f["e"]) - center))
            mid_t = round(float(best_f["e"]), 1)
        else:
            mid_t = round(center, 1)
        out.append(("mid", "中盤", mid_t))

    # 3. 終盤: プラント直後（プラントがある場合のみ）
    l_start = segs["late"].get("start")
    l_end = segs["late"].get("end")
    if l_start is not None and l_end is not None:
        in_late = [f for f in frames if float(l_start) - 1e-9 <= float(f["e"]) <= float(l_end) + 1e-9]
        if in_late:
            late_t = round(float(in_late[0]["e"]), 1)
        else:
            late_t = round(float(l_start), 1)
        out.append(("late", "終盤", late_t))

    return out


_ZONE_FALLBACK_UV = {
    "A": (0.28, 0.28),
    "Mid": (0.50, 0.50),
    "B": (0.72, 0.28),
    "C": (0.50, 0.28),
    "攻撃側": (0.50, 0.85),
    "防衛側": (0.50, 0.15),
}


def _positions_at(item: dict, model, e: float) -> list[dict]:
    """時刻 e（開始からの秒）での自チーム各エージェントの (u, v) を返す（1時刻のみ）。"""
    r = item.get("r") if isinstance(item, dict) and "r" in item else item
    pts_out: dict[str, tuple[float, float]] = {}
    # 1. r["tracks"]（deck.records で 0..1 に変換済みの軌跡）
    tr = (r.get("tracks") or {}) if isinstance(r, dict) else {}
    if isinstance(tr, dict):
        for ag, seq in tr.items():
            if not ag or ag == "?" or not isinstance(seq, (list, tuple)) or not seq:
                continue
            valid = [p for p in seq if isinstance(p, (list, tuple)) and len(p) >= 3]
            if not valid:
                continue
            near = min(valid, key=lambda p: abs(float(p[0]) - e))
            u, v = float(near[1]), float(near[2])
            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and (u > 0.0 or v > 0.0):
                pts_out[str(ag)] = (round(u, 4), round(v, 4))
    # 2. model があれば GM.snap で補完
    if len(pts_out) < 3 and model is not None and isinstance(r, dict):
        try:
            sn = GM.snap(r, model, e, tol=12)
            for ag, (u, v, _ar) in (sn or {}).items():
                if ag and ag != "?" and ag not in pts_out and 0.0 <= float(u) <= 1.0 and 0.0 <= float(v) <= 1.0:
                    pts_out[str(ag)] = (round(float(u), 4), round(float(v), 4))
        except Exception:
            pass
    # 3. _round（report.json の tactics.tracks）から補完
    if len(pts_out) < 3 and isinstance(r, dict):
        rr = _round_dict_of(r)
        norm = P._norm_tracks(P.tracks_of(rr))
        team_no = r.get("no") if r.get("no") in (1, 2) else 1
        ag_map = norm.get(team_no) or norm.get(1) or norm.get(0) or {}
        for idx, (ag, seq) in enumerate(sorted(ag_map.items())):
            if not ag or ag == "?" or ag in pts_out or not seq:
                continue
            near = min(seq, key=lambda p: abs(float(p[0]) - e))
            u = float(near[1]) if len(near) > 1 else 0.0
            v = float(near[2]) if len(near) > 2 else 0.0
            if (u == 0.0 and v == 0.0) or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                zone = P._zone_of_sample(near)
                bu, bv = _ZONE_FALLBACK_UV.get(zone, (0.50, 0.50))
                u = min(0.95, max(0.05, bu + (idx - 2) * 0.03))
                v = min(0.95, max(0.05, bv + (idx % 2) * 0.03))
            pts_out[str(ag)] = (round(u, 4), round(v, 4))
    # 4. su（1:40 の初期配置）から補完
    su = item.get("su") if isinstance(item, dict) else None
    if len(pts_out) < 3 and isinstance(su, dict):
        for ag, (u, v, _ar) in su.items():
            if ag and ag != "?" and ag not in pts_out:
                pts_out[str(ag)] = (round(float(u), 4), round(float(v), 4))
    return [{"label": ag, "xy": [[u, v]]} for ag, (u, v) in sorted(pts_out.items())]


def single_time_board(map_name: str, title: str, mark: str, dots: list[dict], side: str,
                      cfg: dict | None = None) -> dict | None:
    """1枚1時刻の TactiVal ボードJSON（1ステップ・エージェントアイコン駒のみ・時間経過の矢印なし）を作る。"""
    bs = BI.boards(map_name, title, {"marks": [mark], "dots": dots}, side, cfg)
    if not bs:
        return None
    b = bs[0]
    tpl = TV.load_template(cfg)
    ids = TV.agent_ids(tpl)
    fallback_uid = next(iter(ids.values()), None)
    for st in b.get("steps") or []:
        # 1枚1時刻のみ（矢印等の時間経過要素を持たず、すべてTactiValエージェントアイコン駒にする）
        st["elements"] = [el for el in (st.get("elements") or []) if el.get("type") == "agent"]
        for el in st["elements"]:
            d = el.setdefault("data", {})
            if not d.get("agentId") and fallback_uid:
                d["agentId"] = fallback_uid
    b["patchSnapshot"] = tpl.get("patchSnapshot")
    return b


def build_phase_boards(ctx: dict, item: dict, title: str, slug_prefix: str) -> tuple[str, list[dict]]:
    """代表ラウンドの区間（序盤・中盤・終盤）ごとに1枚1時刻の TactiVal ボード（最大3枚）を作る。

    - `board_img` 経路（ボードJSON→本物の TactiVal で画像化）を使う。
    - 実行環境で TactiVal 画像化ができない場合は、ボードJSONを成果物として `<out>/boards/` に残し、
      図の場所にその旨を明示する（旧来の重ね描き図への無言のフォールバックはしない）。
    """
    if not item:
        return "", []
    r = item.get("r") if "r" in item else item
    side = r.get("side") or ATTACK
    times = phase_board_times(item, item.get("flow") if isinstance(item, dict) else None)
    out_dir = Path(ctx["out"])
    boards_dir = out_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    meta_list: list[dict] = []
    for seg_key, seg_label, e in times:
        clock = BI._lab(e)
        mark = f"{seg_label} {clock}"
        dots = _positions_at(item, ctx.get("model"), e)
        b = single_time_board(ctx["map"], title, mark, dots, side, ctx.get("cfg"))
        b_fn = f"{slug_prefix}_{seg_key}.tactival.json"
        b_path = boards_dir / b_fn
        if b is not None:
            b_path.write_text(json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")
        imgs = []
        if b is not None:
            imgs = BI.images(ctx.get("cfg") or {}, ctx["map"], title, None, side,
                             log=ctx.get("log", print), bs=[b], marks=[mark])
        tactival_img = next((p for _m, p, kind in imgs if kind == "tactival" and Path(p).exists()), None)
        rendered = tactival_img is not None
        img_rel = None
        if rendered:
            img_fn = f"{slug_prefix}_{seg_key}.jpg"
            dst_img = boards_dir / img_fn
            if Path(tactival_img).resolve() != dst_img.resolve():
                shutil.copyfile(tactival_img, dst_img)
            img_rel = f"boards/{img_fn}"
            inner = (f'<img class="tactival-img" src="{html.escape(img_rel, quote=True)}" '
                     f'alt="{_e(f"{title} {seg_label}（{clock}）のTactiVal配置図")}" loading="lazy">')
        else:
            inner = (f'<div class="board-note">TactiVal ボードJSONを出力済み（実行環境でTactiVal画像化不可）：'
                     f'<a href="boards/{html.escape(b_fn, quote=True)}" download>{_e(b_fn)}</a></div>')
        cap = f"{seg_label}（{clock} / 開始{e:.0f}秒）"
        cards.append(
            f'<figure class="board-card" data-segment="{_e(seg_key)}" data-time="{e:.1f}" '
            f'data-board="boards/{html.escape(b_fn, quote=True)}">'
            f'{inner}<figcaption>{_e(cap)}</figcaption></figure>'
        )
        meta_list.append({
            "segment": seg_key,
            "label": seg_label,
            "time": e,
            "clock": clock,
            "board_json": f"boards/{b_fn}",
            "image": img_rel,
            "rendered": rendered,
            "single_timestamp": True,
            "steps": len((b or {}).get("steps") or []),
        })
    return f'<div class="boards3">{"".join(cards)}</div>', meta_list


def _ensure_play_gif(ctx: dict, item: dict) -> str:
    """実プレイ例（代表ラウンド）のアニメーション GIF を `<out>/gifs/` に用意し、相対パスを返す。"""
    r = item.get("r") if isinstance(item, dict) and "r" in item else item
    rk_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_key(item)).strip("_") or "round"
    gifs_dir = Path(ctx["out"]) / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    dst = gifs_dir / f"{rk_slug}.gif"
    if dst.exists() and dst.stat().st_size > 0:
        return f"gifs/{dst.name}"

    # 1. report.json の minimap_gif がディスクにあればそれをコピー
    rr = _round_dict_of(r)
    cand_paths = [r.get("gif"), ((rr.get("assets") or {}).get("minimap_gif") if isinstance(rr, dict) else None)]
    for cp in cand_paths:
        if cp and Path(cp).is_file() and Path(cp).stat().st_size > 0:
            shutil.copyfile(cp, dst)
            return f"gifs/{dst.name}"

    # 2. 実プレイの軌跡サンプルからアニメーション GIF を生成
    from PIL import Image, ImageDraw
    size = 280
    fig = ctx.get("fig")
    if fig is not None and getattr(fig, "view", None) is not None:
        v = cv2.resize(fig.view, (size, size), interpolation=cv2.INTER_AREA)
        if v.shape[2] == 4:
            al = (v[:, :, 3:4].astype(np.float32) / 255.0) * 0.85
            bg_arr = np.zeros((size, size, 3), np.uint8)
            bg_arr[:] = (18, 22, 28)
            bgr = (bg_arr * (1.0 - al) + v[:, :, :3] * al).astype(np.uint8)
        else:
            bgr = v[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        base_im = Image.fromarray(rgb)
    else:
        base_im = Image.new("RGB", (size, size), (18, 22, 28))

    norm = P._norm_tracks(P.tracks_of(rr)) if rr else {}
    frames_data = P._frames(norm) if norm else []
    es = [float(f["e"]) for f in frames_data][:18]
    if len(es) < 2:
        es = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    k_rot = ctx.get("k", 0)
    side = r.get("side") or ATTACK
    col_fill = (255, 70, 85) if side == ATTACK else (46, 196, 182)
    pil_frames: list[Image.Image] = []
    for e in es:
        im = base_im.copy()
        dr = ImageDraw.Draw(im)
        dots = _positions_at(item, ctx.get("model"), e)
        for d in dots:
            u, v_coord = d["xy"][0]
            vx, vy = MF.to_view(u, v_coord, k_rot)
            px = int(min(size - 10, max(10, round(vx * size))))
            py = int(min(size - 10, max(10, round(vy * size))))
            dr.ellipse((px - 7, py - 7, px + 7, py + 7), fill=col_fill, outline=(255, 255, 255), width=2)
        dr.rectangle((6, 6, 68, 24), fill=(15, 18, 24))
        dr.text((12, 9), BI._lab(e), fill=(255, 255, 255))
        pil_frames.append(im)

    pil_frames[0].save(
        dst, format="GIF", save_all=True, append_images=pil_frames[1:], duration=360, loop=0
    )
    return f"gifs/{dst.name}"


def play_gif_card(ctx: dict, item: dict) -> tuple[str, str]:
    """代表ラウンド（実際のプレイ例）の GIF カード HTML と相対パスを返す。"""
    if not item:
        return "", ""
    r = item.get("r") if "r" in item else item
    label = sample_label(r)
    gif_rel = _ensure_play_gif(ctx, item)
    card = (f'<figure class="play-gif">'
            f'<img class="gif-img" src="{html.escape(gif_rel, quote=True)}" '
            f'alt="{html.escape(f"{label} の実プレイGIF", quote=True)}" loading="lazy">'
            f'<figcaption>{_e(f"{label}（実プレイGIF）")}</figcaption></figure>')
    return card, gif_rel


# ============================================================== 動く図（後方互換の関数群）
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


# ============================================================== フェーズ型（序盤・中盤・終盤）と文面組み立て
def _observed_action_reaction(members: list[dict], seg: str) -> tuple[str, str] | None:
    """その型のラウンド群から、最も多い「アクション→リアクション」のステップ対を抽出する。"""
    flow_seg = {"entry": "mid", "postplant": "late"}.get(seg, seg)
    pairs: Counter = Counter()
    for it in members:
        fl = it.get("flow") or {}
        flow_str = ((fl.get("flow") or {}).get(flow_seg) or "").strip()
        steps = [s.strip() for s in flow_str.split("→") if s.strip()]
        if len(steps) < 2:
            # 区間内が1ステップ以下ならラウンド全体の流れから拾う
            all_steps = [
                s.strip()
                for k in ("early", "mid", "late")
                for s in (((fl.get("flow") or {}).get(k) or "").split("→"))
                if s.strip()
            ]
            steps = all_steps
        if len(steps) < 2:
            continue
        chosen_pair = None
        for i in range(len(steps) - 1):
            s0, s1 = steps[i], steps[i + 1]
            side0 = s0.split(" ", 1)[0]
            side1 = s1.split(" ", 1)[0]
            if side0 != side1:
                chosen_pair = (s0, s1)
                break
        if chosen_pair is None:
            chosen_pair = (steps[0], steps[1])
        pairs[chosen_pair] += 1
    return pairs.most_common(1)[0][0] if pairs else None


def _defense_late_kind(members: list[dict]) -> tuple[str, int, int]:
    """防衛の終盤ラウンド群から「寄せ」「リテイク」の別と件数を判定する。"""
    rotate_n = 0
    retake_n = 0
    for it in members:
        fl = it.get("flow") or {}
        late_start = ((fl.get("segments") or {}).get("late") or {}).get("start")
        if late_start is None:
            continue
        evs = [
            ev for ev in (fl.get("events") or [])
            if ev.get("side") == DEFENSE and float(ev.get("t") or 0.0) >= float(late_start) - 1e-9
        ]
        if any(ev.get("kind") == "ローテート" for ev in evs):
            rotate_n += 1
        if any(ev.get("kind") in ("取り返し", "テイク") for ev in evs):
            retake_n += 1
    if retake_n > rotate_n:
        return "リテイク", rotate_n, retake_n
    if rotate_n > 0:
        return "寄せ", rotate_n, retake_n
    return "寄せ／リテイク", rotate_n, retake_n


def phase_type_description(seg: str, typ: str, members: list[dict] | None = None,
                           side: str = ATTACK) -> str:
    """フェーズの型と流れ要約から「アクション→リアクション」の形で説明文を組み立てる。

    防衛ブロックは防衛視点（防衛を主語）、攻撃ブロックは攻撃視点（攻撃を主語）で書く。
    旧ノード表（glossary_word 系）は使わない。
    """
    members = members or []
    t = str(typ or "").strip()
    if side == DEFENSE:
        if seg == "early":
            if t.endswith("寄せ"):
                z = t[:-2]
                base = (f"アクション：序盤、防衛は {z} 側に人数を厚く置いて主導権を握る。"
                        f"リアクション：厚い側へ来ればスキルとクロスで受け止め、逆側へ来たら素早く寄せる")
            elif t == "自陣待機":
                base = ("アクション：序盤、防衛は無理に前へ出ず自陣寄りのポジションで構える。"
                        "リアクション：相手の位置が見えてから寄せる方向を判断する")
            else:
                base = (f"アクション：序盤、防衛は {t} の配置でエリアを押さえる。"
                        f"リアクション：初期情報から中盤のローテート判断につなげる")
        elif seg == "mid":
            if t == "上がり":
                base = ("アクション：中盤、防衛は前目へプッシュしてエリアと情報を奪う。"
                        "リアクション：相手の動きを絞って逆サイトの守りを厚くし、エントリーに備える")
            elif t == "ローテート":
                base = ("アクション：中盤、防衛は取った情報に合わせて重心を移動（ローテート）させる。"
                        "リアクション：手薄になったサイトへ素早くカバーを寄せ、入口で迎え撃つ")
            elif t == "取り返し":
                base = ("アクション：中盤、防衛は一度渡したエリアへ人数をかけて取り返しに動く。"
                        "リアクション：エリアを奪い返して相手の攻め筋を狭め、エントリーを遅らせる")
            else:
                base = (f"アクション：中盤、防衛は {t} の動きで主導権を保つ。"
                        f"リアクション：相手の仕掛けに合わせて強い形で迎え撃つ")
        else:
            kind, rot_n, ret_n = _defense_late_kind(members)
            detail = f"{kind}：寄せ {rot_n} 件・リテイク {ret_n} 件"
            if kind == "リテイク":
                base = (f"アクション：スパイク設置後、防衛は味方の合流を待ってからサイトへリテイクに入る（{detail}）。"
                        f"リアクション：スキルと複数方向からの進入でサイト内を制圧し、解除を通す")
            elif kind == "寄せ":
                base = (f"アクション：スパイク設置後、防衛は他サイトから素早く人数を寄せて包囲を作る（{detail}）。"
                        f"リアクション：設置後の配置を崩して解除の時間を確保する")
            else:
                base = (f"アクション：スパイク設置後、防衛は状況に応じて寄せとリテイクを切り替える（{detail}）。"
                        f"リアクション：人数とスキルのタイミングを合わせて解除を狙う")
    elif seg == "early":
        if t == "ラッシュ":
            base = (f"アクション：開幕{int(P.RUSH_SECONDS)}秒以内に攻撃がサイトへ2人以上で一気にエントリーする。"
                    f"リアクション：防衛がサイト内で耐える動きや逆からの寄せが来る前に、人数差で押し切る")
        elif t.endswith("デフォルト"):
            z = t[:-5]
            base = (f"アクション：開幕、攻撃は {z} 側を軸に展開してエリアの主導権を探る。"
                    f"リアクション：防衛の前目プッシュやローテートを見て、手薄な側へエントリーをつなげる")
        elif t == "詰め待ち":
            base = ("アクション：開幕、攻撃は約10秒ほど後方で動かず詰め待ちする。"
                    "リアクション：防衛が前へ情報を探りに来たタイミングを捉え、ピックから主導権を取る")
        else:
            base = (f"アクション：開幕に {t} の形で展開する。"
                    f"リアクション：相手の初期配置とプッシュの有無を見てエントリー先を決める")
    elif seg == "entry":
        z = t[0] if t and t[0] in P.SITES else ""
        if t.endswith("スプリット"):
            base = (f"アクション：攻撃は複数の進入路から同時に {z} サイトへ挟み込む（スプリット）。"
                    f"リアクション：防衛の射線とカバーを分断し、サイト内の守りを崩して設置へつなげる")
        elif t.endswith("セット"):
            base = (f"アクション：攻撃は {z} サイトの1つの進入路に人数をそろえてセットで進入する。"
                    f"リアクション：スキルを合わせてサイト内を一気に確保し、設置まで持ち込む")
        else:
            base = (f"アクション：エントリーで {t} を仕掛ける。"
                    f"リアクション：防衛の配置に合わせてサイト内を制圧する")
    elif seg in ("postplant", "late"):
        if seg == "postplant":
            base = (f"アクション：スパイク設置後、攻撃は「{t}」の配置で防衛の解除を待つ。"
                    f"リアクション：防衛のリテイク進入をクロスファイアと時間稼ぎで迎え撃つ")
        elif t == "プッシュ":
            base = ("アクション：プラント直後に攻撃がサイトより奥（相手スポーン側）へ2人以上で前進する。"
                    "リアクション：リテイクに集まる防衛を深い位置で迎え撃ち、解除の時間を奪う")
        elif t == "サイト耐え":
            base = ("アクション：プラント後に攻撃が2人以上でサイト内に残ってポジションを固める。"
                    "リアクション：防衛の取り返し（リテイク）に対し、サイト内のクロスとカバーで耐え切る")
        elif t == "下がる":
            base = ("アクション：プラント後に攻撃がサイト内からメインや後方エリアへ引いて構える。"
                    "リアクション：防衛がスパイク解除に入る動きに合わせてピークとスキルで阻止する")
        else:
            base = (f"アクション：終盤に {t} の配置を取る。"
                    f"リアクション：防衛のリテイク進入に合わせて撃ち合う")
    else:
        base = (f"アクション：中盤に攻撃は {t} の形で展開する。"
                f"リアクション：防衛の配置とローテートを見てサイトへのエントリーにつなげる")

    pair = _observed_action_reaction(members, seg)
    if pair:
        return f"{base}（実測の応酬：{pair[0]} → {pair[1]}）。"
    return f"{base}。"


def _has_all_3_segments(it: dict) -> bool:
    fl = it.get("flow")
    if not fl:
        return False
    segs = fl.get("segments") or {}
    m0, m1 = (segs.get("mid") or {}).get("start"), (segs.get("mid") or {}).get("end")
    l0 = (segs.get("late") or {}).get("start")
    return m0 is not None and m1 is not None and float(m1) > float(m0) + 1e-9 and l0 is not None


def _pick_phase_reps(members: list[dict], seg: str, used_leads: set[str], max_reps: int = 3) -> list[dict]:
    """型の代表ラウンド（複数件）を選ぶ。VODあり・3区間そろい・full5・勝ち・イベント数を優先。"""
    def sort_key(it: dict):
        r = it["r"]
        fl = it.get("flow") or {}
        seg_flow = (fl.get("flow") or {}).get(seg) or ""
        return (
            not bool(yt_id(r.get("yt"))),
            not _has_all_3_segments(it),
            not is_full5(it),
            not bool(seg_flow),
            not bool(r.get("win")),
            -len(fl.get("events") or []),
            sample_key(it),
        )

    ordered = sorted(members, key=sort_key)
    if not ordered:
        return []
    # 先頭（3枚図と実プレイGIFを描く主代表）は他型の主代表と重複しないものを優先
    lead = next((it for it in ordered if sample_key(it) not in used_leads), ordered[0])
    used_leads.add(sample_key(lead))
    reps = [lead]
    seen = {sample_key(lead)}
    for it in ordered:
        rk = sample_key(it)
        if rk in seen:
            continue
        seen.add(rk)
        reps.append(it)
        if len(reps) >= max_reps:
            break
    return reps


def _postplant_type_name(site: str, med_positions: dict, used: set[str]) -> str:
    """ポストプラントのコールアウト構成から型名を付ける。"""
    counts = Counter(area for _u, _v, area in med_positions.values() if area and area != "?")
    if counts:
        parts = [f"{a}{n}" if n > 1 else str(a) for a, n in counts.most_common(3)]
        base = f"{site}：{'・'.join(parts)}"
    else:
        base = f"{site}：配置"
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}（{i}）" in used:
        i += 1
    cand = f"{base}（{i}）"
    used.add(cand)
    return cand


def _postplant_clusters(classified: list[dict], k: int = 0) -> list[tuple[str, list[dict], str]]:
    """ポストプラント配置をサイト別に PATTERN_DIST でクラスタリングし、サイトごとに上位2種まで返す。"""
    by_site: dict[str, list[tuple[dict, dict, dict]]] = defaultdict(list)
    for it in classified:
        pp = ((it.get("flow") or {}).get("phases") or {}).get("postplant")
        if not pp or not pp.get("site") or not pp.get("positions"):
            continue
        site = pp["site"]
        pos = pp["positions"]
        pt = {ag: MF.to_view(u, v, k) for ag, (u, v, _a) in pos.items()}
        by_site[site].append((it, pos, pt))

    out: list[tuple[str, list[dict], str]] = []
    used_names: set[str] = set()
    for site in sorted(by_site):
        rows = by_site[site]
        points = [pt for _it, _pos, pt in rows]
        clusters, _singles = leader_clusters(points, th=PATTERN_DIST, min_n=1)
        for grp in clusters[:2]:
            members = [rows[i][0] for i in grp]
            win_idx = [i for i in grp if rows[i][0]["r"].get("win")]
            cand = win_idx or grp
            med = min(cand, key=lambda a: sum(round_distance(points[a], points[b]) for b in grp))
            typ_name = _postplant_type_name(site, rows[med][1], used_names)
            out.append((typ_name, members, site))
    out.sort(key=lambda x: (-len(x[1]), x[0]))
    return out


def build_phase_sections(items: list[dict], side: str | None = None,
                         max_types: int | None = 3, k: int = 0) -> dict:
    """構成のラウンド群を `phases.classify_flow` で集計し、サイド別の3部データを返す。

    - 攻撃: 開幕（5種）／エントリー（サイト別セット・スプリット）／ポストプラント（サイト別上位2種）
    - 防衛: 序盤（防衛配置）／中盤（防衛の強い動き）／終盤（寄せ方・リテイク）
    """
    pool = [it for it in items if side is None or (it.get("r") or {}).get("side") == side]
    total = len(pool)
    classified = [it for it in pool if it.get("flow") is not None]
    cls_n = len(classified)
    skipped_n = total - cls_n
    used_leads: set[str] = set()
    raw_sections = []
    all_entries = []
    desc_side = side or ATTACK
    segs = SIDE_SEGS.get(desc_side, SEGS)
    for seg_key, seg_label in segs:
        types_out = []
        if desc_side == ATTACK and seg_key == "postplant":
            pp_groups = _postplant_clusters(classified, k=k)
            typed_n = sum(len(m) for _t, m, _s in pp_groups)
            for typ, members, site in pp_groups:
                n = len(members)
                w = sum(1 for it in members if it["r"].get("win"))
                rate = w / n if n else 0.0
                share = n / cls_n if cls_n else 0.0
                entry = {
                    "side": side,
                    "seg": seg_key,
                    "seg_label": seg_label,
                    "site": site,
                    "type": typ,
                    "n": n,
                    "w": w,
                    "rate": rate,
                    "share": share,
                    "defense_late_kind": None,
                    "description": phase_type_description(seg_key, typ, members, side=desc_side),
                    "lead": None,
                    "reps": [],
                    "members": members,
                }
                types_out.append(entry)
                all_entries.append(entry)
        else:
            by_type: dict[str, list[dict]] = defaultdict(list)
            for it in classified:
                typ = ((it["flow"].get("phases") or {}).get(seg_key) or {}).get("type")
                if typ:
                    by_type[str(typ)].append(it)
            typed_n = sum(len(v) for v in by_type.values())
            sorted_types = sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            if max_types is not None:
                sorted_types = sorted_types[:max_types]
            for typ, members in sorted_types:
                n = len(members)
                w = sum(1 for it in members if it["r"].get("win"))
                rate = w / n if n else 0.0
                share = n / cls_n if cls_n else 0.0
                late_kind = _defense_late_kind(members)[0] if (desc_side == DEFENSE and seg_key == "late") else None
                entry = {
                    "side": side,
                    "seg": seg_key,
                    "seg_label": seg_label,
                    "type": typ,
                    "n": n,
                    "w": w,
                    "rate": rate,
                    "share": share,
                    "defense_late_kind": late_kind,
                    "description": phase_type_description(seg_key, typ, members, side=desc_side),
                    "lead": None,
                    "reps": [],
                    "members": members,
                }
                types_out.append(entry)
                all_entries.append(entry)
        raw_sections.append({
            "side": side,
            "seg": seg_key,
            "label": seg_label,
            "typed_rounds": typed_n,
            "untyped_rounds": cls_n - typed_n,
            "types": types_out,
        })
    # 件数の少ない型から順に主代表ラウンドを割り当て、型間で主代表が重複しないようにする
    for entry in sorted(all_entries, key=lambda e: (e["n"], e["seg"], e["type"])):
        reps = _pick_phase_reps(entry["members"], entry["seg"], used_leads, max_reps=3)
        entry["reps"] = reps
        entry["lead"] = reps[0] if reps else None
    return {
        "side": side,
        "total": total,
        "classified": cls_n,
        "skipped": skipped_n,
        "sections": raw_sections,
    }


# ============================================================== データの組み立て
def _prepare(reports: list[dict], map_name: str, cfg: dict, log=print) -> dict:
    allr = []
    match_yts = []
    for rep in reports:
        if str(rep.get("map") or "").lower() != map_name.lower():
            continue
        for rr in rep.get("rounds") or []:
            if rr.get("yt_url"):
                match_yts.append(rr["yt_url"])
    fallback_yt = match_yts[0] if match_yts else None
    for rep in reports:
        if str(rep.get("map") or "").lower() != map_name.lower():
            continue
        rr_by_n = {int(rr.get("n") or 0): rr for rr in (rep.get("rounds") or []) if rr.get("n") is not None}
        rep_yt = next((rr.get("yt_url") for rr in (rep.get("rounds") or []) if rr.get("yt_url")), fallback_yt)
        for r in D.records(rep):
            r["_round"] = rr_by_n.get(int(r.get("n") or 0))
            if not r.get("yt") and rep_yt:
                r["yt"] = rep_yt
            allr.append(r)
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
    ranks = TC.area_ranks(meta) if callouts else {}
    return {"allr": allr, "fig": fig, "model": model, "callouts": callouts, "ranks": ranks, "walk": walk,
            "kb": kbo, "k": fig.k, "vitals": model.vitals(), "bg": fig._uri, "cfg": cfg, "log": log}


def _apply_filter_to_record(r: dict, rr: dict, ranks: dict | None) -> dict:
    """配置抽出（setup_at_140 等）にも暫定バリアフィルタを効かせるため、r['tracks'] から除外済み初期サンプルを除く。"""
    if not ranks or not rr or not isinstance(r.get("tracks"), dict):
        return r
    sides = P._team_sides(rr)
    norm = P._norm_tracks(P.tracks_of(rr), sides=sides, ranks=ranks)
    team_no = r.get("no") if r.get("no") in (1, 2) else None
    if team_no is None or team_no not in norm:
        return r
    kept_times = {
        ag: {P._snap(P._f(p[0])) for p in pts if isinstance(p, (list, tuple)) and p}
        for ag, pts in norm[team_no].items()
    }
    new_tracks = {}
    for ag, pts in r["tracks"].items():
        if not isinstance(pts, list):
            new_tracks[ag] = pts
            continue
        allowed = kept_times.get(ag)
        if allowed is None:
            new_tracks[ag] = [p for p in pts if P._f(p[0]) > P.RUSH_SECONDS + 1e-9]
        else:
            new_tracks[ag] = [
                p for p in pts
                if P._f(p[0]) > P.RUSH_SECONDS + 1e-9 or P._snap(P._f(p[0])) in allowed
            ]
    out = dict(r)
    out["tracks"] = new_tracks
    return out


def _items_for(rounds: list[dict], allr: list[dict], model, ranks: dict | None = None) -> list[dict]:
    idx = {(r["match_id"], r["order"], r["n"], r["no"]): r for r in allr}
    items = []
    for r in rounds:
        opp = idx.get((r["match_id"], r["order"], r["n"], 3 - r["no"])) or {}
        rr = _round_dict_of(r)
        r_filt = _apply_filter_to_record(r, rr, ranks)
        su = GM.setup_at_140(r_filt, model)
        cls = GM.classify(r_filt, opp, model) if su is not None else {}
        flow, skip = classify_round_flow(r_filt, ranks=ranks)
        fstats = (flow or {}).get("filter_stats") or {"outside_home": 0, "jump": 0, "dropped": 0}
        items.append({"r": r_filt, "opp": opp, "su": su, "cls": cls, "flow": flow,
                      "flow_skip": skip, "filter_stats": fstats})
    return items


def comp_stats(rounds: list[dict], items: list[dict]) -> dict:
    used_def = sum(1 for it in items if it["su"] is not None and it["r"]["side"] == DEFENSE)
    used_atk = sum(1 for it in items if it["su"] is not None and it["r"]["side"] == ATTACK)
    full5 = sum(1 for it in items if it["su"] is not None and len(it["su"]) == 5)
    flow_classified = sum(1 for it in items if it.get("flow") is not None)
    flow_skipped = len(rounds) - flow_classified
    f_out = sum(int((it.get("filter_stats") or {}).get("outside_home", 0)) for it in items)
    f_jmp = sum(int((it.get("filter_stats") or {}).get("jump", 0)) for it in items)
    return {"total": len(rounds), "used": used_def + used_atk, "used_def": used_def, "used_atk": used_atk,
            "excluded": len(rounds) - used_def - used_atk, "full5": full5,
            "flow_classified": flow_classified, "flow_skipped": flow_skipped,
            "barrier_filter": {"outside_home": f_out, "jump": f_jmp, "dropped": f_out + f_jmp},
            "dates": sorted({str(r.get("date")) for r in rounds if r.get("date")}),
            "teams": sorted({str(r.get("tag") or r.get("team")) for r in rounds}),
            "wins": sum(1 for r in rounds if r.get("win"))}


# ============================================================== 基本配置クラスタの構築
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
    """側の基本配置型（1:40 配置クラスタ）。対象＝ピストルでない・1:40 に5人そろったラウンド。"""
    used_names = used_names if used_names is not None else set()
    pool = [it for it in items if it["r"]["side"] == side and not it["r"]["pistol"]
            and it["su"] is not None and len(it["su"]) == 5]
    points = [{a: MF.to_view(u, v, k) for a, (u, v, _ar) in it["su"].items()} for it in pool]
    pats_idx, singles_idx = leader_clusters(points)
    out = []
    for grp in pats_idx:
        members = [pool[i] for i in grp]
        winner = [i for i in grp if pool[i]["r"]["win"]]
        cand = winner or grp
        med = min(cand, key=lambda a: sum(round_distance(points[a], points[b]) for b in grp))
        med_it = pool[med]
        zones = Counter(GM.zone_of(a) for _u, _v, a in med_it["su"].values() if GM.zone_of(a))
        action = _placement_action(members, side)
        name = unique_name(pattern_name(zones, action, side), _pattern_suffix(med_it), used_names)
        used_names.add(name)
        out.append({"side": side, "name": name, "n": len(grp), "w": sum(it["r"]["win"] for it in members),
                    "members": members, "medoid": med_it, "zones": dict(zones)})
    out.sort(key=lambda p: (-p["n"], p["name"]))
    singles = []
    for i in singles_idx:
        it = pool[i]
        zones = Counter(GM.zone_of(a) for _u, _v, a in it["su"].values() if GM.zone_of(a))
        action = _placement_action([it], side)
        name = unique_name(pattern_name(zones, action, side), _pattern_suffix(it), used_names)
        used_names.add(name)
        singles.append({"side": side, "name": name, "n": 1, "item": it})
    return out, singles


# ============================================================== なぜこの構成か
def _pick_claim_example(cands: list[dict]) -> dict | None:
    if not cands:
        return None
    ordered = sorted(
        cands,
        key=lambda it: (
            not bool(yt_id(it["r"].get("yt"))),
            not _has_all_3_segments(it),
            not (it.get("flow") is not None),
            not is_full5(it),
            not bool(it["r"].get("win")),
            sample_key(it),
        ),
    )
    return ordered[0]


def _phase_type_counts(items: list[dict]) -> tuple[int, dict[tuple[str, str, str], list[dict]]]:
    cls_items = [it for it in items if it.get("flow") is not None]
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for it in cls_items:
        side = (it.get("r") or {}).get("side") or ATTACK
        for seg_key, seg_label in SIDE_SEGS.get(side, SEGS):
            if seg_key == "postplant":
                continue
            typ = ((it["flow"].get("phases") or {}).get(seg_key) or {}).get("type")
            if typ:
                by_key[(seg_key, seg_label, str(typ))].append(it)
    return len(cls_items), by_key


def _why_claims(comp_rounds: list[dict], items: list[dict], base_rounds: list[dict],
                base_items: list[dict], model) -> list[dict]:
    """全体平均との差がしきい値を超えた指標だけから主張を作る（旧ノード表や「はまると強い」は使わない）。"""
    claims = []

    # 1. 設置サイトの偏り（全体との差 15pt 以上）
    pa = [r for r in comp_rounds if r.get("plant_site") in ("A", "B")]
    ba = [r for r in base_rounds if r.get("plant_site") in ("A", "B")]
    if len(pa) >= 8 and len(ba) >= 20:
        sha = sum(1 for r in pa if r["plant_site"] == "A") / len(pa)
        bha = sum(1 for r in ba if r["plant_site"] == "A") / len(ba)
        if abs(sha - bha) >= 0.15:
            site = "A" if sha > bha else "B"
            share = sha if site == "A" else 1 - sha
            base_share = bha if site == "A" else 1 - bha
            ex = _pick_claim_example([it for it in items if it["r"].get("plant_site") == site and it["su"]])
            claims.append({
                "text": f"{site} サイトへの設置が {share:.0%}（全体 {base_share:.0%}）。{site} 側のエリア取りから設置へつなげる。",
                "item": ex,
                "metric": "plant_site",
            })

    # 2. ファーストキル取得率（全体との差 10pt 以上）
    fa = [r for r in comp_rounds if r.get("fk_own") is not None]
    fb = [r for r in base_rounds if r.get("fk_own") is not None]
    if len(fa) >= 8 and len(fb) >= 20:
        fr = sum(1 for r in fa if r["fk_own"]) / len(fa)
        br = sum(1 for r in fb if r["fk_own"]) / len(fb)
        if abs(fr - br) >= 0.10:
            ex = _pick_claim_example([it for it in items if it["r"].get("fk_own") and it["su"]]) or \
                 _pick_claim_example([it for it in items if it["su"]])
            if fr >= br:
                txt = f"ファーストキル取得率が {fr:.0%}（全体 {br:.0%}）。序盤のピックから人数差を作る。"
            else:
                txt = f"ファーストキル取得率が {fr:.0%}（全体 {br:.0%}）。序盤は無理に当たらず中盤の展開で勝負する。"
            claims.append({"text": txt, "item": ex, "metric": "fk"})

    # 3. フェーズ型の特徴（構成・全体のどちらも n>=5 で、採用割合または勝率が全体と 10pt 以上差がある）
    cls_n, my_types = _phase_type_counts(items)
    base_cls_n, base_types = _phase_type_counts(base_items)
    best_phase = None
    if cls_n >= 5 and base_cls_n >= 10:
        for key, members in my_types.items():
            seg_key, seg_label, typ = key
            base_members = base_types.get(key) or []
            n, bn = len(members), len(base_members)
            if n < 5 or bn < 5:
                continue
            w = sum(1 for it in members if it["r"].get("win"))
            bw = sum(1 for it in base_members if it["r"].get("win"))
            share, base_share = n / cls_n, bn / base_cls_n
            rate, base_rate = w / n, bw / bn
            diff_share = share - base_share
            diff_rate = rate - base_rate
            if max(abs(diff_share), diff_rate) < 0.10:
                continue
            score = max(abs(diff_share), diff_rate) * n
            if best_phase is None or score > best_phase[0]:
                ex = _pick_claim_example(members)
                if abs(diff_share) >= diff_rate:
                    txt = (f"{seg_label}の「{typ}」が {share:.0%}（全体 {base_share:.0%}、"
                           f"分類 {cls_n} 件中 {n} 件・{_wl(w, n)}）。")
                else:
                    txt = (f"{seg_label}「{typ}」での勝率が {rate:.0%}（全体 {base_rate:.0%}、"
                           f"{_wl(w, n)}）。")
                best_phase = (score, txt, ex)
    if best_phase:
        _sc, txt, ex = best_phase
        claims.append({"text": txt, "item": ex, "metric": "phase"})

    # 4. 守りのバイタル占有（全体との差 15pt 以上）
    vit = model.vitals()
    if vit:
        da = [it for it in items if it["r"]["side"] == DEFENSE and it["su"]]
        oc = sum(1 for it in da if any(a in vit for _u, _v, a in it["su"].values()))
        if len(da) >= 8:
            orate = oc / len(da)
            bda = [it for it in base_items if it["r"]["side"] == DEFENSE and it["su"]]
            if bda:
                base_oc = sum(1 for it in bda if any(a in vit for _u, _v, a in it["su"].values())) / len(bda)
            else:
                base_oc = _base_vital_rate(base_rounds, model, vit)
            if base_oc is not None and abs(orate - base_oc) >= 0.15:
                ex = _pick_claim_example([it for it in da if any(a in vit for _u, _v, a in it["su"].values())])
                claims.append({
                    "text": f"守りで 1:40 にバイタル（{'・'.join(vit)}）へ人がいる割合が {orate:.0%}（全体 {base_oc:.0%}）。",
                    "item": ex,
                    "metric": "vital",
                })
    seen, out = set(), []
    for c in claims:
        if c["metric"] in seen or not c.get("item"):
            continue
        seen.add(c["metric"])
        out.append(c)
    return out[:3]


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
def _slide(eyebrow: str, title: str, body: str, foot: str = "", h1: bool = False,
           cls: str = "", data_side: str | None = None) -> str:
    h = f"<h1>{_e(title)}</h1>" if h1 else f"<h2>{_e(title)}</h2>"
    klass = f"slide {cls}".strip()
    side_attr = f' data-side="{_e(data_side)}"' if data_side else ""
    return (f'<section class="{klass}"{side_attr}><div class="eyebrow">{_e(eyebrow)}</div>{h}{body}'
            f'<div class="foot"><span>{_e(foot)}</span><span>vreplay プレイブック v5</span></div></section>')


def _cover_slide(ctx, comp, stat) -> str:
    agents = agents_of(comp)
    teams = "・".join(stat["teams"])
    period = f"{stat['dates'][0]} 〜 {stat['dates'][-1]}" if stat["dates"] else "-"
    cls_n = stat.get("flow_classified", 0)
    sk_n = stat.get("flow_skipped", stat["total"] - cls_n)
    body = (_s(f"使うチーム：{teams}") +
            _s(f"期間：{period}（{ENV_LABEL}）") +
            _s(f"{ctx['map']} 全 {stat['total']} ラウンド（基本配置の対象＝1:40 の配置が取れた {stat['used']} ラウンド）。") +
            _s(f"フェーズ分類できたラウンド数：{cls_n} / 全 {stat['total']} ラウンド（tracks 無し {sk_n} 件は対象外）。"))
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


def _why_slide(ctx, claims) -> tuple[str, list[dict]]:
    if not claims:
        return _slide("なぜこの構成か", "なぜこの構成か",
                      _s("他の構成とはっきりした差は出ていない。"), ctx["map"]), []
    cols = []
    claims_meta = []
    for idx, c in enumerate(claims, 1):
        b_html, b_meta = build_phase_boards(
            ctx, c["item"], f"根拠_{c['metric']}", f"{ctx['comp_slug']}_why_{idx}_{c['metric']}"
        )
        cols.append(f'<div class="claim">{b_html}{_vod_for(c["item"])}{_s(c["text"])}</div>')
        claims_meta.append({
            "metric": c["metric"],
            "text": c["text"],
            "sample_round": sample_key(c["item"]),
            "boards": b_meta,
            "expected_boards": len(b_meta),
        })
    return _slide("なぜこの構成か", "なぜこの構成か", f'<div class="claims">{"".join(cols)}</div>', ctx["map"]), claims_meta


def side_avg_winrate(items: list[dict], side: str):
    """その側の平均勝率（基本配置と同じ母数＝ピストルを除く・1:40 に5人そろったラウンド）。"""
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
    side_slug = "def" if side == DEFENSE else "atk"
    avg = side_avg_winrate(items, side)
    chosen, avoid = choose_placement(patterns, avg)
    pool_n = sum(1 for it in items if it["r"]["side"] == side and not it["r"]["pistol"] and is_full5(it))
    item, boards_html, boards_meta, reason = None, "", [], ""
    reason_kind = ""
    if chosen:
        item = _pick_claim_example(chosen["members"]) or chosen["medoid"]
        if item:
            used.add(sample_key(item))
            boards_html, boards_meta = build_phase_boards(
                ctx, item, f"{label}の基本配置", f"{ctx['comp_slug']}_placement_{side_slug}"
            )
    texts = []
    if chosen and boards_html:
        reason_kind = "rendered"
        rate = chosen["w"] / chosen["n"] if chosen["n"] else 0.0
        zones = "・".join(f"{z}{n}枚" for z, n in sorted(chosen["zones"].items(), key=lambda kv: -kv[1]))
        texts.append(_s(f"基本配置：{chosen['name']}（{_wl(chosen['w'], chosen['n'])}・勝率 {rate:.0%}、"
                        f"この側の平均 {avg:.0%}／全ラウンド基盤の非ピストル full5 {pool_n} 件中）。"))
        texts.append(_s(f"1:40 の初期人数配分は {zones or '-'}。代表ラウンド（{sample_label(item['r'])}）の序盤・中盤・終盤を並べている。"))
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
    if boards_html:
        body = f'<div class="placement-wrap">{boards_html}<div class="handtxt">{"".join(texts)}</div></div>'
    else:
        body = f'<div class="handtxt">{"".join(texts)}</div>'
    meta = {"side": side, "name": chosen["name"] if chosen else None,
            "n": chosen["n"] if chosen else 0, "w": chosen["w"] if chosen else 0,
            "rate": round(chosen["w"] / chosen["n"], 4) if chosen and chosen["n"] else None,
            "avg": round(avg, 4) if avg is not None else None,
            "meets_winrate": bool(chosen) and (avg is None or chosen["w"] / chosen["n"] >= avg),
            "pool_n": pool_n, "reason_kind": reason_kind,
            "avoid": [{"name": p["name"], "n": p["n"], "w": p["w"]} for p in avoid],
            "rendered": bool(boards_html), "sample_round": sample_key(item) if item else None,
            "boards": boards_meta, "reason": reason}
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
    return _slide("配置 / 型の一覧", "基本配置の型一覧（1:40 配置クラスタ）", table, ctx["map"])


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


SIDE_SECTION_TITLES = {
    ATTACK: {
        "early": "開幕の型",
        "entry": "エントリーの型",
        "postplant": "ポストプラントの形",
    },
    DEFENSE: {
        "early": "序盤の型",
        "mid": "中盤の強い動き",
        "late": "終盤：寄せ方・リテイクの仕方",
    },
}


def _phase_type_slide(ctx: dict, sec: dict, t_info: dict, idx: int,
                      total_rounds: int, classified_rounds: int) -> tuple[str, dict]:
    """序盤・中盤・終盤の各型のスライド（3枚図＋アクション→リアクション説明＋代表ラウンド複数件＋実プレイGIF＋VODカード）。"""
    side = sec.get("side") or ATTACK
    side_label = side
    side_slug = "atk" if side == ATTACK else "def"
    seg_key = sec["seg"]
    seg_label = sec["label"]
    sec_title = SIDE_SECTION_TITLES.get(side, {}).get(seg_key, seg_label)
    typ = t_info["type"]
    late_kind = t_info.get("defense_late_kind")
    n = t_info["n"]
    w = t_info["w"]
    rate = t_info["rate"]
    share = t_info["share"]
    lead = t_info["lead"]
    reps = t_info["reps"]
    sk_n = total_rounds - classified_rounds

    slug_prefix = f"{ctx['comp_slug']}_{side_slug}_{seg_key}_{idx}_{_slug(typ)}"
    boards_html, boards_meta = build_phase_boards(ctx, lead, f"{side_label} {seg_label} {typ}", slug_prefix)

    # 代表ラウンド（複数件）の流れ要約行と VOD カード、実プレイ例の GIF
    rep_lines = []
    vod_cards = []
    reps_meta = []
    gif_html, lead_gif_rel = play_gif_card(ctx, lead) if lead else ("", "")

    for r_idx, rep_it in enumerate(reps[:3]):
        r = rep_it["r"]
        lbl = sample_label(r)
        side_ja = "攻め" if r.get("side") == ATTACK else "守り"
        fl_map = (rep_it.get("flow") or {}).get("flow") or {}
        seg_flow = (fl_map.get(seg_key) or "").strip()
        all_flow = "→".join(s for k in ("early", "mid", "late") for s in [(fl_map.get(k) or "").strip()] if s)
        flow_disp = seg_flow or all_flow or "支配変化なし（構え合い）"
        rep_lines.append(f"代表 {lbl}（{side_ja}）：{flow_disp}")
        vc = _vod_for(rep_it)
        if vc:
            vod_cards.append(vc)
        _, g_rel = (gif_html, lead_gif_rel) if r_idx == 0 else play_gif_card(ctx, rep_it)
        reps_meta.append({
            "round": sample_key(rep_it),
            "team": r.get("tag") or r.get("team"),
            "opp": r.get("opp_tag") or r.get("opp"),
            "label": lbl,
            "side": r.get("side"),
            "flow": flow_disp,
            "has_vod": bool(vc),
            "gif": g_rel,
            "full5": is_full5(rep_it),
        })

    # スライド本文は p.s 2行 + li 最大3行 = 最大5行以内を守る
    stat_line = (f"{side_label} {seg_label}の型「{typ}」：{n} 件・勝率 {rate:.0%}（{_wl(w, n)}／"
                 f"{side_label} 分類{classified_rounds}/{total_rounds}R中 {share:.0%}・"
                 f"tracks 無し {sk_n} 件は対象外）。")
    desc_line = t_info["description"]

    media_row = f'<div class="rep-media">{gif_html}{"".join(vod_cards)}</div>'
    body = (
        f'{boards_html}'
        f'<div class="phase-body">'
        f'<div class="handtxt">{_s(stat_line)}{_s(desc_line)}{_ul(rep_lines, cls="rep-list")}</div>'
        f'{media_row}'
        f'</div>'
    )
    eyebrow = f"{side_label} / {sec_title}（{n} 件・勝率 {rate:.0%}・{side_label} 分類{classified_rounds}/{total_rounds}R）"
    title = f"{side_label}・{seg_label}（{late_kind}）：{typ}" if late_kind else f"{side_label}・{seg_label}：{typ}"
    slide_html = _slide(
        eyebrow, title, body, ctx["map"],
        cls=f"phase-slide phase-{seg_key} phase-{side_slug}", data_side=side_label,
    )
    meta = {
        "side": side,
        "side_label": side_label,
        "seg": seg_key,
        "seg_label": seg_label,
        "section_title": sec_title,
        "title": title,
        "site": t_info.get("site"),
        "defense_late_kind": late_kind,
        "type": typ,
        "n": n,
        "w": w,
        "rate": round(rate, 4),
        "share": round(share, 4),
        "description": desc_line,
        "lead_round": sample_key(lead) if lead else None,
        "boards": boards_meta,
        "expected_boards": len(boards_meta),
        "representatives": reps_meta,
    }
    return slide_html, meta


def _phase_empty_slide(ctx: dict, sec: dict, total_rounds: int, classified_rounds: int) -> str:
    side = sec.get("side") or ATTACK
    side_label = side
    side_slug = "atk" if side == ATTACK else "def"
    seg_key = sec["seg"]
    seg_label = sec["label"]
    sec_title = SIDE_SECTION_TITLES.get(side, {}).get(seg_key, seg_label)
    sk_n = total_rounds - classified_rounds
    body = (
        _s(f"{side_label} {seg_label}の該当する型はありません（{side_label} 分類{classified_rounds}/{total_rounds}R、"
           f"tracks 無し {sk_n} 件は対象外）。") +
        _s("この区間に該当するイベント（プラント等）が発生したラウンドがありませんでした。")
    )
    return _slide(
        f"{side_label} / {sec_title}（0 件・{side_label} 分類{classified_rounds}/{total_rounds}R）",
        f"{side_label}・{seg_label}：該当型なし",
        body, ctx["map"],
        cls=f"phase-slide phase-{seg_key} phase-{side_slug}", data_side=side_label,
    )


def _appendix_slide(ctx, stat, singles, unknown, def_pats, atk_pats) -> str:
    cls_n = stat.get("flow_classified", 0)
    sk_n = stat.get("flow_skipped", stat["total"] - cls_n)
    bf = stat.get("barrier_filter") or {}
    tables = []
    tables.append('<table><thead><tr><th>数合わせ</th><th>値</th></tr></thead><tbody>'
                  f'<tr><td>総数（環境内ラウンド）</td><td>{stat["total"]}</td></tr>'
                  f'<tr><td>守り（1:40 配置あり）</td><td>{stat["used_def"]}</td></tr>'
                  f'<tr><td>攻め（1:40 配置あり）</td><td>{stat["used_atk"]}</td></tr>'
                  f'<tr><td>除外（1:40 が取れない）</td><td>{stat["excluded"]}</td></tr>'
                  f'<tr><td>ピストル</td><td>{stat.get("pistol", 0)}</td></tr>'
                  f'<tr><td>フェーズ分類対象（tracks あり）</td><td>{cls_n}</td></tr>'
                  f'<tr><td>フェーズ分類対象外（tracks 無し）</td><td>{sk_n}</td></tr>'
                  f'<tr><td>暫定バリアフィルタ除外サンプル</td><td>{bf.get("dropped", 0)}（自陣外 {bf.get("outside_home", 0)}・ジャンプ {bf.get("jump", 0)}）</td></tr></tbody></table>')
    tables.append('<table><thead><tr><th>設定値</th><th>値</th><th>意味</th></tr></thead><tbody>'
                  + "".join(f'<tr><td>{_e(k)}</td><td>{_e(v)}</td><td>{_e(d)}</td></tr>'
                            for k, v, d in SETTINGS) + "</tbody></table>")
    tables.append('<table><thead><tr><th>フェーズ区間</th><th>区間の切り方</th><th>判定する型</th></tr></thead><tbody>'
                  + "".join(f'<tr><td>{_e(seg)}</td><td>{_e(cut)}</td><td>{_e(tps)}</td></tr>'
                            for seg, cut, tps in PHASE_RULES) + "</tbody></table>")
    single_txt = "、".join(f"{s['name']}（{s['n']}）" for s in singles) or "なし"
    return _slide("付録", "付録",
                  _s(f"単発の1:40配置（1〜{PATTERN_MIN - 1} 回）：{single_txt}") +
                  _s(f"構成不明で外したラウンド（{ctx['map']} 全体）：{unknown}") +
                  f'<div class="appendix">{"".join(tables)}</div>', ctx["map"], cls="appendix")


def build_page(ctx, comp: str, rounds: list[dict], items: list[dict], stat: dict, comp_rows) -> tuple[str, dict]:
    comp_slug = re.sub(r"[^A-Za-z0-9]+", "-", comp).strip("-").lower() or "comp"
    ctx = {**ctx, "map": ctx["map"], "comp_slug": comp_slug}
    used_names: set[str] = set()
    def_pats, def_singles = build_patterns(items, ctx["model"], ctx["k"], DEFENSE, used_names)
    atk_pats, atk_singles = build_patterns(items, ctx["model"], ctx["k"], ATTACK, used_names)
    claims = _why_claims(rounds, items, ctx["base_rounds"], ctx["base_items"], ctx["model"])

    why_html, claims_meta = _why_slide(ctx, claims)
    slides = [_cover_slide(ctx, comp, stat), _env_slide(ctx, comp, comp_rows), why_html]

    sample_keys = set()
    for c in claims:
        if c.get("item"):
            sample_keys.add(c["item"]["r"]["comp_key"])

    used_samples: set[str] = set()
    placement_meta: list[dict] = []
    for side, pats in ((DEFENSE, def_pats), (ATTACK, atk_pats)):
        html_pl, meta_pl = _placement_slide(ctx, side, pats, items, used_samples)
        slides.append(html_pl)
        placement_meta.append(meta_pl)
    slides.append(_pattern_slide(ctx, def_pats, atk_pats))
    slides.append(_role_slide(ctx, comp, items))

    # サイド別（攻撃・防衛）× 序盤・中盤・終盤の3部構成（上位3型まで）
    phase_sections_meta = []
    side_blocks_meta = {}
    hands_compat_meta = []
    for side_key in (ATTACK, DEFENSE):
        phase_data = build_phase_sections(items, side=side_key, max_types=3, k=ctx["k"])
        block_sections_meta = []
        for sec in phase_data["sections"]:
            sec_types_meta = []
            if not sec["types"]:
                slides.append(_phase_empty_slide(ctx, sec, phase_data["total"], phase_data["classified"]))
            else:
                for idx, t_info in enumerate(sec["types"], 1):
                    s_html, t_meta = _phase_type_slide(
                        ctx, sec, t_info, idx, phase_data["total"], phase_data["classified"]
                    )
                    slides.append(s_html)
                    sec_types_meta.append(t_meta)
                    for rep_it in t_info["reps"]:
                        sample_keys.add(rep_it["r"]["comp_key"])
                    if t_meta["representatives"]:
                        lead_rep = t_meta["representatives"][0]
                        hands_compat_meta.append({
                            "side": lead_rep["side"],
                            "id": f"{side_key}:{sec['seg']}:{t_info['type']}",
                            "name": t_meta["title"],
                            "n": t_info["n"],
                            "w": t_info["w"],
                            "rate": t_meta["rate"],
                            "sample_round": lead_rep["round"],
                            "sample_full5": True,
                            "sample_status": "ok",
                            "sample_team": lead_rep["team"],
                            "sample_opp": lead_rep["opp"],
                            "sample_label": lead_rep["label"],
                        })
            sec_meta = {
                "side": side_key,
                "seg": sec["seg"],
                "label": sec["label"],
                "section_title": SIDE_SECTION_TITLES[side_key][sec["seg"]],
                "typed_rounds": sec["typed_rounds"],
                "untyped_rounds": sec["untyped_rounds"],
                "types": sec_types_meta,
            }
            block_sections_meta.append(sec_meta)
            phase_sections_meta.append(sec_meta)
        side_blocks_meta[side_key] = {
            "side": side_key,
            "label": side_key,
            "total": phase_data["total"],
            "classified": phase_data["classified"],
            "skipped": phase_data["skipped"],
            "sections": block_sections_meta,
        }

    slides.append(_appendix_slide(ctx, stat, def_singles + atk_singles, ctx["unknown"], def_pats, atk_pats))

    for p in def_pats + atk_pats:
        sample_keys.add(p["medoid"]["r"]["comp_key"])

    body = "".join(s for s in slides if s)
    page = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e("・".join(agents_of(comp)))} {_e(ctx["map"])} プレイブック v5</title>'
            f'<style>{ctx["css"]}</style></head><body>{ctx["fig"].defs()}'
            f'<main class="deck"><div class="toolbar"><p>{_e("・".join(agents_of(comp)))} の {_e(ctx["map"])}'
            f'</p><p>印刷すると 1 スライド 1 ページ</p></div>{body}</main></body></html>')
    manifest = {
        "key": comp,
        "agents": agents_of(comp),
        "total": stat["total"],
        "used_def": stat["used_def"],
        "used_atk": stat["used_atk"],
        "excluded": stat["excluded"],
        "full5": stat["full5"],
        "flow_classified": stat["flow_classified"],
        "flow_skipped": stat["flow_skipped"],
        "barrier_filter": stat.get("barrier_filter") or {"outside_home": 0, "jump": 0, "dropped": 0},
        "dates": stat["dates"],
        "teams": stat["teams"],
        "patterns": [{"side": p["side"], "name": p["name"], "n": p["n"], "w": p["w"]}
                     for p in def_pats + atk_pats],
        "hands": hands_compat_meta,
        "placements": placement_meta,
        "claims": claims_meta,
        "side_blocks": side_blocks_meta,
        "phase_sections": phase_sections_meta,
        "sample_comp_keys": sorted(k for k in sample_keys if k),
    }
    return _bump_html(page), manifest


# ============================================================== index
def _index_page(ctx, pages: list[dict], skipped: list[dict], old: list[dict]) -> str:
    rows = []
    for p in sorted(pages, key=lambda x: -x["total"]):
        wr = p["wins"] / p["total"] if p["total"] else 0
        rows.append(f'<tr><td>{"・".join(p["agents"])}</td><td>{_e("・".join(p["teams"]))}</td><td>{p["total"]}</td>'
                    f'<td>{wr:.0%}</td><td>{p.get("flow_classified", 0)}/{p["total"]}</td>'
                    f'<td>{p["dates"][-1] if p["dates"] else "-"}</td>'
                    f'<td><a href="{html.escape(p["file"], quote=True)}">ページ</a></td></tr>')
    main_table = ('<table><thead><tr><th>5人構成</th><th>チーム</th><th>ラウンド</th><th>勝率</th>'
                  '<th>フェーズ分類</th><th>直近</th><th>リンク</th></tr></thead>'
                  f'<tbody>{"".join(rows)}</tbody></table>')
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
.boards3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start;margin:8px 0}
.board-card{border:1px solid var(--line-weak);border-radius:var(--radius);overflow:hidden;background:var(--bg-sub)}
.board-card img.tactival-img{display:block;width:100%;height:auto}
.board-card .board-note{padding:16px 12px;font-size:14px;line-height:1.5;color:var(--text-sub);background:var(--bg)}
.board-card figcaption{font-size:14px;padding:4px 8px;font-weight:700}
.phase-body{display:grid;grid-template-columns:1fr minmax(260px,420px);gap:16px;align-items:start;margin-top:8px}
.rep-media{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;align-items:start}
.play-gif,.vod{margin:0;border:1px solid var(--line-weak);border-radius:var(--radius);overflow:hidden;background:var(--bg-sub)}
.play-gif img,.vod img{display:block;width:100%;height:auto}
.play-gif figcaption,.vod figcaption{font-size:14px;padding:3px 8px}
.placement-wrap{display:grid;gap:12px}
.handtxt ul{margin:4px 0 0;padding-left:1.2em}
.handtxt li{font-size:14px;line-height:1.7;margin:0 0 4px}
.handtxt .s{margin:0 0 6px}
.rolewrap{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
.appendix{display:grid;gap:10px}
.appendix table{font-size:14px}
figure{margin:0}
@media (max-width:900px){.boards3,.phase-body,.claims,.rolewrap{grid-template-columns:1fr}}
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

    ranks = ctx0.get("ranks")
    env_stats = {k: comp_stats(v, _items_for(v, allr, ctx0["model"], ranks=ranks)) for k, v in env_by.items()}
    comp_rows = sorted(env_stats.items(), key=lambda kv: -kv[1]["total"])

    base_items = _items_for(base_rounds, allr, ctx0["model"], ranks=ranks)
    base_stat = comp_stats(base_rounds, base_items)
    ctx = {**ctx0, "map": map_name, "out": out, "base_rounds": base_rounds, "base_items": base_items,
            "unknown": unknown, "css": _css()}

    pages, skipped, page_meta = [], [], []
    try:
        for comp in sorted(env_by, key=lambda k: -len(env_by[k])):
            rounds = env_by[comp]
            items = _items_for(rounds, allr, ctx0["model"], ranks=ranks)
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
                          "wins": stat["wins"], "flow_classified": stat["flow_classified"],
                          "dates": stat["dates"], "file": fn})
            log(f"  ページを作成: {fn}（全{stat['total']}R / フェーズ分類{stat['flow_classified']}R）")
    finally:
        BI.close()

    old_meta = [{"key": k, "agents": agents_of(k), "total": len(v),
                 "teams": sorted({str(r.get("tag") or r.get("team")) for r in v})} for k, v in old_by.items()]
    (out / "index.html").write_text(_index_page(ctx, pages, skipped, old_meta), encoding="utf-8")

    manifest = {"map": map_name, "env_start": ENV_START, "env_label": ENV_LABEL,
                "settings": [{"name": k, "value": v, "note": d} for k, v, d in SETTINGS],
                "barrier_filter": base_stat.get("barrier_filter") or {"outside_home": 0, "jump": 0, "dropped": 0},
                "unknown_rounds": unknown, "pages": page_meta, "skipped": skipped, "old_env": old_meta}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"プレイブック v5: {out / 'index.html'}")
    return out / "index.html"
