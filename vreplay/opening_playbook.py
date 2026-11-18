"""LOUD Summit 分類データを基準に、他マップの初動配置・TactiVal 図解・HTML プレイブックを生成する。

活用するリポジトリ内資産:
  - `config/tactival-barriers/all-maps-barrier-coordinates.json`（全 13 マップの ATK/DEF バリア座標）
  - `vreplay/tactival.py` & `vreplay/data/tactival_template.json`（TactiVal v3 JSON & 座標変換）
  - `vreplay/kb.py` & `config/kb/`（エージェント特性・構成メモ・マップメタ・戦術概念）
  - `vreplay/tactics.py`（CALLOUT_FIX・マップメタ変換）
  - `playbook_JSON/loud_summit_classified (1).json`（LOUD Summit 分類・構成メモ・TactiVal の基準モデル）
"""
from __future__ import annotations

import base64
import copy
import html as _html
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import tactival_barriers as BR
from . import kb as KB
from . import tactival as TV

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_JSON = ROOT / "playbook_JSON" / "loud_summit_classified (1).json"

DEFAULT_COMP = ["Chamber", "Fade", "Neon", "Omen", "Sage"]

# ------------------------------------------------------------------ マップ知識プロファイル
# TactiVal の 1024x1024 回転座標系では、全 13 マップで x 小＝B/C 側、x 中＝Mid、x 大＝A 側。
# 12 マップは y 大＝ATK・y 小＝DEF、Fracture のみ ATK が上下（北・南）・DEF が中央。
MAP_PROFILES: dict[str, dict[str, Any]] = {
    "summit": {
        "display": "Summit",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MID",
        "link_a": "Aリンク",
        "link_b": "Bリンク",
        "post_b": "ドロップ",
        "post_a": "CT/リンク",
        "smoke_mid": "MID膨らみモク",
        "smoke_bottom": "ボトムモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "MIDボトムと両メインの主導権からリンク壁セット・スプリットを展開する2サイトマップ。",
    },
    "sunset": {
        "display": "Sunset",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDトップ",
        "link_a": "Aリンク",
        "link_b": "Bマーケット",
        "post_b": "ボバ(タピオカ)/メイン",
        "post_a": "エルボー/リンク",
        "smoke_mid": "MIDトップ膨らみモク",
        "smoke_bottom": "ボトムモク",
        "default_comp": ["Chamber", "Neon", "Omen", "Sage", "Sova"],
        "summary_note": "Aメインの速い奪取とMIDトップ壁→マーケット(B)/Aリンクへのスプリットが軸。",
    },
    "split": {
        "display": "Split",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDメール/ベント",
        "link_a": "Aランプ",
        "link_b": "ベント",
        "post_b": "Bガレージ/バック",
        "post_a": "Aランプ/ヘブン",
        "smoke_mid": "MIDメール膨らみモク",
        "smoke_bottom": "メール/ベントモク",
        "default_comp": ["Fade", "Neon", "Omen", "Viper", "Yoru"],
        "summary_note": "MID（ベント・メール）の支配権がローテ速度を決め、Aランプ・Bガレージの挟みが強力。",
    },
    "ascent": {
        "display": "Ascent",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDキャット/トップ",
        "link_a": "Aショート(ツリー)",
        "link_b": "Bマーケット",
        "post_b": "ボート/メイン",
        "post_a": "ワイン/ツリー",
        "smoke_mid": "MIDキャット膨らみモク",
        "smoke_bottom": "MIDトップモク",
        "default_comp": ["Chamber", "Neon", "Omen", "Sage", "Sova"],
        "summary_note": "MIDキャット/マーケットの支配からAショート壁・Bマーケット挟みへ繋げる定番2サイトマップ。",
    },
    "bind": {
        "display": "Bind",
        "sites": ["A", "B"],
        "has_mid": False,
        "main_b": "Bロング",
        "main_a": "Aショート",
        "mid_take": "Bフッカー",
        "mid_ctrl": "Aシャワー",
        "link_a": "Aランプ",
        "link_b": "Bフッカー",
        "post_b": "フッカー/エルボー",
        "post_a": "ランプ/シャワー",
        "smoke_mid": "Aシャワー膨らみモク",
        "smoke_bottom": "フッカーモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "Mid無し・TP有り。Bロング/フッカーとAショート/シャワーの2レーンずつを同時に圧迫する。",
    },
    "breeze": {
        "display": "Breeze",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDピラー/ネスト",
        "link_a": "Aホール/ドア",
        "link_b": "Bトンネル/エルボー",
        "post_b": "トンネル/メイン",
        "post_a": "ピラミッド/ケイブ",
        "smoke_mid": "MIDネスト膨らみモク",
        "smoke_bottom": "MIDドアモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "射線が長い広大マップ。MIDのドア/ネスト制圧とAメイン/Bメインの遠距離ピックが軸。",
    },
    "abyss": {
        "display": "Abyss",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDキャットウォーク",
        "link_a": "Aタワー/リンク",
        "link_b": "Bリンク",
        "post_b": "デンジャー/メイン",
        "post_a": "ブリッジ/セキュリティ",
        "smoke_mid": "MIDキャット膨らみモク",
        "smoke_bottom": "MIDボトムモク",
        "default_comp": ["Astra", "Harbor", "Neon", "Sova", "Waylay"],
        "summary_note": "落下死エリアと長いMID構造。MIDコントロールとメイン壁でサイト内の射線を切る。",
    },
    "corrode": {
        "display": "Corrode",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "MIDボトム",
        "mid_ctrl": "MIDトップ",
        "link_a": "Aリンク",
        "link_b": "Bリンク",
        "post_b": "Bエルボー/メイン",
        "post_a": "Aエルボー/リンク",
        "smoke_mid": "MID膨らみモク",
        "smoke_bottom": "ボトムモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "MIDの縦ラインと両リンクのつながりが強く、リンク封鎖壁＋メインの挟み込みが有効。",
    },
    "fracture": {
        "display": "Fracture",
        "sites": ["A", "B"],
        "has_mid": False,
        "main_b": "Bメイン",
        "main_a": "Aメイン",
        "mid_take": "Bアーケード",
        "mid_ctrl": "Aディッシュ",
        "link_a": "Aドロップ/サンド",
        "link_b": "Bタワー/ジェネ",
        "post_b": "タワー/メイン",
        "post_a": "ドロップ/サンド",
        "smoke_mid": "Aディッシュ膨らみモク",
        "smoke_bottom": "アーケードモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "ATK が上下2方向から挟むH型マップ。Bメイン+アーケード、Aメイン+ディッシュの連動が肝。",
    },
    "haven": {
        "display": "Haven",
        "sites": ["A", "B", "C"],
        "has_mid": True,
        "main_b": "Cロング",
        "main_a": "Aメイン",
        "mid_take": "MID/ガレージ",
        "mid_ctrl": "MIDウィンドウ",
        "link_a": "Aショート",
        "link_b": "Cガレージ",
        "post_b": "Cロング/ガレージ",
        "post_a": "Aショート/ヘブン",
        "smoke_mid": "MIDウィンドウ膨らみモク",
        "smoke_bottom": "ガレージモク",
        "default_comp": ["Chamber", "Neon", "Omen", "Sova", "Viper"],
        "summary_note": "3サイト構造。Cロング・ガレージ・Aロビーの初動圧から守りのローテを遅らせ B/C/A を突く。",
    },
    "icebox": {
        "display": "Icebox",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bメイン(イエロー)",
        "main_a": "Aメイン(ベルト)",
        "mid_take": "MIDボイラー",
        "mid_ctrl": "チューブ/キッチン",
        "link_a": "Aメイズ/パイプ",
        "link_b": "Bキッチン/スノーマン",
        "post_b": "イエロー/キッチン",
        "post_a": "ベルト/サイト上",
        "smoke_mid": "MIDボイラー膨らみモク",
        "smoke_bottom": "チューブ下モク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "Aメインの立体戦とBイエロー＋チューブ/キッチンの挟み込み・設置後遅延が強いマップ。",
    },
    "lotus": {
        "display": "Lotus",
        "sites": ["A", "B", "C"],
        "has_mid": True,
        "main_b": "Cマウンド",
        "main_a": "Aメイン(ラブル)",
        "mid_take": "Bメイン",
        "mid_ctrl": "Bピラー/ドア",
        "link_a": "Aツリー/ドア",
        "link_b": "Cリンク(滝)",
        "post_b": "Cマウンド/ウォーター",
        "post_a": "Aツリー/ドロップ",
        "smoke_mid": "Bピラー膨らみモク",
        "smoke_bottom": "Aラブル膨らみモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Viper"],
        "summary_note": "3サイト＋回転扉。Aラブル奪取が試合の主導権になり、Cマウンド/Bドアの連動で防衛を割る。",
    },
    "pearl": {
        "display": "Pearl",
        "sites": ["A", "B"],
        "has_mid": True,
        "main_b": "Bロング",
        "main_a": "Aメイン",
        "mid_take": "MIDショップ",
        "mid_ctrl": "MIDアート/プラザ",
        "link_a": "Aアート",
        "link_b": "Bリンク(コネクター)",
        "post_b": "Bロング/ホール",
        "post_a": "ダグアウト/アート",
        "smoke_mid": "MIDプラザ膨らみモク",
        "smoke_bottom": "ショップ/ドアモク",
        "default_comp": ["Chamber", "Fade", "Neon", "Omen", "Sage"],
        "summary_note": "Bロングの正面衝突とMID→Aアート/Bリンクの挟み込みが鍵を握る2サイトマップ。",
    },
}

# ------------------------------------------------------------------ エージェント・スキル定義
AGENT_SKILLS: dict[str, dict[str, str]] = {
    "Chamber": {"role_ja": "センチネル/オペ", "recon": "トレードマーク", "anchor": "チェンバー", "take": "TP(ランデブー)", "color": "#c9a653"},
    "Sage": {"role_ja": "センチネル/壁", "wall": "セージ壁", "slow": "スローオーブ", "anchor": "セージ", "color": "#45c4b0"},
    "Cypher": {"role_ja": "センチネル/罠", "wall": "サイバーケージ", "recon": "カメラ", "anchor": "サイファー", "color": "#d9d9d9"},
    "Killjoy": {"role_ja": "センチネル/遅延", "slow": "ナノスワーム", "recon": "タレット", "anchor": "キルジョイ", "color": "#e5c84b"},
    "Deadlock": {"role_ja": "センチネル/壁", "wall": "バリアメッシュ", "slow": "グラヴネット", "anchor": "デッドロック", "color": "#7b8fa1"},
    "Vyse": {"role_ja": "センチネル/壁", "wall": "シアー(壁)", "slow": "レイザーバイン", "anchor": "ヴァイス", "color": "#9466ff"},
    "Veto": {"role_ja": "センチネル/抑制", "wall": "インターセプター", "slow": "チョークホールド", "anchor": "ヴェト", "color": "#5588aa"},
    "Fade": {"role_ja": "イニシエーター/索敵", "recon": "プラウラー", "recon2": "ホウント", "slow": "シーズ", "color": "#6a5acd"},
    "Sova": {"role_ja": "イニシエーター/索敵", "recon": "オウルドローン", "recon2": "リコンボルト", "slow": "ショックダーツ", "color": "#3d85c6"},
    "Skye": {"role_ja": "イニシエーター/索敵", "recon": "トレイルブレイザー(犬)", "recon2": "フラッシュ", "slow": "フラッシュ", "color": "#5cb85c"},
    "KAY/O": {"role_ja": "イニシエーター/抑制", "recon": "ゼロポイント(ナイフ)", "recon2": "フラッシュ", "slow": "フラグメント", "color": "#4682b4"},
    "Breach": {"role_ja": "イニシエーター/CC", "recon": "フォールトライン", "recon2": "フラッシュ", "slow": "アフターショック", "color": "#cd661d"},
    "Gekko": {"role_ja": "イニシエーター/索敵", "recon": "ウィングマン", "recon2": "ディジー", "slow": "モッシュピット", "color": "#9acd32"},
    "Tejo": {"role_ja": "イニシエーター/爆撃", "recon": "ステルスドローン", "recon2": "ガイドサルヴォ", "slow": "スペシャルデリバリー", "color": "#b8860b"},
    "Omen": {"role_ja": "コントローラー/モク", "smoke": "スモーク", "take": "シュラステ", "flash": "パラノイア", "color": "#483d8b"},
    "Viper": {"role_ja": "コントローラー/壁", "smoke": "ポイズンオーブ", "wall": "トキシックカーテン", "slow": "スネークバイト", "color": "#2e8b57"},
    "Astra": {"role_ja": "コントローラー/モク", "smoke": "ネビュラ(モク)", "slow": "グラビティウェル", "wall": "コズミックディバイド", "color": "#8a2be2"},
    "Brimstone": {"role_ja": "コントローラー/モク", "smoke": "スカイスモーク", "slow": "インセンディアリー", "wall": "スモーク壁", "color": "#d2691e"},
    "Harbor": {"role_ja": "コントローラー/壁", "smoke": "コーヴ", "wall": "ハイタイド(水壁)", "slow": "カスケード", "color": "#20b2aa"},
    "Clove": {"role_ja": "コントローラー/モク", "smoke": "ルーズ(モク)", "slow": "メドル", "take": "ピックミーアップ", "color": "#db7093"},
    "Miks": {"role_ja": "コントローラー/モク", "smoke": "スモーク", "slow": "音波妨害", "color": "#5f9ea0"},
    "Neon": {"role_ja": "デュエリスト/エントリー", "entry": "スライディング", "wall": "ファストレーン", "slow": "リレーボルト", "color": "#00bfff"},
    "Jett": {"role_ja": "デュエリスト/エントリー", "entry": "テイルウィンド(ダッシュ)", "smoke": "クラウドバースト", "take": "アップドラフト", "color": "#87ceeb"},
    "Raze": {"role_ja": "デュエリスト/エントリー", "entry": "ブラストパック", "recon": "ブームボット", "slow": "ペイント弾", "color": "#ff8c00"},
    "Yoru": {"role_ja": "デュエリスト/欺瞞", "entry": "ゲートクラッシュ", "recon": "フェイクアウト", "flash": "ブラインドサイド", "color": "#4169e1"},
    "Phoenix": {"role_ja": "デュエリスト/エントリー", "entry": "フラッシュ＋エントリー", "wall": "ブレイズ", "slow": "ホットハンズ", "color": "#ff4500"},
    "Reyna": {"role_ja": "デュエリスト/エントリー", "entry": "リーア(目)", "take": "デバウアー", "color": "#9932cc"},
    "Iso": {"role_ja": "デュエリスト/撃ち合い", "entry": "コンティンジェンシー(壁)", "slow": "アンダーカット", "color": "#7b68ee"},
    "Waylay": {"role_ja": "デュエリスト/高速移動", "entry": "ライトスピード", "slow": "サチュレート", "color": "#da70d6"},
}

CATEGORY_ORDER = ["序盤", "スモーク", "索敵", "形", "中盤", "エクセキューション", "ポストプラント", "全体"]


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _canonical_agent(name: str, kb: KB.KB | None = None) -> str:
    k = _norm(name)
    if kb is not None:
        info = kb.agent(name)
        if info.get("name"):
            return info["name"]
    for ag in AGENT_SKILLS:
        if _norm(ag) == k:
            return ag
    return str(name).strip()


def list_maps() -> list[str]:
    """対応する全 13 マップ名（正規名）のリストを返す。"""
    return [v["display"] for v in MAP_PROFILES.values()]


# ------------------------------------------------------------------ キャッシュ探索（オフライン完結）
def _candidate_cache_dirs(cfg: dict | None = None) -> list[Path]:
    dirs: list[Path] = []
    out_dir = Path((cfg or {}).get("output", {}).get("dir") or ".")
    for cand in (
        out_dir / "_cache",
        ROOT / "_cache",
        Path("C:/Users/Aokki/pbv5_local_out/_cache"),
        Path("C:/Users/Aokki/valorant-replay-output/_cache"),
        Path("G:/マイドライブ/valorant-replay/_cache"),
    ):
        if cand.exists() and cand not in dirs:
            dirs.append(cand)
    return dirs


def _load_cached_map_image(map_name: str, cache_dirs: list[Path] | None = None) -> np.ndarray | None:
    """ローカルキャッシュにある公式マップ画像（RGBA）を TactiVal の向きに回転して返す。無ければ None。"""
    mid = _norm(map_name)
    rot_k = TV.MAP_ROT.get(mid, 0)
    search_dirs = cache_dirs if cache_dirs is not None else _candidate_cache_dirs()
    for cd in search_dirs:
        p = cd / "map_icons" / f"{mid}.png"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] >= 3:
                if img.shape[2] == 3:
                    a = np.full(img.shape[:2] + (1,), 255, np.uint8)
                    img = np.concatenate([img, a], axis=2)
                if rot_k % 4:
                    img = np.rot90(img, k=rot_k % 4).copy()
                if img.shape[:2] != (1024, 1024):
                    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
                return img
    return None


def _load_cached_agent_icons(agents: list[str], cache_dirs: list[Path] | None = None) -> dict[str, np.ndarray]:
    """ローカルキャッシュにあるエージェントアイコンを読み込む（無ければ空 dict）。"""
    search_dirs = cache_dirs if cache_dirs is not None else _candidate_cache_dirs()
    out: dict[str, np.ndarray] = {}
    for ag in agents:
        k = _norm(ag)
        for cd in search_dirs:
            p = cd / "agent_icons" / f"{k}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    out[ag] = img
                    break
    return out


def _load_cached_callouts(map_name: str, cache_dirs: list[Path] | None = None) -> list[dict]:
    """ローカルの valorant_maps.json または tests/fixtures から TactiVal 1024x1024 座標のコールアウトを読む。"""
    mid = _norm(map_name)
    cal = TV.exact_calib(mid)
    if not cal:
        return []
    from . import tactics as TC

    fix = TC.CALLOUT_FIX.get(mid)
    search_dirs = cache_dirs if cache_dirs is not None else _candidate_cache_dirs()
    raw_map = None
    for cd in search_dirs:
        p = cd / "valorant_maps.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text("utf-8"))
            raw_map = next((x for x in data if _norm(x.get("displayName")) == mid), None)
            if raw_map:
                break
        except Exception:
            continue

    calls: list[dict] = []
    if raw_map and raw_map.get("callouts"):
        xm, ym = float(raw_map.get("xMultiplier") or 0), float(raw_map.get("yMultiplier") or 0)
        xa, ya = float(raw_map.get("xScalarToAdd") or 0), float(raw_map.get("yScalarToAdd") or 0)
        for c in raw_map["callouts"]:
            loc = c.get("location") or {}
            u = float(loc.get("y") or 0) * xm + xa if (xm or ym) else float(loc.get("x") or 0)
            v = float(loc.get("x") or 0) * ym + ya if (xm or ym) else float(loc.get("y") or 0)
            if fix and (xm or ym):
                u, v = fix[0] * u + fix[1] * v + fix[2], fix[3] * u + fix[4] * v + fix[5]
            bx, by = TV.to_px(u, v, cal)
            calls.append({
                "name": c.get("regionName") or "",
                "super": c.get("superRegionName") or "",
                "x": bx - TV.MAP_ORIGIN[0],
                "y": by - TV.MAP_ORIGIN[1],
            })
        return calls

    # tests/fixtures のフォールバック（Split / Sunset）
    fx_p = ROOT / "tests" / "fixtures" / f"{mid}_callouts.json"
    if fx_p.exists():
        try:
            m = json.loads(fx_p.read_text("utf-8"))
            xm, ym = float(m.get("xMultiplier") or 0), float(m.get("yMultiplier") or 0)
            xa, ya = float(m.get("xScalarToAdd") or 0), float(m.get("yScalarToAdd") or 0)
            for name, sup, x, y in m.get("callouts") or []:
                u, v = float(y) * xm + xa, float(x) * ym + ya
                if fix:
                    u, v = fix[0] * u + fix[1] * v + fix[2], fix[3] * u + fix[4] * v + fix[5]
                bx, by = TV.to_px(u, v, cal)
                calls.append({"name": name, "super": sup, "x": bx - TV.MAP_ORIGIN[0], "y": by - TV.MAP_ORIGIN[1]})
        except Exception:
            pass
    return calls


# ------------------------------------------------------------------ バリア座標解析 & アンカー生成
def load_tactival_barriers(map_name: str, path: Path | None = None) -> dict[str, list[dict]]:
    """TactiVal 1024x1024 座標の ATK / DEF バリア情報を返す。"""
    p = BR.barriers_path(path)
    mid = _norm(map_name)
    if p is None or not mid:
        return {"ATK": [], "DEF": []}
    try:
        raw = json.loads(p.read_text("utf-8"))
    except Exception:
        return {"ATK": [], "DEF": []}
    items = (raw.get("maps") or {}).get(mid) or []
    out: dict[str, list[dict]] = {"ATK": [], "DEF": []}
    for b in items:
        tm = b.get("team")
        pts = b.get("vectorPoints") or []
        if tm in out and len(pts) >= 2:
            cx = sum(float(pt[0]) for pt in pts) / len(pts)
            cy = sum(float(pt[1]) for pt in pts) / len(pts)
            out[tm].append({"pts": [(float(pt[0]), float(pt[1])) for pt in pts], "cx": cx, "cy": cy})
    for tm in out:
        out[tm].sort(key=lambda d: (d["cx"], d["cy"]))
    return out


def _snap_to_floor(x: float, y: float, floor_mask: np.ndarray | None, max_r: int = 90) -> tuple[float, float]:
    """floor_mask (1024x1024) があるとき、近い有効床セルへスナップする。"""
    x = float(min(970.0, max(54.0, x)))
    y = float(min(970.0, max(54.0, y)))
    if floor_mask is None or not floor_mask.any():
        return x, y
    H, W = floor_mask.shape
    ix, iy = min(W - 1, max(0, int(round(x)))), min(H - 1, max(0, int(round(y))))
    if floor_mask[iy, ix]:
        return x, y
    y0, y1 = max(0, iy - max_r), min(H, iy + max_r + 1)
    x0, x1 = max(0, ix - max_r), min(W, ix + max_r + 1)
    sub = floor_mask[y0:y1, x0:x1]
    if not sub.any():
        ys, xs = np.nonzero(floor_mask)
        idx = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
        return float(xs[idx]), float(ys[idx])
    ys, xs = np.nonzero(sub)
    idx = int(np.argmin((xs + x0 - ix) ** 2 + (ys + y0 - iy) ** 2))
    return float(xs[idx] + x0), float(ys[idx] + y0)


def _callout_xy(calls: list[dict], keywords: list[str], super_filter: str | None = None) -> tuple[float, float] | None:
    """コールアウト一覧からキーワード一致する位置を返す。"""
    for c in calls:
        sup = str(c.get("super") or "")
        nm = str(c.get("name") or "")
        full = f"{sup} {nm}".lower()
        if super_filter and super_filter.lower() not in sup.lower():
            continue
        if any(kw.lower() in full for kw in keywords):
            return float(c["x"]), float(c["y"])
    return None


def barrier_anchors(
    map_name: str,
    floor_mask: np.ndarray | None = None,
    calls: list[dict] | None = None,
    barrier_path: Path | None = None,
) -> dict[str, Any]:
    """全 13 マップの ATK/DEF バリア座標とコールアウトから、初動配置アンカー群 (1024x1024) を算出する。"""
    mid = _norm(map_name)
    bars = load_tactival_barriers(mid, barrier_path)
    calls = calls if calls is not None else []
    atk_b = bars.get("ATK") or []
    def_b = bars.get("DEF") or []

    # スポーン基準の向きを判定
    is_fracture = mid == "fracture"
    atk_avg_y = sum(b["cy"] for b in atk_b) / len(atk_b) if atk_b else 680.0
    def_avg_y = sum(b["cy"] for b in def_b) / len(def_b) if def_b else 430.0
    # 通常は ATK が下 (y 大), DEF が上 (y 小)
    atk_sign = 1.0 if atk_avg_y >= def_avg_y else -1.0

    def _pick_x_slice(lst: list[dict], frac_lo: float, frac_hi: float) -> list[dict]:
        if not lst:
            return []
        xs = [b["cx"] for b in lst]
        lo = min(xs) + (max(xs) - min(xs)) * frac_lo
        hi = min(xs) + (max(xs) - min(xs)) * frac_hi
        sub = [b for b in lst if lo <= b["cx"] <= hi]
        return sub or lst

    def _atk_anchor_from(bar_list: list[dict], fallback: tuple[float, float], back_dy: float = 52.0) -> tuple[float, float]:
        if not bar_list:
            return _snap_to_floor(*fallback, floor_mask)
        b = bar_list[len(bar_list) // 2]
        if is_fracture:
            # Fracture は上端 (cy < 450) のバリアは上へ (-back_dy), 下端は下へ (+back_dy)
            dy = -back_dy if b["cy"] < 450 else back_dy
        else:
            dy = atk_sign * back_dy
        return _snap_to_floor(b["cx"], b["cy"] + dy, floor_mask)

    def _def_anchor_from(bar_list: list[dict], fallback: tuple[float, float], back_dy: float = 48.0) -> tuple[float, float]:
        if not bar_list:
            return _snap_to_floor(*fallback, floor_mask)
        b = bar_list[len(bar_list) // 2]
        if is_fracture:
            # Fracture の DEF は中央側へ寄せる
            dy = 25.0 if b["cy"] < 480 else -25.0
            dx = 40.0 if b["cx"] < 450 else -40.0
            return _snap_to_floor(b["cx"] + dx, b["cy"] + dy, floor_mask)
        return _snap_to_floor(b["cx"], b["cy"] - atk_sign * back_dy, floor_mask)

    # ATK レーン別バリア
    if is_fracture:
        # Fracture: 左が B 側 (x < 480)、右が A 側 (x >= 480)
        atk_b_side = [b for b in atk_b if b["cx"] < 480]
        atk_a_side = [b for b in atk_b if b["cx"] >= 480]
        atk_b_main = max(atk_b_side, key=lambda b: b["cy"], default=None)
        atk_b_arc = min(atk_b_side, key=lambda b: b["cy"], default=None)
        atk_a_main = max(atk_a_side, key=lambda b: b["cy"], default=None)
        atk_a_dish = min(atk_a_side, key=lambda b: b["cy"], default=None)
        atk_b_pt = _atk_anchor_from([atk_b_main] if atk_b_main else [], (280, 860), 40)
        atk_mid_pt = _atk_anchor_from([atk_b_arc] if atk_b_arc else [], (340, 140), 40)
        atk_a_pt = _atk_anchor_from([atk_a_main] if atk_a_main else [], (680, 890), 40)
        atk_a2_pt = _atk_anchor_from([atk_a_dish] if atk_a_dish else [], (660, 140), 40)
    else:
        atk_b_pt = _atk_anchor_from(_pick_x_slice(atk_b, 0.0, 0.34), (260, 680), 48)
        atk_mid_pt = _atk_anchor_from(_pick_x_slice(atk_b, 0.34, 0.68), (500, 720), 46)
        atk_a_pt = _atk_anchor_from(_pick_x_slice(atk_b, 0.68, 1.0), (740, 680), 48)
        atk_a2_pt = _atk_anchor_from(_pick_x_slice(atk_b, 0.55, 0.85), (660, 700), 65)

    atk_spawn = _callout_xy(calls, ["attacker side spawn", "spawn"], "attacker") or _snap_to_floor(512, 820 if not is_fracture else 850, floor_mask)
    def_spawn = _callout_xy(calls, ["defender side spawn", "spawn"], "defender") or _snap_to_floor(512, 160 if not is_fracture else 460, floor_mask)

    # サイト位置（コールアウト優先、無ければバリア位置から推定）
    site_b = _callout_xy(calls, ["site"], "B") or _callout_xy(calls, ["site"], "C") or _def_anchor_from(_pick_x_slice(def_b, 0.0, 0.30), (180, 360), 85)
    site_a = _callout_xy(calls, ["site"], "A") or _def_anchor_from(_pick_x_slice(def_b, 0.70, 1.0), (830, 380), 85)
    site_c = _callout_xy(calls, ["site"], "C")
    mid_top = _callout_xy(calls, ["mid top", "top", "vent", "mail", "market", "plaza", "courtyard", "pillar"]) or _def_anchor_from(_pick_x_slice(def_b, 0.35, 0.65), (500, 400), 35)
    mid_bot = _callout_xy(calls, ["mid bottom", "bottom", "tiles", "fountain", "shop"]) or (
        (atk_mid_pt[0] * 0.65 + mid_top[0] * 0.35), (atk_mid_pt[1] * 0.65 + mid_top[1] * 0.35)
    )
    a_link = _callout_xy(calls, ["link", "tree", "ramps", "short", "art", "heaven"], "A") or (
        site_a[0] * 0.65 + mid_top[0] * 0.35, site_a[1] * 0.7 + mid_top[1] * 0.3
    )
    b_link = _callout_xy(calls, ["link", "market", "garden", "drop", "garage", "hookah", "kitchen"], "B") or (
        site_b[0] * 0.65 + mid_top[0] * 0.35, site_b[1] * 0.7 + mid_top[1] * 0.3
    )
    a_main = _callout_xy(calls, ["main", "long", "showers", "rubble", "lobby"], "A") or (
        atk_a_pt[0] * 0.6 + site_a[0] * 0.4, atk_a_pt[1] * 0.55 + site_a[1] * 0.45
    )
    b_main = _callout_xy(calls, ["main", "long", "mound", "yellow", "garage"], "B") or (
        atk_b_pt[0] * 0.6 + site_b[0] * 0.4, atk_b_pt[1] * 0.55 + site_b[1] * 0.45
    )

    # DEF アンカー
    def_b_site = _snap_to_floor(*site_b, floor_mask)
    def_b_main = _def_anchor_from(_pick_x_slice(def_b, 0.0, 0.35), (260, 440), 35)
    def_mid = _def_anchor_from(_pick_x_slice(def_b, 0.35, 0.65), (500, 390), 30)
    def_a_main = _def_anchor_from(_pick_x_slice(def_b, 0.65, 1.0), (740, 440), 35)
    def_a_site = _snap_to_floor(*site_a, floor_mask)
    def_b_push = _snap_to_floor(b_main[0], b_main[1], floor_mask)
    def_a_push = _snap_to_floor(a_main[0], a_main[1], floor_mask)

    pts = {
        "atk_b": _snap_to_floor(*atk_b_pt, floor_mask),
        "atk_mid": _snap_to_floor(*atk_mid_pt, floor_mask),
        "atk_a": _snap_to_floor(*atk_a_pt, floor_mask),
        "atk_a2": _snap_to_floor(*atk_a2_pt, floor_mask),
        "atk_spawn": _snap_to_floor(*atk_spawn, floor_mask),
        "def_spawn": _snap_to_floor(*def_spawn, floor_mask),
        "b_main": _snap_to_floor(*b_main, floor_mask),
        "a_main": _snap_to_floor(*a_main, floor_mask),
        "mid_bot": _snap_to_floor(*mid_bot, floor_mask),
        "mid_top": _snap_to_floor(*mid_top, floor_mask),
        "a_link": _snap_to_floor(*a_link, floor_mask),
        "b_link": _snap_to_floor(*b_link, floor_mask),
        "site_b": def_b_site,
        "site_a": def_a_site,
        "site_c": _snap_to_floor(*site_c, floor_mask) if site_c else def_b_site,
        "def_b_site": def_b_site,
        "def_b_main": def_b_main,
        "def_mid": def_mid,
        "def_a_main": def_a_main,
        "def_a_site": def_a_site,
        "def_b_push": def_b_push,
        "def_a_push": def_a_push,
    }
    return {"barriers": bars, "points": pts, "is_fracture": is_fracture}


# ------------------------------------------------------------------ 構成ロール解析
def analyze_comp(comp: list[str] | None, kb: KB.KB | None = None) -> dict[str, Any]:
    """5 エージェント構成から役割・代表スキル名を解決する。"""
    kb = kb or KB.load()
    agents = [_canonical_agent(a, kb) for a in (comp or DEFAULT_COMP)]
    while len(agents) < 5:
        for d in DEFAULT_COMP:
            if d not in agents:
                agents.append(d)
                break
        else:
            break
    agents = agents[:5]

    by_role: dict[str, list[str]] = defaultdict(list)
    for ag in agents:
        role = kb.role(ag) or ("sentinel" if ag in ("Chamber", "Sage", "Cypher", "Killjoy", "Deadlock", "Vyse", "Veto")
                                else "controller" if ag in ("Omen", "Viper", "Astra", "Brimstone", "Harbor", "Clove", "Miks")
                                else "initiator" if ag in ("Fade", "Sova", "Skye", "KAY/O", "Breach", "Gekko", "Tejo")
                                else "duelist")
        by_role[role].append(ag)

    sents = by_role["sentinel"]
    ctrls = by_role["controller"]
    inits = by_role["initiator"]
    duels = by_role["duelist"]

    # Chamber がいれば op/anchor1 に優先、いなければ最初の sentinel か duelist
    op_anchor = "Chamber" if "Chamber" in agents else (sents[0] if sents else (duels[0] if duels else agents[0]))
    wall_anchor = next((a for a in ("Sage", "Viper", "Vyse", "Deadlock", "Harbor") if a in agents), None)
    anchor2 = wall_anchor or next((a for a in sents if a != op_anchor), None) or (ctrls[0] if ctrls else agents[1])
    smoke_ag = next((a for a in ("Omen", "Astra", "Brimstone", "Clove", "Harbor", "Viper", "Miks") if a in agents), ctrls[0] if ctrls else agents[2])
    recon_ag = inits[0] if inits else (next((a for a in agents if a not in (op_anchor, anchor2, smoke_ag)), agents[3]))
    entry_ag = duels[0] if duels else (next((a for a in agents if a not in (op_anchor, anchor2, smoke_ag, recon_ag)), agents[4]))

    # 役割が重複しないよう 5 人をスロットに割り振る
    used: list[str] = []
    def _pick(pref: str | None) -> str:
        if pref and pref not in used:
            used.append(pref)
            return pref
        for a in agents:
            if a not in used:
                used.append(a)
                return a
        return agents[0]

    s_op = _pick(op_anchor)
    s_anc = _pick(anchor2)
    s_smk = _pick(smoke_ag)
    s_rec = _pick(recon_ag)
    s_ent = _pick(entry_ag)

    def _sk(ag: str, key: str, default: str) -> str:
        return (AGENT_SKILLS.get(ag) or {}).get(key, default)

    return {
        "agents": agents,
        "by_role": dict(by_role),
        "slots": {"op": s_op, "anchor": s_anc, "smoke": s_smk, "recon": s_rec, "entry": s_ent},
        "skills": {
            "recon_b": _sk(s_rec, "recon", "索敵"),
            "recon_mid": _sk(s_rec, "recon2", _sk(s_rec, "recon", "索敵")),
            "recon_delay": _sk(s_rec, "slow", "遅延スキル"),
            "smoke": _sk(s_smk, "smoke", "スモーク"),
            "take_smk": _sk(s_smk, "take", _sk(s_smk, "smoke", "スキル")),
            "wall": _sk(s_anc, "wall", "壁/遅延"),
            "slow": _sk(s_anc, "slow", "スロウ/遅延"),
            "entry": _sk(s_ent, "entry", "エントリー"),
            "op_name": _sk(s_op, "anchor", s_op),
            "anc_name": _sk(s_anc, "anchor", s_anc),
        },
        "is_double_sentinel": len(sents) >= 2,
    }


# ------------------------------------------------------------------ ボード要素・TactiVal 図＆JSON 生成
def _to_board_xy(vx: float, vy: float) -> tuple[float, float]:
    return round(vx + TV.MAP_ORIGIN[0], 2), round(vy + TV.MAP_ORIGIN[1], 2)


def _agent_angle(p0: tuple[float, float], target: tuple[float, float]) -> float:
    return math.atan2(target[1] - p0[1], target[0] - p0[0])


def build_tactival_json(
    map_name: str,
    shape_name: str,
    side: str,
    placements: list[dict],
    arrows: list[dict] | None = None,
    tpl: dict | None = None,
) -> dict:
    """1 つの型から TactiVal v3 互換 JSON を生成する。"""
    tpl = tpl or TV.load_template()
    ids = TV.agent_ids(tpl)
    mid = TV.map_id(tpl, map_name) or _norm(map_name)
    out = {k: v for k, v in tpl.items() if k != "steps"}
    out["schemaVersion"] = tpl.get("schemaVersion", TV.SCHEMA)
    out["mapId"] = mid
    out["mapFlipped"] = False
    out["activeStepIndex"] = 0
    out["savedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    team = "attack" if side == "攻撃" else "defense"
    els: list[dict] = []
    for idx, pl in enumerate(placements):
        ag = pl["agent"]
        bx, by = _to_board_xy(pl["x"], pl["y"])
        el = TV.agent_element(ids.get(ag.lower()), round(bx - TV.AGENT_SIZE / 2, 2), round(by - TV.AGENT_SIZE / 2, 2), idx, team)
        el["data"]["directionAngle"] = round(float(pl.get("angle", -math.pi / 2 if team == "attack" else math.pi / 2)), 3)
        els.append(el)
    sample = TV.arrow_sample(tpl)
    for idx, ar in enumerate(arrows or []):
        x0, y0 = _to_board_xy(*ar["from"])
        x1, y1 = _to_board_xy(*ar["to"])
        ae = TV.arrow_element(x0, y0, x1, y1, 100 + idx, sample, ar.get("label"))
        ae["style"]["stroke"] = "#ff4655" if team == "attack" else "#2ec4b6"
        ae["style"]["strokeStyle"] = ar.get("style", "dashed")
        els.append(ae)

    out["steps"] = [{
        "id": TV._eid(),
        "label": shape_name,
        "transition": 1.5,
        "hold": 0.8,
        "elements": els,
        "zoneVisibility": {},
        "overrides": {},
    }]
    return out


def _draw_dashed_line(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: tuple[int, int, int],
                      thickness: int = 2, dash: int = 10, gap: int = 7) -> None:
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if dist < 1:
        return
    dx, dy = (p1[0] - p0[0]) / dist, (p1[1] - p0[1]) / dist
    cur = 0.0
    while cur < dist:
        nxt = min(dist, cur + dash)
        a = (int(round(p0[0] + dx * cur)), int(round(p0[1] + dy * cur)))
        b = (int(round(p0[0] + dx * nxt)), int(round(p0[1] + dy * nxt)))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        cur += dash + gap


def _draw_fallback_map_canvas(
    map_name: str,
    anchors: dict[str, Any],
    calls: list[dict],
) -> np.ndarray:
    """公式マップ画像キャッシュが無い環境向けの 1024x1024 TactiVal 風マップ下地。"""
    canvas = np.full((1024, 1024, 3), 255, np.uint8)
    pts = anchors["points"]
    bars = anchors["barriers"]

    # コリドーの骨格を接続線で描画し、ライトグレーの通路を作る
    lanes = [
        (pts["atk_spawn"], pts["atk_b"]),
        (pts["atk_spawn"], pts["atk_mid"]),
        (pts["atk_spawn"], pts["atk_a"]),
        (pts["atk_b"], pts["b_main"]),
        (pts["b_main"], pts["site_b"]),
        (pts["atk_mid"], pts["mid_bot"]),
        (pts["mid_bot"], pts["mid_top"]),
        (pts["mid_top"], pts["b_link"]),
        (pts["b_link"], pts["site_b"]),
        (pts["mid_top"], pts["a_link"]),
        (pts["a_link"], pts["site_a"]),
        (pts["atk_a"], pts["a_main"]),
        (pts["a_main"], pts["site_a"]),
        (pts["site_b"], pts["def_spawn"]),
        (pts["site_a"], pts["def_spawn"]),
        (pts["mid_top"], pts["def_spawn"]),
    ]
    if len(MAP_PROFILES.get(_norm(map_name), {}).get("sites", [])) >= 3:
        lanes += [(pts["site_c"], pts["mid_top"]), (pts["site_c"], pts["def_spawn"])]

    floor = np.zeros((1024, 1024), np.uint8)
    for p0, p1 in lanes:
        cv2.line(floor, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 255, 92, cv2.LINE_AA)
    for key in ("site_b", "site_a", "atk_spawn", "def_spawn"):
        p = pts[key]
        cv2.circle(floor, (int(p[0]), int(p[1])), 86, 255, -1, cv2.LINE_AA)

    canvas[floor > 0] = (182, 182, 182)
    # サイトはベージュ (#dfdcb9 → BGR 185, 220, 223)
    for key in ("site_b", "site_a"):
        p = pts[key]
        cv2.rectangle(canvas, (int(p[0]) - 68, int(p[1]) - 58), (int(p[0]) + 68, int(p[1]) + 58), (185, 218, 222), -1)

    # 薄い白縁
    edges = cv2.Canny(floor, 50, 150)
    canvas[edges > 0] = (235, 235, 235)

    # バリアライン
    for side_key, col in (("ATK", (120, 120, 235)), ("DEF", (185, 195, 110))):
        for b in bars.get(side_key) or []:
            poly = np.array([[int(round(x)), int(round(y))] for x, y in b["pts"]], np.int32)
            cv2.polylines(canvas, [poly], True, col, 2, cv2.LINE_AA)

    # サイト文字
    for lbl, p in (("B", pts["site_b"]), ("A", pts["site_a"])):
        cv2.putText(canvas, lbl, (int(p[0]) - 12, int(p[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (110, 110, 95), 2, cv2.LINE_AA)
    return canvas


def render_opening_board(
    map_name: str,
    side: str,
    placements: list[dict],
    skills: list[dict] | None = None,
    arrows: list[dict] | None = None,
    anchors: dict[str, Any] | None = None,
    calls: list[dict] | None = None,
    base_rgba: np.ndarray | None = None,
    agent_icons: dict[str, np.ndarray] | None = None,
    out_size: int = 640,
) -> str:
    """TactiVal 風の初動配置図を 1024x1024 で描き、out_size の PNG Base64 Data URI を返す。"""
    anchors = anchors or barrier_anchors(map_name)
    calls = calls or []
    agent_icons = agent_icons or {}

    if base_rgba is not None and base_rgba.shape[:2] == (1024, 1024):
        canvas = np.full((1024, 1024, 3), 255, np.uint8)
        a = base_rgba[:, :, 3:4].astype(np.float32) / 255.0
        # 公式画像は少し明るめ（白 35% 混ぜ）にして shape_1.png にトーンを合わせる
        fg = base_rgba[:, :, :3].astype(np.float32) * 0.72 + 255.0 * 0.28
        canvas = (fg * a + canvas.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    else:
        canvas = _draw_fallback_map_canvas(map_name, anchors, calls)

    # バリアラインをうっすら重ねる（どの位置が開幕ラインか一目で分かる）
    bars = anchors.get("barriers") or {}
    bar_overlay = canvas.copy()
    for side_key, col in (("ATK", (95, 95, 225)), ("DEF", (175, 195, 90))):
        for b in bars.get(side_key) or []:
            poly = np.array([[int(round(x)), int(round(y))] for x, y in b["pts"]], np.int32)
            cv2.polylines(bar_overlay, [poly], True, col, 3, cv2.LINE_AA)
    canvas = cv2.addWeighted(bar_overlay, 0.55, canvas, 0.45, 0)

    # スキル描画（半透明オーバーレイ）
    over = canvas.copy()
    for sk in skills or []:
        kind = sk.get("kind", "smoke")
        sx, sy = int(round(sk["x"])), int(round(sk["y"]))
        if kind == "smoke":
            r = int(sk.get("r", 48))
            cv2.circle(over, (sx, sy), r, (195, 215, 130), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), r, (165, 185, 80), 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 13, (60, 50, 175), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 13, (255, 255, 255), 2, cv2.LINE_AA)
        elif kind == "slow":
            r = int(sk.get("r", 34))
            cv2.circle(over, (sx, sy), r, (220, 165, 175), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), r, (185, 100, 125), 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 12, (70, 50, 165), -1, cv2.LINE_AA)
        elif kind == "wall":
            ang = float(sk.get("angle", 0.0))
            length = float(sk.get("len", 105))
            dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
            nx, ny = -math.sin(ang) * 10, math.cos(ang) * 10
            poly = np.array([
                [int(round(sx - dx - nx)), int(round(sy - dy - ny))],
                [int(round(sx + dx - nx)), int(round(sy + dy - ny))],
                [int(round(sx + dx + nx)), int(round(sy + dy + ny))],
                [int(round(sx - dx + nx)), int(round(sy - dy + ny))],
            ], np.int32)
            cv2.fillPoly(over, [poly], (215, 225, 130))
            cv2.polylines(canvas, [poly], True, (150, 175, 65), 2, cv2.LINE_AA)
        elif kind == "line":
            ex, ey = int(round(sk["x2"])), int(round(sk["y2"]))
            cv2.line(canvas, (sx, sy), (ex, ey), (165, 105, 35), 3, cv2.LINE_AA)
        elif kind == "recon":
            cv2.circle(canvas, (sx, sy), 13, (55, 50, 200), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 13, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "R", (sx - 5, sy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(over, 0.42, canvas, 0.58, 0)

    # 進行・連携矢印
    team_col = (55, 55, 225) if side == "攻撃" else (180, 185, 45)
    for ar in arrows or []:
        p0 = (int(round(ar["from"][0])), int(round(ar["from"][1])))
        p1 = (int(round(ar["to"][0])), int(round(ar["to"][1])))
        if ar.get("style", "dashed") == "dashed":
            _draw_dashed_line(canvas, p0, p1, team_col, thickness=3)
        else:
            cv2.line(canvas, p0, p1, team_col, 3, cv2.LINE_AA)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        ah = 13
        q1 = (int(round(p1[0] - ah * math.cos(ang - 0.45))), int(round(p1[1] - ah * math.sin(ang - 0.45))))
        q2 = (int(round(p1[0] - ah * math.cos(ang + 0.45))), int(round(p1[1] - ah * math.sin(ang + 0.45))))
        cv2.fillPoly(canvas, [np.array([p1, q1, q2], np.int32)], team_col)

    # エージェント駒（半径 26 px + 向き三角ポインタ）
    R = 25
    ring_col = (55, 55, 225) if side == "攻撃" else (185, 195, 55)
    for pl in placements:
        cx, cy = int(round(pl["x"])), int(round(pl["y"]))
        ag = pl["agent"]
        ang = float(pl.get("angle", -math.pi / 2 if side == "攻撃" else math.pi / 2))

        # 向き三角形
        tip = (int(round(cx + (R + 12) * math.cos(ang))), int(round(cy + (R + 12) * math.sin(ang))))
        b1 = (int(round(cx + (R - 2) * math.cos(ang - 0.35))), int(round(cy + (R - 2) * math.sin(ang - 0.35))))
        b2 = (int(round(cx + (R - 2) * math.cos(ang + 0.35))), int(round(cy + (R - 2) * math.sin(ang + 0.35))))
        cv2.fillPoly(canvas, [np.array([tip, b1, b2], np.int32)], ring_col)

        # 白外枠 + チーム色枠
        cv2.circle(canvas, (cx, cy), R + 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), R + 2, ring_col, -1, cv2.LINE_AA)

        ic = agent_icons.get(ag)
        if ic is not None and ic.ndim >= 2:
            d = 2 * R - 4
            if ic.ndim == 2:
                ic_bgr = cv2.cvtColor(ic, cv2.COLOR_GRAY2BGR)
            elif ic.shape[2] == 4:
                al = ic[:, :, 3:4].astype(np.float32) / 255.0
                ic_bgr = (ic[:, :, :3].astype(np.float32) * al + 40.0 * (1.0 - al)).astype(np.uint8)
            else:
                ic_bgr = ic[:, :, :3]
            face = cv2.resize(ic_bgr, (d, d), interpolation=cv2.INTER_AREA)
            r_in = d // 2
            m = np.zeros((d, d), np.uint8)
            cv2.circle(m, (r_in, r_in), r_in, 255, -1)
            y0, x0 = cy - r_in, cx - r_in
            if 0 <= y0 and 0 <= x0 and y0 + d <= 1024 and x0 + d <= 1024:
                roi = canvas[y0:y0 + d, x0:x0 + d]
                roi[m > 0] = face[m > 0]
        else:
            hexc = (AGENT_SKILLS.get(ag) or {}).get("color", "#4a5568").lstrip("#")
            bgr = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
            cv2.circle(canvas, (cx, cy), R - 2, bgr, -1, cv2.LINE_AA)
            txt = ag[:2].upper()
            cv2.putText(canvas, txt, (cx - 14, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    if out_size != 1024:
        canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ------------------------------------------------------------------ 座標計算・形状テンプレート
def _interp(p0: tuple[float, float], p1: tuple[float, float], t: float) -> tuple[float, float]:
    return (p0[0] * (1 - t) + p1[0] * t, p0[1] * (1 - t) + p1[1] * t)


def _spread_pts(pts: list[tuple[float, float]], min_dist: float = 52.0, floor_mask: np.ndarray | None = None) -> list[tuple[float, float]]:
    """複数エージェントの座標が重なりすぎないよう少し離す。"""
    out: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(pts):
        nx, ny = x, y
        for ox, oy in out:
            d = math.hypot(nx - ox, ny - oy)
            if d < min_dist:
                ang = (i * 2.1) if d < 1e-3 else math.atan2(ny - oy, nx - ox)
                nx += math.cos(ang) * (min_dist - d + 8)
                ny += math.sin(ang) * (min_dist - d + 8)
        out.append(_snap_to_floor(nx, ny, floor_mask))
    return out


def _make_layout(
    spec_key: str,
    comp_info: dict[str, Any],
    anchors: dict[str, Any],
    floor_mask: np.ndarray | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """型の役割 (spec_key) に応じて 5 人の位置・スキル・矢印を生成する。"""
    p = anchors["points"]
    sl = comp_info["slots"]
    s_op, s_anc, s_smk, s_rec, s_ent = sl["op"], sl["anchor"], sl["smoke"], sl["recon"], sl["entry"]

    def _pl(ag: str, pos: tuple[float, float], look: tuple[float, float]) -> dict:
        return {"agent": ag, "x": pos[0], "y": pos[1], "angle": _agent_angle(pos, look)}

    skills: list[dict] = []
    arrows: list[dict] = []

    if spec_key == "atk_b_take":
        raw = [
            (s_op, _interp(p["atk_b"], p["b_main"], 0.35), p["b_main"]),
            (s_rec, _interp(p["atk_b"], p["atk_spawn"], 0.12), p["b_main"]),
            (s_anc, _interp(p["atk_b"], p["atk_mid"], 0.55), p["mid_bot"]),
            (s_ent, _interp(p["atk_mid"], p["atk_a"], 0.55), p["mid_top"]),
            (s_smk, _interp(p["atk_a"], p["atk_spawn"], 0.25), p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        rp = _interp(p["b_main"], p["site_b"], 0.35)
        skills.append({"kind": "recon", "x": rp[0], "y": rp[1]})
        arrows.append({"from": coords[1], "to": rp, "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_mid_take":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, _interp(p["atk_b"], p["atk_mid"], 0.4), p["b_main"]),
            (s_anc, _interp(p["atk_mid"], p["atk_b"], 0.22), p["mid_top"]),
            (s_ent, p["atk_mid"], p["mid_top"]),
            (s_smk, _interp(p["atk_mid"], p["atk_a"], 0.42), p["mid_top"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        smk_pos = _interp(p["mid_bot"], p["mid_top"], 0.55)
        slw_pos = _interp(p["mid_top"], p["a_link"], 0.35)
        skills.append({"kind": "smoke", "x": smk_pos[0], "y": smk_pos[1], "r": 46})
        skills.append({"kind": "slow", "x": slw_pos[0], "y": slw_pos[1], "r": 30})
        arrows.append({"from": coords[3], "to": p["mid_bot"], "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_a_take":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, _interp(p["atk_b"], p["atk_mid"], 0.3), p["b_main"]),
            (s_ent, _interp(p["atk_mid"], p["atk_a"], 0.45), p["a_main"]),
            (s_smk, _interp(p["atk_a"], p["a_main"], 0.25), p["site_a"]),
            (s_anc, _interp(p["atk_a"], p["atk_spawn"], 0.2), p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        smk_pos = _interp(p["a_main"], p["site_a"], 0.42)
        skills.append({"kind": "smoke", "x": smk_pos[0], "y": smk_pos[1], "r": 42})
        arrows.append({"from": coords[3], "to": smk_pos, "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_131":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, _interp(p["atk_mid"], p["mid_bot"], 0.25), p["mid_top"]),
            (s_ent, p["atk_mid"], p["mid_top"]),
            (s_smk, _interp(p["atk_mid"], p["atk_a"], 0.3), p["mid_top"]),
            (s_anc, p["atk_a"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["mid_top"][0], "y": p["mid_top"][1], "r": 44})
        arrows += [
            {"from": coords[2], "to": p["mid_bot"], "style": "dashed"},
            {"from": coords[1], "to": p["mid_top"], "style": "dashed"},
        ]
        return pls, skills, arrows

    if spec_key in ("atk_41_b", "atk_b_rush"):
        raw = [
            (s_ent, _interp(p["atk_b"], p["b_main"], 0.38), p["site_b"]),
            (s_rec, _interp(p["atk_b"], p["b_main"], 0.2), p["site_b"]),
            (s_smk, p["atk_b"], p["site_b"]),
            (s_anc, _interp(p["atk_b"], p["atk_mid"], 0.28), p["b_main"]),
            (s_op, p["atk_a"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["b_link"][0], "y": p["b_link"][1], "r": 44})
        arrows.append({"from": coords[0], "to": p["site_b"], "style": "dashed"})
        return pls, skills, arrows

    if spec_key in ("atk_41_a", "atk_a_rush"):
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_anc, _interp(p["atk_a"], p["atk_mid"], 0.25), p["a_main"]),
            (s_smk, p["atk_a"], p["site_a"]),
            (s_rec, _interp(p["atk_a"], p["a_main"], 0.22), p["site_a"]),
            (s_ent, _interp(p["atk_a"], p["a_main"], 0.42), p["site_a"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["a_link"][0], "y": p["a_link"][1], "r": 44})
        arrows.append({"from": coords[4], "to": p["site_a"], "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_a_exec":
        raw = [
            (s_anc, _interp(p["a_main"], p["site_a"], 0.45), p["a_link"]),
            (s_ent, _interp(p["a_main"], p["site_a"], 0.65), p["site_a"]),
            (s_op, _interp(p["a_main"], p["atk_a"], 0.25), p["site_a"]),
            (s_rec, _interp(p["a_main"], p["atk_a"], 0.35), p["site_a"]),
            (s_smk, p["atk_a"], p["site_a"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        w_ang = _agent_angle(p["a_link"], p["site_a"]) + math.pi / 2
        skills += [
            {"kind": "wall", "x": p["a_link"][0], "y": p["a_link"][1], "angle": w_ang, "len": 110},
            {"kind": "smoke", "x": p["def_a_site"][0], "y": p["def_a_site"][1] - 40, "r": 42},
        ]
        arrows.append({"from": coords[1], "to": p["site_a"], "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_a_split":
        raw = [
            (s_ent, _interp(p["mid_top"], p["a_link"], 0.55), p["site_a"]),
            (s_anc, _interp(p["mid_top"], p["a_link"], 0.35), p["a_link"]),
            (s_op, _interp(p["mid_bot"], p["a_link"], 0.35), p["site_a"]),
            (s_smk, p["a_main"], p["site_a"]),
            (s_rec, _interp(p["a_main"], p["atk_a"], 0.25), p["site_a"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        w_ang = _agent_angle(p["a_link"], p["site_a"]) + math.pi / 2
        skills.append({"kind": "wall", "x": p["a_link"][0], "y": p["a_link"][1], "angle": w_ang, "len": 96})
        arrows += [
            {"from": coords[0], "to": p["site_a"], "style": "dashed"},
            {"from": coords[3], "to": p["site_a"], "style": "dashed"},
        ]
        return pls, skills, arrows

    if spec_key == "atk_b_split":
        raw = [
            (s_ent, _interp(p["mid_top"], p["b_link"], 0.45), p["site_b"]),
            (s_anc, _interp(p["mid_top"], p["b_link"], 0.25), p["site_b"]),
            (s_rec, _interp(p["mid_bot"], p["b_link"], 0.3), p["site_b"]),
            (s_smk, _interp(p["mid_bot"], p["b_link"], 0.1), p["b_link"]),
            (s_op, p["b_main"], p["site_b"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        w_ang = _agent_angle(p["b_link"], p["site_b"]) + math.pi / 2
        skills += [
            {"kind": "wall", "x": p["b_link"][0], "y": p["b_link"][1], "angle": w_ang, "len": 92},
            {"kind": "smoke", "x": p["def_b_site"][0], "y": p["def_b_site"][1] - 42, "r": 42},
        ]
        arrows += [
            {"from": coords[0], "to": p["site_b"], "style": "dashed"},
            {"from": coords[4], "to": p["site_b"], "style": "dashed"},
        ]
        return pls, skills, arrows

    if spec_key == "atk_b_exec":
        raw = [
            (s_ent, _interp(p["b_main"], p["site_b"], 0.65), p["site_b"]),
            (s_anc, _interp(p["b_main"], p["site_b"], 0.45), p["b_link"]),
            (s_rec, _interp(p["b_main"], p["atk_b"], 0.2), p["site_b"]),
            (s_smk, p["b_main"], p["site_b"]),
            (s_op, _interp(p["atk_mid"], p["mid_bot"], 0.4), p["mid_top"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        w_ang = _agent_angle(p["b_link"], p["site_b"]) + math.pi / 2
        skills += [
            {"kind": "wall", "x": p["b_link"][0], "y": p["b_link"][1], "angle": w_ang, "len": 96},
            {"kind": "smoke", "x": p["def_b_site"][0] + 35, "y": p["def_b_site"][1] - 35, "r": 42},
        ]
        arrows.append({"from": coords[0], "to": p["site_b"], "style": "dashed"})
        return pls, skills, arrows

    if spec_key == "atk_smoke_mid":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, _interp(p["atk_b"], p["atk_mid"], 0.4), p["b_main"]),
            (s_anc, _interp(p["atk_mid"], p["atk_b"], 0.25), p["mid_top"]),
            (s_ent, p["atk_mid"], p["mid_top"]),
            (s_smk, _interp(p["atk_mid"], p["atk_a"], 0.3), p["mid_top"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["mid_top"][0], "y": p["mid_top"][1], "r": 54})
        skills.append({"kind": "smoke", "x": _interp(p["mid_bot"], p["b_link"], 0.45)[0], "y": _interp(p["mid_bot"], p["b_link"], 0.45)[1], "r": 42})
        return pls, skills, arrows

    if spec_key == "atk_smoke_a":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, p["atk_mid"], p["mid_top"]),
            (s_ent, _interp(p["atk_mid"], p["atk_a"], 0.55), p["a_main"]),
            (s_smk, p["atk_a"], p["a_main"]),
            (s_anc, _interp(p["atk_a"], p["atk_spawn"], 0.25), p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["a_main"][0], "y": p["a_main"][1], "r": 48})
        return pls, skills, arrows

    if spec_key == "atk_smoke_b":
        raw = [
            (s_op, p["atk_b"], p["b_main"]),
            (s_rec, _interp(p["atk_b"], p["atk_mid"], 0.3), p["b_main"]),
            (s_smk, p["atk_mid"], p["b_main"]),
            (s_ent, _interp(p["atk_mid"], p["atk_a"], 0.45), p["mid_top"]),
            (s_anc, p["atk_a"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["b_main"][0], "y": p["b_main"][1], "r": 48})
        return pls, skills, arrows

    if spec_key == "atk_fake_b":
        raw = [
            (s_rec, p["atk_b"], p["b_main"]),
            (s_smk, _interp(p["atk_mid"], p["atk_a"], 0.25), p["a_main"]),
            (s_ent, p["atk_a"], p["a_main"]),
            (s_op, _interp(p["atk_a"], p["atk_spawn"], 0.15), p["a_main"]),
            (s_anc, _interp(p["atk_a"], p["atk_mid"], 0.35), p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "recon", "x": p["b_main"][0], "y": p["b_main"][1]})
        arrows += [
            {"from": coords[0], "to": p["b_main"], "style": "dashed"},
            {"from": coords[2], "to": p["site_a"], "style": "dashed"},
        ]
        return pls, skills, arrows

    if spec_key == "atk_fake_a":
        raw = [
            (s_rec, p["atk_a"], p["a_main"]),
            (s_smk, _interp(p["atk_mid"], p["atk_b"], 0.3), p["b_main"]),
            (s_ent, p["atk_b"], p["b_main"]),
            (s_op, _interp(p["atk_b"], p["atk_spawn"], 0.15), p["b_main"]),
            (s_anc, _interp(p["atk_b"], p["atk_mid"], 0.35), p["b_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "recon", "x": p["a_main"][0], "y": p["a_main"][1]})
        arrows += [
            {"from": coords[0], "to": p["a_main"], "style": "dashed"},
            {"from": coords[2], "to": p["site_b"], "style": "dashed"},
        ]
        return pls, skills, arrows

    if spec_key == "atk_post_a":
        raw = [
            (s_anc, p["a_link"], p["def_spawn"]),
            (s_op, p["a_main"], p["site_a"]),
            (s_ent, _interp(p["site_a"], p["a_main"], 0.4), p["def_a_site"]),
            (s_rec, _interp(p["a_main"], p["atk_a"], 0.25), p["site_a"]),
            (s_smk, _interp(p["a_link"], p["mid_top"], 0.3), p["site_a"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        w_ang = _agent_angle(p["a_link"], p["site_a"]) + math.pi / 2
        skills += [
            {"kind": "wall", "x": p["a_link"][0], "y": p["a_link"][1], "angle": w_ang, "len": 100},
            {"kind": "slow", "x": p["site_a"][0], "y": p["site_a"][1], "r": 34},
        ]
        return pls, skills, arrows

    if spec_key == "atk_post_b":
        raw = [
            (s_anc, p["b_link"], p["def_spawn"]),
            (s_op, p["b_main"], p["site_b"]),
            (s_ent, _interp(p["site_b"], p["b_main"], 0.4), p["def_b_site"]),
            (s_rec, _interp(p["b_main"], p["atk_b"], 0.25), p["site_b"]),
            (s_smk, _interp(p["b_link"], p["mid_top"], 0.3), p["site_b"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "slow", "x": p["site_b"][0], "y": p["site_b"][1], "r": 34})
        return pls, skills, arrows

    # ---- 防衛配置 ----
    if spec_key == "def_b_op":
        raw = [
            (s_op, p["def_b_main"], p["b_main"]),
            (s_rec, _interp(p["def_b_site"], p["def_mid"], 0.3), p["b_main"]),
            (s_smk, p["def_mid"], p["mid_bot"]),
            (s_ent, _interp(p["def_mid"], p["def_a_main"], 0.55), p["mid_top"]),
            (s_anc, p["def_a_site"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "line", "x": coords[0][0], "y": coords[0][1], "x2": p["b_main"][0], "y2": p["b_main"][1]})
        return pls, skills, arrows

    if spec_key == "def_b_double":
        raw = [
            (s_op, p["def_b_main"], p["b_main"]),
            (s_anc, p["def_b_site"], p["b_main"]),
            (s_rec, _interp(p["def_mid"], p["def_b_site"], 0.35), p["mid_bot"]),
            (s_smk, p["def_mid"], p["mid_bot"]),
            (s_ent, p["def_a_main"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "wall", "x": p["b_link"][0], "y": p["b_link"][1], "angle": 0.3, "len": 88})
        skills.append({"kind": "line", "x": coords[0][0], "y": coords[0][1], "x2": p["b_main"][0], "y2": p["b_main"][1]})
        return pls, skills, arrows

    if spec_key == "def_b_anc":
        raw = [
            (s_anc, p["def_b_site"], p["b_main"]),
            (s_rec, p["def_b_main"], p["b_main"]),
            (s_smk, p["def_mid"], p["mid_bot"]),
            (s_ent, _interp(p["def_mid"], p["def_a_main"], 0.5), p["a_main"]),
            (s_op, p["def_a_main"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "slow", "x": p["b_main"][0], "y": p["b_main"][1], "r": 32})
        return pls, skills, arrows

    if spec_key == "def_a_anc":
        raw = [
            (s_op, p["def_b_main"], p["b_main"]),
            (s_smk, p["def_mid"], p["mid_bot"]),
            (s_rec, _interp(p["def_mid"], p["def_a_main"], 0.45), p["a_main"]),
            (s_anc, p["def_a_site"], p["a_main"]),
            (s_ent, p["def_a_main"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "wall", "x": p["a_main"][0], "y": p["a_main"][1], "angle": 0.2, "len": 88})
        return pls, skills, arrows

    if spec_key == "def_mid_op":
        raw = [
            (s_anc, p["def_b_site"], p["b_main"]),
            (s_rec, _interp(p["def_mid"], p["def_b_main"], 0.35), p["mid_bot"]),
            (s_op, p["def_mid"], p["mid_bot"]),
            (s_smk, _interp(p["def_mid"], p["def_a_main"], 0.35), p["mid_bot"]),
            (s_ent, p["def_a_site"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "line", "x": coords[2][0], "y": coords[2][1], "x2": p["mid_bot"][0], "y2": p["mid_bot"][1]})
        return pls, skills, arrows

    if spec_key == "def_131":
        raw = [
            (s_anc, p["def_b_site"], p["b_main"]),
            (s_rec, _interp(p["def_mid"], p["def_b_main"], 0.3), p["mid_bot"]),
            (s_ent, p["def_mid"], p["mid_bot"]),
            (s_smk, _interp(p["def_mid"], p["def_a_main"], 0.3), p["mid_bot"]),
            (s_op, p["def_a_site"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        skills.append({"kind": "smoke", "x": p["mid_bot"][0], "y": p["mid_bot"][1], "r": 42})
        return pls, skills, arrows

    if spec_key == "def_b_push":
        raw = [
            (s_ent, p["def_b_push"], p["atk_b"]),
            (s_rec, _interp(p["def_b_main"], p["def_b_push"], 0.65), p["atk_b"]),
            (s_smk, p["def_b_site"], p["b_main"]),
            (s_op, p["def_mid"], p["mid_bot"]),
            (s_anc, p["def_a_site"], p["a_main"]),
        ]
        coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
        pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
        arrows.append({"from": coords[0], "to": p["atk_b"], "style": "dashed"})
        return pls, skills, arrows

    # 汎用フォールバック（広くやる形・中盤等）
    raw = [
        (s_op, p["atk_b"] if spec_key.startswith("atk") else p["def_b_site"], p["b_main"]),
        (s_rec, p["atk_mid"] if spec_key.startswith("atk") else p["def_b_main"], p["mid_bot"]),
        (s_smk, _interp(p["atk_mid"], p["atk_a"], 0.25) if spec_key.startswith("atk") else p["def_mid"], p["mid_top"]),
        (s_ent, p["atk_mid"] if spec_key.startswith("atk") else p["def_a_main"], p["a_main"]),
        (s_anc, p["atk_a"] if spec_key.startswith("atk") else p["def_a_site"], p["a_main"]),
    ]
    coords = _spread_pts([r[1] for r in raw], floor_mask=floor_mask)
    pls = [_pl(r[0], coords[i], r[2]) for i, r in enumerate(raw)]
    return pls, skills, arrows


# ------------------------------------------------------------------ 構成メモ & 型の生成
def _build_comp_memo(map_name: str, team: str, mp: dict[str, Any], comp_info: dict[str, Any], kb: KB.KB) -> str:
    sl = comp_info["slots"]
    sk = comp_info["skills"]
    agents = comp_info["agents"]
    kb_map = kb.map(map_name)
    kb_comps = kb.comp_notes(agents, map_name)

    extra_lines: list[str] = []
    for c in kb_comps[:1]:
        if c.get("plan"):
            extra_lines.append(f"・構成プラン：{c['plan']}")
        for s in (c.get("attack") or {}).get("strong", [])[:2]:
            extra_lines.append(f"・強み：{s}")
    if kb_map.get("summary"):
        extra_lines.append(f"・マップメタ：{kb_map['summary']}")

    header = f"{mp['display']} {team} comp ({' / '.join(agents)})"
    if comp_info["is_double_sentinel"]:
        role_desc = f"{sl['anchor']}と{sl['op']}を使用した2センチネル構成。防衛のじゃんけんに強い手札を持ちつつ、攻撃のセットやポストプラントも強力。"
    else:
        role_desc = f"{sl['op']}・{sl['anchor']}・{sl['smoke']}・{sl['recon']}・{sl['entry']}を軸にした構成。初動のエリア取りから中盤以降のローテ・セットを連動させる。"

    memo = [
        header,
        "",
        role_desc,
        *extra_lines,
        "",
        "■ 攻めのデフォルト",
        f"・{mp['main_b']} {sk['recon_b']}＋{sk['op_name']}でテイク",
        "  → カウンターに対して強く、ドライの取り返しが困難",
        f"・{mp['mid_take']} {sk['smoke']}＋{sk['slow']}",
        f"  → {mp['mid_ctrl']}の主導権を握り 1-3-1 や 1-4 から振り回す展開が可能",
        f"・{mp['main_a']}テイク（{sk['slow']}＋{sk['take_smk']}）",
        f"  → 敵がいる場合は {sk['recon_b']}＋{sk['entry']} 等の強気なテイクが可能、ピックが取れたらそのままいける",
        "  → スキル音だけ出して行かないフェイクも可",
        "→ 全部の動きがスキルだけ入れてグループが可能",
        "",
        "■ エクセキューション",
        f"・{mp['main_a']}セット（{sk['wall']}）",
        f"  → {mp['link_a']}に{sk['wall']}／視界遮断を入れ、射線を通せない状態でサイトを挟む",
        "  → ポストプラントにも使え、エントリーの射線組みも可能",
        "・Aスプリット",
        f"  → {mp['link_a']}に{sk['wall']}を炊きサイトをスプリット。壁を占めて敵のローテルートをカットして挟む強力なセット",
        "・Bスプリット",
        f"  → {mp['mid_take']}テイク後から{mp['link_b']}経由でスプリット。{sk['wall']}やスモークでリンク奥・CTを閉じる",
        f"・{mp['main_b']}セット",
        "  → 狭く切る形から壁・スモークとフランクを利用して勝つ形（ウルトが強力な時・少スキルエントリー時に有効）",
        "",
        "■ ポストプラント",
        "・Aサイト",
        f"  - {mp['post_a']}の射線組みと{sk['wall']}によるリテイク分断",
        f"  - {sk['recon_delay']}や{sk['slow']}で1ウェーブ遅延し、人数をかけて切り返す",
        "・Bサイト",
        f"  - {mp['post_b']}の死守が肝。複数人でアングルを組みつつ侵入タイミングをケア",
        "  - 広く取れている時も前めの無理は避け、敵の前目スキルは壊してリソースを吐かせる",
    ]
    return "\n".join(memo)


def _shape_specs(mp: dict[str, Any], comp_info: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    """[(型名, side, category, desc, layout_key), ...] を LOUD Summit 粒度で生成する。"""
    sl = comp_info["slots"]
    sk = comp_info["skills"]
    op_n, anc_n = sk["op_name"], sk["anc_name"]
    specs: list[tuple[str, str, str, str, str]] = [
        # --- 序盤（攻撃の3本柱） ---
        (f"{mp['main_b']} テイク（{sk['recon_b']}＋{op_n}）", "攻撃", "序盤",
         f"{mp['main_b']}を{sk['recon_b']}（{sl['recon']}）＋{sl['op']}でテイク。\n"
         f"カウンターに対して強く、ドライの取り返しが困難。スキルだけ入れてグループも可能。", "atk_b_take"),
        (f"{mp['mid_take']} テイク（{sk['smoke']}＋{sk['slow']}）", "攻撃", "序盤",
         f"{mp['mid_take']}に{sk['smoke']}＋{sk['slow']}を入れて{mp['mid_ctrl']}の主導権を握る。\n"
         "1-3-1や1-4から振り回す展開が可能。スキルだけ入れてグループも可能。", "atk_mid_take"),
        (f"{mp['main_a']} テイク（{sk['slow']}＋{sk['take_smk']}）", "攻撃", "序盤",
         f"{sk['slow']}＋{sk['take_smk']}で{mp['main_a']}をテイク。\n"
         f"敵がいる場合は{sk['recon_b']}＋{sk['entry']}等の強気なテイクが可能で、ピックが取れたらそのままいける。スキル音だけ出して行かないフェイクも可。",
         "atk_a_take"),

        # --- エクセキューション ---
        (f"{mp['main_a']}セット（{sk['wall']}）", "攻撃", "エクセキューション",
         f"{mp['link_a']}に{sk['wall']}を炊き、敵が覗けない状態でAサイトを挟む。\n"
         "ポストプラントにも使え、エントリーの射線組みも可能になる。", "atk_a_exec"),
        (f"Aスプリット（{mp['link_a']}）", "攻撃", "エクセキューション",
         f"{mp['link_a']}に{sk['wall']}・スモークを入れAサイトをスプリット。敵のローテルートをカットした状態でサイトを挟める強力なセット。",
         "atk_a_split"),
        (f"Bスプリット（{mp['mid_take']}後）", "攻撃", "エクセキューション",
         f"{mp['mid_take']}テイク後からのBスプリット。\n"
         f"{mp['link_b']}に{sk['wall']}・スモークを使いつつ、抜けた後は裏やCTを閉じて挟み込む。", "atk_b_split"),
        (f"{mp['main_b']}セット（狭く切り＋{sk['wall']}）", "攻撃", "エクセキューション",
         f"狭く切る形から{sk['wall']}と{mp['mid_ctrl']}フランクを利用して勝つ形。\n"
         "サイトでキルが発生すれば有利。ウルトが強力な場合や少ないスキルでエントリーできる時に使う。", "atk_b_exec"),

        # --- ポストプラント ---
        ("Aサイト ポストプラント", "攻撃", "ポストプラント",
         f"・CT前目で射線組み / {mp['link_a']}を{sk['wall']}で塞ぎ残りにフォーカス\n"
         f"・{sk['recon_delay']}や{sk['slow']}等で1ウェーブ遅延する\n"
         "・人数かけてファイトして切り返しや押し切りを狙う", "atk_post_a"),
        (f"Bサイト ポストプラント（{mp['post_b']}）", "攻撃", "ポストプラント",
         f"{mp['post_b']}の死守が肝。2人以上でアングルを組みつつ、サイトに入ってくる敵とのアングルをケアする。\n"
         "広く取れているときも前めのポジションは使わず、リソースを吐かせるため前目のスキルは壊す。", "atk_post_b"),

        # --- 形・中盤・スモーク・索敵・全体タグ ---
        ("1－3－1", "攻撃", "形",
         f"{mp['mid_ctrl']}に3人を置き、両メイン（{mp['main_b']}・{mp['main_a']}）の情報から攻め先を決める基本陣形。", "atk_131"),
        ("4－1デフォ", "攻撃", "形",
         f"主力4人で{mp['main_b']}または{mp['main_a']}に圧を掛け、逆サイド1人がラークでローテを釣る。", "atk_41_b"),
        (f"{mp['main_a']}4-1デフォ", "攻撃", "形",
         f"{mp['main_a']}側に4人を寄せて圧を掛け、逆サイド1人で情報を取る。", "atk_41_a"),
        ("広くやる形", "攻撃", "中盤",
         f"{mp['main_b']}・{mp['mid_take']}・{mp['main_a']}の全レーンでスキルを小出しにして、守りの薄い場所を探す。", "atk_131"),
        (mp["smoke_mid"], "攻撃", "スモーク",
         f"{sl['smoke']}の{sk['smoke']}を{mp['mid_ctrl']}手前に膨らませて炊き、安全な進行ルートを作る。", "atk_smoke_mid"),
        (f"{mp['main_a']}モク", "攻撃", "スモーク",
         f"{mp['main_a']}の射線を切るスモーク。テイク・フェイク両方に使用。", "atk_smoke_a"),
        (f"{mp['main_b']}モク", "攻撃", "スモーク",
         f"{mp['main_b']}入口のスモーク。相手の前詰め牽制・フェイクに使用。", "atk_smoke_b"),
        (f"{sl['recon']}Bフェイク", "攻撃", "エクセキューション",
         f"{sl['recon']}の{sk['recon_b']}と音で{mp['main_b']}を見せ、守りが寄った隙に本命のA側へ攻める。", "atk_fake_b"),
        ("AフェイクB", "攻撃", "エクセキューション",
         f"{mp['main_a']}でスキルと音を見せてA寄りを誘い、Bサイトへ切り替える。", "atk_fake_a"),
        ("Aラッシュ", "攻撃", "エクセキューション",
         f"{mp['main_a']}から5人で一気にチョークを抜けてサイトを取り切る。", "atk_a_rush"),
        ("Bセット", "攻撃", "エクセキューション",
         f"{mp['main_b']}からスキルを合わせてBサイトへ進入する。", "atk_b_exec"),
        ("エコBラッシュ", "攻撃", "エクセキューション",
         f"エコ・フォース時の{mp['main_b']}一斉ラッシュ。トレード距離を詰めて武器回収・設置を狙う。", "atk_b_rush"),

        # --- 防衛（全体初動配置） ---
        (f"B{op_n}デフォ", "防衛", "全体",
         f"{sl['op']}がB側（{mp['main_b']}）の初動ワンピック・射線管理を担当し、A側は{sl['anchor']}で遅らせる標準配置。", "def_b_op"),
        (f"B{anc_n}{op_n}", "防衛", "全体",
         f"B側に{sl['anchor']}＋{sl['op']}の2枚を厚く置き、Bの進行を完全に封じてAへ回す配置。", "def_b_double"),
        (f"B{anc_n}デフォ", "防衛", "全体",
         f"{sl['anchor']}が{mp['main_b']}を壁・遅延で1人受け持ち、{sl['op']}はA側またはMIDのフリーピックを狙う配置。", "def_b_anc"),
        (f"A{anc_n}", "防衛", "全体",
         f"{sl['anchor']}が{mp['main_a']}を固く守り、{sl['op']}がB/MIDで勝負するAアンカー配置。", "def_a_anc"),
        (f"MID{op_n}", "防衛", "全体",
         f"{sl['op']}が{mp['mid_ctrl']}で初動のラインを押し上げ、両サイドの挟み込みを警戒・カットする配置。", "def_mid_op"),
        ("1－3－1", "防衛", "全体",
         f"{mp['mid_ctrl']}に3人を寄せてMID主導権を奪い、両サイトを1人＋スキルで遅延する攻撃的防衛。", "def_131"),
        ("Bプッシュ", "防衛", "全体",
         f"開幕から{sl['entry']}＋{sl['recon']}で{mp['main_b']}を前詰めして裏取り・情報奪取を行う配置。", "def_b_push"),
    ]
    if "C" in mp["sites"]:
        specs.append(("Cサイト ポストプラント", "攻撃", "ポストプラント",
                      f"{mp['main_b']}・{mp['link_b']}の2方向からクロスを組み、遅延スキルでリテイクを遅らせる。", "atk_post_b"))
    return specs


# ------------------------------------------------------------------ 公開API：生成＆統合
def _match_shape_key(existing_shapes: dict, cand_name: str) -> str | None:
    """既存 JSON の型名（例：末尾の括弧有無の違いなど）をゆるく一致させる。"""
    if cand_name in existing_shapes:
        return cand_name
    head = cand_name.split("（", 1)[0].strip()
    for k in existing_shapes:
        if k.split("（", 1)[0].strip() == head:
            return k
    return None


def generate_map_playbook_data(
    map_name: str,
    comp: list[str] | None = None,
    team: str = "LOUD",
    base_json: Path | str | None = None,
    cfg: dict | None = None,
    kb: KB.KB | None = None,
    cache_dirs: list[Path] | None = None,
    out_size: int = 640,
) -> dict[str, Any]:
    """指定マップ・構成の初動配置データ（comp_memo, shapes, rounds）を生成する。

    base_json（例：`loud_summit_classified (1).json`）が指定され、かつ対象マップが一致する場合は
    既存の分類済みラウンド・手動説明・手動 tactival_img を温存し、未登録の型だけ図解・説明を補完する。
    別マップが指定された場合は base_json の構造・命名粒度をお手本に、そのマップ用の配置・図解を新規生成する。
    """
    kb = kb or KB.load(cfg)
    mid = _norm(map_name)
    mp = MAP_PROFILES.get(mid)
    if mp is None:
        mp = {
            "display": str(map_name or "Unknown").title(),
            "sites": ["A", "B"],
            "has_mid": True,
            "main_b": "Bメイン",
            "main_a": "Aメイン",
            "mid_take": "MIDボトム",
            "mid_ctrl": "MID",
            "link_a": "Aリンク",
            "link_b": "Bリンク",
            "post_b": "Bサイト裏",
            "post_a": "Aリンク/CT",
            "smoke_mid": "MID膨らみモク",
            "smoke_bottom": "ボトムモク",
            "default_comp": list(DEFAULT_COMP),
            "summary_note": f"{map_name} の初動配置テンプレート。",
        }

    use_comp = comp or mp.get("default_comp") or DEFAULT_COMP
    comp_info = analyze_comp(use_comp, kb)
    base_rgba = _load_cached_map_image(mp["display"], cache_dirs=cache_dirs)
    floor_mask = (base_rgba[:, :, 3] > 10) if base_rgba is not None else None
    calls = _load_cached_callouts(mp["display"], cache_dirs=cache_dirs)
    anchors = barrier_anchors(mp["display"], floor_mask=floor_mask, calls=calls)
    icons = _load_cached_agent_icons(comp_info["agents"], cache_dirs=cache_dirs)
    tpl = TV.load_template(cfg)

    base_data: dict[str, Any] | None = None
    same_map_base = False
    if base_json is not None and Path(base_json).exists():
        try:
            base_data = json.loads(Path(base_json).read_text("utf-8"))
            memo_head = str(base_data.get("comp_memo") or "").lower()
            if mid in memo_head or (mid == "summit" and "summit" in memo_head):
                same_map_base = True
        except Exception:
            base_data = None

    shapes: dict[str, Any] = {}
    if same_map_base and base_data:
        shapes = copy.deepcopy(base_data.get("shapes") or {})

    for name, side, cat, desc, layout_key in _shape_specs(mp, comp_info):
        pls, sks, arrs = _make_layout(layout_key, comp_info, anchors, floor_mask)
        tv_json = build_tactival_json(mp["display"], name, side, pls, arrs, tpl=tpl)

        exist_k = _match_shape_key(shapes, name)
        if exist_k:
            entry = shapes[exist_k]
            if not entry.get("desc"):
                entry["desc"] = desc
            if not entry.get("tactival_img"):
                entry["tactival_img"] = render_opening_board(
                    mp["display"], side, pls, sks, arrs, anchors=anchors, calls=calls,
                    base_rgba=base_rgba, agent_icons=icons, out_size=out_size,
                )
            if not entry.get("tactival_json"):
                entry["tactival_json"] = tv_json
            if not entry.get("placements"):
                entry["placements"] = pls
        else:
            img_uri = render_opening_board(
                mp["display"], side, pls, sks, arrs, anchors=anchors, calls=calls,
                base_rgba=base_rgba, agent_icons=icons, out_size=out_size,
            )
            shapes[name] = {
                "side": side,
                "category": cat,
                "desc": desc,
                "tactival_img": img_uri,
                "tactival_note": f"{mp['display']} 自動配置（バリア座標基準）",
                "tactival_json": tv_json,
                "placements": pls,
            }

    # 既存の shape のうち tactival_img / tactival_json が空のものがあれば補完
    for k, entry in shapes.items():
        side = entry.get("side") or "攻撃"
        if not entry.get("tactival_img") or not entry.get("tactival_json"):
            lkey = _infer_layout_key_from_name(k, side)
            pls, sks, arrs = _make_layout(lkey, comp_info, anchors, floor_mask)
            if not entry.get("tactival_json"):
                entry["tactival_json"] = build_tactival_json(mp["display"], k, side, pls, arrs, tpl=tpl)
            if not entry.get("tactival_img"):
                entry["tactival_img"] = render_opening_board(
                    mp["display"], side, pls, sks, arrs, anchors=anchors, calls=calls,
                    base_rgba=base_rgba, agent_icons=icons, out_size=out_size,
                )
            if not entry.get("placements"):
                entry["placements"] = pls

    comp_memo = (
        base_data["comp_memo"]
        if (same_map_base and base_data and base_data.get("comp_memo"))
        else _build_comp_memo(mp["display"], team, mp, comp_info, kb)
    )
    rounds = copy.deepcopy(base_data.get("rounds") or []) if (same_map_base and base_data) else []

    return {
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "map": mp["display"],
        "team": team,
        "comp": comp_info["agents"],
        "comp_memo": comp_memo,
        "shapes": shapes,
        "rounds": rounds,
    }


def _infer_layout_key_from_name(name: str, side: str) -> str:
    n = str(name)
    if side == "防衛":
        if "プッシュ" in n:
            return "def_b_push"
        if "1" in n and "3" in n:
            return "def_131"
        if "MID" in n.upper():
            return "def_mid_op"
        if n.startswith("A"):
            return "def_a_anc"
        if "セージ" in n and "チェンバー" in n:
            return "def_b_double"
        if "セージ" in n:
            return "def_b_anc"
        return "def_b_op"
    if "ポストプラント" in n:
        return "atk_post_a" if "A" in n else "atk_post_b"
    if "フェイク" in n:
        return "atk_fake_b" if ("Bフェイク" in n or "Bﾌｪｲｸ" in n) else "atk_fake_a"
    if "Aスプリット" in n:
        return "atk_a_split"
    if "Bスプリット" in n:
        return "atk_b_split"
    if "A" in n and "セット" in n:
        return "atk_a_exec"
    if "B" in n and "セット" in n:
        return "atk_b_exec"
    if "Aラッシュ" in n:
        return "atk_a_rush"
    if "Bラッシュ" in n:
        return "atk_b_rush"
    if "モク" in n:
        if "A" in n:
            return "atk_smoke_a"
        if "Bメイン" in n or "Bﾛﾝｸﾞ" in n:
            return "atk_smoke_b"
        return "atk_smoke_mid"
    if "4" in n and "1" in n:
        return "atk_41_a" if "A" in n else "atk_41_b"
    if "MID" in n.upper() or "ボトム" in n:
        return "atk_mid_take"
    if "A" in n:
        return "atk_a_take"
    if "B" in n or "C" in n:
        return "atk_b_take"
    return "atk_131"


# ------------------------------------------------------------------ report.json → ラウンド自動分類
def _canon_side(s: str | None) -> str | None:
    if not s:
        return None
    if "攻" in str(s):
        return "攻撃"
    if "防" in str(s) or "衛" in str(s) or "衞" in str(s):
        return "防衛"
    return None


def _auto_tag_round(
    side: str,
    opening: str,
    execution: str,
    mid_round: str,
    site: str | None,
    shapes: dict[str, Any],
    eco: str | None = None,
    setup_info: dict[str, Any] | None = None,
    smokes: list[dict[str, Any]] | None = None,
    turns_db: str = "",
) -> list[str]:
    """ラウンドの集計ラベル・エージェント配置(who/zones)・スモーク検知結果から最も合う shapes キーを自動推定する。"""
    picked: list[str] = []
    op_l = (opening or "").lower()
    ex_l = (execution or "").lower()
    mr_l = (mid_round or "").lower()
    td_l = (turns_db or "").lower()
    su = setup_info or {}
    zones = {k: int(v) for k, v in (su.get("zones") or {}).items() if v}
    who = su.get("who") or {}
    sm_list = list(smokes or su.get("smokes") or [])
    available = {k: v for k, v in shapes.items() if v.get("side") == side}

    def _add_by_pred(pred) -> None:
        for k in available:
            if k not in picked and pred(k, available[k]):
                picked.append(k)
                return

    if side == "攻撃":
        if "エコ" in str(eco or "") and "エコBラッシュ" in available:
            picked.append("エコBラッシュ")

        # 1) スモーク検知結果からのタグ付け（MID膨らみモク / Aメインモク / Bメインモク）
        for sm in sm_list:
            sz = str(sm.get("zone") or "")
            sa = str(sm.get("area") or sm.get("callout") or "").lower()
            if sz == "Mid" or "mid" in sa:
                _add_by_pred(lambda k, v: "モク" in k and "MID" in k.upper())
            elif sz == "A" or "a main" in sa or "a lobby" in sa:
                _add_by_pred(lambda k, v: "モク" in k and ("A" in k or "シャワー" in k))
            elif sz in ("B", "C") or "b main" in sa or "c main" in sa:
                _add_by_pred(lambda k, v: "モク" in k and ("B" in k or "C" in k or "フッカー" in k))

        # 2) 初動形（1-3-1 / 4-1 / 各メインテイク）
        mid_cnt = zones.get("Mid", 0)
        a_cnt = zones.get("A", 0)
        b_cnt = zones.get("B", 0) + zones.get("C", 0)
        if mid_cnt >= 3 or "1-3-1" in op_l or "1－3－1" in op_l:
            _add_by_pred(lambda k, v: "1－3－1" in k or "1-3-1" in k)
        if (b_cnt >= 4 and a_cnt <= 1) or "b4-1" in op_l or "b5" in op_l:
            _add_by_pred(lambda k, v: "4－1" in k and "A" not in k)
        elif (a_cnt >= 4 and b_cnt <= 1) or "a4-1" in op_l:
            _add_by_pred(lambda k, v: "4-1" in k or "4－1" in k)

        if b_cnt >= 2 or "b " in op_l or "b lobby" in op_l or "b main" in op_l:
            _add_by_pred(lambda k, v: v.get("category") in ("序盤", "形") and ("B" in k or "フッカー" in k or "C" in k))
        if mid_cnt >= 2 or "mid" in op_l:
            _add_by_pred(lambda k, v: v.get("category") in ("序盤", "形") and ("MID" in k.upper() or "1－3－1" in k))
        if a_cnt >= 2 or "a " in op_l or "a lobby" in op_l or "a main" in op_l:
            _add_by_pred(lambda k, v: v.get("category") in ("序盤", "形") and "A" in k and "4-1" not in k)

        # 3) 中盤フェイク・転換
        if "b 見せ" in mr_l or "b見せ" in mr_l or ("b → a" in td_l and site == "A") or "bフェイク" in mr_l:
            _add_by_pred(lambda k, v: "Bフェイク" in k or "フェイク" in k)
        elif "a 見せ" in mr_l or "a見せ" in mr_l or ("a → b" in td_l and site == "B") or "aフェイク" in mr_l:
            _add_by_pred(lambda k, v: "Aフェイク" in k or "フェイク" in k)
        elif "見せ" in mr_l or "フェイク" in mr_l:
            _add_by_pred(lambda k, v: "フェイク" in k)

        # 4) エクセキューション（スプリット / ラッシュ / セット）
        if "スプリット" in ex_l:
            _add_by_pred(lambda k, v: "スプリット" in k and (not site or site in k))
        elif "ラッシュ" in ex_l:
            _add_by_pred(lambda k, v: "ラッシュ" in k and (not site or site in k))
        elif "セット" in ex_l:
            _add_by_pred(lambda k, v: "セット" in k and (not site or site in k))
    else:
        # 防衛：エージェント別配置 (who) とゾーン人数 (zones) を優先
        b_agents = set(who.get("B", []) + who.get("C", []))
        a_agents = set(who.get("A", []))
        mid_agents = set(who.get("Mid", []))
        has_op_b = any(ag in b_agents for ag in ("Chamber", "Jett"))
        has_anc_b = any(ag in b_agents for ag in ("Sage", "Vyse", "Deadlock", "Cypher", "Killjoy", "Viper"))
        has_anc_a = any(ag in a_agents for ag in ("Sage", "Vyse", "Deadlock", "Cypher", "Killjoy", "Viper"))
        has_op_mid = any(ag in mid_agents for ag in ("Chamber", "Jett"))

        if "前詰め" in op_l or "前目" in op_l or "プッシュ" in op_l:
            _add_by_pred(lambda k, v: "プッシュ" in k)
        elif zones.get("Mid", 0) >= 3 or "1-3-1" in op_l or "1－3－1" in op_l:
            _add_by_pred(lambda k, v: "1－3－1" in k or "1-3-1" in k)
        elif has_op_mid or (not who and "mid" in op_l):
            _add_by_pred(lambda k, v: "MID" in k.upper() or "1－3－1" in k)
        elif has_anc_b and has_op_b:
            _add_by_pred(lambda k, v: "B" in k and ("セージ" in k or "アンカー" in k or "チェンバー" in k) and "プッシュ" not in k)
        elif has_anc_a and not has_anc_b:
            _add_by_pred(lambda k, v: k.startswith("A"))
        elif has_anc_b and not has_op_b:
            _add_by_pred(lambda k, v: k.startswith("B") and "チェンバー" not in k and "プッシュ" not in k)
        elif (not who and ("a " in op_l or "a寄せ" in op_l)) or zones.get("A", 0) >= 3:
            _add_by_pred(lambda k, v: k.startswith("A"))
        else:
            _add_by_pred(lambda k, v: "デフォ" in k or "B" in k)

    if not picked and available:
        picked.append(next(iter(available)))
    return picked[:3]


def classify_round_record(
    r: dict[str, Any],
    rt: dict[str, Any],
    target_no: int,
    shapes: dict[str, Any],
    rep_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """1 ラウンドの report データからプレイブック分類レコードを組み立てる。"""
    rep = rep_meta or {}
    rn = int(r.get("n") or 0)
    s1 = _canon_side(r.get("side1") or r.get("side"))
    side = s1 if target_no == 1 else ("防衛" if s1 == "攻撃" else "攻撃")
    wno = r.get("winner_no")
    win = (wno == target_no) if wno in (1, 2) else bool(r.get("win"))

    t1 = str(rep.get("team1") or rep.get("tag1") or "Team 1")
    t2 = str(rep.get("team2") or rep.get("tag2") or "Team 2")
    opp = t2 if target_no == 1 else t1
    mid = str(rep.get("match_id") or "match")
    score = (
        f"{rep.get('score1', '?')}-{rep.get('score2', '?')}"
        if target_no == 1
        else f"{rep.get('score2', '?')}-{rep.get('score1', '?')}"
    )

    site = rt.get("plant_site") or rt.get("site") or r.get("site")
    eco_raw = str(r.get("buy1" if target_no == 1 else "buy2") or r.get("eco") or "")
    eco = "ピストル" if rn in (1, 13) else ("エコ" if "eco" in eco_raw.lower() else ("フルバイ" if eco_raw else "標準"))

    su = ((rt.get("setups") or {}).get(str(target_no)) or {})
    opening = str(r.get("setup_name") or rt.get("opening") or su.get("name") or f"{su.get('code') or '標準'} 配置")
    if su.get("who") and opening == f"{su.get('code') or '標準'} 配置":
        opening = f"{su.get('code', '')} ({', '.join(f'{z}:' + '・'.join(ags) for z, ags in su['who'].items())})".strip()

    # スモーク検知結果の抽出（rt.smokes の 1:40 / 1:30 / 1:15）
    smokes_all: list[dict[str, Any]] = []
    for lbl in ("1:40", "1:30", "1:15", "1:00"):
        for sm in (rt.get("smokes") or {}).get(lbl) or []:
            if sm.get("team") in (None, target_no):
                smokes_all.append({**sm, "mark": lbl})

    ex_raw = str(rt.get("execute") or r.get("execution") or "")
    execution = ex_raw or (f"{site} セット" if site else ("設置前 決着" if not rt.get("plant") else "サイト攻防"))
    mid_round = str(rt.get("mid") or r.get("mid_round") or "中盤展開")
    turns_db = str(r.get("turns_db") or "")

    sel = _auto_tag_round(
        side,
        opening,
        execution,
        mid_round,
        site,
        shapes,
        eco=eco,
        setup_info=su,
        smokes=smokes_all,
        turns_db=turns_db,
    )
    assets = r.get("assets") or {}
    gif = assets.get("minimap_gif") or ""
    if gif:
        gif = f"gifs/{Path(gif).name}" if "/" not in str(gif) and "\\" not in str(gif) else str(gif).replace("\\", "/")
    pe = None
    if rt.get("plant") is not None and rt.get("start") is not None:
        pe = round(float(rt["plant"]) - float(rt["start"]), 1)

    return {
        "id": f"{mid}_R{rn:02d}",
        "match_id": mid,
        "date": str(rep.get("date") or ""),
        "stage": str(rep.get("stage") or rep.get("event") or ""),
        "score": score,
        "round_n": rn,
        "opp": opp,
        "side": side,
        "win": win,
        "how_ja": str(r.get("how_ja") or r.get("how") or ("勝利" if win else "敗北")),
        "eco": eco,
        "site": site,
        "plant_site": site if rt.get("plant") is not None else None,
        "plant_e": pe,
        "opening": opening,
        "mid_round": mid_round,
        "execution": execution,
        "post_plant": f"{site} ポスト" if rt.get("plant") is not None and site else "設置なし",
        "smokes": smokes_all,
        "hand_db": "",
        "turns_db": turns_db,
        "reason": str((r.get("notes") or [""])[0] if isinstance(r.get("notes"), list) else ""),
        "ults": ", ".join(str(u) for u in (r.get("ults") or []) if u),
        "yt_url": str(r.get("yt_url") or ""),
        "duration": str(r.get("duration") or ""),
        "gif": gif,
        "shapes_selected": sel,
        "shape": " / ".join(sel),
        "focus_phase": "序盤",
        "is_rep": False,
        "rep_note": "",
    }


def classify_report_rounds(
    reports: list[dict] | dict | Path,
    shapes: dict[str, Any],
    team_filter: str | None = None,
    map_filter: str | None = None,
) -> list[dict[str, Any]]:
    """report.json（単体・リスト・ディレクトリ）からプレイブック用ラウンド一覧を生成・自動分類する。"""
    rep_list: list[dict] = []
    if isinstance(reports, Path):
        paths = [reports] if reports.is_file() else sorted(reports.rglob("report.json"))
        for p in paths:
            try:
                rep_list.append(json.loads(p.read_text("utf-8")))
            except Exception:
                continue
    elif isinstance(reports, dict):
        rep_list = [reports]
    else:
        rep_list = list(reports)

    tf = _norm(team_filter) if team_filter else ""
    mf = _norm(map_filter) if map_filter else ""
    out: list[dict[str, Any]] = []

    for rep in rep_list:
        rmap = str(rep.get("map") or "")
        if mf and _norm(rmap) != mf:
            continue
        t1 = str(rep.get("team1") or rep.get("tag1") or "Team 1")
        t2 = str(rep.get("team2") or rep.get("tag2") or "Team 2")
        target_no: int | None = None
        if tf:
            if tf in _norm(t1) or tf in _norm(rep.get("tag1")):
                target_no = 1
            elif tf in _norm(t2) or tf in _norm(rep.get("tag2")):
                target_no = 2
            else:
                continue
        else:
            target_no = 1

        opp = t2 if target_no == 1 else t1
        mid = str(rep.get("match_id") or "match")
        score = f"{rep.get('score1', '?')}-{rep.get('score2', '?')}" if target_no == 1 else f"{rep.get('score2', '?')}-{rep.get('score1', '?')}"
        tac_rounds = (rep.get("tactics") or {}).get("rounds") or {}

        for r in rep.get("rounds") or []:
            rn = int(r.get("n") or 0)
            s1 = _canon_side(r.get("side1") or r.get("side"))
            if s1 not in ("攻撃", "防衛"):
                continue
            rt = r.get("tactics") or tac_rounds.get(str(rn)) or {}
            out.append(classify_round_record(r, rt, target_no, shapes, rep_meta=rep))
    return out


# ------------------------------------------------------------------ HTML プレイブック生成（DADS スタイル）
def _e(v: Any) -> str:
    return _html.escape(str(v if v is not None else ""))


def _render_memo_blocks(memo: str) -> str:
    """comp_memo テキストを DADS の構造化ブロック HTML に変換する。"""
    blocks: list[tuple[str, list[str]]] = []
    cur_title = ""
    cur_lines: list[str] = []
    for raw in (memo or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("■"):
            if cur_lines or cur_title:
                blocks.append((cur_title, cur_lines))
            cur_title = line.lstrip("■").strip()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_lines or cur_title:
        blocks.append((cur_title, cur_lines))

    html_blocks: list[str] = []
    for title, lines in blocks:
        lis = []
        for ln in lines:
            if ln.startswith("・"):
                lis.append(f"<li><strong>{_e(ln[1:].strip())}</strong></li>")
            elif ln.startswith("→"):
                lis.append(f'<li class="memo-arrow">→ {_e(ln[1:].strip())}</li>')
            else:
                lis.append(f"<li>{_e(ln)}</li>")
        h3 = f'<h3 class="memo-sub">{_e(title)}</h3>' if title else ""
        html_blocks.append(f'<div class="memo-block">{h3}<ul class="memo-list">{"".join(lis)}</ul></div>')
    return f'<div class="memo-panel">{"".join(html_blocks)}</div>'


def _round_card_html(r: dict, is_rep: bool = False) -> str:
    side = r.get("side") or "攻撃"
    win = bool(r.get("win"))
    s_badge = '<span class="dads-badge badge-atk">攻撃</span>' if side == "攻撃" else '<span class="dads-badge badge-def">防衛</span>'
    w_badge = f'<span class="dads-badge {"badge-win" if win else "badge-lose"}">{"WIN" if win else "LOSE"}（{_e(r.get("how_ja") or "")}）</span>'
    rep_badge = '<span class="dads-badge badge-star">★ 代表例</span>' if is_rep else ""
    eco = f'<span class="dads-badge badge-neutral">{_e(r.get("eco"))}</span>' if r.get("eco") else ""
    gif = r.get("gif") or ""
    media_parts = []
    if gif:
        media_parts.append(f'<img src="{_e(gif)}" loading="lazy" alt="R{r.get("round_n")}" data-title="{_e(r.get("id"))}" onclick="openImgModal(this.src, this.dataset.title)">')
    if r.get("yt_url"):
        media_parts.append(f'<a class="dads-btn-vod" href="{_e(r.get("yt_url"))}" target="_blank" rel="noopener">▶ YouTubeで該当シーンを見る</a>')
    media_html = f'<div class="ex-media">{"".join(media_parts)}</div>' if media_parts else ""
    note_html = f'<div class="rep-comment">{_e(r.get("rep_note"))}</div>' if r.get("rep_note") else ""
    reason_html = f'<div style="margin-top:8px;font-size:12px;color:var(--dads-text-muted);">{_e(r.get("reason"))}</div>' if r.get("reason") else ""
    dt_str = f"{r.get('date', '')} {r.get('stage', '')} vs {r.get('opp', '')} ({r.get('score', '')})".strip()
    return f"""
    <article class="ex-card">
      <div class="ex-head">
        <div class="ex-title">{rep_badge}{s_badge}<strong>R{int(r.get('round_n') or 0):02d}</strong>{w_badge}{eco}</div>
        <div class="ex-meta">{_e(dt_str)}</div>
      </div>
      <div class="ex-body">
        {media_html}
        <div>
          {note_html}
          <dl class="phase-dl">
            <div><dt>序盤</dt><dd>{_e(r.get("opening") or "-")}</dd></div>
            <div><dt>中盤</dt><dd>{_e(r.get("mid_round") or "-")}</dd></div>
            <div><dt>エクセキューション</dt><dd>{_e(r.get("execution") or "-")}</dd></div>
            <div><dt>ポストプラント</dt><dd>{_e(r.get("post_plant") or "-")}</dd></div>
          </dl>
          {reason_html}
        </div>
      </div>
    </article>"""


def _shape_section_html(sid: str, name: str, info: dict, matched: list[dict], share_only: bool) -> str:
    side = info.get("side") or "攻撃"
    cat = info.get("category") or "全体"
    n = len(matched)
    w = sum(1 for r in matched if r.get("win"))
    rate = f"{round(100 * w / n)}%" if n else "-"
    s_badge = '<span class="dads-badge badge-atk">攻撃</span>' if side == "攻撃" else '<span class="dads-badge badge-def">防衛</span>'

    img = info.get("tactival_img") or ""
    has_tv_json = bool(info.get("tactival_json"))
    tv_btn = (
        f'<div class="tv-controls"><button type="button" class="dads-btn" data-shape="{_e(name)}" onclick="openInTactival(this.dataset.shape)">'
        f'TactiVal JSON 保存 / 開く</button>'
        f'<span style="font-size:11px;color:var(--dads-text-muted);">{"v3 ボード生成済み" if has_tv_json else ""}</span></div>'
        if has_tv_json else ""
    )
    if img:
        img_box = f'<img src="{img}" alt="{_e(name)}" data-title="{_e(name)}" onclick="openImgModal(this.src, this.dataset.title)">{tv_btn}'
    else:
        img_box = '<div class="tactival-placeholder">TactiVal 画像未登録</div>'

    desc = str(info.get("desc") or "").strip()
    desc_html = "<br>".join(_e(ln) for ln in desc.splitlines()) if desc else '<span style="color:var(--dads-text-muted)">（説明文はまだ登録されていません）</span>'

    reps = [r for r in matched if r.get("is_rep")]
    others = [r for r in matched if not r.get("is_rep")]
    top_cards = (reps + others)[:4]
    rest_cards = (reps + others)[4:]

    cards_html = "".join(_round_card_html(r, is_rep=bool(r.get("is_rep"))) for r in top_cards)
    if not cards_html:
        cards_html = '<p style="color:var(--dads-text-muted);font-size:13px;">この型に紐づいた実戦ラウンドはまだありません（初動テンプレート図解のみ）。</p>'
    acc_html = ""
    if rest_cards:
        rest_inner = "".join(_round_card_html(r) for r in rest_cards)
        acc_html = f'<details class="dads-accordion"><summary>その他の該当ラウンド（{len(rest_cards)} 件）をすべて表示</summary><div class="accordion-body"><div class="ex-grid">{rest_inner}</div></div></details>'

    search_blob = " ".join([
        name.lower(), side, cat, desc.lower(),
        " ".join(f"{r.get('opp','')} {r.get('opening','')} {r.get('execution','')}" for r in matched[:8]).lower()
    ])
    return f"""
      <section class="shape-section" id="{_e(sid)}" data-shape="{_e(name)}" data-search="{_e(search_blob)}">
        <div class="shape-head">
          <div class="shape-head-left">{s_badge}<span class="dads-badge badge-neutral">{_e(cat)}</span><h3>{_e(name)}</h3></div>
          <div class="shape-stats">採用: <strong>{n}</strong> ラウンド ｜ 勝率: <strong>{rate}</strong>（{w}勝 {n - w}敗）</div>
        </div>
        <div class="shape-overview">
          <div class="tactival-box"><div class="box-label">TactiVal 図解</div>{img_box}</div>
          <div class="desc-box"><div class="box-label">戦術のポイント・説明</div><div class="desc-text">{desc_html}</div></div>
        </div>
        <div>
          <h4 style="margin:0 0 10px;font-size:14px;">代表的なラウンド（実戦例）</h4>
          <div class="ex-grid">{cards_html}</div>
        </div>
        {acc_html}
      </section>"""


DADS_CSS = """
  :root {
    --dads-bg: #FFFFFF; --dads-bg-sub: #F4F5F6; --dads-bg-hover: #E8E9EC;
    --dads-border: #D8D8DB; --dads-border-strong: #94949B;
    --dads-text: #1A1A1C; --dads-text-sub: #4D4D50; --dads-text-muted: #6E6E73;
    --dads-blue: #0017C1; --dads-blue-hover: #00118F; --dads-blue-bg: #E8EEFF;
    --dads-red: #EC0000; --dads-red-bg: #FFF0F0;
    --dads-green: #197A4B; --dads-green-bg: #E6F4EC;
    --dads-amber: #9E6B00; --dads-amber-bg: #FFF8E1;
    --radius-md: 8px; --radius-lg: 12px;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--dads-bg); color: var(--dads-text); font-family: "Noto Sans JP", "BIZ UDPGothic", "Yu Gothic UI", sans-serif; font-size: 15px; line-height: 1.65; }
  .site-header { position: sticky; top: 0; z-index: 40; background: rgba(255,255,255,0.97); backdrop-filter: blur(8px); border-bottom: 2px solid var(--dads-blue); padding: 10px 20px; }
  .header-inner { max-width: 1440px; margin: 0 auto; display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }
  .brand { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .site-title { margin: 0; font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .tag-official { background: var(--dads-blue); color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 700; }
  .progress-pill { font-size: 12px; background: var(--dads-bg-sub); border: 1px solid var(--dads-border); padding: 3px 10px; border-radius: 999px; color: var(--dads-text-sub); }
  .nav-tabs, .header-actions { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .tab-btn, .dads-btn { background: #fff; color: var(--dads-blue); border: 1px solid var(--dads-blue); border-radius: 6px; padding: 6px 12px; font-size: 12px; font-weight: 700; cursor: pointer; text-decoration: none; }
  .tab-btn { color: var(--dads-text-sub); border-color: var(--dads-border-strong); }
  .tab-btn.active, .dads-btn.primary { background: var(--dads-blue); color: #fff; border-color: var(--dads-blue); }
  .dads-btn:hover { background: var(--dads-blue-bg); }
  .dads-btn.primary:hover { background: var(--dads-blue-hover); }
  .dads-input, .dads-select, .dads-textarea { width: 100%; padding: 8px 10px; font-size: 13px; border: 1px solid var(--dads-border-strong); border-radius: 6px; background: #fff; color: var(--dads-text); font-family: inherit; }
  .dads-badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; line-height: 1.4; }
  .badge-atk { background: var(--dads-red-bg); color: var(--dads-red); border: 1px solid rgba(236,0,0,0.25); }
  .badge-def { background: var(--dads-blue-bg); color: var(--dads-blue); border: 1px solid rgba(0,23,193,0.25); }
  .badge-win { background: var(--dads-green-bg); color: var(--dads-green); }
  .badge-lose { background: var(--dads-red-bg); color: var(--dads-red); }
  .badge-neutral { background: var(--dads-bg-sub); color: var(--dads-text-sub); border: 1px solid var(--dads-border); }
  .badge-star { background: var(--dads-amber-bg); color: var(--dads-amber); border: 1px solid rgba(158,107,0,0.3); }
  .view-section { display: none; }
  .view-section.active { display: block; }
  .page-wrap { max-width: 1380px; margin: 0 auto; display: grid; grid-template-columns: 280px 1fr; gap: 28px; padding: 20px 24px 60px; align-items: start; }
  .page-wrap.toc-hidden { grid-template-columns: 1fr; }
  .page-wrap.toc-hidden .toc-nav { display: none; }
  @media (max-width: 1024px) { .page-wrap { grid-template-columns: 1fr; } .toc-nav { position: static !important; max-height: none !important; } }
  .toc-nav { position: sticky; top: 72px; max-height: calc(100vh - 90px); overflow-y: auto; background: var(--dads-bg-sub); border: 1px solid var(--dads-border); border-radius: var(--radius-md); padding: 14px; font-size: 13px; }
  .toc-heading { font-size: 11px; font-weight: 700; color: var(--dads-text-muted); margin: 12px 0 6px; }
  .toc-list { list-style: none; margin: 0; padding: 0; }
  .toc-list li { margin-bottom: 3px; }
  .toc-list a { display: flex; align-items: center; justify-content: space-between; gap: 6px; padding: 4px 8px; border-radius: 6px; color: var(--dads-text); text-decoration: none; }
  .toc-list a:hover { background: var(--dads-bg-hover); color: var(--dads-blue); }
  .toc-cat { font-size: 10px; font-weight: 700; color: var(--dads-text-sub); background: #fff; border: 1px solid var(--dads-border); padding: 1px 5px; border-radius: 4px; flex-shrink: 0; }
  .toc-cnt { font-size: 11px; color: var(--dads-text-muted); flex-shrink: 0; }
  .content-col { min-width: 0; }
  .playbook-search-bar { display: flex; gap: 10px; align-items: center; margin-bottom: 18px; background: var(--dads-bg-sub); padding: 12px 16px; border-radius: var(--radius-md); border: 1px solid var(--dads-border); }
  .summary-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: var(--dads-bg-sub); border: 1px solid var(--dads-border); border-radius: var(--radius-md); padding: 12px 16px; }
  .stat-card .lbl { font-size: 12px; color: var(--dads-text-sub); font-weight: 700; }
  .stat-card .val { font-size: 20px; font-weight: 700; color: var(--dads-blue); }
  h2.section-h2 { font-size: 22px; font-weight: 700; border-left: 6px solid var(--dads-blue); padding-left: 12px; margin: 34px 0 16px; }
  h2.section-h2.atk-h2 { border-left-color: var(--dads-red); }
  .memo-panel { background: var(--dads-bg-sub); border: 1px solid var(--dads-border); border-radius: var(--radius-lg); padding: 18px; display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; }
  .memo-block { background: #fff; border: 1px solid var(--dads-border); border-radius: var(--radius-md); padding: 14px 16px; }
  .memo-sub { margin: 0 0 8px; font-size: 15px; color: var(--dads-blue); border-bottom: 1px solid var(--dads-border); padding-bottom: 5px; }
  .memo-list { margin: 0; padding-left: 18px; font-size: 13.5px; }
  .memo-list li.memo-arrow { list-style: none; margin-left: -12px; color: var(--dads-text-sub); }
  .shape-section { background: #fff; border: 1px solid var(--dads-border); border-radius: var(--radius-lg); padding: 20px 22px; margin-bottom: 22px; }
  .shape-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding-bottom: 12px; border-bottom: 1px solid var(--dads-border); margin-bottom: 16px; }
  .shape-head-left { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .shape-head h3 { margin: 0; font-size: 19px; font-weight: 700; }
  .shape-overview { display: grid; grid-template-columns: minmax(260px, 400px) 1fr; gap: 18px; margin-bottom: 18px; }
  @media (max-width: 820px) { .shape-overview { grid-template-columns: 1fr; } }
  .box-label { font-size: 12px; font-weight: 700; color: var(--dads-text-muted); margin-bottom: 6px; }
  .tactival-box, .desc-box { background: var(--dads-bg-sub); border: 1px solid var(--dads-border); border-radius: var(--radius-md); padding: 12px; }
  .tactival-box img { width: 100%; max-height: 320px; object-fit: contain; border-radius: 6px; cursor: zoom-in; display: block; background: #fff; }
  .tv-controls { display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
  .ex-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  @media (max-width: 860px) { .ex-grid { grid-template-columns: 1fr; } }
  .ex-card { border: 1px solid var(--dads-border); border-radius: var(--radius-md); overflow: hidden; background: #fff; }
  .ex-head { background: var(--dads-bg-sub); padding: 8px 14px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 6px; border-bottom: 1px solid var(--dads-border); }
  .ex-body { display: flex; flex-direction: column; gap: 12px; padding: 12px 14px; }
  .ex-media img { width: 100%; height: 260px; object-fit: contain; background: #0b0f14; border-radius: 6px; cursor: zoom-in; }
  .dads-btn-vod { display: block; text-align: center; background: var(--dads-blue); color: #fff; text-decoration: none; font-weight: 700; font-size: 13px; padding: 7px 12px; border-radius: 6px; margin-top: 6px; }
  .phase-dl { margin: 0; display: grid; gap: 5px; font-size: 13.5px; }
  .phase-dl > div { display: grid; grid-template-columns: 115px 1fr; gap: 8px; align-items: baseline; }
  .phase-dl dt { font-size: 11px; font-weight: 700; color: var(--dads-text-sub); background: var(--dads-bg-sub); padding: 2px 6px; border-radius: 4px; text-align: center; }
  .phase-dl dd { margin: 0; }
  .dads-table-wrap { overflow-x: auto; border: 1px solid var(--dads-border); border-radius: var(--radius-md); }
  table.dads-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.dads-table th, table.dads-table td { padding: 9px 11px; border-bottom: 1px solid var(--dads-border); text-align: left; }
  table.dads-table th { background: var(--dads-bg-sub); font-weight: 700; color: var(--dads-text-sub); }
  .sub-container { max-width: 1360px; margin: 0 auto; padding: 20px 24px 60px; }
  .shapes-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; margin-top: 14px; }
  .shape-card { background: var(--dads-bg-sub); border: 1px solid var(--dads-border); border-radius: var(--radius-md); padding: 14px; display: flex; flex-direction: column; gap: 8px; }
  .shape-card img { width: 100%; max-height: 220px; object-fit: contain; background: #fff; border-radius: 6px; }
  .modal { display: none; position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,0.78); align-items: center; justify-content: center; padding: 24px; }
  .modal.open { display: flex; }
  .modal-box { background: #fff; border-radius: var(--radius-lg); max-width: 960px; width: 100%; padding: 18px; position: relative; }
  .modal-box img { width: 100%; max-height: 78vh; object-fit: contain; background: #111; border-radius: 6px; }
  .modal-close { position: absolute; top: 10px; right: 14px; background: transparent; border: none; font-size: 26px; cursor: pointer; }
  .hidden-by-search { display: none !important; }
  @media print { .site-header, .toc-nav, .playbook-search-bar { display: none !important; } .page-wrap { grid-template-columns: 1fr; padding: 0; } }
"""


def build_playbook_html(data: dict[str, Any], out_path: Path, share_only: bool = False) -> Path:
    """プレイブック HTML（share_only=True なら共有閲覧版、False なら編集タブ付きワークスペース版）を生成する。"""
    map_name = str(data.get("map") or "Map")
    team = str(data.get("team") or "LOUD")
    comp = data.get("comp") or []
    shapes: dict[str, Any] = data.get("shapes") or {}
    rounds: list[dict] = data.get("rounds") or []
    memo = str(data.get("comp_memo") or "")

    # カテゴリ順・出現数順で並べる
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    by_shape_rounds: dict[str, list[dict]] = defaultdict(list)
    for r in rounds:
        for s in (r.get("shapes_selected") or []):
            by_shape_rounds[s].append(r)

    def _ordered(side: str) -> list[tuple[str, dict]]:
        items = [(k, v) for k, v in shapes.items() if (v.get("side") or "攻撃") == side]
        items.sort(key=lambda kv: (cat_rank.get(kv[1].get("category") or "全体", 99), -len(by_shape_rounds.get(kv[0], [])), kv[0]))
        return items

    atk_items = _ordered("攻撃")
    def_items = _ordered("防衛")

    toc_atk = "".join(
        f'<li><a href="#atk-{i}"><span class="toc-cat">{_e(v.get("category") or "全体")}</span> '
        f'{_e(k)} <span class="toc-cnt">{len(by_shape_rounds.get(k, []))}</span></a></li>'
        for i, (k, v) in enumerate(atk_items, 1)
    )
    toc_def = "".join(
        f'<li><a href="#def-{i}"><span class="toc-cat">{_e(v.get("category") or "全体")}</span> '
        f'{_e(k)} <span class="toc-cnt">{len(by_shape_rounds.get(k, []))}</span></a></li>'
        for i, (k, v) in enumerate(def_items, 1)
    )

    atk_sections = "".join(
        _shape_section_html(f"atk-{i}", k, v, by_shape_rounds.get(k, []), share_only)
        for i, (k, v) in enumerate(atk_items, 1)
    )
    def_sections = "".join(
        _shape_section_html(f"def-{i}", k, v, by_shape_rounds.get(k, []), share_only)
        for i, (k, v) in enumerate(def_items, 1)
    )

    def _round_row_html(r: dict) -> str:
        tags = "".join(
            f'<span class="dads-badge badge-neutral">{_e(s)}</span> '
            for s in (r.get("shapes_selected") or [])
        ) or "-"
        yt = r.get("yt_url")
        vod_cell = f'<a href="{_e(yt)}" target="_blank" rel="noopener">VOD</a>' if yt else "-"
        search_txt = (
            f"{r.get('id', '')} {r.get('opp', '')} {r.get('side', '')} "
            f"{r.get('shape', '')} {r.get('opening', '')}"
        ).lower()
        side_cls = "badge-atk" if r.get("side") == "攻撃" else "badge-def"
        win_cls = "badge-win" if r.get("win") else "badge-lose"
        win_lbl = "WIN" if r.get("win") else "LOSE"
        return (
            f'<tr data-search="{_e(search_txt)}">'
            f'<td>{_e(r.get("date", ""))} vs {_e(r.get("opp", ""))}</td>'
            f'<td><strong>R{int(r.get("round_n") or 0):02d}</strong></td>'
            f'<td><span class="dads-badge {side_cls}">{_e(r.get("side"))}</span></td>'
            f'<td><span class="dads-badge {win_cls}">{win_lbl}</span></td>'
            f'<td>{tags}</td>'
            f'<td>{_e(r.get("opening") or "-")}</td>'
            f'<td>{_e(r.get("execution") or "-")}</td>'
            f'<td>{vod_cell}</td></tr>'
        )

    round_rows = "".join(_round_row_html(r) for r in rounds)

    n_classified = sum(1 for r in rounds if r.get("shapes_selected"))
    n_matches = len({r.get("match_id") for r in rounds if r.get("match_id")})
    title = f"{team} {map_name} プレイブック" + ("（共有版）" if share_only else "")

    # TactiVal ダウンロード用の JSON マップ
    tv_payload = {
        k: {"tactival_json": v.get("tactival_json"), "tactival_url": "https://tactival.app/board"}
        for k, v in shapes.items() if v.get("tactival_json")
    }
    tv_json_str = json.dumps(tv_payload, ensure_ascii=False).replace("</", "<\\/")

    def _shape_edit_card(k: str, v: dict) -> str:
        side_cls = "badge-atk" if v.get("side") == "攻撃" else "badge-def"
        tv_img = v.get("tactival_img") or ""
        img_tag = f'<img src="{tv_img}" alt="{_e(k)}">' if tv_img else ""
        return (
            f'<div class="shape-card">'
            f'<div><span class="dads-badge {side_cls}">{_e(v.get("side"))}</span> '
            f'<span class="dads-badge badge-neutral">{_e(v.get("category"))}</span> <strong>{_e(k)}</strong></div>'
            f'{img_tag}'
            f'<textarea class="dads-textarea" rows="4" data-shape="{_e(k)}" onchange="updateShapeDesc(this.dataset.shape, this.value)">{_e(v.get("desc") or "")}</textarea>'
            f'<button type="button" class="dads-btn" data-shape="{_e(k)}" onclick="openInTactival(this.dataset.shape)">TactiVal JSON 保存</button>'
            f'</div>'
        )

    # ワークスペース版用ヘッダータブ＆追加ビュー
    nav_html = ""
    extra_views = ""
    extra_js = ""
    if not share_only:
        state_payload = json.dumps({
            "exported_at": data.get("exported_at"),
            "map": map_name,
            "team": team,
            "comp": comp,
            "comp_memo": memo,
            "shapes": {k: {sk: sv for sk, sv in v.items() if sk != "tactival_json"} for k, v in shapes.items()},
            "rounds": rounds,
        }, ensure_ascii=False).replace("</", "<\\/")
        nav_html = """
        <div class="nav-tabs">
          <button type="button" class="tab-btn active" onclick="switchView('Playbook', this)">1. プレイブック</button>
          <button type="button" class="tab-btn" onclick="switchView('Rounds', this)">2. ラウンド分類</button>
          <button type="button" class="tab-btn" onclick="switchView('Shapes', this)">3. 初動配置・TactiVal</button>
          <button type="button" class="tab-btn" onclick="switchView('Memo', this)">4. 構成メモ</button>
        </div>
        <div class="header-actions">
          <label class="dads-btn">JSON 読込<input type="file" accept=".json,application/json" style="display:none" onchange="if(this.files[0]) loadClassifiedJson(this.files[0])"></label>
          <button type="button" class="dads-btn" onclick="downloadClassifiedJson()">JSON 保存</button>
          <button type="button" class="dads-btn primary" onclick="exportShareHtml()">共有版 HTML 出力</button>
        </div>"""
        shape_cards_edit = "".join(_shape_edit_card(k, v) for k, v in atk_items + def_items)
        extra_views = f"""
  <div id="viewRounds" class="view-section">
    <div class="sub-container">
      <h2>2. ラウンド分類一覧（{len(rounds)} 件）</h2>
      <div class="dads-table-wrap"><table class="dads-table">
        <thead><tr><th>試合</th><th>R</th><th>攻防</th><th>勝敗</th><th>分類タグ</th><th>序盤</th><th>エクセキューション</th><th>VOD</th></tr></thead>
        <tbody>{round_rows or '<tr><td colspan="8">ラウンドデータなし（テンプレート生成モード）</td></tr>'}</tbody>
      </table></div>
    </div>
  </div>
  <div id="viewShapes" class="view-section">
    <div class="sub-container">
      <h2>3. 初動配置・大きな形の TactiVal 図解＆説明（{len(shapes)} 種）</h2>
      <div class="shapes-grid">{shape_cards_edit}</div>
    </div>
  </div>
  <div id="viewMemo" class="view-section">
    <div class="sub-container">
      <div style="max-width:920px;margin:0 auto;">
        <h2>4. 構成メモ編集</h2>
        <textarea id="memoEditor" class="dads-textarea" rows="26" onchange="updateMemo(this.value)">{_e(memo)}</textarea>
      </div>
    </div>
  </div>"""
        file_stem = _e(_norm(team) + "_" + _norm(map_name))
        extra_js = f"""
const STATE = {state_payload};
function _escHtml(s) {{
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}}
function _renderMemoHtml(memoText) {{
  const lines = String(memoText || "").split(/\\r?\\n/);
  const blocks = [];
  let curTitle = "", curLines = [];
  for (const raw of lines) {{
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("■")) {{
      if (curLines.length || curTitle) blocks.push([curTitle, curLines]);
      curTitle = line.replace(/^■+/, "").trim();
      curLines = [];
    }} else {{
      curLines.push(line);
    }}
  }}
  if (curLines.length || curTitle) blocks.push([curTitle, curLines]);
  const htmlBlocks = blocks.map(([title, blines]) => {{
    const lis = blines.map(ln => {{
      if (ln.startsWith("・")) return "<li><strong>" + _escHtml(ln.slice(1).trim()) + "</strong></li>";
      if (ln.startsWith("→")) return '<li class="memo-arrow">→ ' + _escHtml(ln.slice(1).trim()) + "</li>";
      return "<li>" + _escHtml(ln) + "</li>";
    }}).join("");
    const h3 = title ? '<h3 class="memo-sub">' + _escHtml(title) + "</h3>" : "";
    return '<div class="memo-block">' + h3 + '<ul class="memo-list">' + lis + "</ul></div>";
  }}).join("");
  return '<div class="memo-panel">' + htmlBlocks + "</div>";
}}
function switchView(name, btn) {{
  document.querySelectorAll('.view-section').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const el = document.getElementById('view' + name);
  if (el) el.classList.add('active');
  if (btn) btn.classList.add('active');
}}
function updateShapeDesc(key, val) {{
  if (STATE.shapes[key]) STATE.shapes[key].desc = val;
  document.querySelectorAll('#viewPlaybook .shape-section').forEach(sec => {{
    if (sec.dataset.shape === key) {{
      const box = sec.querySelector('.desc-text');
      if (box) {{
        const trimmed = String(val || "").trim();
        box.innerHTML = trimmed ? trimmed.split(/\\r?\\n/).map(_escHtml).join("<br>") : '<span style="color:var(--dads-text-muted)">（説明文はまだ登録されていません）</span>';
      }}
    }}
  }});
}}
function updateMemo(val) {{
  STATE.comp_memo = val;
  const sec = document.getElementById('sec-comp');
  if (!sec) return;
  const panel = sec.querySelector('.memo-panel');
  if (panel) panel.outerHTML = _renderMemoHtml(val);
}}
function loadClassifiedJson(file) {{
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      const loaded = JSON.parse(String(reader.result || "{{}}"));
      if (typeof loaded.comp_memo === "string") {{
        const ed = document.getElementById("memoEditor");
        if (ed) ed.value = loaded.comp_memo;
        updateMemo(loaded.comp_memo);
      }}
      if (loaded.shapes && typeof loaded.shapes === "object") {{
        for (const [k, v] of Object.entries(loaded.shapes)) {{
          if (!STATE.shapes[k]) STATE.shapes[k] = {{}};
          Object.assign(STATE.shapes[k], v);
          if (typeof v.desc === "string") updateShapeDesc(k, v.desc);
        }}
        document.querySelectorAll('#viewShapes textarea[data-shape]').forEach(ta => {{
          const s = STATE.shapes[ta.dataset.shape];
          if (s && typeof s.desc === "string") ta.value = s.desc;
        }});
      }}
      if (Array.isArray(loaded.rounds)) STATE.rounds = loaded.rounds;
    }} catch (err) {{
      alert("JSON の読み込みに失敗しました: " + err);
    }}
  }};
  reader.readAsText(file, "utf-8");
}}
function downloadClassifiedJson() {{
  STATE.exported_at = new Date().toISOString();
  const blob = new Blob([JSON.stringify(STATE, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '{file_stem}_classified.json';
  a.click();
}}
function exportShareHtml() {{
  const clone = document.documentElement.cloneNode(true);
  clone.querySelectorAll('.nav-tabs, .header-actions, #viewRounds, #viewShapes, #viewMemo').forEach(el => el.remove());
  const vp = clone.querySelector('#viewPlaybook');
  if (vp) vp.classList.add('active');
  const titleEl = clone.querySelector('title');
  if (titleEl && !titleEl.textContent.includes('共有版')) titleEl.textContent += '（共有版）';
  const html = '<!DOCTYPE html>\\n' + clone.outerHTML;
  const blob = new Blob([html], {{ type: 'text/html;charset=utf-8' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '{file_stem}_playbook_share.html';
  a.click();
}}"""

    html_doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_e(title)}</title>
<style>{DADS_CSS}</style>
</head>
<body>
<header class="site-header">
  <div class="header-inner">
    <div class="brand">
      <h1 class="site-title"><span class="tag-official">PLAYBOOK</span> {_e(team)} {_e(map_name)} プレイブック</h1>
      <span class="progress-pill">構成: <b>{_e(" / ".join(comp))}</b></span>
    </div>
    {nav_html}
  </div>
</header>
<div id="viewPlaybook" class="view-section active">
  <div id="playbookWrap" class="page-wrap">
    <nav class="toc-nav" aria-label="目次">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-weight:700;font-size:12px;color:var(--dads-text-sub);">目次ナビ</span>
        <button type="button" class="dads-btn" style="padding:2px 8px;font-size:11px;" onclick="toggleToc()">閉じる</button>
      </div>
      <div class="toc-heading">構成コンセプト</div>
      <ul class="toc-list"><li><a href="#sec-comp">{_e(map_name)} {_e(team)} comp メモ</a></li></ul>
      <div class="toc-heading">攻撃の型・タグ（Attack）</div>
      <ul class="toc-list">{toc_atk}</ul>
      <div class="toc-heading">防衛の型（Defense）</div>
      <ul class="toc-list">{toc_def}</ul>
      <div class="toc-heading">データ索引</div>
      <ul class="toc-list"><li><a href="#sec-appendix">全 {len(rounds)} ラウンド一覧</a></li></ul>
    </nav>
    <main class="content-col">
      <div class="playbook-search-bar">
        <button type="button" class="dads-btn" id="tocToggleBtn" onclick="toggleToc()">目次を隠す</button>
        <input type="search" class="dads-input" style="max-width:none;" placeholder="プレイブック内を検索（例: スモーク, 索敵, スプリット, フェイク...）" oninput="filterPlaybook(this.value)">
        <button type="button" class="dads-btn" onclick="window.print()">印刷 / PDF</button>
      </div>
      <div class="summary-strip">
        <div class="stat-card"><div class="lbl">マップ / 構成</div><div class="val" style="font-size:16px">{_e(map_name)} / {_e(", ".join(comp))}</div></div>
        <div class="stat-card"><div class="lbl">対象試合 / 分類ラウンド</div><div class="val">{n_matches} 試合 / {n_classified} / {len(rounds)} R</div></div>
        <div class="stat-card"><div class="lbl">登録された型・タグ</div><div class="val">{len(shapes)} 種類</div></div>
      </div>
      <section id="sec-comp">
        <h2 class="section-h2">1. 構成コンセプト（{_e(map_name)} {_e(team)} comp）</h2>
        {_render_memo_blocks(memo)}
      </section>
      <section id="sec-atk">
        <h2 class="section-h2 atk-h2">2. 攻撃側の戦術・タグ（Attack）</h2>
        {atk_sections}
      </section>
      <section id="sec-def">
        <h2 class="section-h2">3. 防衛側の配置（Defense）</h2>
        {def_sections}
      </section>
      <section id="sec-appendix">
        <h2 class="section-h2">4. 全ラウンド索引</h2>
        <div class="dads-table-wrap">
          <table class="dads-table" id="roundTable">
            <thead><tr><th>試合</th><th>R</th><th>攻防</th><th>勝敗</th><th>分類（タグ）</th><th>序盤</th><th>エクセキューション</th><th>VOD</th></tr></thead>
            <tbody>{round_rows or '<tr><td colspan="8" style="color:var(--dads-text-muted)">実戦ラウンドはまだ紐づいていません（<code>--report</code> で追加可能）。</td></tr>'}</tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
</div>
{extra_views}
<div id="imgModal" class="modal" onclick="if(event.target===this) closeImgModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeImgModal()" aria-label="閉じる">&times;</button>
    <div id="imgModalTitle" style="font-weight:700;margin-bottom:10px;"></div>
    <img id="imgModalSrc" src="" alt="">
  </div>
</div>
<script>
const SHARE_SHAPES = {tv_json_str};
let tocHidden = false;
function toggleToc() {{
  tocHidden = !tocHidden;
  const wrap = document.getElementById("playbookWrap");
  const btn = document.getElementById("tocToggleBtn");
  if (wrap) wrap.classList.toggle("toc-hidden", tocHidden);
  if (btn) btn.textContent = tocHidden ? "目次を表示" : "目次を隠す";
}}
function filterPlaybook(val) {{
  const terms = val.trim().toLowerCase().split(/\\s+/).filter(Boolean);
  document.querySelectorAll(".shape-section").forEach(sec => {{
    const ok = !terms.length || terms.every(t => (sec.dataset.search || "").includes(t));
    sec.classList.toggle("hidden-by-search", !ok);
  }});
  document.querySelectorAll("#roundTable tbody tr").forEach(tr => {{
    const ok = !terms.length || terms.every(t => (tr.dataset.search || "").includes(t));
    tr.classList.toggle("hidden-by-search", !ok);
  }});
}}
function openInTactival(name) {{
  const info = SHARE_SHAPES[name];
  if (!info) return;
  if (info.tactival_json) {{
    const blob = new Blob([JSON.stringify(info.tactival_json, null, 2)], {{ type: "application/json" }});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name.replace(/[^A-Za-z0-9_-ぁ-んァ-ヶ一-龠]/g, "_") + ".tactival.json";
    a.click();
  }}
  window.open((info.tactival_url || "").trim() || "https://tactival.app/board", "_blank", "noopener");
}}
function openImgModal(src, title) {{
  document.getElementById("imgModalSrc").src = src;
  document.getElementById("imgModalTitle").textContent = title || "";
  document.getElementById("imgModal").classList.add("open");
}}
function closeImgModal() {{
  document.getElementById("imgModal").classList.remove("open");
  document.getElementById("imgModalSrc").src = "";
}}
document.addEventListener("keydown", e => {{ if (e.key === "Escape") closeImgModal(); }});
{extra_js}
</script>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def build_index_html(entries: list[dict[str, Any]], out_path: Path, title: str = "全マップ 初動配置＆プレイブック一覧") -> Path:
    """生成された複数マップのプレイブックをまとめたポータル HTML を出力する。"""
    cards = []
    for e in entries:
        mp = e["map"]
        comp = " / ".join(e.get("comp") or [])
        thumb = e.get("thumb") or ""
        img_tag = f'<img src="{thumb}" alt="{_e(mp)}" style="width:100%;max-height:220px;object-fit:contain;background:#fff;border-radius:6px;border:1px solid var(--dads-border);">' if thumb else ""
        atk_list = "".join(f"<li>{_e(x)}</li>" for x in (e.get("atk_openings") or [])[:4])
        def_list = "".join(f"<li>{_e(x)}</li>" for x in (e.get("def_openings") or [])[:4])
        links = []
        if e.get("workspace_href"):
            links.append(f'<a class="dads-btn primary" href="{_e(e["workspace_href"])}">ワークスペース版を開く</a>')
        if e.get("share_href"):
            links.append(f'<a class="dads-btn" href="{_e(e["share_href"])}">共有版を開く</a>')
        cards.append(f"""
        <article class="shape-card" style="background:#fff">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
            <h3 style="margin:0;font-size:18px;color:var(--dads-blue)">{_e(mp)}</h3>
            <span class="dads-badge badge-neutral">{_e(e.get("team") or "LOUD")} ・ {e.get("n_shapes", 0)} 型</span>
          </div>
          <div style="font-size:12px;color:var(--dads-text-sub)">構成: <strong>{_e(comp)}</strong></div>
          {img_tag}
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
            <div><strong>攻撃 初動テイク</strong><ul style="margin:4px 0;padding-left:16px">{atk_list}</ul></div>
            <div><strong>防衛 初動配置</strong><ul style="margin:4px 0;padding-left:16px">{def_list}</ul></div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:auto">{"".join(links)}</div>
        </article>""")

    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{_e(title)}</title><style>{DADS_CSS}</style></head>
<body>
<header class="site-header"><div class="header-inner">
  <div class="brand"><h1 class="site-title"><span class="tag-official">INDEX</span> {_e(title)}</h1>
  <span class="progress-pill">全 <b>{len(entries)}</b> マップ</span></div>
</div></header>
<main class="sub-container">
  <p style="color:var(--dads-text-sub)">Summit の分類知識・<code>config/kb/</code>・バリア座標（<code>config/tactival-barriers/</code>）を元に生成した各マップの初動配置プレイブック一覧です。</p>
  <div class="shapes-grid">{"".join(cards)}</div>
</main>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def generate_all(
    out_dir: Path | str,
    base_json: Path | str | None = DEFAULT_BASE_JSON,
    maps: list[str] | None = None,
    comp: list[str] | None = None,
    use_map_comp: bool = False,
    team: str = "LOUD",
    reports: Path | str | None = None,
    share_only: bool = False,
    cfg: dict | None = None,
    cache_dirs: list[Path] | None = None,
    log=print,
) -> list[dict[str, Any]]:
    """指定マップ（省略時は全 13 マップ）の JSON + HTML（ワークスペース版/共有版）+ index.html を生成する。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = KB.load(cfg)
    target_maps = maps or list_maps()
    index_entries: list[dict[str, Any]] = []

    for mp_name in target_maps:
        mid = _norm(mp_name)
        use_c = comp if comp else (None if use_map_comp else DEFAULT_COMP)
        data = generate_map_playbook_data(
            mp_name, comp=use_c, team=team, base_json=base_json, cfg=cfg, kb=kb, cache_dirs=cache_dirs
        )
        if reports is not None:
            r_path = Path(reports)
            if r_path.exists():
                extra_rounds = classify_report_rounds(r_path, data["shapes"], team_filter=team, map_filter=mp_name)
                if extra_rounds:
                    exist_ids = {r.get("id") for r in data["rounds"]}
                    for r in extra_rounds:
                        if r.get("id") not in exist_ids:
                            data["rounds"].append(r)

        slug = f"{_norm(team)}_{mid}"
        json_p = out_dir / f"{slug}_classified.json"
        json_p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        ws_p = out_dir / f"{slug}_playbook.html"
        sh_p = out_dir / f"{slug}_playbook_share.html"
        if not share_only:
            build_playbook_html(data, ws_p, share_only=False)
        build_playbook_html(data, sh_p, share_only=True)

        shapes = data["shapes"]
        atk_open = [k for k, v in shapes.items() if v.get("side") == "攻撃" and v.get("category") in ("序盤", "形")]
        def_open = [k for k, v in shapes.items() if v.get("side") == "防衛"]
        thumb = next((v.get("tactival_img") for v in shapes.values() if v.get("tactival_img")), "")
        index_entries.append({
            "map": data["map"],
            "team": team,
            "comp": data["comp"],
            "n_shapes": len(shapes),
            "n_rounds": len(data["rounds"]),
            "atk_openings": atk_open,
            "def_openings": def_open,
            "thumb": thumb,
            "json_path": str(json_p),
            "workspace_href": None if share_only else ws_p.name,
            "share_href": sh_p.name,
        })
        if log:
            log(f"  [{data['map']}] {len(shapes)} 型 / {len(data['rounds'])} R → {sh_p.name}")

    idx_p = out_dir / "index.html"
    build_index_html(index_entries, idx_p, title=f"{team} 全マップ初動配置プレイブック一覧")
    if log:
        log(f"一覧ページを作成しました: {idx_p}")
    return index_entries


def parse_comp_arg(comp_str: str | None) -> list[str] | None:
    """カンマ・スラッシュ区切りのエージェント構成文字列をリストへ変換する。"""
    if not comp_str:
        return None
    parts = [p.strip() for p in re.split(r"[,/]+", comp_str) if p.strip()]
    return parts or None


def parse_maps_arg(maps_arg: list[str] | str | None, all_maps: bool = False) -> list[str] | None:
    """CLI の --map / --all-maps 指定をマップ名リストへ変換する。"""
    if all_maps or not maps_arg:
        return None
    raw = [maps_arg] if isinstance(maps_arg, str) else list(maps_arg)
    out = [p.strip() for item in raw for p in item.split(",") if p.strip()]
    return out or None


def add_openings_args(parser) -> None:
    """openings コマンド共通の argparse オプションを登録する。"""
    parser.add_argument(
        "--base-json",
        default=str(DEFAULT_BASE_JSON),
        help="ベースにする classified JSON（既定: LOUD Summit JSON）",
    )
    parser.add_argument(
        "--no-base-json",
        action="store_true",
        help="ベース JSON を読み込まず、知識ベースのみから生成する",
    )
    parser.add_argument("--all-maps", action="store_true", help="全13マップ分をまとめて生成する")
    parser.add_argument("--map", nargs="+", help="生成するマップ名（例: Sunset または Sunset Split）")
    parser.add_argument("--comp", help="カンマ区切りの5人構成（例: Sage,Chamber,Omen,Sova,Neon）")
    parser.add_argument(
        "--use-map-comp",
        action="store_true",
        help="--comp 未指定時に各マップの推奨構成を使って生成する",
    )
    parser.add_argument("--team", default="LOUD", help="チーム名（既定: LOUD）")
    parser.add_argument("--report", help="実戦ラウンドを自動分類する report.json またはフォルダ")
    parser.add_argument(
        "--out",
        default=str(Path("playbook_JSON") / "generated"),
        help="出力先フォルダ、または単一マップ時の .html ファイルパス",
    )
    parser.add_argument("--share", action="store_true", help="共有閲覧版 HTML のみを出力する")
    parser.add_argument("--config", help="設定 YAML のパス")


def run_openings_cli(args, cfg: dict | None = None, log=print) -> list[dict[str, Any]]:
    """openings CLI の実行エントリポイント。"""
    base_json = None if getattr(args, "no_base_json", False) else getattr(args, "base_json", None)
    if base_json and not Path(base_json).exists() and str(base_json) == str(DEFAULT_BASE_JSON):
        base_json = None
    maps = parse_maps_arg(getattr(args, "map", None), all_maps=bool(getattr(args, "all_maps", False)))
    comp = parse_comp_arg(getattr(args, "comp", None))
    team = str(getattr(args, "team", None) or "LOUD")
    share_only = bool(getattr(args, "share", False))
    reports = getattr(args, "report", None)
    out_raw = getattr(args, "out", None) or str(Path("playbook_JSON") / "generated")
    out_path = Path(out_raw)

    # 単一マップ + .html 指定時は指定ファイルを直接生成する
    if maps and len(maps) == 1 and out_path.suffix.lower() in (".html", ".htm"):
        data = generate_map_playbook_data(
            maps[0],
            comp=comp,
            team=team,
            base_json=base_json,
            cfg=cfg,
        )
        if reports:
            rep_rounds = classify_report_rounds(reports, data["shapes"], team_filter=team, map_filter=data["map"])
            if rep_rounds:
                existing = {r.get("id") for r in data["rounds"]}
                data["rounds"].extend(r for r in rep_rounds if r.get("id") not in existing)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_p = out_path.with_suffix(".json")
        json_p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        build_playbook_html(data, out_path, share_only=share_only)
        if not share_only:
            sh_p = out_path.with_name(f"{out_path.stem}_share{out_path.suffix}")
            build_playbook_html(data, sh_p, share_only=True)
        if log:
            log(f"[{data['map']}] 生成完了: {out_path} ({len(data['shapes'])} 型 / {len(data['rounds'])} R)")
        return [{
            "map": data["map"],
            "team": team,
            "comp": data["comp"],
            "n_shapes": len(data["shapes"]),
            "n_rounds": len(data["rounds"]),
            "json_path": str(json_p),
            "share_href": out_path.name if share_only else f"{out_path.stem}_share{out_path.suffix}",
            "workspace_href": None if share_only else out_path.name,
        }]

    return generate_all(
        out_dir=out_path,
        base_json=base_json,
        maps=maps,
        comp=comp,
        use_map_comp=bool(getattr(args, "use_map_comp", False)),
        team=team,
        reports=reports,
        share_only=share_only,
        cfg=cfg,
        log=log,
    )
