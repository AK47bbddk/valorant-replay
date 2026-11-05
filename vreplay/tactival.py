"""TactiVal（作戦ボードのアプリ）に読み込ませる JSON を書く。

プレイブックの型（`deck.playbook_rows`）と節目ごとの動き（`deck.motion_frames`）を、
TactiVal の 1 ボード＝1 作戦にする。節目（1:30 → 1:15 → 1:00 → 0:40 → 設置後）は
そのまま TactiVal の step になるので、アプリの中で再生・編集できる。

形式（ユーザーが書き出した `schemaVersion: 3` のファイルから分かったもの）
  - トップ：schemaVersion / patchId / gamePatch / mapVersion / mapFlipped / appVersion /
    savedAt / mapId / activeStepIndex / overrides / zones / steps / patchSnapshot
  - `steps[]` = {id, label, transition（秒）, hold（秒）, elements, zoneVisibility, overrides}
  - 駒 = {"id", "type": "agent", "x", "y", "width": 48, "height": 48, "angle", "style",
         "order", "groupId", "data": {"agentId": <uuid>, "team": "neutral", …}}
  - `patchSnapshot.agents` が uuid → 表示名、`patchSnapshot.maps` が mapId → {name, multiplier}

**座標の基準（マップごとの calibration）**
x・y はアプリのマップ（ベクター画像）のピクセルで、valorant-api の 0..1 とは原点も倍率も違う。
対応表は公開されていないので、マップごとに 1 回だけ合わせる：
アプリで**攻撃側スポーンと防衛側スポーンのちょうど上に駒を 1 体ずつ置いたボード**を書き出し、

    vreplay tactival --calibrate <書き出した.json> --map Sunset

を実行すると、倍率と原点を計算して `output.dir/_cache/tactival_calib.json` に覚える。
合わせていないマップのボードは（位置がずれて誤解のもとになるので）書き出さない。
"""
from __future__ import annotations

import json
import math
import random
import string
import time
from pathlib import Path

SCHEMA = 3
BUNDLED = Path(__file__).parent / "data" / "tactival_template.json"
CALIB = "tactival_calib.json"
# ボードにする型（多すぎると使いにくいので、覚える順に近いものから）
KINDS = ("Default", "Push", "Rush", "Split", "Execute", "Retake", "Counter")
AGENT_SIZE = 48


# ---------------------------------------------------------------- 雛形
def template_path(cfg: dict | None = None) -> Path:
    p = ((cfg or {}).get("tactival") or {}).get("template")
    return Path(p) if p else BUNDLED


def load_template(cfg: dict | None = None) -> dict:
    return json.loads(template_path(cfg).read_text("utf-8"))


def map_id(tpl: dict, map_name: str) -> str | None:
    """「Split」→ "split"。patchSnapshot.maps の表示名で探す。"""
    maps = ((tpl.get("patchSnapshot") or {}).get("maps")) or {}
    key = str(map_name or "").strip().lower()
    for mid, m in maps.items():
        if mid.lower() == key or str(m.get("name", "")).lower() == key:
            return mid
    return None


def agent_ids(tpl: dict) -> dict[str, str]:
    """表示名（小文字）→ エージェントの uuid。"""
    out = {}
    for uid, a in ((tpl.get("patchSnapshot") or {}).get("agents") or {}).items():
        nm = str(a.get("displayName") or "").lower()
        if nm and nm != "bot":
            out[nm] = uid
    return out


def elements(data: dict) -> list[dict]:
    """ボードの中の駒・スキルを全部。"""
    return [el for st in data.get("steps") or [] for el in st.get("elements") or []
            if isinstance(el, dict)]


# ---------------------------------------------------------------- 座標合わせ
def calib_path(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "_cache" / CALIB


BUNDLED_CALIB = Path(__file__).parent / "data" / "tactival_calib.json"


def load_calib(cfg: dict | None) -> dict:
    """{mapId: 変換}。同梱の既定値 < 自分で合わせたもの < config の tactival.calibration。"""
    out = {}
    try:
        if BUNDLED_CALIB.exists():
            out.update(json.loads(BUNDLED_CALIB.read_text("utf-8")))
    except Exception:
        pass
    try:
        p = calib_path(cfg or {})
        if p.exists():
            out.update(json.loads(p.read_text("utf-8")))
    except Exception:
        pass
    out.update(((cfg or {}).get("tactival") or {}).get("calibration") or {})
    return out


# 座標合わせ用に、どのエージェントをどこに置いてもらうか
TARGETS = {"jett": ("spawn", "攻撃"), "sage": ("spawn", "防衛"),
           "omen": ("site", "A"), "viper": ("site", "B"), "cypher": ("site", "C")}


def _center(el: dict) -> tuple[float, float]:
    """駒の x/y は左上なので、真ん中に直す。"""
    return float(el["x"]) + float(el.get("width") or 0) / 2, float(el["y"]) + float(el.get("height") or 0) / 2


def _targets(meta: dict) -> dict[str, tuple[float, float]]:
    """「攻撃スポーン」「A サイト」などの 0..1 座標。"""
    out = {}
    for side, xy in ((meta or {}).get("spawns") or {}).items():
        out[f"spawn:{side}"] = xy
    for c in (meta or {}).get("callouts") or []:
        if str(c.get("name", "")).lower() == "site" and c.get("super") in ("A", "B", "C"):
            out[f"site:{c['super']}"] = (float(c["x"]), float(c["y"]))
    return out


def _agent_names(data: dict, tpl: dict | None = None) -> dict[str, str]:
    src = (data.get("patchSnapshot") or {}).get("agents") or \
        ((tpl or {}).get("patchSnapshot") or {}).get("agents") or {}
    return {u: str(a.get("displayName") or "") for u, a in src.items()}


def calibrate(data: dict, meta: dict, tpl: dict | None = None) -> dict:
    """置いた駒から、0..1 → ピクセルの倍率と原点を出す。

    Jett＝攻撃側スポーン／Sage＝防衛側スポーン／Omen＝A サイト／Viper＝B サイト／Cypher＝C サイト
    の約束で置いてもらう。3 点以上あればアフィン変換（回転・鏡映・縦横の倍率）で合わせる。
    TactiVal のマップは valorant-api の画像に対して回っていることがある（Split は 90 度）ので、
    倍率と原点だけでは合わない。どのエージェントか分からない 2 体だけのときは、スポーン 2 つと
    みなして 1 つの倍率で合わせる（回転は見られない）。

    返り値 {"m": [a,b,c,d,e,f], "rotation", "rms", "points"}、2 点のときは
    {"scale", "ox", "oy", "rms", "points"}。合わせられなければ ValueError。
    """
    ags = [el for el in elements(data) if el.get("type") == "agent"]
    if len(ags) < 2:
        raise ValueError(f"駒が 2 体以上のボードが要ります（いまは {len(ags)} 体）")
    tg = _targets(meta)
    names = _agent_names(data, tpl)
    pairs = []
    for el in ags:
        nm = names.get((el.get("data") or {}).get("agentId") or "", "").lower()
        key = TARGETS.get(nm)
        if key and f"{key[0]}:{key[1]}" in tg:
            pairs.append((tg[f"{key[0]}:{key[1]}"], _center(el)))
    if len(pairs) >= 3:
        # アフィン変換（回転・鏡映・縦横の倍率をまとめて）。TactiVal の Split は
        # valorant-api の画像に対して 90 度回っているので、倍率と原点だけでは合わない。
        import numpy as np
        A = np.array([[q[0][0], q[0][1], 1.0] for q in pairs], float)
        m = []
        for i in (0, 1):
            b = np.array([q[1][i] for q in pairs], float)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            m += [float(v) for v in sol]
        if abs(m[0] * m[4] - m[1] * m[3]) < 1e-9:
            raise ValueError("置いた駒が一直線で、向きを決められません")
        err = sum(math.hypot(m[0] * q[0][0] + m[1] * q[0][1] + m[2] - q[1][0],
                             m[3] * q[0][0] + m[4] * q[0][1] + m[5] - q[1][1]) for q in pairs) / len(pairs)
        return {"m": [round(v, 4) for v in m], "rms": round(err, 2), "points": len(pairs),
                "rotation": round(math.degrees(math.atan2(m[3], m[0])), 1)}
    # 2 点：1 つの倍率で合わせる（スポーンが縦に並ぶマップでは横の精度が落ちる）
    if len(pairs) == 2:
        cand = [(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])]
    else:
        sp = (meta or {}).get("spawns") or {}
        if not (sp.get("攻撃") and sp.get("防衛")):
            raise ValueError("このマップのスポーンの座標が valorant-api から取れません")
        P = [_center(a) for a in ags[:2]]
        cand = [(sp["攻撃"], P[0], sp["防衛"], P[1]), (sp["攻撃"], P[1], sp["防衛"], P[0])]
    best = None
    for na, pa, nb, pb in cand:
        dn = math.hypot(nb[0] - na[0], nb[1] - na[1])
        if dn <= 1e-9:
            continue
        s = math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / dn
        ox = (pa[0] - na[0] * s + pb[0] - nb[0] * s) / 2
        oy = (pa[1] - na[1] * s + pb[1] - nb[1] * s) / 2
        err = (math.hypot(na[0] * s + ox - pa[0], na[1] * s + oy - pa[1])
               + math.hypot(nb[0] * s + ox - pb[0], nb[1] * s + oy - pb[1])) / 2
        if best is None or err < best[0]:
            best = (err, s, ox, oy)
    if best is None:
        raise ValueError("座標を合わせられませんでした")
    err, s, ox, oy = best
    return {"scale": round(s, 3), "ox": round(ox, 2), "oy": round(oy, 2),
            "rms": round(err, 2), "points": 2}


def save_calib(cfg: dict, mid: str, c: dict) -> Path:
    p = calib_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if p.exists():
        try:
            cur = json.loads(p.read_text("utf-8"))
        except Exception:
            cur = {}
    cur[mid] = c
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")
    return p


def to_px(x: float, y: float, c: dict) -> tuple[float, float]:
    """0..1 → TactiVal のピクセル。アフィン（m）があればそれ、無ければ倍率と原点。"""
    x, y = float(x), float(y)
    m = c.get("m")
    if m:
        return round(m[0] * x + m[1] * y + m[2], 2), round(m[3] * x + m[4] * y + m[5], 2)
    sx = c.get("sx", c.get("scale"))
    sy = c.get("sy", c.get("scale"))
    return round(x * sx + c["ox"], 2), round(y * sy + c["oy"], 2)


# ---------------------------------------------------------------- ボードを作る
def _eid() -> str:
    tail = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"{int(time.time() * 1000)}-{tail}"


def agent_element(agent_uuid: str | None, x: float, y: float, order: int = 0,
                  team: str = "neutral") -> dict:
    """駒 1 個（アプリの書き出しと同じ形）。"""
    return {
        "id": _eid(), "type": "agent", "x": x, "y": y,
        "width": AGENT_SIZE, "height": AGENT_SIZE, "angle": 0,
        "style": {"stroke": "transparent", "fill": "transparent", "strokeWidth": 0,
                  "strokeStyle": "solid"},
        "order": order, "groupId": None,
        "data": {"agentId": agent_uuid, "name": None, "team": team, "alive": True, "health": 100,
                 "showHealth": True, "directionAngle": 0, "showDirection": True,
                 "showAudioRange": False, "castRanges": [], "activeBuffs": [],
                 "buffAngle": math.pi, "buffSize": 0.6, "buffModifier": "none",
                 "deathGreyscale": False, "deathMarker": "none", "ownerId": None,
                 "altIcon": None, "useAltIcon": False},
    }


def calib_board(cfg: dict | None, map_name: str, meta: dict, tpl: dict | None = None,
                guess: dict | None = None) -> dict:
    """座標合わせ用のボード。TARGETS の約束どおりに駒を置いた状態で作る。

    guess（すでに分かっている変換）があればその位置に、無ければマップの真ん中あたりに固めて置く。
    アプリで正しい場所へドラッグして書き出し、`vreplay tactival --calibrate` に渡す。
    """
    tpl = tpl or load_template(cfg)
    ids = agent_ids(tpl)
    tg = _targets(meta)
    out = {k: v for k, v in tpl.items() if k != "steps"}
    out.update({"mapId": map_id(tpl, map_name) or map_name.lower(), "mapFlipped": False,
                "activeStepIndex": 0, "savedAt": ""})
    els, labels = [], []
    for i, (ag, key) in enumerate((("jett", "spawn:攻撃"), ("sage", "spawn:防衛"),
                                   ("omen", "site:A"), ("viper", "site:B"), ("cypher", "site:C"))):
        if key not in tg or ag not in ids:
            continue
        if guess:
            x, y = to_px(tg[key][0], tg[key][1], guess)
        else:                               # 分からないので真ん中に固めて置く（ドラッグ前提）
            x, y = 900 + (i % 2) * 90, 420 + (i // 2) * 90
        els.append(agent_element(ids[ag], round(x - AGENT_SIZE / 2, 2), round(y - AGENT_SIZE / 2, 2), i))
        labels.append(f"{ag.title()}={key.split(':')[1]}")
    out["steps"] = [{"id": _eid(), "label": "座標合わせ：" + " / ".join(labels), "transition": 1.5,
                     "hold": 0, "elements": els, "zoneVisibility": {}, "overrides": {}}]
    return out


# 矢印の要素の形はアプリの書き出しに例が無く、公開された仕様も無い。
# 矢印を 1 本引いたボードを `tactival.template` に指定すると、その形をまねて書き出す。
ARROW_ENDS = (("x2", "y2"), ("toX", "toY"), ("endX", "endY"), ("x1", "y1"))


def arrow_sample(tpl: dict) -> dict | None:
    """雛形の中にある矢印らしい要素（駒・スキル以外で、2 つ目の点を持つもの）。"""
    for el in elements(tpl):
        if el.get("type") in ("agent", "ability"):
            continue
        if any(all(k in el for k in ks) for ks in ARROW_ENDS):
            return el
        for k in ("points", "path", "end", "to"):
            if isinstance(el.get(k), (list, dict)):
                return el
    return None


def arrow_element(sample: dict, x0: float, y0: float, x1: float, y1: float, order: int = 0) -> dict:
    """矢印 1 本。雛形の矢印をまねて、両端だけ差し替える。"""
    el = json.loads(json.dumps(sample))
    el["id"] = _eid()
    el["order"] = order
    if "x" in el and "y" in el:
        el["x"], el["y"] = round(x0, 2), round(y0, 2)
    for kx, ky in ARROW_ENDS:
        if kx in el and ky in el:
            el[kx], el[ky] = round(x1, 2), round(y1, 2)
            break
    for k in ("points", "path"):
        v = el.get(k)
        if isinstance(v, list) and len(v) >= 2:
            if isinstance(v[0], (list, tuple)):
                el[k] = [[round(x0, 2), round(y0, 2)], [round(x1, 2), round(y1, 2)]]
            elif isinstance(v[0], dict):
                el[k] = [{**v[0], "x": round(x0, 2), "y": round(y0, 2)},
                         {**v[-1], "x": round(x1, 2), "y": round(y1, 2)}]
            else:
                el[k] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
            break
    for k in ("end", "to"):
        if isinstance(el.get(k), dict):
            el[k] = {**el[k], "x": round(x1, 2), "y": round(y1, 2)}
            break
    return el


def board(cfg: dict | None, map_name: str, name: str, frames: dict, calib: dict,
          tpl: dict | None = None, transition: float = 1.5, hold: float = 0.8,
          arrows: list | None = None, areas_xy: dict | None = None) -> dict:
    """1 つの型 → 1 つのボード。節目の数だけ step を作り、同じ駒を動かす。"""
    tpl = tpl or load_template(cfg)
    ids = agent_ids(tpl)
    out = {k: v for k, v in tpl.items() if k != "steps"}
    out["schemaVersion"] = tpl.get("schemaVersion", SCHEMA)
    out["mapId"] = map_id(tpl, map_name) or tpl.get("mapId")
    out["mapFlipped"] = False
    out["activeStepIndex"] = 0
    out["savedAt"] = ""
    steps = []
    for i, mark in enumerate(frames.get("marks") or []):
        els = []
        for j, d in enumerate(frames.get("dots") or []):
            if i >= len(d["xy"]):
                continue
            x, y = to_px(d["xy"][i][0], d["xy"][i][1], calib)
            els.append(agent_element(ids.get(str(d["label"]).lower()),
                                     x - AGENT_SIZE / 2, y - AGENT_SIZE / 2, j))
        steps.append({"id": _eid(), "label": f"{name}　{mark}", "transition": transition,
                      "hold": hold, "elements": els, "zoneVisibility": {}, "overrides": {}})
    sample = arrow_sample(tpl)
    if sample and arrows and areas_xy and steps:
        for j, (a0, a1) in enumerate(arrows):
            if a0 in areas_xy and a1 in areas_xy:
                p0 = to_px(areas_xy[a0][0], areas_xy[a0][1], calib)
                p1 = to_px(areas_xy[a1][0], areas_xy[a1][1], calib)
                steps[0]["elements"].append(arrow_element(sample, p0[0], p0[1], p1[0], p1[1], 100 + j))
    out["steps"] = steps or [{"id": _eid(), "label": name, "transition": transition, "hold": hold,
                              "elements": [], "zoneVisibility": {}, "overrides": {}}]
    return out


def export(cfg: dict | None, key: dict, prs: dict, out_dir: Path, frames_of,
           log=print, k: int = 12, areas_xy: dict | None = None) -> list[Path]:
    """型ごとに 1 ファイルずつ書き出す。座標を合わせていないマップでは何も書かない。

    frames_of(g, side) は `deck.motion_frames` と同じもの（型 → 節目ごとの駒の位置）。
    """
    from . import deck as D
    try:
        tpl = load_template(cfg)
    except Exception as e:
        log(f"  TactiVal の雛形を読めませんでした（{e}）")
        return []
    mid = map_id(tpl, key["map"])
    if not mid:
        log(f"  TactiVal にこのマップ（{key['map']}）がありません")
        return []
    calib = load_calib(cfg).get(mid)
    if not calib:
        log(f"  TactiVal：{key['map']} の座標合わせがまだです（スポーンに 2 体置いたボードを書き出して"
            f" `vreplay tactival --calibrate <file> --map {key['map']}`）。ボードは作りません")
        return []
    d = out_dir / "tactival"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for side in (D.ATTACK, D.DEFENSE):
        for g in prs[side]:
            if g["kind"] not in KINDS or len(out) >= k:
                continue
            fr = frames_of(g, side)
            if not fr:
                continue
            b = board(cfg, key["map"], g["name"], fr, calib, tpl,
                      arrows=D.row_arrows(g, side), areas_xy=areas_xy)
            f = d / f"{key['slug']}_{D._slug(g['name'])}.tactival.json"
            f.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
            out.append(f)
    if out:
        log(f"  TactiVal のボード {len(out)} 個：{d}"
            + ("" if arrow_sample(tpl) else "（矢印は雛形に例が無いので入れていません）")
            + ("（座標は概算）" if calib.get("approx") else ""))
    return out
