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


def load_calib(cfg: dict | None) -> dict:
    """{mapId: {"scale", "ox", "oy"}}。config の tactival.calibration が優先。"""
    out = {}
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
    の約束で置いてもらう。3 点以上あれば縦横べつべつの倍率で合わせる（横に離れた点があるほど正確）。
    どのエージェントか分からない 2 体だけのときは、スポーン 2 つとみなして 1 つの倍率で合わせる。

    返り値 {"scale" | "sx","sy", "ox", "oy", "rms", "points"}。合わせられなければ ValueError。
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
    if len(pairs) >= 3:                       # 縦横べつべつに最小二乗
        res = {}
        for i, (a, b) in enumerate((("sx", "ox"), ("sy", "oy"))):
            n = len(pairs)
            sn = sum(q[0][i] for q in pairs)
            sp_ = sum(q[1][i] for q in pairs)
            snn = sum(q[0][i] ** 2 for q in pairs)
            snp = sum(q[0][i] * q[1][i] for q in pairs)
            den = n * snn - sn * sn
            if abs(den) < 1e-12:
                raise ValueError("置いた駒が一直線で、倍率を決められません")
            res[a] = (n * snp - sn * sp_) / den
            res[b] = (sp_ - res[a] * sn) / n
        err = sum(math.hypot(q[0][0] * res["sx"] + res["ox"] - q[1][0],
                             q[0][1] * res["sy"] + res["oy"] - q[1][1]) for q in pairs) / len(pairs)
        return {"sx": round(res["sx"], 3), "sy": round(res["sy"], 3),
                "ox": round(res["ox"], 2), "oy": round(res["oy"], 2),
                "rms": round(err, 2), "points": len(pairs)}
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
    """0..1 → TactiVal のピクセル。縦横べつべつの倍率（sx/sy）があればそれを使う。"""
    sx = c.get("sx", c.get("scale"))
    sy = c.get("sy", c.get("scale"))
    return round(float(x) * sx + c["ox"], 2), round(float(y) * sy + c["oy"], 2)


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


def board(cfg: dict | None, map_name: str, name: str, frames: dict, calib: dict,
          tpl: dict | None = None, transition: float = 1.5, hold: float = 0.8) -> dict:
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
    out["steps"] = steps or [{"id": _eid(), "label": name, "transition": transition, "hold": hold,
                              "elements": [], "zoneVisibility": {}, "overrides": {}}]
    return out


def export(cfg: dict | None, key: dict, prs: dict, out_dir: Path, frames_of,
           log=print, k: int = 12) -> list[Path]:
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
            b = board(cfg, key["map"], g["name"], fr, calib, tpl)
            f = d / f"{key['slug']}_{D._slug(g['name'])}.tactival.json"
            f.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
            out.append(f)
    if out:
        log(f"  TactiVal のボード {len(out)} 個：{d}")
    return out
