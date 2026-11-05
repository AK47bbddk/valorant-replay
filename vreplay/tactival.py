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


def calibrate(data: dict, meta: dict) -> dict:
    """スポーンに 1 体ずつ置いたボードから、0..1 → ピクセルの倍率と原点を出す。

    data: TactiVal の書き出し（駒がちょうど 2 体）。meta: `tactics.load_map` の返り値。
    返り値 {"scale", "ox", "oy", "rms"}。合わせられなければ ValueError。
    """
    ags = [el for el in elements(data) if el.get("type") == "agent"]
    if len(ags) != 2:
        raise ValueError(f"駒がちょうど 2 体のボードが要ります（いまは {len(ags)} 体）")
    sp = (meta or {}).get("spawns") or {}
    pts = [sp.get("攻撃"), sp.get("防衛")]
    if not all(pts):
        raise ValueError("このマップのスポーンの座標が valorant-api から取れません")
    P = [(float(a["x"]), float(a["y"])) for a in ags]
    best = None
    for order in ((0, 1), (1, 0)):
        (ax, ay), (bx, by) = pts[order[0]], pts[order[1]]
        dn = math.hypot(bx - ax, by - ay)
        dp = math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1])
        if dn <= 1e-9:
            continue
        s = dp / dn
        ox = (P[0][0] - ax * s + P[1][0] - bx * s) / 2
        oy = (P[0][1] - ay * s + P[1][1] - by * s) / 2
        err = math.hypot(ax * s + ox - P[0][0], ay * s + oy - P[0][1]) + \
            math.hypot(bx * s + ox - P[1][0], by * s + oy - P[1][1])
        if best is None or err < best[0]:
            best = (err, s, ox, oy)
    if best is None:
        raise ValueError("座標を合わせられませんでした")
    err, s, ox, oy = best
    return {"scale": round(s, 3), "ox": round(ox, 2), "oy": round(oy, 2), "rms": round(err / 2, 2)}


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
    return round(float(x) * c["scale"] + c["ox"], 2), round(float(y) * c["scale"] + c["oy"], 2)


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
