"""手・構成の「動き」を作戦ボード（TactiVal）の絵にする。

1 つの手（例：B スプリット）の節目（1:30 → 1:15 → 1:00 → 0:40）ごとに、
TactiVal のボード（1 ステップ）を作って TactiVal 自身に描かせ、その絵をスライドに貼る。
TactiVal には画像の書き出しが無いので、ヘッドレスの Chromium（playwright）で
https://tactival.app/board を開き、「Load from File」と同じ input にボードの JSON を入れて、
描かれたキャンバスのマップの部分を切り抜く。

  - 攻めの駒は赤、守りの駒は青緑（TactiVal の team: attack / defense の色）
  - 前の節目からの動きは点線の矢印（床の上を通る道を 2〜3 本の矢印に分けて描く）
  - 絵と同じボードの JSON も残すので、TactiVal で開いて直せる（添削 → 修正）

playwright・Chromium が無い／TactiVal に届かないときは、公式のマップ画像に駒と矢印を
描いた「TactiVal 風」の絵にする（スライドが欠けないように）。
絵は内容のハッシュで `output.dir/_cache/tactival_png/` に覚えるので、作り直しでは描き直さない。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

URL = "https://tactival.app/board"
COL = {"attack": "#ff4655", "defense": "#2ec4b6"}
MARKS = [("1:30", 10), ("1:15", 25), ("1:00", 40), ("0:40", 60)]
AGENT = 64             # 駒の大きさ（ボードの px。アプリの既定 48 より大きめにして、縮めた絵でも顔が見えるように）
_R = None              # 立ち上げた描き手（1 回の実行で使い回す）
_R_FAILED = False


# ---------------------------------------------------------------- どの位置を描くか
def nearest_area(xy: dict, u: float, v: float) -> str | None:
    best, bd = None, 9.0
    for name, (x, y) in (xy or {}).items():
        d = (x - u) ** 2 + (y - v) ** 2
        if d < bd:
            best, bd = name, d
    return best


def from_round(r: dict, marks=MARKS) -> dict | None:
    """1 ラウンドのエージェントの位置（アイコン照合）→ 節目ごとの駒。設置より後の節目は使わない。"""
    ag = r.get("agents") or {}
    pe = r.get("plant_e")
    use = [(m, e) for m, e in marks if ag.get(m) and (pe is None or e <= float(pe) + 2)]
    if len(use) < 2:
        return None
    names = []
    for m, _ in use:
        for a, *_ in ag[m]:
            if a not in names:
                names.append(a)
    dots = []
    for a in names:
        seq, last = [], None
        for m, _ in use:
            p = next(((u, v) for a2, u, v, *_ in ag[m] if a2 == a and 0 <= u <= 1 and 0 <= v <= 1), None)
            seq.append(p)
        if sum(1 for p in seq if p) < 2:
            continue
        last = next(p for p in seq if p)
        for i, p in enumerate(seq):
            if p is None:
                seq[i] = last
            else:
                last = p
        dots.append({"label": a, "xy": [[round(u, 4), round(v, 4)] for u, v in seq]})
    if len(dots) < 3:
        return None
    return {"marks": [m for m, _ in use], "dots": dots}


def pick_round(rows: list[dict], med: dict | None = None) -> dict | None:
    """代表のラウンド。勝ったラウンドを先に、その中で「いちばん典型的」なもの。

    med（エージェントごとの medoid の立ち位置 {"marks", "agents"}）があれば、それに近いラウンド。
    無ければ、エージェントが多く見えているもの・新しいもの。"""
    cands = []
    for r in rows:
        ag = r.get("agents") or {}
        if not ag or not r.get("align_ok", True):
            continue
        seen = sum(len(ag.get(m) or []) for m, _ in MARKS)
        if seen < 8:
            continue
        d = None
        if med:
            ds = []
            for i, m in enumerate(med["marks"]):
                for a, u, v, *_ in ag.get(m) or []:
                    p = (med["agents"].get(a) or [None] * 9)[i]
                    if p:
                        ds.append(math.dist((u, v), p))
            d = sum(ds) / len(ds) if len(ds) >= 6 else None
        cands.append((r, seen, d))
    if not cands:
        return None
    wins = [c for c in cands if c[0].get("win")] or cands
    if any(c[2] is not None for c in wins):
        return min((c for c in wins if c[2] is not None), key=lambda c: c[2])[0]
    return max(wins, key=lambda c: (min(c[1], 20), str(c[0].get("date") or ""), c[0].get("n") or 0))[0]


def motion(rows: list[dict], side: str, calls, walk=None) -> dict | None:
    """手のラウンド → 描く駒。実際の 1 ラウンド（勝ったラウンドのうち、いちばん典型的なもの）を描く。

    同じ構成のラウンドなら、エージェントごとの典型の立ち位置（medoid）にいちばん近いラウンドを選ぶ。
    平均の位置をそのまま描くと、ラウンドごとの速さの違いで「戻っている」ように見えるので使わない。
    返り値 {"marks", "dots": [{"label", "xy", "route"}], "how": 説明, "row": 代表のラウンド}
    """
    from . import deck as D
    med = None
    comps = {tuple(r.get("comp") or ()) for r in rows}
    if len(comps) == 1 and len(rows) >= 3:
        med = D.agent_tracks({"rows": rows})
    r = pick_round(rows, med)
    fr = from_round(r) if r else None
    if not fr:
        return None
    for d in fr["dots"]:
        d["route"] = [D.route(walk, d["xy"][i], d["xy"][i + 1]) for i in range(len(d["xy"]) - 1)]
    who = f"{r.get('tag') or r.get('team') or ''} vs {r.get('opp_tag') or r.get('opp') or ''} R{r.get('n')}"
    how = (f"代表のラウンド：{who}（{'勝ち' if r.get('win') else '負け'}）"
           + (f"・{len(rows)} ラウンドの典型の立ち位置にいちばん近いラウンド" if med else ""))
    return pick_marks({**fr, "row": r, "how": how}, walk)


def pick_marks(fr: dict, walk=None, want=("1:30", "1:15", "1:00", "0:40"), k: int = 4) -> dict:
    """スライドに並べる節目を k 個まで（1:30 → 1:15 → 1:00 → 0:40 を優先）。道は選んだ節目の間で引き直す。"""
    from . import deck as D
    idx = [i for i, m in enumerate(fr["marks"]) if m in want]
    if len(idx) < 2:
        idx = list(range(len(fr["marks"])))
    idx = idx[:k]
    if idx == list(range(len(fr["marks"]))):
        return fr
    dots = []
    for d in fr["dots"]:
        xy = [d["xy"][i] for i in idx]
        dots.append({**d, "xy": xy, "route": [D.route(walk, xy[j], xy[j + 1]) for j in range(len(xy) - 1)]})
    return {**fr, "marks": [fr["marks"][i] for i in idx], "dots": dots}


def moves(fr: dict, i: int, xy: dict) -> tuple[list[str], list[str]]:
    """節目 i で何が起きたか（前の節目から場所が変わった駒・同じ場所にいる駒）。"""
    mv, stay = [], []
    for d in fr["dots"]:
        a1 = nearest_area(xy, *d["xy"][i])
        if i == 0:
            stay.append(f"{d['label']}（{a1 or '?'}）")
            continue
        a0 = nearest_area(xy, *d["xy"][i - 1])
        dist = math.dist(d["xy"][i - 1], d["xy"][i])
        if a0 != a1 and dist > 0.03:
            mv.append(f"{d['label']}：{a0 or '?'} → {a1 or '?'}")
        else:
            stay.append(f"{d['label']}（{a1 or '?'}）")
    return mv, stay


# ---------------------------------------------------------------- ボード
def _simplify(path: list, k: int = 3) -> list:
    """床の上の道（折れ線）を k 本くらいの矢印に間引く。"""
    if not path or len(path) < 2:
        return path or []
    if len(path) <= k + 1:
        return path
    idx = sorted({round(j * (len(path) - 1) / k) for j in range(k + 1)})
    return [path[j] for j in idx]


def boards(map_name: str, name: str, fr: dict, side: str, cfg: dict | None = None) -> list[dict] | None:
    """節目ごとに 1 枚（1 ステップ）のボード。座標を合わせていないマップは None。"""
    from . import tactival as TV
    tpl = TV.load_template(cfg)
    mid = TV.map_id(tpl, map_name)
    calib = TV.load_calib(cfg).get(mid) if mid else None
    if not calib:
        return None
    ids = TV.agent_ids(tpl)
    team = "attack" if side in ("attack", "攻撃") else "defense"
    base = {k: v for k, v in tpl.items() if k not in ("steps", "patchSnapshot")}
    base.update({"mapId": mid, "mapFlipped": False, "activeStepIndex": 0, "savedAt": "", "zones": []})
    out = []
    for i, mark in enumerate(fr["marks"]):
        els, order = [], 0
        for d in fr["dots"]:
            if i > 0:
                path = (d.get("route") or [None] * len(fr["marks"]))[i - 1] or [d["xy"][i - 1], d["xy"][i]]
                if math.dist(d["xy"][i - 1], d["xy"][i]) > 0.03:
                    pts = [TV.to_px(p[0], p[1], calib) for p in _simplify(path)]
                    for j in range(len(pts) - 1):
                        el = TV.arrow_element(pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1], order)
                        el["style"].update({"stroke": COL[team], "strokeWidth": 4, "strokeStyle": "dashed", "opacity": 0.9})
                        el["data"]["arrowhead"] = "arrow" if j == len(pts) - 2 else "none"
                        el["data"]["label"] = None
                        els.append(el)
                        order += 1
        for d in fr["dots"]:
            x, y = TV.to_px(d["xy"][i][0], d["xy"][i][1], calib)
            el = TV.agent_element(ids.get(str(d["label"]).lower()), x - AGENT / 2, y - AGENT / 2, 100 + order, team)
            el["width"] = el["height"] = AGENT
            els.append(el)
            order += 1
        b = dict(base)
        b["steps"] = [{"id": TV._eid(), "label": f"{name}　{mark}", "transition": 1.0, "hold": 1.0,
                       "elements": els, "zoneVisibility": {}, "overrides": {}}]
        out.append(b)
    return out


def full_board(map_name: str, name: str, fr: dict, side: str, cfg: dict | None = None) -> dict | None:
    """TactiVal で開いて直す用：節目を全部ステップに持つ 1 つのボード。"""
    bs = boards(map_name, name, fr, side, cfg)
    if not bs:
        return None
    from . import tactival as TV
    tpl = TV.load_template(cfg)
    b = dict(bs[0])
    b["steps"] = [s for x in bs for s in x["steps"]]
    b["patchSnapshot"] = tpl.get("patchSnapshot")
    return b


# ---------------------------------------------------------------- 描く（TactiVal）
def tight(src: Path, out: Path, margin: float = 0.03, thr: int = 45) -> Path:
    """マップ（床）のある所だけを正方形で切り抜く（ボードの余白を落として、縮めても見えるように）。"""
    from PIL import Image
    import numpy as np
    im = Image.open(src).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where(a.max(axis=2) > thr)
    if len(xs) > 100:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        W, H = im.size
        side = min(side, W, H)
        L = int(min(max(0, cx - side / 2), W - side))
        T = int(min(max(0, cy - side / 2), H - side))
        im = im.crop((L, T, L + int(side), T + int(side)))
    if im.size[0] > 900:
        im = im.resize((900, 900), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "JPEG", quality=88)
    return out


class Renderer:
    """ヘッドレスの Chromium で TactiVal を開き、ボードを読み込ませてマップの部分を撮る。"""

    def __init__(self, width: int = 2560, height: int = 1500, timeout: float = 30.0):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._br = self._pw.chromium.launch(args=["--no-sandbox"])
        self._pg = self._br.new_page(viewport={"width": width, "height": height})
        self._pg.set_default_timeout(timeout * 1000)
        self._pg.goto(URL, wait_until="networkidle")
        self._pg.wait_for_selector("input[type=file]", state="attached")
        self._pg.wait_for_selector("canvas")

    def shot(self, board: dict, out: Path, wait_ms: int = 900) -> Path:
        pg = self._pg
        pg.set_input_files("input[type=file]", files=[{
            "name": "board.tactival.json", "mimeType": "application/json",
            "buffer": json.dumps(board, ensure_ascii=False).encode()}])
        try:
            pg.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        pg.wait_for_timeout(wait_ms)
        # いちばん大きいキャンバス＝ボード。1920×1080 の論理座標がキャンバスに収まるように描かれている
        box = pg.evaluate("""() => {
            const cs=[...document.querySelectorAll('canvas')].map(c=>c.getBoundingClientRect())
                .filter(r=>r.width>50).sort((a,b)=>b.width*b.height-a.width*a.height);
            const r=cs[0]; return r?{x:r.x,y:r.y,w:r.width,h:r.height}:null }""")
        if not box:
            raise RuntimeError("TactiVal のボードが見つかりません")
        s = min(box["w"] / 1920, box["h"] / 1080)
        ox, oy = box["x"] + (box["w"] - 1920 * s) / 2, box["y"] + (box["h"] - 1080 * s) / 2
        clip = {"x": ox + 448 * s, "y": oy + 28 * s, "width": 1024 * s, "height": 1024 * s}
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".png")
        pg.screenshot(path=str(tmp), clip=clip)
        tight(tmp, out)
        tmp.unlink(missing_ok=True)
        return out

    def close(self):
        for f in (self._br.close, self._pw.stop):
            try:
                f()
            except Exception:
                pass


def renderer(cfg: dict | None, log=print):
    global _R, _R_FAILED
    if _R is not None or _R_FAILED:
        return _R
    if ((cfg or {}).get("tactival") or {}).get("render") is False:
        _R_FAILED = True
        return None
    try:
        _R = Renderer()
        log("  TactiVal で動きの絵を描きます（https://tactival.app/board）")
    except Exception as e:
        _R_FAILED = True
        log(f"  TactiVal を開けませんでした（{str(e).splitlines()[0][:120]}）→ 公式のマップ画像に描いた絵にします"
            "（Colab ではセル 1 の `playwright install` を確認）")
    return _R


def close():
    global _R
    if _R is not None:
        _R.close()
        _R = None


# ---------------------------------------------------------------- 描く（予備：TactiVal 風）
def _fallback(cfg: dict, map_name: str, fr: dict, i: int, side: str, out: Path, size: int = 720) -> Path | None:
    import cv2
    import numpy as np
    from . import tactics as TC
    art = TC.map_art(cfg, map_name)
    if art is None:
        return None
    bg = np.zeros((size, size, 3), np.uint8)
    bg[:] = (18, 13, 11)
    a = cv2.resize(art, (size, size), interpolation=cv2.INTER_AREA)
    if a.shape[2] == 4:
        al = (a[:, :, 3:4].astype(np.float32) / 255) * 0.85
        bg = (bg * (1 - al) + a[:, :, :3] * al).astype(np.uint8)
    hexc = COL["attack" if side in ("attack", "攻撃") else "defense"].lstrip("#")
    col = tuple(int(hexc[j:j + 2], 16) for j in (4, 2, 0))
    P = lambda p: (int(p[0] * size), int(p[1] * size))
    for d in fr["dots"]:
        if i > 0 and math.dist(d["xy"][i - 1], d["xy"][i]) > 0.03:
            path = (d.get("route") or [None] * 9)[i - 1] or [d["xy"][i - 1], d["xy"][i]]
            pts = [P(p) for p in path]
            for j in range(len(pts) - 1):
                cv2.line(bg, pts[j], pts[j + 1], col, 2, cv2.LINE_AA)
            if len(pts) >= 2:
                cv2.arrowedLine(bg, pts[-2], pts[-1], col, 2, cv2.LINE_AA, tipLength=0.4)
    try:
        from . import agents as AG
        icons = AG.load_icons(cfg, [d["label"] for d in fr["dots"]]) or {}
    except Exception:
        icons = {}
    R = max(12, size // 34)
    for d in fr["dots"]:
        x, y = P(d["xy"][i])
        cv2.circle(bg, (x, y), R + 3, col, -1, cv2.LINE_AA)
        ic = icons.get(d["label"])
        if ic is not None:
            ic = cv2.resize(ic[:, :, :3] if ic.ndim == 3 else cv2.cvtColor(ic, cv2.COLOR_GRAY2BGR), (2 * R, 2 * R))
            m = np.zeros((2 * R, 2 * R), np.uint8)
            cv2.circle(m, (R, R), R, 255, -1)
            y0, x0 = y - R, x - R
            if 0 <= y0 and 0 <= x0 and y0 + 2 * R <= size and x0 + 2 * R <= size:
                roi = bg[y0:y0 + 2 * R, x0:x0 + 2 * R]
                roi[m > 0] = ic[m > 0]
        else:
            cv2.putText(bg, str(d["label"])[:2], (x - R + 2, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return tight(out, out)


# ---------------------------------------------------------------- まとめ
def images(cfg: dict, map_name: str, name: str, fr: dict, side: str, log=print) -> list[tuple[str, Path, str]]:
    """節目ごとの絵 [(節目, 画像, "tactival"|"fallback")]。"""
    cache = Path((cfg.get("output") or {}).get("dir") or ".") / "_cache" / "tactival_png"
    bs = boards(map_name, name, fr, side, cfg)
    out = []
    for i, mark in enumerate(fr["marks"]):
        b = bs[i] if bs else None
        if b is not None:
            key = json.dumps({**b, "steps": [{**b["steps"][0], "id": "", "label": "",
                                               "elements": [{k: v for k, v in e.items() if k != "id"}
                                                            for e in b["steps"][0]["elements"]]}]},
                             sort_keys=True, ensure_ascii=False)
            h = hashlib.sha1(key.encode()).hexdigest()[:16]
            p = cache / f"{h}.jpg"
            if p.exists():
                out.append((mark, p, "tactival"))
                continue
            R = renderer(cfg, log=log)
            if R is not None:
                try:
                    out.append((mark, R.shot(b, p), "tactival"))
                    continue
                except Exception as e:
                    log(f"  TactiVal で描けませんでした（{name} {mark}：{str(e).splitlines()[0][:100]}）")
        h = hashlib.sha1(json.dumps([map_name, side, fr["dots"], i], sort_keys=True).encode()).hexdigest()[:16]
        p = cache / f"fb_{h}.jpg"
        if p.exists() or _fallback(cfg, map_name, fr, i, side, p):
            out.append((mark, p, "fallback"))
    return out


def write_board(cfg: dict, map_name: str, name: str, fr: dict, side: str, path: Path) -> Path | None:
    b = full_board(map_name, name, fr, side, cfg)
    if not b:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    return path


def summary_text(fr: dict, xy: dict) -> list[str]:
    """節目ごとの一文（スライドの絵の下に置く）。"""
    out = []
    for i, mark in enumerate(fr["marks"]):
        mv, stay = moves(fr, i, xy)
        if i == 0:
            zs = Counter(nearest_area(xy, *d["xy"][0]) or "?" for d in fr["dots"])
            out.append("配置：" + "・".join(f"{z} {c}" for z, c in zs.most_common()))
        else:
            out.append("動き：" + ("／".join(mv) if mv else "大きな動きなし（その場で情報を待つ）"))
    return out
