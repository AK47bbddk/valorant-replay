"""照合の生の検出（ラウンド・時刻ごとの match の結果）から 1 秒格子の軌跡を組み、軌跡の「あり得なさ」の指標
（vreplay/track_metrics.py の層 A）と生存時間の被覆率を、ベースの track.py と今の track.py で比べる。

使い方: python tools/track_measure.py <detections.json> <report.json> [--base 0d58aae] [--rounds 2,5,6] [--json 出力]
  detections.json: {"icon_frac", "frames": [{"round", "t"（開始からの秒）, "found": [{team, agent, x, y, score}]}]}
  report.json: 解析レポート（players・rounds[].kills・tactics.rounds[].start/end/side1・tactics.callouts_placed）
ベースは git show <rev>:vreplay/track.py を読み込んで使う（deaths は渡さない＝ベースの tactics と同じ）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreplay import tactics as TT          # noqa: E402
from vreplay import track as TR            # noqa: E402
from vreplay import track_metrics as TM    # noqa: E402

ASPECT = 320 / 304          # この放送のミニマップ（320×304）。spawn_scale がこの比で相談票の 0.748 になる


def load_base(rev: str):
    src = subprocess.run(["git", "show", f"{rev}:vreplay/track.py"], capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    mod = types.ModuleType("track_base")
    exec(compile(src, f"{rev}:vreplay/track.py", "exec"), mod.__dict__)
    return mod


def spawns_of(placed: list[dict]) -> dict | None:
    sp = {}
    for c in placed or []:
        if str(c.get("name") or "").lower() != "spawn":
            continue
        low = str(c.get("super") or "").lower()
        key = "attack" if "attacker" in low else "defense" if "defender" in low else None
        if key:
            sp[key] = (float(c["x"]), float(c["y"]))
    return sp if len(sp) == 2 else None


def context(rep: dict, n: int) -> dict:
    """ラウンド n の開始・終わり・攻撃側・死亡時刻（開始からの秒）・最初のキル・生存の窓。"""
    rt = rep["tactics"]["rounds"][str(n)]
    rr = next(r for r in rep["rounds"] if r["n"] == n)
    start, end = float(rt["start"]), float(rt["end"])
    who = {p["name"]: (int(p["team"]), p["agent"]) for p in rep["players"]}
    side1 = TT._canon(rt.get("side1"))
    attack_team = 1 if side1 == TT.ATTACK else 2 if side1 == TT.DEFENSE else None
    deaths: dict = {}
    for k in rr.get("kills") or []:
        key = who.get(k.get("victim"))
        if key and k.get("t") is not None:
            deaths[key] = min(float(k["t"]) - start, deaths.get(key, 1e9))
    kts = [float(k["t"]) - start for k in rr.get("kills") or [] if k.get("t") is not None]
    alive = {key: (0.0, min(deaths.get(key, end - start), end - start)) for key in who.values()}
    ults = [{**u, "e": round(float(u["t"]) - start, 1)} for u in rr.get("ults") or []]
    return {"start": start, "end": end, "attack_team": attack_team, "deaths": deaths,
            "first_kill": min(kts) if kts else None, "alive": alive, "ults": ults}


def build_round(mod, frames: list, ctx: dict, scale: float, deaths: bool, fixed: dict | None = None) -> dict:
    fr = [(ctx["start"] + float(f["t"]), f["found"]) for f in frames]
    kw = {}
    if deaths:
        kw["deaths"] = {k: ctx["start"] + e for k, e in ctx["deaths"].items()}
    if fixed is not None:
        kw["fixed"] = fixed
    return mod.build(fr, ctx["start"], ctx["end"], scale, every=1.0, aspect=ASPECT, **kw)


def metrics(trk: dict, ctx: dict, scale: float, spawns, icon_frac) -> dict:
    m = TM.round_metrics(trk, scale, ASPECT, deaths=ctx["deaths"], first_kill=ctx["first_kill"],
                         attack_team=ctx["attack_team"], spawns=spawns, icon_frac=icon_frac)
    m["coverage"] = TM.coverage(trk, ctx["alive"])
    return m


def _keyed(d: dict) -> dict:
    return {f"{t}:{ag}": v for (t, ag), v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections")
    ap.add_argument("report")
    ap.add_argument("--base", default="0d58aae")
    ap.add_argument("--rounds", default="2,5,6")
    ap.add_argument("--json")
    a = ap.parse_args()
    det = json.loads(Path(a.detections).read_text("utf-8"))
    rep = json.loads(Path(a.report).read_text("utf-8"))
    base = load_base(a.base)
    placed = rep["tactics"]["callouts_placed"]
    scale = TT.spawn_scale(placed, ASPECT)
    spawns = spawns_of(placed)
    icon_frac = det.get("icon_frac")
    variants = {"base": (base, False), "fix1_no_deaths": (TR, False), "fix1": (TR, True)}
    result: dict = {"scale": round(scale, 4), "aspect": round(ASPECT, 4), "rounds": {}}
    for n in [int(x) for x in a.rounds.split(",")]:
        frames = [f for f in det["frames"] if int(f["round"]) == n]
        ctx = context(rep, n)
        rr: dict = {"context": {**ctx, "deaths": _keyed(ctx["deaths"]), "alive": _keyed(ctx["alive"])},
                    "metrics": {}, "tracks": {}, "fixed": {}}
        for name, (mod, use_deaths) in variants.items():
            fixed: dict | None = {} if mod is TR else None
            trk = build_round(mod, frames, ctx, scale, use_deaths, fixed)
            rr["metrics"][name] = metrics(trk, ctx, scale, spawns, icon_frac)
            rr["tracks"][name] = _keyed(trk)
            if fixed:
                rr["fixed"][name] = _keyed(fixed)
        result["rounds"][str(n)] = rr
    if a.json:
        Path(a.json).write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), "utf-8")
    print_table(result)


def print_table(result: dict) -> None:
    keys = ["coverage", "impossible", "impossible_per100s", "roundtrips", "zigzag_windows", "path_over",
            "path_ratio_max", "overlap", "mirror_jumps", "start_violations", "cross_before_contact",
            "after_death", "mass_jump_frames", "track_seconds"]
    print(f"scale={result['scale']} aspect={result['aspect']}")
    for n, rr in result["rounds"].items():
        names = list(rr["metrics"])
        print(f"\n== ラウンド {n} ==")
        print(f"{'指標':<22}" + "".join(f"{nm:>16}" for nm in names))
        for k in keys:
            print(f"{k:<22}" + "".join(f"{str(rr['metrics'][nm].get(k)):>16}" for nm in names))
        for nm, fx in rr["fixed"].items():
            print(f"  外れ値の置き換え（{nm}）: {fx}")


if __name__ == "__main__":
    main()
