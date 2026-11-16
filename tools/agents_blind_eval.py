"""ブラインド配置（agents_check_html.py --blind で人が置いた正解）で、エージェント照合の P/R を数える。

使い方:
  python tools/agents_blind_eval.py <dump_dir> <blind.json> [--tol 0.03] [--json out.json]
      [--config config/default.yaml] [--set key=value ...]

照合は tools/agents_eval.py の replay でやり直す（meta.json の params に --set を上書き）。
突き合わせはチェック HTML と同じ：
  1) 同じチーム・同じエージェントで距離 ≤ tol を近い順に貪欲に組む → 一致（TP）
  2) 残りどうしを、同じチームで距離 ≤ tol を近い順に組む → 取り違え（swap。TP には数えない）
  3) 採用点のうち一致しないもの = FP（取り違えも含む）、配置のうち一致しないもの = FN
距離は正規化座標（x は幅、y は高さで割った値）の差のユークリッド距離。
FN は理由に分ける：swap / rejected:<棄却理由>（近くの棄却候補が同じエージェント）/
near:<棄却理由>（近くの棄却候補が別のエージェント）/ no-candidate（tol 以内に候補が無い）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents_eval as AE  # noqa: E402


def load_blind(path: str) -> dict[str, list[dict]]:
    """ブラインド配置の JSON を {file: [{team, agent, x, y}]} にする。"""
    raw = json.loads(Path(path).read_text("utf-8"))
    out: dict[str, list[dict]] = {}
    for fr in raw.get("frames") or []:
        out[fr["file"]] = [{"team": int(p[0]), "agent": p[1], "x": float(p[2]), "y": float(p[3])}
                           for p in fr.get("placed") or []]
    return out


def _d(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _pair(P, K, usedP, usedK, cond, tol):
    cs = sorted((_d(p, k), i, j) for i, p in enumerate(P) for j, k in enumerate(K)
                if i not in usedP and j not in usedK and cond(p, k) and _d(p, k) <= tol)
    out = []
    for d, i, j in cs:
        if i in usedP or j in usedK:
            continue
        usedP.add(i)
        usedK.add(j)
        out.append((i, j, d))
    return out


def match_frame(placed: list[dict], found: list[dict], rejected: list[dict], tol: float = 0.03) -> dict:
    P, K = placed, found
    usedP: set = set()
    usedK: set = set()
    tp = _pair(P, K, usedP, usedK, lambda p, k: p["team"] == k.get("team") and p["agent"] == k.get("agent"), tol)
    sw = _pair(P, K, usedP, usedK, lambda p, k: p["team"] == k.get("team"), tol)
    swap_p = {i: K[j] for i, j, _ in sw}
    misses = []
    for i, p in enumerate(P):
        if any(i == a for a, _, _ in tp):
            continue
        if i in swap_p:
            reason = "swap"
        else:
            near = [r for r in rejected if _d(p, r) <= tol]
            if not near:
                reason = "no-candidate"
            else:
                r = min(near, key=lambda r: _d(p, r))
                rr = r.get("reason") or "thr"
                reason = f"rejected:{rr}" if r.get("agent") == p["agent"] else f"near:{rr}"
        misses.append({**p, "reason": reason,
                       "as": swap_p[i].get("agent") if i in swap_p else None})
    fps = [K[j] for j in range(len(K)) if j not in {j for _, j, _ in tp}]
    return {"tp": [P[i] for i, _, _ in tp], "misses": misses, "fp": fps, "swap": len(sw),
            "n_placed": len(P), "n_kept": len(K)}


def evaluate(results: list[dict], blind: dict, tol: float = 0.03) -> dict:
    tot = Counter()
    by_agent = defaultdict(lambda: [0, 0])
    reasons: Counter = Counter()
    agent_reasons: dict = defaultdict(Counter)
    fp_by_agent: Counter = Counter()
    swaps: Counter = Counter()
    frames = []
    for r in results:
        if r["file"] not in blind:
            continue
        m = match_frame(blind[r["file"]], r.get("found") or [], r.get("rejected") or [], tol)
        tot["tp"] += len(m["tp"])
        tot["kept"] += m["n_kept"]
        tot["placed"] += m["n_placed"]
        tot["swap"] += m["swap"]
        for p in blind[r["file"]]:
            by_agent[p["agent"]][0] += 1
        for p in m["tp"]:
            by_agent[p["agent"]][1] += 1
        for x in m["misses"]:
            reasons[x["reason"]] += 1
            agent_reasons[x["agent"]][x["reason"]] += 1
            if x["reason"] == "swap":
                swaps[f"{x['agent']}->{x['as']}"] += 1
        for f in m["fp"]:
            fp_by_agent[f.get("agent")] += 1
        frames.append({"file": r["file"], "misses": m["misses"],
                       "fp": [{k: f.get(k) for k in ("team", "agent", "x", "y", "score")} for f in m["fp"]]})
    P = tot["tp"] / tot["kept"] if tot["kept"] else None
    R = tot["tp"] / tot["placed"] if tot["placed"] else None
    return {"tp": tot["tp"], "kept": tot["kept"], "placed": tot["placed"], "fp": tot["kept"] - tot["tp"],
            "swap": tot["swap"], "precision": P, "recall": R,
            "by_agent": {a: {"n": n, "tp": c, "recall": c / n, "fp": fp_by_agent.get(a, 0),
                             "miss_reasons": dict(agent_reasons[a].most_common())}
                         for a, (n, c) in sorted(by_agent.items())},
            "fp_by_agent": dict(fp_by_agent.most_common()),
            "miss_reasons": dict(reasons.most_common()), "swaps": dict(swaps.most_common()),
            "frames": frames}


def _parse_set(items) -> dict:
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def summary_text(ev: dict) -> str:
    """評価の要約。画像が 1 枚も読めず kept・placed が 0 のときも落ちない（P・R は「—」）。"""
    lines = [f"precision {_pct(ev['precision'])} ({ev['tp']}/{ev['kept']}) / "
             f"recall {_pct(ev['recall'])} ({ev['tp']}/{ev['placed']}) / FP {ev['fp']}（うち取り違え {ev['swap']}）"]
    if ev.get("fp_by_agent"):
        lines.append("FP のエージェント別: " + " / ".join(f"{a} {n}" for a, n in ev["fp_by_agent"].items()))
    lines.append("エージェント別 recall: " + "・".join(
        f"{a} {v['recall'] * 100:.0f}% ({v['tp']}/{v['n']})"
        for a, v in sorted(ev["by_agent"].items(), key=lambda kv: kv[1]["recall"])))
    lines.append("取りこぼしの理由: " + " / ".join(f"{k} {v}" for k, v in ev["miss_reasons"].items()))
    if ev["swaps"]:
        lines.append("取り違え（正解->採用）: " + " / ".join(f"{k} {v}" for k, v in ev["swaps"].items()))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ブラインド配置の正解でエージェント照合を数える")
    ap.add_argument("dump_dir")
    ap.add_argument("blind")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--json", default=None, help="結果（フレームごとの見落とし・FP 付き）を書き出す")
    ap.add_argument("--config", default=None,
                    help="config の yaml（tactics.agent_match を照合の引数にする。meta.json の params より優先）")
    ap.add_argument("--set", nargs="*", default=[], help="照合の引数の上書き（例 thr=0.55 face_search=true）。--config より優先")
    a = ap.parse_args(argv)
    params = {}
    if a.config:
        import yaml
        cfg = yaml.safe_load(Path(a.config).read_text("utf-8")) or {}
        params.update((cfg.get("tactics") or {}).get("agent_match") or {})
    params.update(_parse_set(a.set))
    results = AE.replay(a.dump_dir, **params)
    ev = evaluate(results, load_blind(a.blind), a.tol)
    print(summary_text(ev))
    if a.json:
        Path(a.json).write_text(json.dumps(ev, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
