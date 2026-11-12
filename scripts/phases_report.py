"""report.json の glob から、ラウンドの「フェーズ分類」（段階A）を集計して JSON と Markdown で出す。

    python scripts/phases_report.py "<data>/**/report.json" --map Split --out <出力フォルダ>

構成（マップ×5人）ごとに、序盤・中盤・終盤の型の件数と代表ラウンド（マッチ・ラウンド番号・流れ要約）を出す。
1 ラウンドの分類は、そのラウンドに出た両チームの構成それぞれに数える（流れ要約は両チームの動きを 1 本にしたもの）。
先頭に、分類できたラウンド数と tracks 無しで対象外になった数を出す（落とさず・混ぜない）。
判定は `vreplay.phases`、エリアの距離は `vreplay.tactics.area_ranks`（report.json の callouts_placed）から取る。
"""
import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vreplay import phases, tactics  # noqa: E402

SEGS = (("early", "序盤"), ("mid", "中盤"), ("late", "終盤"))


def _comp_of(rep: dict) -> dict[int, str | None]:
    agents: dict[int, list] = {1: [], 2: []}
    for p in rep.get("players") or []:
        if p.get("team") in (1, 2) and p.get("agent"):
            agents[p["team"]].append(p["agent"])
    return {no: "|".join(sorted(agents[no])) if agents[no] else None for no in (1, 2)}


def _tag(rep: dict, no: int) -> str:
    return rep.get(f"tag{no}") or rep.get(f"team{no}") or f"team{no}"


def collect(reports: list[dict], map_name: str) -> dict:
    target = str(map_name).lower()
    reps = [r for r in reports if str(r.get("map") or "").lower() == target]
    classified = skipped = 0
    distribution: dict[str, Counter] = {s: Counter() for s, _ in SEGS}
    comps: dict = {}
    for rep in reps:
        comp = _comp_of(rep)
        ranks = tactics.area_ranks((rep.get("tactics") or {}).get("callouts_placed") or [])
        for rr in rep.get("rounds") or []:
            if phases.skip_reason(rr):
                skipped += 1
                continue
            res = phases.classify_flow(rr, ranks)
            if res is None:
                skipped += 1
                continue
            classified += 1
            for seg, _label in SEGS:
                typ = res["phases"][seg]["type"]
                if typ:
                    distribution[seg][typ] += 1
            for no in (1, 2):
                ck = comp.get(no)
                if not ck or "?" in ck:
                    continue
                b = comps.setdefault(ck, {"key": ck, "agents": ck.split("|"), "teams": set(),
                                          "rounds": 0, "types": {s: Counter() for s, _ in SEGS},
                                          "reps": {}})
                b["teams"].add(_tag(rep, no))
                b["rounds"] += 1
                for seg, _label in SEGS:
                    typ = res["phases"][seg]["type"]
                    if not typ:
                        continue
                    b["types"][seg][typ] += 1
                    prev = b["reps"].get((seg, typ))
                    cand = {"match_id": str(rep.get("match_id")), "map": rep.get("map"),
                            "date": rep.get("date"), "team": _tag(rep, no),
                            "round": rr.get("n"), "flow": res["flow"].get(seg, ""),
                            "events": len(res["events"])}
                    if prev is None or cand["events"] > prev["events"]:
                        b["reps"][(seg, typ)] = cand
    out_comps = []
    for ck, b in sorted(comps.items(), key=lambda kv: -kv[1]["rounds"]):
        reps_out = {}
        for (seg, typ), r in b["reps"].items():
            reps_out.setdefault(seg, {})[typ] = r
        out_comps.append({"key": ck, "agents": b["agents"], "teams": sorted(b["teams"]),
                          "rounds": b["rounds"],
                          "types": {s: dict(b["types"][s]) for s, _ in SEGS},
                          "representatives": reps_out})
    return {"map": map_name, "reports": len(reps), "classified": classified, "skipped_no_tracks": skipped,
            "distribution": {s: dict(distribution[s]) for s, _ in SEGS},
            "comps": out_comps}


def markdown(data: dict) -> str:
    lines = [f"# フェーズ分類レポート: {data['map']}", "",
             f"- 対象 report.json: {data['reports']} 件",
             f"- 分類できたラウンド: {data['classified']}",
             f"- tracks 無しで対象外: {data['skipped_no_tracks']}",
             f"- 構成: {len(data['comps'])}", ""]
    lines.append("## 型の分布（全体）")
    lines.append("")
    for seg, label in SEGS:
        types = (data.get("distribution") or {}).get(seg) or {}
        lines.append(f"### {label}")
        if not types:
            lines.append("- 型なし")
            lines.append("")
            continue
        lines.append("| 型 | 件数 |")
        lines.append("| --- | --- |")
        for typ, n in sorted(types.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {typ} | {n} |")
        lines.append("")
    for b in data["comps"]:
        lines.append(f"## {' / '.join(b['agents'])}")
        lines.append(f"- ラウンド: {b['rounds']} ・チーム: {', '.join(b['teams']) or '-'}")
        for seg, label in SEGS:
            types = b["types"].get(seg) or {}
            reps = (b["representatives"].get(seg) or {})
            if not types:
                lines.append(f"### {label}")
                lines.append("- 型なし")
                continue
            lines.append(f"### {label}")
            lines.append("| 型 | 件数 | 代表 |")
            lines.append("| --- | --- | --- |")
            for typ, n in sorted(types.items(), key=lambda kv: -kv[1]):
                r = reps.get(typ) or {}
                where = f"{r.get('match_id')} R{r.get('round')}" if r else "-"
                lines.append(f"| {typ} | {n} | {where}　{r.get('flow') or ''} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="report.json の glob（例 'data/**/report.json'）")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True, help="出力フォルダ（phases.json と phases.md を書く）")
    a = ap.parse_args(argv)
    reports = []
    for f in sorted(glob.glob(a.pattern, recursive=True)):
        try:
            reports.append(json.loads(Path(f).read_text("utf-8")))
        except Exception as e:                                   # noqa: BLE001
            print(f"読めませんでした: {f}（{e}）")
    data = collect(reports, a.map)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phases.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "phases.md").write_text(markdown(data), encoding="utf-8")
    print(f"report.json {data['reports']} 件 / 分類 {data['classified']} / tracks 無し {data['skipped_no_tracks']}")
    print(f"構成 {len(data['comps'])} 件 → {out / 'phases.json'} ・ {out / 'phases.md'}")


if __name__ == "__main__":
    main()
