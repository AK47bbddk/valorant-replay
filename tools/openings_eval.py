"""初動の分類 v1 / v2 をコーチの正解ラベルと比べる（指示書 ticket_barrier_v2 の 4. 検証）。

使い方:
  python tools/openings_eval.py --data C:/Users/Aokki/barrier_v2_data --out eval.json [--param window_t=18 ...]

- tracks/openings_mass.json・openings_batch3.json の全ラウンドに v2（vreplay/openings.py）を当てる。v1 は軌跡の
  attack_pred（v1 の予測）をそのまま使う。
- labels/ の正解と比べる：攻撃は類型＋エリアに正規化して一致率（全体・マップ別・類型別の混同）、防衛は人数配置
  （数値形式）の再現率とキー配置のレーン一致率。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import fields, replace
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vreplay import barriers as BR  # noqa: E402
from vreplay import openings as OP  # noqa: E402

TRACK_FILES = ("openings_mass.json", "openings_batch3.json")
AREA_WORDS = (("MID", "Mid"), ("ミッド", "Mid"), ("Mid", "Mid"), ("A", "A"), ("B", "B"), ("C", "C"))
TYPE_WORDS = (("詰め待ち", "wait"), ("ラッシュ", "rush"), ("スプリット", "split"), ("複合", "default"),
              ("デフォルト", "default"))
AGENT_JA = {"チェンバー": "Chamber", "セージ": "Sage", "サイファー": "Cypher", "ソーヴァ": "Sova", "Sova": "Sova",
            "Chamber": "Chamber", "Sage": "Sage", "Cypher": "Cypher"}
SEQ_RE = re.compile(r"(?<![\d])\d(?:-\d){1,3}(?![\d])")
COUNT_RE = re.compile(r"(MID|Mid|A|B|C)(\d)(?![\d-])")
EMPTY_RE = re.compile(r"(MID|Mid|A|B|C)空け")
KEY_RE = re.compile(r"(MID|Mid|A|B|C)\d?((?:引き目)?(?:チェンバー|セージ|サイファー|ソーヴァ|Sova|Chamber|Sage|Cypher)+)")


# ------------------------------------------------------------------ ラベルの正規化
def norm_attack(label: str | None) -> tuple[str, frozenset] | None:
    """攻撃のラベル → (類型, エリアの集合)。複合（v2）・「MID-Bデフォルト」は default ＋ 2 エリア。"""
    s = (label or "").strip()
    kind = next((k for w, k in TYPE_WORDS if w in s), None) if s else None
    if kind is None:
        return None
    if kind == "wait":
        return "wait", frozenset()
    word = {"rush": "ラッシュ", "split": "スプリット"}.get(kind, "複合" if "複合" in s else "デフォルト")
    head = s.split(word)[0]
    if head.startswith("Attacker Side"):
        return kind, frozenset()
    areas = set()
    for part in re.split(r"[-・]", head):
        a = next((a for w, a in AREA_WORDS if part.startswith(w)), None)
        if a:
            areas.add(a)
    return kind, frozenset(areas)


def attack_match(pred: str | None, truth: str | None) -> bool:
    p, t = norm_attack(pred), norm_attack(truth)
    return p is not None and t is not None and p == t


def _lane(s: str) -> str:
    return "Mid" if s.upper() == "MID" else s


def defense_numeric_items(label: str) -> list[tuple]:
    """防衛のラベルの数値形式：('seq', [1,3,1]) / ('count', 'A', 3) / ('empty', 'C')。"""
    items = [("seq", [int(x) for x in m.group(0).split("-")]) for m in SEQ_RE.finditer(label)]
    items += [("count", _lane(m.group(1)), int(m.group(2))) for m in COUNT_RE.finditer(label)]
    items += [("empty", _lane(m.group(1))) for m in EMPTY_RE.finditer(label)]
    return items


def _group_count(d: dict, lane: str) -> int:
    groups = [g.split("+") for g in d["groups"]]
    g = next((g for g in groups if lane in g), [lane])
    return sum(d["counts"].get(ln, 0) for ln in g)


def defense_item_ok(item: tuple, d: dict) -> bool:
    if item[0] == "seq":
        got = [int(x) for x in d["distribution"].split("-")]
        nz = [x for x in got if x]
        return item[1] in (got, got[::-1], nz, nz[::-1])
    if item[0] == "count":
        return _group_count(d, item[1]) == item[2]
    if item[0] == "empty":
        return _group_count(d, item[1]) == 0
    return False


def defense_key_items(label: str) -> list[tuple[str, str]]:
    """「Aチェンバー配置」「A1Sova配置」「Bチェンバーセージ配置」→ [(agent, lane)]。レーンの無いものは数えない。"""
    out = []
    for m in KEY_RE.finditer(label):
        for ja, en in AGENT_JA.items():
            if ja in m.group(2) and (en, _lane(m.group(1))) not in out:
                out.append((en, _lane(m.group(1))))
    return out


# ------------------------------------------------------------------ 読み込み
def load_labels(data: Path) -> dict:
    out = {}
    for f in sorted((data / "labels").glob("*.json")):
        for it in json.loads(f.read_text(encoding="utf-8"))["items"]:
            out[it["key"]] = dict(it, _file=f.name)
    return out


def unit_of(r: dict) -> str:
    return r.get("unit") or str(r["map"]).lower()


class Maps:
    """unit ごとの tactics と BarrierMap（作るのは 1 度）。"""

    def __init__(self, data: Path):
        self.data = data
        self.cache: dict = {}

    def get(self, unit: str, map_name: str):
        if unit not in self.cache:
            m = map_name.lower()
            tac = json.loads((self.data / "reports" / f"{unit}.json").read_text(encoding="utf-8"))["tactics"]
            kb = yaml.safe_load((ROOT / "config" / "kb" / "maps" / f"{m}.yaml").read_text(encoding="utf-8"))
            bm = None
            if BR.has_barriers(kb):
                bg = cv2.imread(str(self.data / "backgrounds" / f"{unit}.png"))
                icon = cv2.imread(str(self.data / "map_icons" / f"{m}.png"), cv2.IMREAD_UNCHANGED)
                if bg is None or icon is None or icon.ndim != 3 or icon.shape[2] != 4:
                    print(f"[warn] {unit}: 背景かアイコン（RGBA）が読めない → v1 相当で評価", file=sys.stderr)
                else:
                    try:
                        bm = BR.for_match(kb, icon[:, :, 3], tac, bg.shape)
                    except ValueError as e:
                        print(f"[warn] {unit}: バリアの領域が作れない（{e}）→ v1 相当で評価", file=sys.stderr)
            self.cache[unit] = (tac, bm, kb)
        return self.cache[unit]


def run(data: Path, prm: OP.Params, maps: Maps | None = None) -> list[dict]:
    maps = maps or Maps(data)
    labels = load_labels(data)
    rows = []
    for tf in TRACK_FILES:
        for r in json.loads((data / "tracks" / tf).read_text(encoding="utf-8")):
            tac, bm, kb = maps.get(unit_of(r), r["map"])
            res = OP.classify_round(r, tac, bm, prm, kb_map=kb)
            lab = labels.get(r["key"]) or {}
            rows.append({"key": r["key"], "set": tf, "map": r["map"], "v1": r.get("attack_pred"),
                         "v1_defense": v1_defense(r, res["defense"]),
                         "v2": res["attack"]["label"], "v2_type": res["attack"]["type"],
                         "features": res["features"], "defense": res["defense"], "fallback_v1": res["fallback_v1"],
                         "truth_attack": lab.get("attack"),
                         "truth_defense": " ".join(x for x in (lab.get("defense"), lab.get("note")) if x),
                         "label_file": lab.get("_file")})
    return rows


def v1_defense(r: dict, v2: dict) -> dict:
    """v1 の防衛（軌跡の defense_counts7 / defense_key_agents7）を v2 と同じ形にする（並びは v2 の groups）。"""
    counts: dict = {}
    for area, n in (r.get("defense_counts7") or {}).items():
        z = OP._zone(area)
        if z:
            counts[z] = counts.get(z, 0) + int(n)
    seq = [sum(counts.get(ln, 0) for ln in g.split("+")) for g in v2["groups"]]
    keys = {a: {"lane": OP._zone(area)} for a, area in (r.get("defense_key_agents7") or {}).items()}
    return {"counts": counts, "groups": v2["groups"], "distribution": "-".join(map(str, seq)), "keys": keys}


# ------------------------------------------------------------------ 集計
def _group(rows, key) -> dict:
    g: dict = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return g


def _acc(rs: list, k: str) -> tuple[int, int]:
    return sum(attack_match(r[k], r["truth_attack"]) for r in rs), len(rs)


def _confusion(rs: list, k: str) -> dict:
    c = collections.Counter()
    for r in rs:
        p = norm_attack(r[k])
        c[(norm_attack(r["truth_attack"])[0], p[0] if p else "none")] += 1
    return {f"{a}->{b}": n for (a, b), n in sorted(c.items())}


def summarize(rows: list[dict]) -> dict:
    lab = [r for r in rows if norm_attack(r["truth_attack"])]
    b3 = [r for r in lab if r["set"] == "openings_batch3.json"]
    bar = [r for r in lab if not r["fallback_v1"]]
    rush = [r for r in lab if norm_attack(r["truth_attack"])[0] == "rush"]
    kind = lambda r: norm_attack(r["truth_attack"])[0]  # noqa: E731
    return {
        "n_rounds": len(rows), "n_attack_labeled": len(lab),
        "attack": {
            "all": {"v1": _acc(lab, "v1"), "v2": _acc(lab, "v2")},
            "batch3": {"v1": _acc(b3, "v1"), "v2": _acc(b3, "v2"),
                       "v1_exact_string": (sum(r["v1"] == r["truth_attack"] for r in b3), len(b3))},
            "barrier_maps_only": {"v1": _acc(bar, "v1"), "v2": _acc(bar, "v2")},
        },
        "by_map": {m: {"v1": _acc(rs, "v1"), "v2": _acc(rs, "v2")} for m, rs in sorted(_group(lab, lambda r: r["map"]).items())},
        "by_truth_type": {k: {"v1": _acc(rs, "v1"), "v2": _acc(rs, "v2")} for k, rs in sorted(_group(lab, kind).items())},
        "confusion_type_v1": _confusion(lab, "v1"),
        "confusion_type_v2": _confusion(lab, "v2"),
        "rush_recall": {"v1": _acc(rush, "v1"), "v2": _acc(rush, "v2"),
                        "v2_type_only": (sum(r["v2_type"] == "rush" for r in rush), len(rush))},
        "defense": {"v2": defense_summary(rows), "v1": defense_summary(rows, "v1_defense")},
    }


def defense_summary(rows: list[dict], which: str = "defense") -> dict:
    by_kind: dict = collections.defaultdict(lambda: [0, 0])
    key = [0, 0]
    for r in rows:
        lab = r["truth_defense"] or ""
        for it in defense_numeric_items(lab):
            by_kind[it[0]][0] += defense_item_ok(it, r[which])
            by_kind[it[0]][1] += 1
        for agent, lane in defense_key_items(lab):
            key[0] += r[which]["keys"].get(agent, {}).get("lane") == lane
            key[1] += 1
    tot = [sum(v[0] for v in by_kind.values()), sum(v[1] for v in by_kind.values())]
    return {"numeric": tuple(tot), "numeric_by_kind": {k: tuple(v) for k, v in by_kind.items()},
            "key_lane": tuple(key)}


FEATURES = ("t_first_neutral", "t_first_def_barrier", "n_neutral_wait", "lanes_with_2plus", "max_in_top_lane",
            "n_site_rush", "n_deep_rush")


def feature_table(rows: list[dict]) -> dict:
    """正解の類型ごとの特徴量の分布（最小・中央・最大）。バリアのあるマップだけ。"""
    out = {}
    rs_all = [r for r in rows if norm_attack(r["truth_attack"]) and not r["fallback_v1"]]
    for kind, rs in sorted(_group(rs_all, lambda r: norm_attack(r["truth_attack"])[0]).items()):
        out[kind] = {"n": len(rs)}
        for n in FEATURES:
            v = sorted(r["features"][n] for r in rs if r["features"][n] is not None)
            out[kind][n] = (v[0], v[len(v) // 2], v[-1], len(v)) if v else None
    return out


def parse_params(items: list[str]) -> OP.Params:
    types = {f.name: f.type for f in fields(OP.Params)}
    kw = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in types:
            raise SystemExit(f"不明なパラメータ: {k}")
        t = str(types[k])
        kw[k] = (v.lower() in ("1", "true")) if "bool" in t else int(v) if "int" in t else float(v)
    return replace(OP.Params(), **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--param", action="append", default=[])
    a = ap.parse_args(argv)
    prm = parse_params(a.param)
    rows = run(a.data, prm)
    summ = summarize(rows)
    summ["features_by_truth_type"] = feature_table(rows)
    summ["params"] = dict(prm.__dict__)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1, default=str),
                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
