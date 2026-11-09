"""プレイブック v4 を report.json から作る（Colab・手元どちらでも）。

    python scripts/playbook_v4.py --team PRX --map Split --reports "/content/drive/MyDrive/valorant-replay/**/report.json" \
        --out /content/drive/MyDrive/valorant-replay/playbook_v4_test/PRX_Split_v4_test.html
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vreplay import playbook_v4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--reports", required=True, help="report.json の glob（** 可）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reps, seen = [], set()
    for f in sorted(glob.glob(a.reports, recursive=True), key=lambda f: -Path(f).stat().st_size):
        try:
            r = json.loads(Path(f).read_text("utf-8"))
        except Exception as e:
            print(f"  読めません：{f}（{e}）")
            continue
        if str(r.get("map", "")).lower() != a.map.lower():
            continue
        key = (r.get("match_id"), r.get("order"))
        if key in seen:                     # 同じマップの report が 2 つあれば大きい方（先に読んだ方）を使う
            continue
        seen.add(key)
        reps.append(r)
    playbook_v4.build(reps, team=a.team, map_name=a.map, out=Path(a.out))


if __name__ == "__main__":
    main()
