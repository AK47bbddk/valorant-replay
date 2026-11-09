"""report.json の glob から、プレイブック v4 を 1 冊作る。

    python scripts/playbook_v4.py "<output.dir>/**/report.json" --team PRX --map Split --out <出力フォルダ> [--config config.yaml]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vreplay import config, playbook_v4  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="report.json の glob（例 'drive/vreplay/**/report.json'）")
    ap.add_argument("--team", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    reps = []
    for f in sorted(glob.glob(a.pattern, recursive=True)):
        try:
            reps.append(json.loads(Path(f).read_text("utf-8")))
        except Exception as e:
            print(f"読めませんでした: {f}（{e}）")
    print(f"{len(reps)} 件の report.json")
    print(playbook_v4.build(reps, a.team, a.map, a.out, cfg))


if __name__ == "__main__":
    main()
