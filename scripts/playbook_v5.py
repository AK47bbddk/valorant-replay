"""report.json の glob から、マップ × 5人構成のプレイブック v5 を一式作る。

    python scripts/playbook_v5.py "<data>/**/*.json" --map Split --out <出力フォルダ> [--config config.yaml]

出力フォルダには構成ごとのページ（comp_*.html）、入口の index.html、manifest.json ができる。
マップ画像のキャッシュは <out>/_work/_cache に自己完結させる。
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vreplay import config, playbook_v5  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="report.json の glob（例 'data/by_map/Split/*.json'）")
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    reps = []
    for f in sorted(glob.glob(a.pattern, recursive=True)):
        try:
            reps.append(json.loads(Path(f).read_text("utf-8")))
        except Exception as e:                                   # noqa: BLE001
            print(f"読めませんでした: {f}（{e}）")
    print(f"{len(reps)} 件の report.json")
    print(playbook_v5.generate(reps, a.map, a.out, cfg))


if __name__ == "__main__":
    main()
