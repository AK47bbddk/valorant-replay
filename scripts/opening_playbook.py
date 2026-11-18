"""全マップの初動配置＆DADS HTMLプレイブックを生成する CLI スクリプト。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vreplay import config as cfg_mod
from vreplay import opening_playbook as op


def _load_safe_cfg(config_path: str | None) -> dict:
    if config_path:
        return cfg_mod.load(config_path)
    try:
        return cfg_mod.load()
    except SystemExit:
        return yaml.safe_load(cfg_mod.DEFAULT.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opening_playbook.py",
        description="LOUD Summit テンプレートと KB・TactiVal バリア知識から全マップの初動配置とプレイブック HTML を生成する",
    )
    op.add_openings_args(ap)
    args = ap.parse_args(argv)
    cfg = _load_safe_cfg(args.config)
    op.run_openings_cli(args, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
