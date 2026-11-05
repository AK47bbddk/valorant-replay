"""毎朝の Notion 更新（GitHub Actions の daily-notion）は requests・bs4・lxml・pyyaml・numpy しか入れない。
その環境で `run --upsert-only` が最後まで通ること（cv2 などを読むと毎朝失敗する）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import importlib.abc, sys, sysconfig
from pathlib import Path

# ワークフローで入れるもの（と、その依存）以外のサードパーティは読めないようにする
OK = {"requests", "bs4", "lxml", "yaml", "_yaml", "numpy", "urllib3", "certifi", "charset_normalizer",
      "idna", "soupsieve", "typing_extensions", "_pytest", "pytest", "pluggy", "iniconfig", "packaging"}
STD = set(sys.stdlib_module_names)

import importlib.machinery

def _third_party(name, path):
    spec = importlib.machinery.PathFinder.find_spec(name, path)
    o = str((spec and (spec.origin or (spec.submodule_search_locations or [""])[0])) or "")
    return "site-packages" in o or "dist-packages" in o

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in STD or top in OK or top == "vreplay" or "." in name:
            return None
        if _third_party(name, path):
            raise ModuleNotFoundError(f"daily-notion の環境には無い: {name}", name=name)
        return None
sys.meta_path.insert(0, Block())

import os, tempfile
from datetime import date
import vreplay.__main__  # noqa: F401  （CLI の読み込み）
from vreplay import config, pipeline, vlr

m = vlr.Match(id="8", url="https://www.vlr.gg/8/x", team1="A", team2="B", event="VCT 2026: Pacific")
m.maps = [vlr.MapData(order=1, name="Bind", game_id="1", score1=13, score2=5)]
vlr.parse_match = lambda url: m
vlr.list_results = lambda d: [{"url": m.url, "event": m.event, "team1": "A", "team2": "B"}]
os.environ["NOTION_TOKEN"] = "dummy"
calls = []
pipeline.upsert_match = lambda nt, cfg, mm: calls.append(mm.id) or ("pid", {1: {"id": "p1"}})
cfg = config.load()
cfg["output"]["dir"] = tempfile.mkdtemp()
pipeline.run(cfg, day=date(2026, 10, 21), match_urls=[m.url], upsert_only=True, log=lambda *a: None)
assert calls == ["8"], calls
print("ok")
'''


def test_upsert_only_runs_with_workflow_packages_only():
    r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "ok" in r.stdout
