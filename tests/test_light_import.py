"""毎朝の Notion 更新（GitHub Actions）は cv2 を入れていない。その経路で cv2 を読まないこと。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_upsert_path_imports_without_cv2():
    code = (
        "import sys; sys.modules['cv2'] = None\n"
        "import vreplay.__main__, vreplay.pipeline, vreplay.notion, vreplay.vlr, vreplay.db\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
