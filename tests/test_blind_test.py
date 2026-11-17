import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import blind_test as BT


def test_collect_items_and_build_html(tmp_path):
    gif = tmp_path / "r01.gif"
    board = tmp_path / "_tactics" / "board_r01_140.jpg"
    board.parent.mkdir(parents=True)
    gif.write_bytes(b"GIF89a\x01\x00\x01\x00\x00\xff\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")
    board.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")

    rep = {
        "map": "Sunset",
        "teams": [{"name": "Team Alpha"}, {"name": "Team Beta"}],
        "rounds": [
            {
                "n": 1,
                "side1": "攻撃",
                "assets": {
                    "minimap_gif": "/content/drive/MyDrive/match/r01.gif",
                    "minimap_start": "/content/drive/MyDrive/match/start_r01.jpg",
                },
                "tactics": {
                    "boards": {"1:40": "/content/drive/MyDrive/match/_tactics/board_r01_140.jpg"},
                    "site": "A",
                    "agents": {"1:40": [{"agent": "Jett", "team": 1, "callout": "A Lobby", "zone": "A"}]},
                },
            }
        ],
    }
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")

    items = BT.collect_items(tmp_path, max_items=10)
    assert len(items) == 1
    assert items[0]["gif"].startswith("data:image/gif;base64,")
    assert items[0]["minimap"].startswith("data:image/jpeg;base64,")
    assert items[0]["answer"]["site"] == "A"
    assert items[0]["answer"]["agents_140"][0]["agent"] == "Jett"

    out = BT.build_html(items, tmp_path / "blind.html")
    text = out.read_text("utf-8")
    assert "位置確認ブラインドテスト" in text
    assert "Team Alpha vs Team Beta" in text


def test_demo_items_build_html(tmp_path):
    items = BT.demo_items()
    assert len(items) == 2
    out = BT.build_html(items, tmp_path / "demo.html")
    assert out.exists() and out.stat().st_size > 1000
