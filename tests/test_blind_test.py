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
    text = out.read_text("utf-8")
    assert out.exists() and out.stat().st_size > 1000
    assert "スモーク・壁検知の正確性" in text
    assert "初動配置・戦術の分類" in text


def test_collect_from_dumps_and_30_questions(tmp_path):
    import cv2
    import numpy as np

    dump = tmp_path / "dump_Sunset"
    (dump / "icons").mkdir(parents=True)
    img = np.full((240, 240, 3), 120, np.uint8)
    cv2.circle(img, (120, 100), 14, (65, 65, 65), -1)
    cv2.circle(img, (120, 100), 14, (40, 40, 220), 2)
    frames_meta = []
    for i in range(1, 35):
        fn = f"r{i:02d}_t015.png"
        cv2.imwrite(str(dump / fn), img)
        frames_meta.append({"file": fn, "round": i, "t": 15.0, "color_team": {"warm": 1, "cool": 2}, "alive": {"1": 5, "2": 5}})
    (dump / "meta.json").write_text(json.dumps({"teams": {"1": [], "2": []}, "icon_frac": 0.055, "frames": frames_meta}), encoding="utf-8")

    items = BT.collect_from_dumps_and_classified([dump], max_items=30, seed=42)
    assert len(items) == 30
    assert all("shape_choices" in it and len(it["shape_choices"]) >= 2 for it in items)
    assert all("overlay" in it for it in items)
    out = BT.build_html(items, tmp_path / "blind30.html")
    assert out.exists() and "ITEMS" in out.read_text("utf-8")
