from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from tools import expand_dataset, train_yolo_onnx
from vreplay import yolo_onnx


def test_letterbox_and_decode_both_yolo_layouts():
    img = np.full((304, 320, 3), 40, dtype=np.uint8)
    padded, ratio, pad = yolo_onnx.letterbox(img, new_shape=320)
    assert padded.shape == (320, 320, 3)

    # Create 2 detections (1 agent at center, 1 overlapping lower-conf agent, 1 smoke at top-left)
    # In letterbox coords: center = (160, 160), smoke = (80, 90)
    raw_n_first = np.array(
        [[
            [160.0, 160.0, 16.0, 16.0, 0.92, 0.02, 0.01],
            [161.0, 160.0, 16.0, 16.0, 0.65, 0.02, 0.01],  # suppressed by NMS
            [80.0, 90.0, 22.0, 22.0, 0.01, 0.88, 0.02],
            [200.0, 200.0, 12.0, 12.0, 0.10, 0.10, 0.10],  # below conf_thres
        ]],
        dtype=np.float32,
    )
    raw_c_first = np.transpose(raw_n_first, (0, 2, 1))  # (1, 7, 4) YOLOv8/v11 format

    for raw in (raw_c_first, raw_n_first):
        dets = yolo_onnx.decode_yolo_output(raw, (304, 320), ratio, pad, conf_thres=0.35, iou_thres=0.45)
        assert len(dets) == 2
        assert dets[0]["class_name"] == "agent" and abs(dets[0]["x"] - 0.5) < 0.02 and abs(dets[0]["y"] - 0.5) < 0.02
        assert dets[1]["class_name"] == "smoke_dome"


def test_detect_minimap_objects_filters_transparent_void_and_fallbacks():
    H, W = 304, 320
    img = np.full((H, W, 3), 80, dtype=np.uint8)
    art_alpha = np.zeros((512, 512), dtype=np.uint8)
    art_alpha[64:448, 64:448] = 255
    art_alpha[210:302, 210:302] = 0  # 中央の透過壁穴
    T_art = {"ok": True, "mirror": False, "theta": 0.0, "scale": 1.0, "tx": 0.5 * W / H, "ty": 0.5}

    # Mock ONNX session outputting 1 smoke on floor (100,100) and 1 smoke in transparent hole (160,152)
    padded, ratio, (px, py) = yolo_onnx.letterbox(img, new_shape=320)
    raw = np.array(
        [[
            [100 * ratio + px, 100 * ratio + py, 20 * ratio, 20 * ratio, 0.02, 0.90, 0.01],
            [160 * ratio + px, 152 * ratio + py, 20 * ratio, 20 * ratio, 0.02, 0.95, 0.01],
        ]],
        dtype=np.float32,
    )
    fake_sess = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images")],
        run=lambda _out, _feed: [np.transpose(raw, (0, 2, 1))],
    )

    res = yolo_onnx.detect_minimap_objects(img, session=fake_sess, art_alpha=art_alpha, T_art=T_art)
    assert res is not None
    assert len(res["smokes"]) == 1
    assert abs(res["smokes"][0]["x"] - 100 / W) < 0.03

    # Graceful fallback when model does not exist or input is empty
    assert yolo_onnx.detect_minimap_objects(None, session=fake_sess) is None
    assert yolo_onnx.detect_minimap_objects(img, model_path="/nonexistent/model.onnx") is None


def test_train_yolo_validate_and_expand_dataset_from_gif(tmp_path):
    # 1) validate_data_yaml
    dy = tmp_path / "data.yaml"
    dy.write_text("train: images/train\nval: images/val\nnames:\n  0: agent\n", "utf-8")
    assert train_yolo_onnx.validate_data_yaml(dy)["train"] == "images/train"

    # 2) expand_dataset end-to-end on synthetic GIF
    gif_path = tmp_path / "r01_minimap.gif"
    pil_frames = []
    for k in range(4):
        arr = np.full((200, 200, 3), 35, dtype=np.uint8)
        cv2.circle(arr, (60 + k * 5, 80), 4, (255, 0, 0), -1)  # RGB red dot
        pil_frames.append(Image.fromarray(arr))
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=200, loop=0)

    out_ds = tmp_path / "expanded"
    rep = expand_dataset.expand_from_gifs([gif_path], out_ds, step_frames=1, max_frames_per_gif=4)
    assert rep["frames"] == 4
    assert Path(rep["data_yaml"]).is_file()
    assert Path(rep["review_html"]).is_file()
    assert len(list((out_ds / "images" / "train").glob("*.png"))) == 4
    assert len(list((out_ds / "labels" / "train").glob("*.txt"))) == 4
