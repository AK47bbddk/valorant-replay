"""Google Colab (T4 GPU) でミニマップ 3 クラス検出 YOLO を学習し ONNX へ書き出すスクリプト。

Colab での使い方:
  !pip -q install ultralytics onnx onnxruntime-gpu
  !python tools/train_yolo_onnx.py \\
      --data /content/drive/MyDrive/valorant-replay/_yolo_poc/dataset_v3_allmaps/data.yaml \\
      --model yolo11n.pt --epochs 120 --imgsz 320 \\
      --out /content/valorant-replay/vreplay/data/minimap_yolo.onnx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def validate_data_yaml(data_yaml: Path) -> dict:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml が見つかりません: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text("utf-8")) or {}
    if not isinstance(cfg, dict) or "train" not in cfg or "val" not in cfg:
        raise ValueError(f"data.yaml に train / val の定義が必要です: {data_yaml}")
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ミニマップ 3 クラス YOLO 学習＆ONNX エクスポート")
    ap.add_argument("--data", required=True, help="YOLO data.yaml のパス")
    ap.add_argument("--model", default="yolo11n.pt", help="ベースモデル (例: yolo11n.pt, yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=120, help="エポック数")
    ap.add_argument("--imgsz", type=int, default=640, help="入力解像度（小アイコン向けに 640 推奨）")
    ap.add_argument("--batch", type=int, default=16, help="バッチサイズ")
    ap.add_argument("--device", default="0", help="学習デバイス (0=GPU, cpu=CPU)")
    ap.add_argument("--project", default="_work/yolo_runs", help="学習ログ出力先")
    ap.add_argument("--name", default="minimap_3cls", help="実験名")
    ap.add_argument(
        "--out",
        default="vreplay/data/minimap_yolo.onnx",
        help="エクスポート先 ONNX ファイルパス",
    )
    args = ap.parse_args(argv)

    data_yaml = Path(args.data)
    validate_data_yaml(data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics がインストールされていません。Colab 等で "
            "`pip install ultralytics onnx` を実行してから再実行してください。",
            file=sys.stderr,
        )
        return 2

    project_dir = str(Path(args.project).resolve())
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project_dir,
        name=args.name,
        # ミニマップでは上下反転・回転・小スケール劣化に強くする
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.25,
    )

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(project_dir) / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        print(f"best.pt が見つかりません: {best_pt}", file=sys.stderr)
        return 1

    best = YOLO(str(best_pt))
    metrics = best.val(data=str(data_yaml), imgsz=args.imgsz)
    print(f"Validation mAP50: {getattr(metrics.box, 'map50', 0.0):.4f}")

    exported = Path(best.export(format="onnx", imgsz=args.imgsz, opset=13, simplify=True))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out_path)
    print(f"Exported ONNX model to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
