# タスク指示書：軽量 YOLO ONNX 3クラス推論・Colab GPU 学習・半自動データ拡充パイプライン

- **タスクID / ブランチ**: `task/yolo-onnx-pipeline`
- **作成日**: 2026-11-27
- **参照すべき `PROMPT.md` の節**: `## ミニマップの配置とキルの正確さ`

## 背景・目的
現行の純 NumPy 小型 CNN（`agent_dots.py` / `smoke_detector.py`）は未知データで Agent F1=0.935 / Smoke F1=0.740 を達成したが、95〜98% クラスへの引き上げと推論高速化（150ms → 10〜20ms/frame）には、
1. `agent` / `smoke_dome` / `ability_other` の 3 クラスを同時検出する軽量 YOLOv8n/v11n ONNX 推論基盤（既存の `onnxruntime` を活用、未配置時は安全に `None` フォールバック）
2. Google Colab T4 GPU で学習・ONNX 書き出しを行うスクリプト（`tools/train_yolo_onnx.py`）
3. Google Drive の最新試合ミニマップ GIF から仮アノテーション付きデータセット＋目視チェック HTML を自動生成する拡充ツール（`tools/expand_dataset.py`）
が必要となる。

## 触ってよいファイル
- `docs/tasks/yolo-onnx-pipeline.md`
- `vreplay/yolo_onnx.py`（新規）
- `tools/train_yolo_onnx.py`（新規）
- `tools/expand_dataset.py`（新規）
- `tests/test_yolo_onnx.py`（新規）

## 変更内容
1. `vreplay/yolo_onnx.py`: ONNX Runtime による 3 クラス同時推論・レターボックス・YOLOv8/v11 出力デコード＆NMS・`playable_region` 透過領域フィルタ・安全な `None` フォールバック。
2. `tools/train_yolo_onnx.py`: Colab GPU での YOLOv8n/v11n 学習および `vreplay/data/minimap_yolo.onnx` エクスポート CLI。
3. `tools/expand_dataset.py`: 実戦ミニマップ GIF 群からフレーム抽出 → 現行モデルで仮 YOLO ラベル生成 → 不確実度順の目視確認 HTML 出力を行う CLI。
4. `tests/test_yolo_onnx.py`: 上記すべてをネットワーク・GPU・`onnxruntime` 不要で検証するオフライン単体テスト。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_yolo_onnx.py
```

## 受け入れ条件
- [x] `tests/test_yolo_onnx.py` がネットワーク・`onnxruntime` なしで全件通る
- [x] `onnxruntime` 不在・`.onnx` モデル不在時に例外で落ちず `None` を返す
