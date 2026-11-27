# タスク指示書：定半径ドームスモーク検知器＋クリーンミニマップ構造事前知識 (`smoke-detection-accuracy`)

- **タスクID / ブランチ**: `task/smoke-detection-accuracy`
- **作成日**: 2026-11-25
- **参照すべき `PROMPT.md` の節**: ミニマップ解析・スモーク検知関連節

## 背景・目的
既存の `vreplay/smokes.py` は二値差分・輝度閾値＋輪郭/連結成分フィルタに依存しており、
1. 暗い壁柱・矩形部屋や小半径アビリティ (`ability_other`) を多数誤検知する（ブラインド検証ベースライン `P=0.443, R=0.500, F1=0.470`, FP=34）
2. 白壁線やエージェントアイコンが重なったスモークで輪郭が分断され取りこぼす
という課題があった。

全てのコントローラーでミニマップ上のドームスモーク半径がほぼ一定（正規化高さ `H=304` で `R ≈ 9.0 px`, `R/H ≈ 0.0296`）であること、およびクリーンミニマップの構造事前知識（プレイ可能床は一様な中間灰、壁柱は白枠で囲まれた静的暗領域）を活用し、純粋 NumPy/OpenCV（新規依存ゼロ）の 2 段階定半径学習検知器を導入してスモーク検知精度を大幅に改善する。

## 触ってよいファイル
- `vreplay/smoke_detector.py`（新規）
- `vreplay/data/smoke_model.npz`（新規、`< 300 KB`）
- `tools/train_smoke_detector.py`（新規）
- `vreplay/smokes.py`
- `tests/test_smokes.py`（既存テスト無改変・追記のみ）
- `docs/tasks/smoke-detection-accuracy.md`（新規）
- `docs/BACKLOG.md`

## 変更内容
1. **`vreplay/smoke_detector.py`**:
   - 正規化スケール `H_CANON = 304`, `R_SMOKE = 9.0` での 2 段階検知パイプラインを実装（純粋 NumPy/OpenCV）。
   - Stage 1: `dilate(floor_mask, 5x5)` 内で固定半径 `R=9.0 px` の `cv2.filter2D` 応答マップ（外向き放射勾配 `KX9/KY9`、内部ディスク `DISC9`、外周リング `OUT9`、有効 `bg` 時の `bg_gray - gray`）を用いた stride-3 密提案生成。
   - Stage 2A: ベクトル化 BLAS による同心リング・マルチ半径（`R=5.5..13.0`）・回転不変ソート 8 セクタ特徴のデュアルヘッド MLP（単一フレーム 74 次元 / BG 付き 88 次元）＋粗密 `±2 px` ピーク精緻化。
   - Stage 2B: 上位候補に対する `20x20` パッチ（6ch SF / 8ch BG）の純粋 NumPy 3 層 2D CNN アンサンブル。Stage 2A/2B ロジット統合・NMS（`min_sep=11.5 px`）・ソフトなエージェント重なりガード。
   - `load_model()` は `@lru_cache` かつ `allow_pickle=False` で読み込み、欠損・破損 `.npz` 時は例外を投げず `None` を返す。
2. **`tools/train_smoke_detector.py`**:
   - `/tmp/blind_train`（106 枚・110 GT スモーク・7 マップ）のみを使用した 7-fold Leave-One-Map-Out CV、8 倍 D4 対称拡張（`(gx, gy)` ベクトル変換込み）、`±1 px` シフト、低解像度ブラー拡張（正負ペア）、`ability_other` / 静的 `bg_med` ハードネガティブ重視、2 ラウンドのハードネガティブマイニングを実装し、float16 圧縮で `vreplay/data/smoke_model.npz`（118.7 KB）を出力。
3. **`vreplay/smokes.py`**:
   - 既存シグネチャ・入力検証・戻り値スキーマを完全維持。
   - `mask` 指定かつ有効 `bg_bgr` 時に `playable_region` が暗い壁柱を埋め戻さないよう保護。
   - レガシーフォールバックの `cand_static` を `bg_is_distinct` でゲート（焼き込み `bg=frame.copy()` は維持）。
   - 実ミニマップ＋デフォルト半径時は `smoke_detector.detect_smokes_learned` にルーティングし、明色カラー付きドーム補完および合成テスト・カスタム半径・モデル未ロード時のフォールバックを維持。
4. **`tests/test_smokes.py`**:
   - 既存 9 テストを一切変更せず、`load_model` 耐性（欠損・破損・不完全 `.npz`）、決定性・スキーマ、非スモーク形状拒否、モデル未ロード時フォールバックの 4 テストを追加。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_minimap.py tests/test_tactics.py tests/test_smokes.py tests/test_eval_all.py -k "not test_fit_mask_reads_site_letters_on_symmetric_map"
```

## 受け入れ条件
- [x] 指定したテストがすべてネットワークなしで通る（`45 passed, 1 deselected`）
- [x] 空入力・欠損/破損 `.npz`・部分失敗時に例外で停止しない
- [x] 学習・チューニングは `/tmp/blind_train` の 7-fold LOMO-CV のみで完結し、`/tmp/blind_val` リークがない
- [ ] 実装と別モデルのレビューで「マージしてよい」判定を取得済み

## 質問・ブロッカー / 相談票
なし

## レビュー結果・修正履歴
- Iteration 1 実装・ローカルコミット完了（M4 Verification Wave 待ち）

## 完了レポート
- **変えたファイル**: `vreplay/smoke_detector.py`, `vreplay/data/smoke_model.npz` (118.7 KB), `tools/train_smoke_detector.py`, `vreplay/smokes.py`, `tests/test_smokes.py`, `docs/tasks/smoke-detection-accuracy.md`, `docs/BACKLOG.md`
- **テスト結果**:
  - オフライン `pytest`: `45 passed, 1 deselected`
  - 7-fold LOMO-CV (`/tmp/blind_train`): Dual `P=0.787, R=0.773, F1=0.780` / Single-Frame `P=0.726, R=0.745, F1=0.735`
  - ブラインド評価 (`/tmp/blind_val`, `tol=0.06`, 1 回限り): `TP=27, FP=11, FN=27, P=0.711, R=0.500, F1=0.587`（ベースライン `P=0.443, F1=0.470` から FP 67.6% 削減・Precision +26.8pt・F1 +11.7pt）
- **迷ったところ・確認してほしい点**: `main` 側 `minimap.py` の `smoke_floor=0.7593`（54 中 13 スモークがマスク外）が `/tmp/blind_val` での Recall 上限になっている点、および 2-CPU 環境での推論速度（`detect_smokes_learned` `124.5 ms/frame` @ `OPENBLAS_NUM_THREADS=1`）。
