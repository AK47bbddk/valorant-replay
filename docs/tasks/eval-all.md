# タスク指示書：統合ベンチマーク・ブラインド評価ツール (`tools/eval_all.py`) の整備

- **タスクID / ブランチ**: `task/eval-all`
- **作成日**: 2026-11-24
- **参照すべき `PROMPT.md` の節**: `## 大量データ（2024〜2026）とプレイブック`

## 背景・目的
Google Drive 上のブラインド検証データセット（`dataset_v3_allmaps/val` 等、YOLO 形式の画像＋ラベル）およびリポジトリ内の既存テストを使って、エージェント点検出・ドームスモーク検知の精度（Precision / Recall / F1 / 中心距離誤差）を 1 コマンドで定量計測できる `tools/eval_all.py` を整備する。
これにより、今後の改善 PR ごとに「Before → After」の客観スコアと誤検知・見逃し事例を PR 本文へ自動記載できる基盤を作る。

## 触ってよいファイル
- `docs/tasks/eval-all.md`（本指示書）
- `tools/eval_all.py`（新規作成）
- `tests/test_eval_all.py`（新規作成）

## 変更内容
1. **`tools/eval_all.py` の新規作成**
   - CLI 引数:
     - `--blind-dir PATH`: YOLO 形式のブラインドデータセット（`images/val` と `labels/val`、または直下の画像・`.txt` ペア）のディレクトリ
     - `--tol FLOAT`: 中心距離の許容誤差（正規化座標 0..1、既定 `0.06`）
     - `--json PATH`: 詳細結果 JSON の出力先（任意）
     - `--md PATH`: PR 貼り付け用 Markdown サマリーの出力先（任意）
     - `--limit INT`: 評価枚数の上限（0 = 全件）
   - 評価対象:
     - **エージェント点検出**（クラス `0: agent`）: `vreplay.tactics.detect_dots` による検出結果と GT ボックス中心のマッチング（貪欲法、距離 `<= tol`）。Precision / Recall / F1 / 平均距離誤差を算出。
     - **ドームスモーク検知**（クラス `1: smoke_dome`）: `vreplay.smokes` が利用可能な場合、同一解像度の画像群からサイズ別・マップ別の近似背景（中央値合成、1 枚しかない場合はブラー近似）を生成して `smokes.detect_smokes` を実行し、Precision / Recall / F1 を算出（利用不可時は `skipped` 扱い）。
   - 例外・欠損データ（空ディレクトリ・ラベル欠損・壊れた画像）に対してクラッシュせず安全に処理し、0 件時はゼロ除算せず `0.0` を返す。
2. **`tests/test_eval_all.py` の新規作成**
   - 一時ディレクトリに合成ミニマップ画像と YOLO ラベルを作成し、`eval_all.evaluate_blind_dir(...)` がネットワークなしで Precision / Recall / F1 と Markdown / JSON 出力を正しく生成できることを検証する。
   - 空ディレクトリ・不正パスでも例外で落ちないことを検証する。

## 通すテスト
```powershell
python -m pytest -x -q tests/test_eval_all.py
```

## 受け入れ条件
- [ ] `tests/test_eval_all.py` がネットワークなしで通る
- [ ] `dataset_v3_allmaps/val`（52 枚）に対して `python tools/eval_all.py --blind-dir /tmp/blind_val` が完走し、ベースラインスコアが出力される
- [ ] 実装と別視点（レビューサブエージェント）で「マージしてよい」判定を取得済み

## 質問・ブロッカー / 相談票（3周で解決しない・仕様判断が必要な場合）
<!-- なし -->

## レビュー結果・修正履歴
<!-- Round 1〜3 の判定と対応内容を記録する -->

## 完了レポート
<!-- 変えたファイル / テスト結果 / 迷ったところ・確認してほしい点 -->
