# タスク指示書：temporal-dot-correction（1R @ 2秒間隔の ±2 秒時系列補正によるエージェント点・スモーク補完）

- **タスクID / ブランチ**: `task/temporal-dot-correction`（ベース: `task/non-playable-mask-hardening`）
- **作成日**: 2026-11-27
- **参照すべき `PROMPT.md` の節**: `## 映像からミニマップ位置を読み取る`

## 背景・目的
単体フレームのミニマップ検出（`detect_dots` / `detect_smokes`）は高精度になったものの、以下の課題が残っている：
1. 味方同士・敵味方がミニマップ上で重なる瞬間やスキル・スモーク等で一瞬隠れた瞬間に 1 フレームだけエージェント点が欠損する。
2. 逆に低スコアの孤立 1 フレーム誤検知（特にドームスモークの単発 FP）が混入する。
3. 既存の `vreplay/track.py` は顔アイコン照合済みの *名前付きエージェント* 向けであり、無名のチーム色ドット列やスモーク列に対する ±2 秒時系列補正が存在しない。

本タスクでは、`scipy` 等の新規依存を追加せず純 Python + NumPy のみで厳密な O(N^3) ハンガリアン法 1 対 1 対応付けと ±2 秒時系列補正（重なり・欠損補完、1 パス低閾値候補スナップ、保守的な孤立 1 フレーム FP 抑止、カットアウェイ／プレイ可能マスク／生存数ガード）を実装し、`tactics.py` パイプライン統合と 1R 実戦 GIF @ 2 秒評価ツールを整備する。

## 触ってよいファイル
- `vreplay/temporal.py`（新規）
- `vreplay/agent_dots.py`（`"score"` 出力追加 + 1 パス `detect_agent_dots_with_low`）
- `vreplay/tactics.py`（`analyze_video` のラウンド別時系列補正フック）
- `tools/eval_temporal_gif.py`（新規：1R GIF @ 2 秒評価＆HTML ストリッププレビュー生成）
- `tests/test_temporal.py`（新規：オフライン単体テスト）
- `docs/tasks/temporal-dot-correction.md`（新規）
- `docs/BACKLOG.md`
- `PROMPT.md`（仕様追記が必要な場合のみ）

※ `tests/fixtures/` 画像・`config/kb/` YAML は変更禁止。新規 Python 依存パッケージ追加禁止。

## 変更内容
1. **1 パス低閾値候補＆スコア公開 (`vreplay/agent_dots.py`)**:
   - `assign_teams_and_cap` の各返却 dict に `"score"`（小数第 4 位丸め、既存キーと 100% 後方互換）を追加する。
   - `detect_agent_dots_with_low(img, ..., low_thresh=0.25)` を追加し、MLP+CNN を `min(eff_thresh, low_thresh)` で 1 回だけ推論して `(primary_dots, low_dots)` を追加推論コスト 0 で返す。
2. **純 Python + NumPy ±2 秒時系列補正モジュール (`vreplay/temporal.py`)**:
   - `min_cost_match(pts_a, pts_b, max_dist, aspect=1.0)`: 最大基数・最小総コストの厳密 O(N^3) ハンガリアン法（`scipy` 不使用）。
   - `correct_agent_dots_sequence(frames, ...)`: 色別（`warm`/`cool`）に、Pass 1（内部有効フレーム限定の保守的な低スコア孤立 1 フレーム FP 抑止）→ Pass 2（`prev↔cur` / `cur↔next` / `unmatched_prev↔unmatched_next` の ±2 秒補完＋`low_dots` スナップ＋`dot_mask` クランプ／除外＋`alive_counts` / `per_color` 上限＋`overlap_snap` / `overlap_interp` / `snap_low` / `interp` タグ付け）。
   - `correct_smokes_sequence(frames, ...)`: 孤立 1 フレーム FP 抑止（`conf < suppress_lone_conf`）＋1 フレーム欠損ブリッジ（`temporal="interp"`）。
   - `analyze_round_frames(frames_bgr, timestamps, ...)`: 中央値背景・`register_art_auto`・`detect_agent_dots_with_low`・`detect_smokes`・時系列補正を一括実行する 1R ヘルパー。
3. **パイプライン統合 (`vreplay/tactics.py`)**:
   - `analyze_video` で `tcfg.get("temporal_dots", True)` 有効時に `raw` をラウンド番号 `n` ごとにグループ化（ラウンド境界を跨がない）して `temporal.correct_agent_dots_sequence` を適用する。粗い 5 秒サンプリング時は `dt > max_dt (3.5s)` のため自動的に no-op となり、2 秒系サンプリング時のみ自動発動する。
4. **1R 実戦 GIF 評価・目視プレビュー (`tools/eval_temporal_gif.py`)**:
   - 1R ミニマップ GIF から `--step-sec 2.0` 間隔でフレームを抽出し、補正前後の集計表・JSON・左右並列比較 HTML ストリップを生成する。
5. **オフライン単体テスト (`tests/test_temporal.py`)**:
   - ハンガリアン最適性、味方／敵味方重なり補完、低閾値スナップ、`dot_mask` クランプ／除外、カットアウェイ保護、デス／テレポート安全性、境界フレーム保護、`alive_counts` 上限、不正入力耐性、スモーク補正、`detect_agent_dots_with_low`、`analyze_round_frames` を網羅する。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_temporal.py tests/test_minimap.py tests/test_tactics.py tests/test_smokes.py tests/test_track.py tests/test_eval_all.py
python3 tools/eval_all.py --blind-dir /tmp/blind_val --tol 0.06
python3 tools/eval_temporal_gif.py --gif <1R_GIF> --step-sec 2.0 --html /tmp/temporal_1r_preview.html
```

## 受け入れ条件
- [x] 新規依存なし（純 Python + NumPy）で厳密な O(N^3) ハンガリアン法と ±2 秒補正が動作する
- [x] オフライン `pytest` が全件通過する
- [x] `/tmp/blind_val --tol 0.06` で回帰なし（`mask_ok_frames=52/52`, `gt_in_mask_rate=1.0000`, `detect_dots F1>=0.935`, `detect_smokes F1>=0.740`）
- [x] Drive 実戦 1R GIF 3 本（`lotus_r01`, `summit_r01`, `sunset_r01`）@ 2 秒で補完効果と HTML ストリッププレビューを確認できる
