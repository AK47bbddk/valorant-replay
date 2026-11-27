# タスク指示書：non-playable-mask-hardening（透過＝非プレイ可能領域マスクの強化・全 13 マップ公式アルファ同梱）

- **タスクID / ブランチ**: `task/non-playable-mask-hardening`（ベース: `task/agent-dot-accuracy`）
- **作成日**: 2026-11-27
- **参照すべき `PROMPT.md` の節**: `## 映像からミニマップ位置を読み取る` / `## プレイ可能床マスク`

## 背景・目的
TactiVal 等の透過ミニマップにおいて透過領域（`alpha == 0`）はエージェントやスキルが存在できない非プレイ可能領域（マップ外ボイド・壁・柱）であり、これは `valorant-api.com` 公式 `displayIcon` のアルファチャンネルと 99.997% 一致する。
従来のヒューリスティック `map_mask` + `playable_region` には以下の課題があった：
1. 暗いドームスモークが床セグメンテーションに穴を開けてしまい、スモーク GT の約 24% を `floor_mask` 外に落としていた。
2. `art_alpha` + `T_art` が与えられても `playable_region` がヒューリスティックな穴埋め・`MORPH_CLOSE` 湾曲補完・`_outline` バンド拡張を実行していたため、実在する壁・柱穴が埋まったり白線付近の透過ボイドに `dot_mask` が漏れていた。
3. `register_art` の初期推定が `_outline` パーセンタイルのみだったため、HUD テキスト等のノイズがあると一部ミニマップで局所解に陥っていた。
4. オフライン環境（テストやローカルキャッシュ無し）では `displayIcon` アルファを参照できなかった。

本タスクでは全 13 競技マップの 512×512 2 値アルファマスク（~18 KB）をオフライン同梱し、`register_art` の外接矩形初期化強化＋高速 2 段階 `register_art_auto`、`playable_region` の `art_alpha` 厳密ファストパス、およびパイプライン全体（`tactics.py` / `smokes.py` / `eval_all.py`）への伝搬を実装する。

## 触ってよいファイル
- `tools/build_map_alphas.py`（新規）
- `vreplay/data/map_alphas.npz`（新規：全 13 競技マップの 512×512 パック済み 2 値アルファマスク）
- `vreplay/minimap.py`
- `vreplay/tactics.py`
- `vreplay/smokes.py`
- `tools/eval_all.py`
- `tests/test_minimap.py`
- `tests/test_tactics.py`
- `tests/test_smokes.py`
- `docs/tasks/non-playable-mask-hardening.md`（新規）
- `docs/BACKLOG.md`
- `PROMPT.md`（仕様追記が必要な場合のみ）

※ `tests/fixtures/` 画像・`config/kb/` YAML は変更禁止。`/tmp/blind_val` は評価専用とし、閾値チューニングを行わないこと。

## 変更内容
1. **オフライン 13 マップ公式アルファパック (`tools/build_map_alphas.py` + `vreplay/data/map_alphas.npz` + `vreplay/minimap.py`)**:
   - 競技 13 マップ（`abyss, ascent, bind, breeze, corrode, fracture, haven, icebox, lotus, pearl, split, summit, sunset`）の `displayIcon` アルファを 512×512 (`alpha > 128`) に変換し `np.packbits(..., axis=1)` + `np.savez_compressed` で `vreplay/data/map_alphas.npz` に保存する。
   - `vreplay/minimap.py` にメモリキャッシュ付き・100% オフラインの `load_bundled_alpha(map_name)` と `load_all_bundled_alphas()` を追加する。
2. **`register_art` 堅牢化 & 高速 2 段階 `register_art_auto` (`vreplay/minimap.py`)**:
   - `register_art(bg, alpha, T0=None, mask_hint=None)` に `mask_hint`（未指定時は `map_mask(bg)`）外接矩形からの 4 回転初期化を追加し、座標軸別降下＋粗段階サブサンプリングで高速化する。
   - `register_art_auto(bg, map_name=None, mask_hint=None)` を追加する（`map_name` 既知時は直接 `register_art`、未知時は全 13 マップの粗スコア→上位 `top_k` 精密探索の 2 段階実行）。
3. **`playable_region` の `art_alpha` 厳密ファストパス (`vreplay/minimap.py`)**:
   - 有効な `art_alpha` + `T_art` (`T_art.get("ok")`) が与えられたときは、HUD CC 除去・穴埋め・`MORPH_CLOSE` 湾曲補完・`_outline` バンド拡張をスキップし、`tol_px is None` 時は幾何由来のタイトな `tol = max(2, int(round(0.015 * H)))` で `dot_mask` を生成する。
   - `art_alpha` / `T_art` が無い場合は既存のヒューリスティック動作を完全維持する。
4. **パイプライン全体への `art_alpha` / `T_art` 伝搬 (`vreplay/tactics.py`, `vreplay/smokes.py`, `tools/eval_all.py`)**:
   - `detect_dots` / `detect_smokes` に省略可能引数 `art_alpha=None, T_art=None` を追加して `MM.playable_region` へ渡す。
   - `analyze_video` で `map_art()` が `None` のときに `MM.load_bundled_alpha(...)` へフォールバックし、`playable_region` および `SM.detect_smokes` へ `art_alpha` + `T_art` を渡す。
   - `tools/eval_all.py` で参照画像単位に `MM.register_art_auto(ref, mask_hint=raw_m)` を実行し、`ok=True` 時は `art_alpha` + `T_art` を `playable_region` / `detect_dots` / `detect_smokes` に渡す（`mask_metrics["art_reg_ok_frames"]` を追記）。
5. **オフライン単体テスト (`tests/test_minimap.py`, `tests/test_tactics.py`, `tests/test_smokes.py`)**:
   - 全 13 マップ同梱アルファ読み込み、`playable_region` の壁穴保持・タイト `dot_mask`、`register_art` / `register_art_auto`、`detect_dots` / `detect_smokes` の透過壁穴除外テストを追加する。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_minimap.py tests/test_tactics.py tests/test_smokes.py tests/test_track.py tests/test_eval_all.py
python3 -m pytest -q --ignore=tests/test_loadout.py --ignore=tests/test_rounds.py -k "not test_riot_scores_not_mixed_with_seed and not test_killfeed_reads_kills_with_real_ocr"
python3 tools/eval_all.py --blind-dir /tmp/blind_val --tol 0.06
```

## 受け入れ条件
- [x] 全 13 マップ同梱アルファ (`vreplay/data/map_alphas.npz`, ~18 KB) がオフラインで読み込める
- [x] `/tmp/blind_val` で `register_art_auto` が 52/52 (100%)・高速（< 15s）で成功する
- [x] `playable_region` の `art_alpha` 厳密ファストパスが壁穴・透過ボイドを保持し、`gt_in_mask_rate = 1.0000` (agent 100%, smoke 100%) を達成する
- [x] オフライン `pytest` 全件通過（既存テストの削除・緩和なし）
- [x] `docs/BACKLOG.md` を更新し、`task/non-playable-mask-hardening` ブランチへコミット済み
