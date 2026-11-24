# タスク指示書：ミニマップ領域判定とプレイ可能床マスクによる誤検知抑制

- **タスクID / ブランチ**: `task/playable-minimap-region`
- **作成日**: 2026-11-24
- **参照すべき `PROMPT.md` の節**: `## 大量データ（2024〜2026）とプレイブック`（2. ミニマップの位置合わせ / 3. サイト・位置）、`## ミニマップの配置とキルの正確さ`

## 背景・目的
ブラインド検証データセット（`/tmp/blind_val` 52 枚）および実戦放送の切り抜きにおいて、`detect_dots`（`vreplay/tactics.py`）と `detect_smokes`（`vreplay/smokes.py`）が以下の原因でマップ外背景・壁内部・UI・非ミニマップ画面を大量に誤検知していた：
1. `MM.map_mask()` が非ミニマップ画面（スコアボード・キルフィード・矩形 UI）を弾けず、かつ `sq` の HSV 範囲（`H=10..35, S>=35`）が広すぎて茶・橙の 3D 背景で漏れたり `None` になったりしていた。
2. `detect_dots()` が切り抜き全体から面積順に `[:per_color]` を取り出した後でしか領域フィルタを行わないため、マップ外の大きな色塊が真の選手アイコンを押し出し、斜線・バリア帯などの細長い形状も拾っていた。
3. `smokes.detect_smokes()` にプレイ可能床マスク・ミニマップ存在検証がなく、`min_radius_px=10` の固定値により低〜中解像度（`H=197..304`）のドームスモークを見逃し、壁穴の 3D 背景変化や単一フレーム時の `medianBlur` アーティファクトを誤検知していた。

本タスクでは、ミニマップ存在判定・プレイ可能領域（床マスク・ドット許容マスク）を `vreplay/minimap.py` に導入し、`tactics.py` / `smokes.py` / `tools/eval_all.py` に統合することで、マップ外・壁内部・HUD 文字・非ミニマップ画面からの誤検知を根本的に抑止し、プレビュー HTML 出力も整備する。

## 触ってよいファイル
- `docs/tasks/playable-minimap-region.md`（本指示書）
- `PROMPT.md`
- `vreplay/minimap.py`
- `vreplay/tactics.py`
- `vreplay/smokes.py`
- `tools/eval_all.py`
- `tests/test_minimap.py`
- `tests/test_tactics.py`
- `tests/test_smokes.py`
- `tests/test_eval_all.py`
- `tests/test_tactival_barriers.py`

※ 座標変換の不変条件（`valorant-api` の x/y 入れ替え、`CALLOUT_FIX["sunset"]` / `CALLOUT_OVERRIDE["sunset"]["B Boba"]`、TactiVal `MAP_ROT` と `(448, 28)` 1024×1024）は変更しない。`tests/fixtures/` 画像および `config/kb/` YAML は変更しない。

## 変更内容
1. **`vreplay/minimap.py`**
   - `map_mask(bg)` の `sq` をサイト四角相当の `H=20..34, S=35..80, V>=120` に絞り、中立床候補の近傍にのみ接続するとともに、k-means 中立クラスタにも `hsv[..., 1] <= 110` を課して単一フレーム高彩度外縁点の混入を防ぐ。
   - `_frame_matches_mask(mask, line, min_f=0.30, max_line_frac=0.048)` を追加：最小白縁画素数・最大白縁率・`_edge_f` 調和平均で高密度文字の暗転カットアウェイを弾く。
   - `detect_minimap_region(img, mask=None, edge_min_f=0.30) -> tuple[bool, np.ndarray | None]` を追加：厳格な 2D `mask` 検証、アスペクト比・サイズ・床被覆率・白縁画素数/上限率・`_edge_f`・枠端接触率・多角形頂点数/`iso`・閉ループ壁穴構造・マスク外孤立白縁率・内部輝度/色均質性により、実ミニマップを受け入れ、UI 切り抜き・キルフィード・ギザギザ UI・黒画面・一様画像・ノイズを拒否する。
   - `playable_region(img_or_bg, *, mask=None, art_alpha=None, T_art=None, tol_px=None) -> tuple[np.ndarray | None, np.ndarray | None]` を追加：厳格な 2D `mask` 検証、主床以外の左上隅オフセット HUD 断片除外、小ドット穴補完、4-連結（`MORPH_CROSS`）＋白縁制約の湾曲穴補完、および壁境界許容 `dot_mask`（既定 `max(2, round(0.015 * H))`＋白縁近傍帯）を返す。
2. **`vreplay/tactics.py`**
   - `detect_dots(..., mask=None, require_minimap=False, edge_tol_px=None)` に拡張：厳格な 2D `mask` 検証、`require_minimap=True` の不在拒否、`dot_mask` による `[:per_color]` 前段フィルタ、回転外接矩形（`minAreaRect`）と選択的 `MORPH_OPEN` による斜め・連結視線の分離除去、`0.38*D / 0.09*D^2` 最小閾値とコンパクト度重み付きソートを実装。`mask=None, require_minimap=False` 時は従来の `np.zeros` 合成テスト互換を維持。
   - `analyze_video()`：背景サンプル間隔カウンタ（`n_seen`）修正、カットアウェイ優勢時の有効ミニマップ再構築と全カットアウェイ時の `None` 返却（黒キャンバス単体テスト互換維持）、`_frame_matches_mask` によるフレーム／節目スモークバッファ検証、`SM.detect_smokes(..., mask=mask, icon_frac=icon_frac)` 連携を実装。
3. **`vreplay/smokes.py`**
   - `detect_smokes(..., mask=None, icon_frac=None, require_minimap=False, found_agents=None)` に拡張：厳格な 2D `mask` 検証、`require_minimap=True` かつ `bg_bgr` 指定時の `frame_bgr` 白縁検証（`_frame_matches_mask`）、`floor_inner` 限定、高さ相対半径（明示 `max_radius_px` 尊重）、縁除去 `deep_mask`＋平滑ドーム/Harbor シアン免除、`floor_med` 相対の暗化閾値、壁穴・エージェントアイコン抑止を統合。
4. **`tools/eval_all.py`**
   - `_frame_matches_mask` によるカットアウェイ除外、`found_agents=dots` 連携、`smoke_gt_in_floor_mask_rate` / `smoke_gt_in_floor_inner_rate` 追加、`--preview-html` の `</` エスケープ対応を実装。
5. **`tests/` & `PROMPT.md`**
   - `tests/test_tactival_barriers.py::test_opening_masks_sunset_fixture` を `CALLOUT_FIX["sunset"]` + `CALLOUT_OVERRIDE["sunset"]` + `MM.register_art` に更新（厳格アサーションは維持）。
   - `tests/test_minimap.py`, `tests/test_tactics.py`, `tests/test_smokes.py`, `tests/test_eval_all.py` に回帰テストを追加し、`PROMPT.md` の該当節を同期。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_minimap.py tests/test_tactics.py tests/test_smokes.py tests/test_eval_all.py tests/test_tactival.py tests/test_tactival_barriers.py tests/test_track_walls.py tests/test_barriers.py tests/test_opening_playbook.py
python3 tools/eval_all.py --blind-dir /tmp/blind_val --tol 0.06
```

## 受け入れ条件
- [x] 指定したテストがすべてネットワークなしで通る
- [x] 空入力・部分失敗時に例外で停止しない
- [x] `/tmp/blind_val` でベースラインから誤検知抑制・F1 改善を達成する（注記：`dot_mask` の壁穴漏れ抑止のため `tol=0.015*H` へ厳格化した結果、`/tmp/blind_val` の `gt_in_mask_rate` は `0.9279`、`smoke_gt_in_floor_mask_rate` は `0.6667 -> 0.7593`）
- [x] `blind_check_preview.html` が `--preview-html` により更新される
- [x] 実装と別モデルのレビューで「マージしてよい」判定を取得済み

## 質問・ブロッカー / 相談票（3周で解決しない・仕様判断が必要な場合）
- なし（Round 1 でブロッカーだった `tests/test_tactival_barriers.py::test_opening_masks_sunset_fixture` は本指示書の対象ファイルへ追加のうえ `CALLOUT_FIX["sunset"]` + `CALLOUT_OVERRIDE["sunset"]` + `MM.register_art` で修正済み）。

## レビュー結果・修正履歴
- **Round 1（`9eef040`）**: Reviewer 1/2・Challenger 1/2 より `REQUEST_CHANGES` / `FAIL`（UI 切り抜き誤通過、高密度文字の片側 `_edge_agreement` 通過、オフセット HUD 漏れ、`dot_mask` `0.028*H` の壁穴漏れ、斜め/連結視線、`analyze_video` の `mask=None`/バッファ/ストライド漏れ、`smokes.py` の 3D/不一致マスク・`frame_bgr` 未検証・縁付き/Harbor シアン・`diff=16`・`dark_max_r` 上書き、`test_opening_masks_sunset_fixture` 失敗）。
- **Round 2**: 上記すべてに対する一般化修正とオフライン回帰テストを実装・検証。レビューサブエージェント判定：**マージしてよい**（必ず直すこと：なし）。

## 完了レポート
- **変えたファイル**:
  - `docs/tasks/playable-minimap-region.md`, `PROMPT.md`
  - `vreplay/minimap.py`, `vreplay/tactics.py`, `vreplay/smokes.py`, `tools/eval_all.py`
  - `tests/test_minimap.py`, `tests/test_tactics.py`, `tests/test_smokes.py`, `tests/test_eval_all.py`, `tests/test_tactival_barriers.py`
- **テスト結果**:
  - 対象スイート（`test_minimap.py`, `test_tactics.py`, `test_smokes.py`, `test_eval_all.py`, `test_tactival.py`, `test_tactival_barriers.py`, `test_track_walls.py`, `test_barriers.py`, `test_opening_playbook.py`）：**`96 passed`（0 failed）**。
  - オフライン全スイート（`rapidocr_onnxruntime` / `ffmpeg` 依存の `test_loadout.py`, `test_rounds.py`, `test_riot_scores_not_mixed_with_seed`, `test_killfeed_reads_kills_with_real_ocr` 除外）：**`490 passed, 5 skipped, 2 deselected`**。
- **ブラインド評価 (`/tmp/blind_val`, 52 枚, `tol=0.06`)**:
  - `mask_ok_frames=52/52`, `gt_in_mask_rate=0.9279` (`agent=0.9410`, `smoke=0.8333`, `smoke_floor=0.7593`, `smoke_inner=0.7407`), `fp_outside_mask_filtered=115`
  - `detect_dots`: `TP=239, FP=174, FN=151, P=0.5787, R=0.6128, F1=0.5953`（ベースライン `F1=0.538` → `+0.057`）
  - `detect_smokes`: `TP=27, FP=34, FN=27, P=0.4426, R=0.5000, F1=0.4696`（ベースライン `F1=0.037` → `+0.433`）

