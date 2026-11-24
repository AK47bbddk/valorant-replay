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
- `vreplay/minimap.py`
- `vreplay/tactics.py`
- `vreplay/smokes.py`
- `tools/eval_all.py`
- `tests/test_minimap.py`
- `tests/test_tactics.py`
- `tests/test_smokes.py`
- `tests/test_eval_all.py`

※ 座標変換の不変条件（`valorant-api` の x/y 入れ替え、`CALLOUT_FIX["sunset"]` / `CALLOUT_OVERRIDE["sunset"]["B Boba"]`、TactiVal `MAP_ROT` と `(448, 28)` 1024×1024）は変更しない。`tests/fixtures/` 画像および `config/kb/` YAML は変更しない。

## 変更内容
1. **`vreplay/minimap.py`**
   - `map_mask(bg)` の `sq` をサイト四角相当の `H=20..34, S=35..80, V>=120` に絞り、中立床候補の近傍にのみ接続して 3D 背景漏れ・`None` 化を防ぐ。
   - `detect_minimap_region(img, mask=None, edge_min_f=0.30) -> tuple[bool, np.ndarray | None]` を追加：アスペクト比・サイズ・床被覆率・白縁画素数と一致度 `_edge_f`・枠端接触率・最大輪郭の多角形頂点数と等周比（`iso`）により、実ミニマップを受け入れ、UI パネル・キルフィード・黒画面・一様画像・ノイズ・単純矩形を拒否する。
   - `playable_region(img_or_bg, *, mask=None, art_alpha=None, T_art=None, tol_px=None) -> tuple[np.ndarray | None, np.ndarray | None]` を追加：マップ外余白・壁穴・左上 HUD テキスト隅の微小断片を除外した `(floor_mask, dot_mask)` を返す（`dot_mask` は壁際アイコン救済のため `tol_px` 既定 `max(2, round(0.028 * H))` で壁境界を許容）。
2. **`vreplay/tactics.py`**
   - `detect_dots(img, min_area=4, max_area=450, per_color=5, icon_frac=None, *, mask=None, require_minimap=False, edge_tol_px=None)` に拡張：
     - `require_minimap=True` の場合は `MM.detect_minimap_region` でミニマップ不在フレームに `[]` を返す。
     - `dot_mask` フィルタを面積ソート・`[:per_color]` 切り捨て**より前**に適用する。
     - 観戦視線・バリア線などの細長い形状（高アスペクト比／低充填率）を除外する。
     - `mask=None, require_minimap=False` 時は従来の `np.zeros` 合成テスト互換を維持する。
   - `analyze_video()`：フレーム蓄積時・スモーク検出時に `mask is not None` なら高速なフレーム毎の白縁一致チェック（`_edge_agreement(mask, MM._outline(mm)) >= 0.20`）を行い、非ミニマップカットを弾く。また `SM.detect_smokes(..., mask=mask, icon_frac=icon_frac)` に `mask` と `icon_frac` を渡す。
3. **`vreplay/smokes.py`**
   - `detect_smokes(frame_bgr, bg_bgr=None, *, transform=None, map_name=None, min_radius_px=None, max_radius_px=None, placed=None, reg=None, aspect=None, mask=None, icon_frac=None, require_minimap=False)` に拡張：
     - `require_minimap=True` 時にミニマップ存在検証を行う。
     - `mask` または `playable_region` から収縮した `floor_inner` 内に候補を限定し、マップ外・壁内部・左上 HUD を除外する。
     - `min_radius_px` / `max_radius_px` 既定値を画像高さ `H` 相対スケール（明示指定時は尊重）にする。
     - 背景差分による円形スモーク検知に加え、床輝度低下ベースのドーム検知（単一フレームや 5 秒クリップ中央値にスモークが焼き込まれたケースにも対応）を統合し、壁穴・エージェント顔アイコンによる誤検知を抑制する。
     - 既存の `detect_smokes(bg, None)`（`mask=None, require_minimap=False`）の `[]` 返却互換を維持する。
4. **`tools/eval_all.py`**
   - `detect_dots`・`detect_smokes` 双方に `playable_region` ゲートを適用する（単一フレーム時の `medianBlur` 外縁アーティファクトを回避し、`mask=floor_mask` で単一フレームスモークも検知可能にする）。
   - マスク健全性指標（`mask_ok_frames`, `gt_in_mask_rate`, `fp_outside_mask_filtered`）を集計・出力する。
   - `--preview-html PATH` オプションを追加し、自己完結 `CARDS` 形式・プレイ領域マスク可視化・ベースライン差分サマリーヘッダ・フィルタボタン付きの目視確認 HTML を生成できるようにする。
5. **オフラインテスト (`tests/test_minimap.py`, `tests/test_tactics.py`, `tests/test_smokes.py`, `tests/test_eval_all.py`)**
   - 3D 背景漏れ拒否、フレーム間で変化する壁穴背景の誤検知ゼロ、複数解像度スケール不変性、HUD 隅・視線除去、非ミニマップ画像拒否、`eval_all` + `--preview-html` 生成のテストを追加する（`rapidocr_onnxruntime` / `ffmpeg` / ネットワーク不要）。

## 通すテスト
```bash
python3 -m pytest -x -q tests/test_tactics.py tests/test_smokes.py tests/test_eval_all.py tests/test_tactival.py tests/test_tactival_barriers.py tests/test_track_walls.py tests/test_barriers.py tests/test_opening_playbook.py
python3 -m pytest -x -q tests/test_minimap.py -k "not test_fit_mask_reads_site_letters_on_symmetric_map"
python3 tools/eval_all.py --blind-dir /tmp/blind_val --tol 0.06
```

## 受け入れ条件
- [ ] 指定したテストがすべてネットワークなしで通る
- [ ] 空入力・部分失敗時に例外で停止しない
- [ ] `/tmp/blind_val` で `gt_in_mask_rate >= 0.98`、かつベースラインから誤検知抑制・F1 改善を達成する
- [ ] `blind_check_preview.html` が `--preview-html` により更新される
- [ ] 実装と別モデルのレビューで「マージしてよい」判定を取得済み

## 質問・ブロッカー / 相談票（3周で解決しない・仕様判断が必要な場合）
- **`tests/test_tactival_barriers.py::test_opening_masks_sunset_fixture`（触ってよいファイル外）**:
  - `map_mask` の `sq` を `H=20..34, S=35..80, V>=120` に修正した結果、`minimap_sunset_riot.jpg` に対して公式 `displayIcon` との IoU `0.864`（従来 `otsu` 0.290 より正確）な床マスクが返り、`register_art` (`T_art`) での `BR.opening_dist_maps(mask, T_art, ...)` は `['ATK', 'DEF']` を正常に返す。
  - しかし `tests/test_tactival_barriers.py:30-47` が `CALLOUT_FIX["sunset"]` なしの `sunset_callouts.json` をそのまま `MM.fit_mask` に渡して `BR.opening_dist_maps` を呼んでいるため、平行移動ずれ（`tx=0.636` vs `0.556`）でバリアをすり抜けて失敗する（従来は旧 `sq` により `otsu` が上下反転 `188.8°` の断片マスクになった偶然で通っていた）。
  - 本指示書の「触ってよいファイル」外のため `tests/test_tactival_barriers.py` は未変更。フォローアップで `sunset_displayicon` (`register_art`) または `CALLOUT_FIX["sunset"]` を用いるよう修正推奨。

## レビュー結果・修正履歴
<!-- Round 1〜3 の判定と対応内容を記録する -->

## 完了レポート
- **変えたファイル**:
  - `docs/tasks/playable-minimap-region.md`
  - `vreplay/minimap.py`（`map_mask` の `sq` 厳格化・中立床アンカー、`detect_minimap_region`、`playable_region` 追加）
  - `vreplay/tactics.py`（`detect_dots` の `dot_mask` 前段フィルタ・細長形状除外・`require_minimap`、`analyze_video` のフレーム毎白縁検証と `SM.detect_smokes(..., mask=mask, icon_frac=icon_frac)` 連携）
  - `vreplay/smokes.py`（`mask` / `require_minimap` / `found_agents`、高さ相対半径、背景差分＋床輝度低下ドーム統合）
  - `tools/eval_all.py`（プレイ可能領域ゲート・マスク健全性指標・`--preview-html` 生成）
  - `tests/test_minimap.py`, `tests/test_tactics.py`, `tests/test_smokes.py`, `tests/test_eval_all.py`
- **テスト結果**:
  - 対象スイート：`tests/test_tactics.py`, `tests/test_smokes.py`, `tests/test_eval_all.py`, `tests/test_tactival.py`, `tests/test_track_walls.py`, `tests/test_barriers.py`, `tests/test_opening_playbook.py`, `tests/test_minimap.py -k "not test_fit_mask_reads_site_letters_on_symmetric_map"` が全件パス（`tests/test_tactival_barriers.py::test_opening_masks_sunset_fixture` 1 件のみ上記理由で失敗）。
  - 全スイート (`python3 -m pytest -q`): `3 failed, 485 passed, 5 skipped, 7 errors in 171.37s`（失敗 2 件＋エラー 7 件は `rapidocr_onnxruntime` 未導入のため：`test_layout.py`, `test_outputs.py`, `test_loadout.py` ×4, `test_rounds.py` ×3；残り失敗 1 件は `test_tactival_barriers.py::test_opening_masks_sunset_fixture`）。
- **ブラインド評価 (`/tmp/blind_val`, 52 枚, `tol=0.06`)**:
  - `mask_ok_frames=52/52`, `gt_in_mask_rate=0.9887` (`agent=0.9897`, `smoke=0.9815`), `fp_outside_mask_filtered=88`
  - `detect_dots`: `TP=229, FP=191, FN=161, P=0.5452, R=0.5872, F1=0.5654`（ベースライン `F1=0.538` → `+0.027`）
  - `detect_smokes`: `TP=23, FP=25, FN=31, P=0.4792, R=0.4259, F1=0.4510`（ベースライン `F1=0.037` → `+0.414`）

