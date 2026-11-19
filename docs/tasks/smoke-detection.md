# タスク指示書：ミニマップからのスモーク位置検知と初動分類への連携

- **タスクID / ブランチ**: `task/smoke-detection`
- **作成日**: 2026-11-19
- **参照すべき `PROMPT.md` の節**: `## 大量データ（2024〜2026）とプレイブック`（2. ミニマップの位置合わせ / 3. サイト・位置）

## 背景・目的

`vreplay/opening_playbook.py`（1880行目付近）では、ラウンドごとのタイムライン情報 `rt["smokes"]`（`1:40 / 1:30 / 1:15 / 1:00` の節目におけるスモーク一覧）を受け取って初動分類に活かす受け皿がすでに用意されている。
一方で、実際のミニマップ画像からスモーク（ドーム型スモーク等）を検知して `rt["smokes"]` を生成する処理がまだ存在しないため、実戦データではスモーク情報が空のままになっている。

本タスクでは、ミニマップ静止画（および背景中央値画像との差分）からスモークの円形領域を検知するモジュール `vreplay/smokes.py` を新設し、`vreplay/tactics.py` のタイムライン出力に `smokes` フィールドを統合することで、初動分類とプレイブックの精度・分かりやすさを引き上げる。

## 触ってよいファイル
- `vreplay/smokes.py`（新規作成）
- `vreplay/tactics.py`（`rt["smokes"]` への統合のみ。既存のプレイヤー検出・位置合わせロジックは壊さない）
- `tests/test_smokes.py`（新規作成）
- `tests/test_opening_playbook.py`（必要なら連携テストを 1 件追加）

※ `vreplay/minimap.py` の座標変換（`x`/`y` 入れ替えや Sunset 補正）や `tests/fixtures/` の画像は変更しない。

## 変更内容

1. **`vreplay/smokes.py` の新規作成**
   - `detect_smokes(frame_bgr, bg_bgr, *, transform=None, map_name=None, min_radius_px=10, max_radius_px=42) -> list[dict]`
     - `frame_bgr`（現在フレームのミニマップ BGR 画像）と `bg_bgr`（試合全体の背景中央値 BGR 画像）の差分から、スモーク特有の円形・半透明／高彩度／低エッジ領域（ドームスモーク）を抽出する。
     - プレイヤーアイコン（半径が小さく高コントラストな円）やスパイク設置エフェクトを誤検知しないよう、半径範囲・円形度・内部均一性でフィルタする。
     - `transform`（`minimap.py` の相似変換）と `map_name` が与えられた場合、マップ正規化座標 `(x, y)` と `zone`（`A` / `B` / `C` / `Mid` 等）および `area`（コールアウト名）を付与する。変換不能な場合も例外を出さず `zone=None, area=None` で返す。
     - 戻り値の各要素は `opening_playbook.py` と互換な辞書形式とする：
       ```python
       {
           "kind": "smoke",
           "subkind": "dome",  # 将来 wall / orb を拡張可能にする
           "x": float,         # 正規化座標 0..1（未変換時は画像相対 0..1）
           "y": float,
           "r": float,         # 正規化半径
           "zone": str | None,
           "area": str | None,
           "team": int | None, # 判定不能時は None（色手がかりがある場合 1 or 2）
           "conf": float,      # 0.0..1.0
       }
       ```
     - 空画像・`None`・サイズ不一致の場合は例外を投げず空リスト `[]` を返す。

2. **`vreplay/tactics.py` への統合**
   - 各ラウンドの節目（`1:40`, `1:30`, `1:15`, `1:00`）のミニマップフレームに対して `detect_smokes(...)` を呼び出し、ラウンド結果辞書に `"smokes": {"1:40": [...], "1:30": [...], ...}` を格納する。
   - スモーク検知が例外や部分失敗を起こしても、既存の `tactics.py` の解析・出力を一切停止させない（`try/except` で安全にフォールバックし空辞書・空リストとする）。

3. **ユニットテストの作成 (`tests/test_smokes.py`)**
   - 合成ミニマップ画像（背景＋模擬ドームスモーク＋模擬プレイヤー点）を使い、以下をネットワークなしで検証する：
     - ドームスモークが 1〜2 個あるフレームで正しい位置・半径・`kind == "smoke"` が検出されること
     - プレイヤー点のみのフレームで誤検知（0 件）にならないこと
     - `None` / 空配列 / サイズ不一致入力で例外を出さず `[]` を返すこと
     - `detect_smokes` の出力が `opening_playbook.classify_round_record` の `smokes` スキーマとそのまま噛み合うこと

## 通すテスト
```powershell
python -m pytest -x -q tests/test_smokes.py tests/test_opening_playbook.py tests/test_tactics.py
```

## 受け入れ条件
- [ ] 指定したテストがすべてネットワークなしで通る
- [ ] 空入力・サイズ不一致・部分失敗時に例外で停止しない
- [ ] 既存の `tests/test_tactics.py` と `tests/test_opening_playbook.py` にデグレがない
- [ ] 実装と別モデルのレビューで「マージしてよい」判定を取得済み

## 質問・ブロッカー / 相談票（3周で解決しない・仕様判断が必要な場合）
<!--
相談先の希望: Claude (Opus / Claude Code) / Muse / 上位モデル
件名:
詰まっていること:（1〜3文）
試したこと:
証拠:（ログ末尾30行とエラー）
聞きたいこと:（1〜3問）
-->

## レビュー結果・修正履歴
<!-- Round 1〜3 の判定と対応内容を記録する -->

## 完了レポート
<!-- 変えたファイル / テスト結果 / 迷ったところ・確認してほしい点 -->
