# choke-boundary-model（エリア境界をチョークポイントに一致させる領域モデルの再作成）

※この指示書は Muse（窓口）が 2026-11-10 に作成。青木が Tactival 上に赤丸で示した Sunset のチョークポイント7点が正本。
先行タスク main-region-fix（プラントのアンカー方式）は本タスクに吸収される（境界がチョークに合えば Main の色はプラント帯域に入らない）。main-region-fix の実装が先に終わっていたら、その差分を読んでから着手し、不要になる部分は戻してよい。

## 青木のモデル（2026-11-10）
- チョークポイントは「防衛側が守り、攻め側が抜けるために策を講じる点」。エリアはこの点を境目として切り替わる。
- Sunset のチョーク7点（Tactival 座標から変換した 0..1 座標。検証用重ね図は窓口の output/sunset_chokes_overlay.png）:
  - (0.113, 0.587) B Main → B Site の入口
  - (0.244, 0.512) B Market → B Site のドア
  - (0.385, 0.458) Mid → B Market の境
  - (0.499, 0.362) Mid → Mid Top（上段）の境
  - (0.766, 0.629) A Lobby → A Main の入口（A Main 側のチョーク）
  - (0.698, 0.474) A Main → A Site の入口
  - (0.899, 0.319) A Alley → A Site の裏口

## やること
1. kb にチョークを持てるようにする: `config/kb/maps/sunset.yaml` に `chokes: [[x, y], ...]`（0..1）で上記7点を入れる。読み込みは `vreplay/kb.py`。
2. `vreplay/mapfig.py` の `area_regions` を、境界がチョークに合う方式に作り替える。本命は**通路上の経路距離（測地距離）による割り当て**: 下地マスクの通路画素について、各コールアウトからの最短経路距離（256×256 グリッドの多始点ダイクストラ等）で最も近いコールアウトのエリアに割り当てる。直線距離のボロノイと違い、境界が自然に狭い通路（チョーク）に寄る。計算量が問題になる場合は、チョーク点を障壁としてマスクを区画に割る方式でもよい（その場合は理由を報告に）。
   - 既存の重み（バイタルの shrink、kb の shrink 指定）は維持すること。
   - キャッシュ（map_areas）のキーは方式変更が分かるようにすること（旧キャッシュを誤って使わない）。
3. 検証テストを追加する:
   - Sunset の7チョークについて、各点の近傍（半径 0.03 程度）に「境目の両側のエリア」の画素が両方存在すること（期待ペア: B Main|B Site、B Market|B Site、Mid|B Market、Mid|Mid Top、A Main|A Site、A Alley|A Site。A Lobby→A Main の点は同一エリア内のチョークなので、ペア検査の対象外とし、その旨をテストにコメントする）。
   - A Main / B Main の領域が A Site / B Site のコールアウト点とプラント帯域（A Elbow・B Boba の点）を含まないこと。
   - 既存テスト（test_kb / test_playbook_v4 / test_gamemodel / test_tactival / test_meta / test_deck / test_mapfig）が全て通ること。Split の領域が大きく壊れていないこと（Split のバイタル多角形テストで担保）。
4. LOUD の Sunset プレイブックを再生成し、バイタルの地図で境界がチョークに合っていることを窓口が目視確認できるよう、生成物パスを報告に含めること。

## 触ってよいファイル
- `vreplay/mapfig.py`、`vreplay/kb.py`（chokes の読み込み分のみ）
- `config/kb/maps/sunset.yaml`（chokes の追加のみ）
- `tests/test_mapfig.py`、`tests/test_playbook_v4.py`（検証テストの追加）
- 触らないもの: 他マップの YAML（チョークは当面 Sunset のみ）、deck.py、PROMPT.md

## 完了の定義
- 上記の検証テストが通り、既存テストに退行がないこと。レビューパック（変更ファイル・テスト結果・方式の説明・残課題）を作ること。push・マージはしない。

## 座標の注意（2026-11-10、窓口の重ね図ズレの切り分けで判明・必読）
- Sunset の valorant-api 素の変換係数はズレていることが分かっており、本番（`vreplay/tactics.py` の `load_map`）は `CALLOUT_FIX` のアフィン補正を適用した座標を使っている。Tactival のチョーク座標（displayIcon 基準）と揃うのは補正後の空間。
- 一方 `tests/deckfx.py` の `areas()` は補正前の素の座標を返す。チョークの検証テストで素の座標を使うと、座標系が混ざって必ずズレる。テストでは補正後の座標（`tactics.load_map` 経由か、deckfx の値に CALLOUT_FIX を適用）を使うこと。
- 未切り分けの残件: B Boba は補正後 (0.230, 0.205) でも displayIcon の床の外（北西の黒い領域）に落ちる。補正の外挿誤差か元座標の問題かは未判定。B Boba は B Site のアンカー点なので、本タスクの検証で床の上に落ちているかを確認し、外れている場合は報告に明記すること（座標そのものの修正は本タスクの範囲外。別途判断する）。
