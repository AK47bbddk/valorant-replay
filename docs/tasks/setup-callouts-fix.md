# setup-callouts-fix（setup の callouts にエリア名でなく実際のコールアウトを入れる）

※窓口 (Muse) が 2026-11-11 に作成。青木の指摘「エルボーが消えている」の根本原因。

## 原因（窓口が特定済み）
- `vreplay/tactics.py` の `setup_of` と `_agent_setups` の両方で、setup の `"callouts"` に `areas`（エリア名の Counter）の要素を入れている。つまり callouts と称してエリア名（A Site 等）が入っており、実際のコールアウト名（A Elbow、B Boba 等）が setup から漏れる。
- そのため deck の「よく映る場所」（setup.callouts の集計）にコールアウトが一切出てこない。データ側には A Elbow の出現はある（例: 2026-11-06 の Sunset、ラウンド0の 1:40 で Omen が A Elbow）。

## やること
1. `setup_of` と `_agent_setups` の両方で、setup の `"callouts"` に、そのサンプルに実際に立っていた地点のコールアウト名（データの callout 表記そのまま、例 "A Elbow"）を入れる。`"areas"`（エリア名の Counter）と `"zones"` の意味は変えない。
2. deck 側の集計（「よく映る場所」等）がコールアウト名で動くことを確認する。表示件数や閾値の調整が必要なら最小限で行う。
3. テストを追加する: A Elbow に立つラウンドを含む入力で、setup の callouts に "A Elbow" が含まれること。既存テストが callouts=エリア名 前提で落ちる場合は、新しい意味に合わせて直す（理由を報告に書く）。
4. 既存テストを全て通す。レビューパックを作る。push・マージはしない。

## 触ってよいファイル
- vreplay/tactics.py、vreplay/deck.py、tests/、PROMPT.md（必要なら）
