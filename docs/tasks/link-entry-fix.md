# link-entry-fix（Link の入口分類を正す＋入口判定の誤検出を減らす）

※この指示書は Muse（窓口）が 2026-11-10 朝の GLM-5.3 レビュー指摘をもとに作成。第1弾（areas-choke-split）のフォローアップ。第2弾の本体（他マップの自動分割）とは別タスク。

## 背景（確認済み 2026-11-10）
- GLM レビューは第1弾を「要確認」とし、「必ず直すこと」はなし。ただし `ENTRY_CALLOUTS` の `"link"` が誤分類の疑いありと指摘した。Sunset の A Link／Split の B Link は Mid からサイトへ入るサイト側の回廊で、攻めの入口ではない、という指摘。
- 実態を確認したところ、現行の Split／Sunset の kb YAML（`config/kb/maps/split.yaml`・`sunset.yaml`）の `areas[].callouts` に Link は一切含まれていない。つまり今朝届けた HTML 2件のバイタル図にこの誤分類の影響はない。影響が出るのは kb が無いマップの自動生成（第2弾）で Link を Main 系と同一エリアにまとめた場合だけ。
- ただしテスト側に矛盾がある。`tests/test_kb.py` は「A Link は入口系」と固定する一方、`tests/test_playbook_v4.py` は Sunset の A Main ポリゴンが A Link を含まないことを確認しており、こちらは A Link をサイト側として扱っている。どちらかに統一する。
- GLM の残り2点も同じタスクで直す。(1) `is_entry_callout` は部分一致（`w in n`）のため将来の名前で誤検出し得る。(2) バイタル図の多角形テストは Sunset の A Main のみで Split 側のカバーが薄い。

## 決定事項（2026-11-10 青木が回答済み・実装済み）
- Q1 は **A（サイト側）** で確定。同日、ブランチ `task/areas-kb-allmaps` 上で本指示書の内容（ENTRY_CALLOUTS から link を除去、is_entry_callout の単語一致化、Split 多角形テスト追加、PLAYBOOK_v4.md の一覧修正）を実装し、対象テスト77件の通過を確認した。あわせてチョーク跨ぎの例外3件（Sunset B Site の Boba、Bind A Short の Cubby、Bind B Short の Window）を `CHOKE_EXCEPTIONS` として固定した。未マージのため、本ファイルの着手条件のうち「第1弾のマージ判断」は引き続き未決。

## 決定事項（元の質問）
- Q1: A Link（Sunset）／B Link（Split）は「チョーク手前（入口側）」か「サイト側」か。
  - A: サイト側（GLM の指摘どおり。`tests/test_playbook_v4.py` の扱いとも一致）→ `"link"` を `ENTRY_CALLOUTS` から外し、`test_kb.py` の入口系リストから "A Link" をサイト側リストへ移す。
  - B: 入口側（現状維持）→ `ENTRY_CALLOUTS` は変えず、`test_playbook_v4.py` の A Link 扱いがなぜサイト側なのかを理由付きで記録する（ポリゴンから外すのはバイタルの範囲の問題で、分類とは別、等）。
  - 回答が無いうちは実装しないこと。

## やること（Q1 の回答後）
1. Q1 の結論どおり `ENTRY_CALLOUTS` と `tests/test_kb.py` の分類を統一する。`docs/PLAYBOOK_v4.md` の入口系一覧（Main / Lobby / Long / Short / Market / Link / Boba / Ramps / Garage）も同じ結論に合わせる。
2. `is_entry_callout` の部分一致を、単語境界での一致に寄せる（例: "Mid Market" や "Long Alley" が入口側に落ちないこと）。完全な解決にならない場合は、誤検出する名前の例を報告に列挙し、許可リスト方式への切替を「質問」に書いて止める。テストを緩めて通さない。
3. Split のバイタル（MID／A Ramps）について、Sunset の A Main と同じ形の多角形テストを追加する（チョークの先のコールアウトを含まないこと）。

## 着手条件
- 青木の Q1 回答と、第1弾ブランチ `task/areas-choke-split`（b29fceb）のマージ判断が出てから着手する。ブランチは `task/link-entry-fix`。第1弾が未マージなら第1弾ブランチから切り、マージ済みなら main から切る。

## 触ってよいファイル
- `vreplay/kb.py`（`ENTRY_CALLOUTS`・`is_entry_callout` の範囲のみ）
- `tests/test_kb.py`、`tests/test_playbook_v4.py`
- `PROMPT.md`、`docs/PLAYBOOK_v4.md`（入口系一覧の追記のみ）
- 触らないもの: `config/kb/maps/*.yaml`（Link は現行エリアに無いため変更不要のはず。必要なら止めて理由を「質問」に書く）

## 通すテスト
- `python -m pytest -q tests/test_kb.py tests/test_playbook_v4.py tests/test_gamemodel.py`
- 全体: `python -m pytest -q`（既存の失敗・エラーが増えていないこと）

## 終わりの条件
- Q1 の結論と `ENTRY_CALLOUTS`・両テストの分類が一致している
- `is_entry_callout` の誤検出テスト（"Mid Market"・"Long Alley" 等）が通り、残る誤検出の余地があれば報告にある
- Split のバイタル多角形テストが追加され、対象テストが全通過している
- push・マージはしない。レビューパックは第1弾と同じ手順で作る

## 質問
（実装担当が書く。無ければ空のまま）
