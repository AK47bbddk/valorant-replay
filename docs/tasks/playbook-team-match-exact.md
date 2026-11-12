# playbook-team-match-exact（プレイブックのチーム照合を完全一致に）

※この指示書は Muse（窓口）が 2026-11-11 の他マップ一括生成の検証結果をもとに作成。青木の承認後に実装する。範囲はチーム照合の修正と汚染された12件の再生成のみ。

## 目的
他マップ一括生成（161チーム×マップ）で、12件のプレイブックに他チームのラウンドが混ざっていることが分かった。原因は `vreplay/playbook_v4.py` の `_match` がチーム名の部分一致（`t in team.lower()`）で照合しているためで、短いタグほど別チームを拾う。7件はファイル名まで別チームになり、5件はファイル名は正しいが中身が混合のまま残っている。照合を完全一致に直し、12件を作り直す。

## 確認済みの事実（2026-11-11、VMで検証）
- 一括生成は完了済み: Driveの report.json 171件を全てダウンロード（ok=171 fail=0）、重複排除で143マップ分、12マップ・161チーム×マップ。ログは `~/workspace/valo-opencode/make_all.log`（GENERATE DONE made=154 failed=7）。
- 汚染12件（`_match` で拾ったラウンドのタグ内訳。`mine` は照合で拾われた全ラウンド）:
  - Ascent TH: 124R中 TH 19R / 100T 105R（`100 Thieves` の `Thieves` に `th` が含まれる）
  - Lotus TH: 94R中 TH 24R / 100T 70R
  - Split TH: 52R中 TH 16R / 100T 36R（ファイル名は TH のまま＝黙って混入）
  - Bind AST: 51R中 AST 19R / UB 19R / MC 13R
  - Haven AST: 43R中 AST 14R / UB 14R / MC 15R
  - Lotus RA: 44R中 RA 22R / DRG 22R（`Dragon Ranger` に `ra` が含まれる）
  - Summit RA: 42R中 RA 21R / DRG 21R
  - Split TL: 61R中 TL 44R / M8 17R
  - Abyss TL: 34R中 TL 20R / RELE 14R（ファイル名は TL のまま）
  - Haven TL: 35R中 TL 20R / RELE 15R（ファイル名は TL のまま）
  - Split GE: 38R中 GE 21R / M8 17R（`Gentle Mates` に `ge` が含まれる。ファイル名は GE のまま）
  - Summit GE: 63R中 GE 42R / DRG 21R（ファイル名は GE のまま）
- ファイル名が別チームになった7件（`build()` は `mine[0]` のタグで命名するため）: Ascent_TH→`playbook_v4_100T_Ascent.html`、Bind_AST→`playbook_v4_UB_Bind.html`、Haven_AST→`playbook_v4_UB_Haven.html`、Lotus_RA→`playbook_v4_DRG_Lotus.html`、Lotus_TH→`playbook_v4_100T_Lotus.html`、Split_TL→`playbook_v4_M8_Split.html`、Summit_RA→`playbook_v4_DRG_Summit.html`。`make_all_playbooks.py` は期待するファイル名が無いため failed=7 と数えた（生成スクリプト自体は異常終了していない）。
- 上記12件以外は汚染なし（全161件を同じ方法で走査し、タグ内訳が自チームのみであることを確認済み）。
- 原因の箇所: `vreplay/playbook_v4.py` の `_match`（`t in (r.get("team") or "").lower() or t == (r.get("tag") or "").lower()`）。`deck.records()` が返す1ラウンド記録の `team` は正式チーム名（例: `100 Thieves`、`Team Heretics`、`Dragon Ranger Gaming`、`Gentle Mates`）、`tag` は略称（例: `100T`、`TH`、`DRG`、`M8`）。部分一致の前半が正式名の部分文字列に当たる。

## 規則（青木の指示ではない。今回の修正で決める仕様）
1. チーム照合は完全一致のみ: `tag` の完全一致（大文字小文字を区別しない）または正式チーム名 `team` の完全一致（大文字小文字を区別しない）。部分一致は使わない。
2. 出力ファイル名は要求されたチーム名（`--team` の値）で付ける。`mine[0]` のタグで命名しない（照合が完全一致になれば両者は一致するはずだが、命名の根拠を要求側に固定する）。
3. 照合で拾うラウンドが0件の場合は従来どおりエラーにする（`{team} の {map} のラウンドがありません`）。黙って他チームで埋めない。

## 触ってよいファイル
- `vreplay/playbook_v4.py`（`_match` と `build()` のファイル名決定のみ）
- `tests/test_playbook_v4.py`（回帰テストの追加。無ければ作る場所は既存のテスト構成に従う）
- `PROMPT.md`、`docs/PLAYBOOK_v4.md`（チーム照合の仕様を書いてある節があれば、その節だけ追記）
- 生成物の置き場（リポジトリ外）: `~/workspace/valo-opencode/output/playbooks/<Map>_<Team>/`（12件の再生成と誤ったファイル名の削除のみ。ほかの149件には触らない）

## やること
1. ブランチ `task/playbook-team-match-exact` を切る（AGENTS.md の手順どおり）。
2. `_match` を規則1の完全一致に直す。`build()` の出力ファイル名を規則2どおり要求チーム名で付ける。
3. 回帰テストを追加する: 短いタグの衝突ケースを合成レコードで確認する（例: 正式名 `100 Thieves` / タグ `100T` のラウンドが `--team TH` で拾われないこと、`Dragon Ranger Gaming` / `DRG` が `--team RA` で拾われないこと、`Gentle Mates` / `M8` が `--team GE` で拾われないこと。逆に `--team TH` が `Team Heretics` / `TH` を拾うこと）。既存のテストデータで再現できる場合はそちらを使ってもよい。
4. 再生成の前に、汚染12件の出力フォルダから既存の HTML を削除する（`make_all_playbooks.py` は HTML が存在すると生成をスキップするため、そのまま再実行しても直らない）。7件の誤ったファイル名の HTML も削除する（残すと別チームのプレイブックの重複として残る）。
5. 12件だけを `scripts/playbook_v4.py`（`.venv` の python）で再生成する。ステージ済みデータは `~/workspace/valo-opencode/data/by_map/<Map>/*.json` を使う。生成後は全161件について、ファイル名が `<Map>_<Team>` と一致することと、照合で拾われるラウンドのタグが自チームのみであることを機械的に確認し、結果を報告に列挙する（確認方法は 2026-11-11 の検証で使った走査と同じ考え方でよい）。
6. 仕様の追記（PROMPT.md / docs/PLAYBOOK_v4.md にチーム照合の節があれば）。

## 通すテスト
- `python -m pytest -x -q tests/test_playbook_v4.py`
- 全体: `python -m pytest -q`（既存の失敗・エラーが増えていないこと。既知の失敗は `tests/test_video_cache.py::test_range_download_matches_source` の1件のみで、今回の変更と無関係）

## 終わりの条件
- 回帰テストが通り、短いタグの衝突ケースが再現しない
- 汚染12件が正しいチームのみの内容で再生成され、誤ったファイル名の HTML が残っていない
- 全161件のファイル名とタグ内訳の確認結果が報告にある
- 仕様としてチーム照合が完全一致であることが PROMPT.md または docs/PLAYBOOK_v4.md に書かれている（書く場所が無ければ報告に明記する）

## 質問
（実装担当が書く。無ければ空のまま）
