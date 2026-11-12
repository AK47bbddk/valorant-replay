# sunset-test-icon-fixture（Sunset エリアテストの画像を追跡対象のフィクスチャにする）

※窓口 (Muse) が 2026-11-12 に作成。青木の承認済み。目的は GitHub の test ワークフロー（CI）を緑にすること。

## 目的と確認済みの事実
- CI（クリーンなチェックアウト）では `tests/test_playbook_v4.py` の Sunset エリア系テスト11件が落ちる。原因はテストヘルパー `_sunset_base()` が参照する公式 displayIcon のキャッシュ（`<repo>/_review/_cache/map_icons/sunset.png` と `<repo>/.review_tmp/sunset_displayicon.png`）が未追跡ファイルで、クリーン環境に存在しないこと。無い場合は全面が通路の合成画像 `_flat_base()` に落ちて、エリアの形が成り立たず落ちる。
- コード側（kb・mapfig・playbook_v4）は正常で、キャッシュ画像を置けば同テストは 39/39 通ることを窓口が確認済み（2026-11-12、マージ後ツリーでの再現）。
- ネットワークから valorant-api を取る方法は、この VM からは IncompleteRead が頻発して安定しないため採らない（窓口が 2026-11-12 に再確認）。画像をリポジトリに含める方式にする。
- 対象の画像は現在キャッシュにあるもの（27316 バイト、アルファ付き PNG）で、テストが実際に使って通っているものと同一ファイルを使う。

## やること
1. ブランチ `task/sunset-test-icon-fixture` を切る。
2. 現在のキャッシュ画像 `_review/_cache/map_icons/sunset.png` を `tests/fixtures/sunset_displayicon.png` としてコピーし、追跡対象にする（このファイルの追加に限り tests/fixtures への追加を許可する。既存フィクスチャの変更はしない）。
3. `tests/test_playbook_v4.py` の `_sunset_base()` を直す: 最初に `tests/fixtures/sunset_displayicon.png` を見る。次に既存のキャッシュ2か所、最後に `_flat_base()` の順は維持する。docstring も実態に合わせる。
4. 検証: クリーンな作業ツリー（`git worktree add --detach <一時先> HEAD` 等、_review も .review_tmp も無い状態）で `tests/test_playbook_v4.py` が全て通り、全体テストで失敗が増えていないことを確認する。通常の作業ツリーでも対象テストを通す。
5. レビューパックを作る。push・マージはしない。

## 触ってよいファイル
- `tests/fixtures/sunset_displayicon.png`（新規追加のみ）
- `tests/test_playbook_v4.py`（`_sunset_base()` の探索順と docstring のみ。他のテストの変更はしない）
- `PROMPT.md`（テストの前提を書いた節があれば、その節だけ）

## 通すテスト
- `python -m pytest -x -q tests/test_playbook_v4.py`（通常ツリーとクリーンツリーの両方）
- 全体: `python -m pytest -q`（失敗が増えていないこと）

## 終わりの条件
- クリーンな作業ツリーで Sunset 11件を含む test_playbook_v4.py が全て通る
- フィクスチャ画像が追跡対象に入っている（git status で確認できる）
- テストの判定条件（しきい値・期待値）は変えていない

## 質問
（実装担当が書く。無ければ空のまま）

## 追記（2026-11-12 夜、窓口）: 表現を base64 テキストに変更する
- 理由: この環境から GitHub へ送る正規の経路（コネクタの snapshot push、および MCP の create_or_update_file）は UTF-8 テキスト専用で、バイナリの PNG を送れない。青木は「PNGの送り方は任せる」とし、窓口はテキストで送れる表現への変更を選んだ。
- 変更後の形: フィクスチャは `tests/fixtures/sunset_displayicon.png.b64`（元の PNG バイト列を base64 化したテキスト、76桁折り返し）とする。PNG ファイル自体は追跡対象から外す。`_sunset_base()` は最初にこの .b64 を読み、base64 復号 → cv2.imdecode で画像化して使う（復号したバイト列が従来の PNG と一致することは窓口が blob で照合する）。以降の探索順（既存キャッシュ2か所→全面通路）は不変。docstring に .b64 の作り方（公式 displayIcon を base64 化したもの）も書く。
- ブランチは未 push のため、コミット 2ef3dc4 を amend して1コミットに保ってよい。
