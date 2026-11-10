# areas-choke-split（エリア分割はチョークを境に）

※この指示書は Muse（窓口）が青木の指示（2026-11-10）をもとに作成。範囲は青木が決定済み: 第1弾は Split / Sunset のみ。第2弾（他マップ）は第1弾の通過後に青木が判断する。

## 目的
エリア（バイタルを含む）の分割がマップの構造とずれると、バイタルの色がサイト奥まで回り込む等の誤りが出る（Sunset で A Elbow を A Main に含めていた件）。エリアはチョークポイントを境に分ける規則を、コードとテストで固定する。

## 規則（青木の指示 2026-11-10）
1. エリアはチョークを跨がない。チョーク手前（ロビー〜チョーク本体）とチョークの先（サイト内・スポーン側）は別エリアにする。
2. バイタル（vital: true のエリア）の範囲はチョーク手前まで。バイタルに指定されたエリアは、チョークの先のコールアウトを含めない。
3. チョーク自体は手前側のエリアに属する（Sunset の A Main が choke: true の前例どおり）。
4. kb（config/kb/maps/*.yaml）が無いマップでエリアを自動生成する場合も同じ規則を適用する: 入口系コールアウト（Main / Lobby / Long / Short / Market / Link 等）とサイト内コールアウトを同一エリアに混ぜない。自動判定に使った対応一覧は、実装報告に必ず表で出す（青木が朝チェックで直す前提）。
5. エリア境界が変わると守りの配置判定（zone 集計・hands の when）にも影響する。zone が変わるマップ・手がある場合は、テストと報告で明示する。

## 触ってよいファイル
- vreplay/kb.py、vreplay/gamemodel.py、vreplay/deck.py、vreplay/playbook_v4.py、vreplay/mapfig.py（エリア生成・ポリゴン・バイタル判定に関わる範囲のみ）
- config/kb/maps/split.yaml、config/kb/maps/sunset.yaml（規則1〜3に反する callouts の付け替えのみ。hands や note は変えない）
- tests/test_playbook_v4.py、tests/test_gamemodel.py、tests/test_kb.py（無ければ作る）
- PROMPT.md、docs/PLAYBOOK_v4.md（仕様の追記。第1弾の範囲が決まってから）

## やること
### 第1弾（Split / Sunset）
1. 両マップの全エリアについて「チョーク跨ぎが無いか」を機械的に検査するテストを書く（kb の callouts と choke フラグから判定できる形）。
2. 検査で落ちる箇所があれば規則どおりに直し、直した箇所を報告に列挙する。
3. バイタルのポリゴンがチョーク先に広がらないことを、図の生成テスト（既存の test_playbook_v4 系）で担保する。

### 第2弾（他マップの自動分割）※第1弾の通過後に別途判断
4. kb が無いマップの自動エリア生成に規則4を適用し、対応一覧を出す。

## 通すテスト
- `python -m pytest -x -q tests/test_playbook_v4.py tests/test_gamemodel.py`
- 全体: `python -m pytest -q`（既存の失敗・エラーが増えていないこと）

## 終わりの条件
- 第1弾の検査テストが通り、違反箇所の一覧（直した／無かった）が報告にある
- Drive の実データ（2026-11-06 分の report.json）で Split / Sunset のプレイブック HTML を生成でき、バイタル図にチョーク跨ぎが無いことをレビュー担当（別モデル）が確認する
- 仕様として PROMPT.md と docs/PLAYBOOK_v4.md に規則が追記されている

## 質問
（実装担当が書く。無ければ空のまま）
