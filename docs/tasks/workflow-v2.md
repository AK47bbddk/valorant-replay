# workflow-v2：Antigravity 内で「実装 → 別モデルのレビュー → 改善」を回す形に作り直す

作成：Claude（2026-11-18）／承認：青木（未）

## 目的

今の方式 A は、1 つの Orchestrator が設計からマージまでを抱えていて、レビュー役のモデルも決まっていない。そのため「実装した本人がそのまま確かめる」形にずれやすい。
これを次の形に作り直す。

- 実装は Gemini 4 Hydrogen、レビューは Sonnet（実装とは別のモデル）が行う。
- 「必ず直すこと」が無くなるまで、実装とレビューを最大 3 周回す。
- 詰まったら止めて相談票を出す（相談先は Opus か Argon。Argon は Muse の許可制）。
- agy を起動できるのは青木本人だけ（IDE か Remote Control）。スクリプト、Bot、Muse からは起動しない。

## 前提

1. 済み（2026-11-18、Claude）：ノート PC で Antigravity が作ったルールファイル（`AGENTS.md`、`GEMINI.md`、`docs/WORKFLOW.md`、`docs/tasks/_TEMPLATE.md`、`.agents/skills/`、`tools/agy_orchestrator.py`、`tools/discord_bot.py`、`scripts/setup_remote_ssh.ps1`）と、この指示書を、ブランチ `task/workflow-v2`（`origin/main` の `addf5d1` から分岐）にコミットした。ノート PC の main と作業ツリーには手を付けていない。
2. 青木：ノート PC で `git push origin task/workflow-v2` を実行する。
3. デスクトップでの作業：`git fetch origin` のあと、`git worktree add ../vr-workflow-v2 task/workflow-v2` で作った worktree の中で作業する。デスクトップの main と、ほかの worktree（`vr-barriers` など）には触らない。デスクトップの main に同じ名前の未コミットのファイルがあっても触らず、違いがあれば「質問・ブロッカー」に書く。
4. 推奨：デスクトップにしか無いブランチを、バックアップのために push する（`main` の `edff44e`、`task/color-half-fix`、`task/barriers-v2`）。push するかどうかは青木が決める。

## 触ってよいファイル

- `.agents/skills/vreplay-orchestrator/SKILL.md`
- `.agents/skills/vreplay-review/SKILL.md`
- `docs/WORKFLOW.md`
- `docs/tasks/_TEMPLATE.md`
- `GEMINI.md`（役割分担の節だけ）
- `AGENTS.md`（「起動とモデル」と「完了レポート」の節の追加だけ）
- `tools/discord_bot.py`
- `tools/agy_orchestrator.py`
- `tests/test_discord_bot.py`、`tests/test_agy_orchestrator.py`（無ければ新しく作る）

上に無いファイルが必要になったら、作業を止めてこの指示書の末尾の「質問・ブロッカー」に書く。

## やること

### 1. 役割とモデルを固定する（`vreplay-orchestrator` の SKILL.md、`GEMINI.md`）

次の表をそのまま書き込む。

| 役割 | モデル | やること | してはいけないこと |
|---|---|---|---|
| 進行と実装 | Gemini 4 Hydrogen（起動時に固定） | `task/<名前>` の worktree で実装、テスト、コミット | main の変更、push、マージ |
| レビュー | Sonnet（サブエージェント、または別セッション） | `git diff main...task/<名前>` と指示書だけを読み、`vreplay-review` の形式で判定する | ファイルの編集、コミット |
| 相談 | Opus（claude.ai）か Argon | 相談票に答える | — |

- 実装役とレビュー役は、同じモデルにしない。
- サブエージェントごとにモデルを指定できない場合は、実装が終わったところで止める。そして「レビュー起動文」（Sonnet に固定した別セッション用の文）を完了レポートに書き、青木の 2 回目の起動を待つ。
- 1 回の起動で扱うのは 1 タスク・1 ブランチだけ。

### 2. ループと止める条件（`vreplay-orchestrator` の SKILL.md）

1. Hydrogen が実装し、指示書の「通すテスト」を実行してコミットする。
2. Sonnet がレビューする。
3. 判定が「直してからマージ」なら、「必ず直すこと」だけを Hydrogen に戻す。最大 3 周まで。
4. 「マージしてよい」が出たら止めて、完了レポートを出す。

次のどれかに当たったら、周の途中でも止める。そして指示書の末尾に「相談票」を書く。

- 仕様や定義の判断が要る（エリアの境界、ラベルの定義など）
- 3 周しても「必ず直すこと」が残る
- 触ってよいファイルの外を変える必要がある
- テストが環境の都合で動かない（既知の 5 件以外の失敗）

相談票の形：

```
相談先の希望: Opus / Argon
件名:
詰まっていること:（1〜3 文）
試したこと:
証拠:（ログの末尾 30 行とエラー）
聞きたいこと:（1〜3 問）
```

Argon に相談するのは、指示書の頭に `Argon 相談: 可（Muse 許可 YYYY-MM-DD）` と書いてあるときだけ。書いていなければ、相談先は Opus（青木が claude.ai に貼る）とする。

### 3. 完了レポート（`AGENTS.md` に節を追加、`_TEMPLATE.md` に欄を追加）

止まるたびに、必ず次の形で出す。

```
## 完了レポート：<名前>
判定（レビュー役）: マージしてよい / 直してからマージ / やり直し / 止めた（相談票あり）
ブランチとコミット: task/<名前> <短いハッシュ>
変えたこと:（ファイルごとに 1 行）
終わりの条件:（条件ごとに OK / NG / 不明と根拠）
テスト: <コマンド> → passed / failed（既知の 5 件以外の失敗の有無）
周回数: n / 3
使ったモデルと回数: Hydrogen n、Sonnet n、Argon n
範囲外の変更: なし / あり（内容）
秘密情報: 触れていない
青木に決めてほしいこと: なし / あり（内容）
```

`_TEMPLATE.md` には次の 2 行を足す。

- `担当モデル: 実装 Hydrogen ／ レビュー Sonnet`
- `Argon 相談: 不可`

### 4. マージ（`docs/WORKFLOW.md`、`vreplay-orchestrator` の SKILL.md）

- マージと push は、青木が「マージしてよい」と言ったあとだけ行う。
- 実行役は Claude か Antigravity のどちらでもよい。Antigravity がマージするのは、起動文に「マージまで」と書いてあるときだけ。
- マージしたあとは、`PROMPT.md` と `README.md` の該当する節を更新する。

### 5. agy をスクリプトから起動する経路を止める（`tools/discord_bot.py`、`tools/agy_orchestrator.py`）

- `discord_bot.py`：`!vr prompt` と、`!vr task` の agy エンジン経路と、`!vr review` の SDK 自動判定を外す。呼ばれたら「agy の起動は Remote Control から行ってください」と返す。`!vr status` と `!vr test` はそのまま残す。
- `agy_orchestrator.py`：`run-task --engine agy` と `review --use-sdk` を外す（呼ばれたら同じ文言で終了コード 2 を返す）。`status`、`new-task`、レビューパックの生成（SDK なし）は残す。
- コードの中で `agy` の CLI か `google-antigravity` SDK を呼んでいる箇所が無いことを、テストで確かめる。

### 6. `docs/WORKFLOW.md` を書き直す

- **方式 A（標準）**：この指示書の 1〜4。起動は青木が IDE か Remote Control で行う。
- **方式 B（外出先）**：起動は Remote Control で行う。SSH と Discord Bot は状態の確認とテストの実行だけに使う。
- **方式 C（予備）**：Google Pro の枠が切れたときに使う。OpenCode＋DeepSeek V4.1 Flash Free で実装し、別のモデルでレビューする（今の手順のまま）。
- 冒頭に「agy を起動できるのは青木本人だけ。スクリプト・Bot・Muse からは起動しない」と書く。

## 通すテスト

```
python -m pytest -x -q tests/test_discord_bot.py tests/test_agy_orchestrator.py
python -m pytest -q
```

全体の失敗が既知の環境依存の 5 件だけであること。テストを消したり、スキップしたり、緩めたりしない。

## 終わりの条件

1. `vreplay-orchestrator` の SKILL.md に、役割とモデルの表、ループ、止める条件、相談票、Argon の許可制が書いてある。
2. `tools/` の中で agy の CLI と `google-antigravity` SDK を呼ぶ箇所が 0 件で、そのことをテストで確かめている。`!vr status` と `!vr test` は今までどおり動く。
3. `docs/WORKFLOW.md` が方式 A・B・C の新しい形になっていて、冒頭に起動のルールがある。
4. 完了レポートの形が `AGENTS.md` にあり、`_TEMPLATE.md` に「担当モデル」と「Argon 相談」の欄がある。

## この指示書の担当

担当モデル: 実装 Hydrogen ／ レビュー Sonnet
Argon 相談: 不可（2026-11-21 まで枠なし）

## 質問・ブロッカー

（実装担当が書く）
