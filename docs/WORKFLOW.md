# 開発の回し方（Antigravity オーケストレーション・モデル分担）

## 方式 A：Antigravity (`agy`) オーケストレーション（推奨・一元管理）

設計・実装・テスト・レビューまでを Antigravity（デスクトップアプリ / `agy` CLI / Python SDK）内で完結させる標準ワークフロー。

| 段階 | 担当 | どこで・どうやって |
|---|---|---|
| 1. 設計・指示書作成 | Orchestrator (`Argon` / Plan Mode) | `docs/tasks/_TEMPLATE.md` から `docs/tasks/<名前>.md`（または `ticket_*.md`）を作成（`python tools/agy_orchestrator.py new-task <名前>` でも可） |
| 2. 実装・テスト・ローカルコミット | Execution Model / Subagent (`flash`) | Plan/Execute、または `invoke_subagent`（`Workspace: "share"`）／`swarm`／`python tools/agy_orchestrator.py run-task <名前>` で実装し、`python -m pytest -x -q` を通す |
| 3. レビュー | Review Subagent (`.agents/skills/vreplay-review`) | ローカルで `main..HEAD` の差分と指示書を突き合わせて判定（Google ドライブへの `.txt` 手動アップロードやブラウザ変換は不要） |
| 4. 確認・マージ・仕様同期 | Orchestrator | 「マージしてよい」確認後に `main` へマージし、`PROMPT.md` / `README.md` を更新 |

### クイックコマンド（`tools/agy_orchestrator.py`）

```powershell
# 現在のブランチ・ワークツリー・チケット一覧を確認
python tools/agy_orchestrator.py status

# タスク指示書の雛形を作成
python tools/agy_orchestrator.py new-task barrier_v2 --files vreplay/tactics.py tests/test_tactics.py

# 実装 → pytest 検証 → レビューパック生成 → 通知を一貫実行
python tools/agy_orchestrator.py run-task barrier_v2 --engine agy

# 差分レビューパック生成（--use-sdk で Python SDK による自動レビューも実行）
python tools/agy_orchestrator.py review barrier_v2 --range main..HEAD --use-sdk
```

---

## 方式 B：外出先・ノート PC からの遠隔オーケストレーション（SSH / Discord Bot / `vr-barriers`）

自宅デスクトップ PC 上の最新ワークツリー（例：`vr-barriers`）や `ticket_barrier_v2.md` をノート PC やスマホから操作する場合（詳細は `.agents/skills/vreplay-remote-sync/SKILL.md`）。

1. **ノート PC の SSH 鍵準備**：
   `powershell -ExecutionPolicy Bypass -File scripts/setup_remote_ssh.ps1` を実行し、`~/.ssh/id_ed25519` を生成・公開鍵をデスクトップ PC の `authorized_keys` に登録する。
2. **自宅デスクトップ PC で Discord Bot を常駐**：
   環境変数（`DISCORD_BOT_TOKEN`, `DISCORD_ALLOWED_USER_IDS`, `VREPLAY_WORKTREE_DIR`）を設定し、`python tools/discord_bot.py` を起動する。
3. **外出先からのコマンド**：
   - `!vr status` : ブランチ・`git worktree list`・チケット一覧を確認
   - `!vr test` : 対象ワークツリーで `pytest -x -q` を実行
   - `!vr review <name>` : レビューパック生成（SDK があれば自動判定まで実施）
   - `!vr task <name>` : `agy_orchestrator.py run-task` を実行
   - `!vr prompt <指示>` : `google-antigravity` SDK または `agy` CLI へ直接指示を送る

---

## 方式 C：従来方式（claude.ai + OpenCode + Google Drive 経由レビュー）

Claude（Opus）の使用量を減らすため、書く・見るを分ける。

| 段階 | 担当 | どこで |
|---|---|---|
| 1. 設計・指示書 `docs/tasks/<名前>.md` | Claude Opus 5.5 | claude.ai（このプロジェクト） |
| 2. 実装・テスト・ローカルのコミット | OpenCode ＋ DeepSeek V4.1 Flash **Free**（`opencode.json` で固定） | 自宅 PC の `~/valorant-replay` |
| 2'. 無料版が使えないとき | Claude Sonnet（Claude が下請けに出す） | claude.ai |
| 3. レビュー | Gemini 4 Argon（Google AI Pro の Gemini アプリ。API は使わない） | Claude がアプリ内ブラウザで Gemini を操作 |
| 4. 指摘の確認・マージ・push | Claude Opus 5.5（差分全体は読まない） | claude.ai |

## 手順

1. Claude に「◯◯を直したい」と言う → Claude が指示書を書く。
2. PC で：`opencode run "docs/tasks/<名前>.md を AGENTS.md の手順で実装して"`
   - 無料版が 403・上限・遅すぎるときは Claude に「Sonnet で」と言う。
3. 実装が終わると `G:/マイドライブ/valorant-replay/_review/review_<名前>.txt` ができる。
   Claude に「review_<名前> をレビューに出して」と言う → Claude がドライブでその .txt を「開く → Google ドキュメント」で変換し（Gemini はドライブの .txt を読めない。ドキュメントなら読める）、Gemini（Argon）に読ませて結果を受け取る。
   - OpenCode を Claude から動かすときは、PC で `opencode serve --port 4096` を開いておく。Claude はアプリ内ブラウザから http://127.0.0.1:4096 の API（`/session`・`prompt_async`・`shell`）で指示を出す。
4. 「必ず直すこと」があれば、Claude が指摘を指示書の末尾に足し、2 に戻る（OpenCode に直させる）。
5. 「マージしてよい」なら、Claude がテストを確かめて main に入れ、PROMPT.md（プロジェクトの要約も）を直す。

## 節約のきまり

- Colab の見張りは Claude にさせない。セルの最後に通知（終わった・失敗した）を出し、Claude は終わってからまとめるだけ。
- 1 つの話題は 1 つのチャット。区切りで `HANDOFF` を残して新しいチャットへ。
- ログは末尾とエラーだけ貼る。スクショは必要なときだけ。
- 放送の画像を見て原因を探すデバッグと設計の判断は Opus が担当（安いモデルだとやり直しが増える）。
