# 開発の回し方（デスクトップ PC 一元管理・Antigravity Neon + Swarm 運用）

## 0. ファイル散乱を防ぐ 3 つの鉄則

1. **作業場所は「自宅デスクトップ PC」1 台に完全固定**
   - コード編集・テスト・ブランチ操作はすべてデスクトップ PC の `C:\Users\Aokki\valorant-replay` で行う。
   - ノート PC（SSH / Remote-SSH）やスマホ（Antigravity Remote Control / Discord Bot）はデスクトップ PC を操作するクライアントに徹し、ノート PC のローカルで直接編集・コミットしない。
2. **1 タスク = 1 指示書 = 1 ブランチ**
   - 作業は必ず `docs/tasks/<名前>.md`（`_TEMPLATE.md` から作成）を起点にし、「触ってよいファイル」を限定する。
   - マージ済みのブランチ・worktree は即座に削除し、`main` だけを常に綺麗に保つ。
3. **リポジトリを直接触るのは Antigravity（またはデスクトップ上 CLI）のみ**
   - 上位相談（Claude / Muse 等）はテキストの「相談票」ベースで行い、複数 AI が同時にファイルを書き換える競合を防ぐ。

---

## 1. 役割分担とモデル構成

| 役割 | 担当 | やること | してはいけないこと |
|---|---|---|---|
| **司令塔 (Orchestrator)** | **Antigravity (`Gemini 4 Neon`)** | 要望から指示書 `docs/tasks/<名前>.md` を作成、Swarm で実装・レビューを割り振り、完了レポートをまとめる | 確認なしの `main` マージ・`git push` |
| **実装担当 (Implementer)** | **Swarm ワーカー**（軽量・高速モデル）または **OpenCode**（OpenRouter 枠） | `workspace: "share"` / ブランチ `task/<名前>` で実装、`python -m pytest -x -q` を通してコミット | `main` 変更、`push`、指示書外の編集 |
| **レビュー担当 (Reviewer)** | **Swarm の別モデルワーカー**（例：Claude Sonnet 等、実装とは別モデル）または **Claude Code** | `git diff main...HEAD` と指示書を読み、`.agents/skills/vreplay-review` の形式で判定 | ファイル編集、コミット |
| **上位相談役 (Escalation)** | **Claude（Opus / Claude Code）** / **上位モデル** / **Muse** | 詰まったときの「相談票」に回答し、設計・仕様判断を返す | リポジトリの直接編集 |

---

## 2. 標準の開発手順（1 サイクル：バックログ → PR 自動作成 → 人間はチェックのみ）

1. **開始前確認**：`main` が最新かつクリーンであることを確認（`git status -sb`）。
2. **タスク選定・指示書作成**：
   - `docs/BACKLOG.md` から最優先の `TODO` を 1 つ選び、対応する `docs/tasks/<名前>.md`（未作成なら `_TEMPLATE.md` から作成）を準備する。
   - `/teamwork` や `/goal` による自律実行時は、人間の事前 GO なしで既存指示書・バックログから着手してよい（外部 DB スキーマ変更など破壊的変更のみ事前確認）。
3. **Swarm 実行（実装 → テスト → 別モデルレビュー、最大 3 周）**：
   - 実装ワーカーが `task/<名前>` で実装・`pytest` 実行・コミット。
   - 別モデルのレビューワーカーが差分を検査。
   - 判定が「直してからマージ」なら「必ず直すこと」だけを実装ワーカーに戻す（最大 3 周）。
   - ※単発の小修正は Swarm ではなく通常のサブエージェント（`invoke_subagent`）や OpenCode を使いクォータを節約する。
4. **詰まったら自動停止 → 相談票**：
   - 以下のいずれかでループを止め、指示書末尾に「相談票」を出力する：
     - 3 周しても「必ず直すこと」が消えない／同じ問題で堂々巡りする
     - 仕様や定義の判断が必要（エリア境界・スモーク分類定義など）
     - 「触ってよいファイル」の外を変える必要がある
   - 相談票を Claude（Opus / Claude Code）や Muse に渡し、回答を指示書へ反映して再開する。
5. **PR 自動作成 → 人間はチェック＆マージのみ**：
   - レビュー判定「マージしてよい」かつローカル `pytest` が通ったら、`task/<名前>` ブランチを `git push -u origin task/<名前>` し、`gh pr create` で **GitHub Pull Request を自動作成** する（`.github/pull_request_template.md` 準拠）。
   - GitHub Actions（CI）の `pytest` が通ったことを人間が確認し、GitHub 上で **Merge** を押す（エージェントは `main` への直接マージを行わない）。
   - `docs/BACKLOG.md` のステータスを更新し、仕様変更がある場合は `PROMPT.md` 該当節と `README.md` を同期する。
   - マージ済みの作業ブランチ・worktree は削除して `main` をクリーンに保つ。

---

## 3. 3 台デバイスからの接続・運用

### 自宅デスクトップ PC（母艦）
- Antigravity の `Settings → App Settings` で `Keep computer awake` と `Run in background` を ON にする。
- Windows 電源設定で電源接続時のスリープを「なし」にする。

### ノート PC（外出先での開発）
- SSH（構築済み）でデスクトップ PC に接続し、Remote-SSH またはターミナルから直接デスクトップ PC の `C:\Users\Aokki\valorant-replay` を操作する（詳細は `.agents/skills/vreplay-remote-sync/SKILL.md`）。

### スマホ（進捗確認・指示出し）
- **Antigravity Remote Control** でデスクトップ PC のセッションへ接続する。
  - オフラインと表示される場合は、デスクトップ側のスリープ状態・Antigravity アプリの再起動・アカウント一致を確認する。
- **代替手段（Discord Bot）**：デスクトップ PC で `python tools/discord_bot.py` を常駐させておけば、Discord から以下のコマンドで操作可能：
  - `!vr status` : ブランチ・`git worktree list`・チケット一覧を確認
  - `!vr test` : `pytest -x -q` を実行
  - `!vr review <name>` : レビューパック生成

---

## 4. クイックコマンド

```powershell
# 状態確認
git status -sb
python tools/agy_orchestrator.py status

# タスク指示書の雛形作成
python tools/agy_orchestrator.py new-task <name> --files vreplay/xxx.py tests/test_xxx.py

# レビューパック生成
python tools/review_pack.py main..HEAD --task docs/tasks/<name>.md --out _work/review_<name>.txt

# マージ後のブランチ掃除
git branch -d task/<name>
```
