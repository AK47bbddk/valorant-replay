---
name: vreplay-remote-sync
description: >-
  Guides remote development and orchestration between the laptop and desktop PC
  (including git worktrees like vr-barriers, SSH setup, and Discord Bot remote
  execution) for valorant-replay.
---

# valorant-replay リモート開発・同期ガイド

ノート PC（外出先）と自宅デスクトップ PC（`vr-barriers` ワークツリーや `ticket_barrier_v2.md` を含む作業環境）を連携させて開発するための手順。

## 1. ノート PC の SSH 鍵生成とデスクトップ PC への登録

1. ノート PC で `scripts/setup_remote_ssh.ps1` を実行し、`~/.ssh/id_ed25519` を生成して公開鍵をクリップボードにコピーする：
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup_remote_ssh.ps1
   ```
2. 表示・コピーされた公開鍵（`id_ed25519.pub`）を、自宅デスクトップ PC 側の `authorized_keys`（Windows OpenSSH Server の場合は `C:\Users\<User>\.ssh\authorized_keys` または管理者ユーザーなら `C:\ProgramData\ssh\administrators_authorized_keys`）へ追記する。
3. ノート PC の `~/.ssh/config` に自宅デスクトップ PC への接続設定（Tailscale / WireGuard IP 等）を追加し、`ssh desktop-pc` で接続できることを確認する。

## 2. デスクトップ PC 側ワークツリー（例：`vr-barriers`）の確認・作業

SSH 接続後、またはデスクトップ PC 上の Antigravity から対象ワークツリーを確認する：

```powershell
# ワークツリー一覧の確認
git worktree list

# vr-barriers ワークツリーの状態とチケット確認
python tools/agy_orchestrator.py status --worktree <vr-barriersのパス>

# テストの実行
python -m pytest -x -q

# タスクのオーケストレーション実行
python tools/agy_orchestrator.py run-task barrier_v2 --worktree <vr-barriersのパス>
```

## 3. 外出先からの Discord Bot 連携（`tools/discord_bot.py`）

自宅デスクトップ PC 側で `tools/discord_bot.py` を起動しておくと、外出先から Discord メッセージで進捗確認・テスト・レビュー・タスク実行が行える。

### 環境変数（リポジトリには書かない）
- `DISCORD_BOT_TOKEN`: Discord Bot トークン
- `DISCORD_ALLOWED_USER_IDS`: 操作を許可する自分の Discord ユーザー ID（カンマ区切り）
- `VREPLAY_WORKTREE_DIR`: 既定の作業ディレクトリ（例：`C:\Users\Aokki\valorant-replay` または `vr-barriers` ワークツリーのパス）

### 主なコマンド
- `!vr status` : ブランチ、`git worktree list`、直近コミット、タスク/チケット一覧を表示
- `!vr test [pytest引数]` : 対象ワークツリーで `pytest` を実行
- `!vr review <name> [range]` : 差分をまとめてレビューパック生成・判定
- `!vr task <name> [engine]` : `agy_orchestrator.py run-task` を実行
- `!vr prompt <指示文>` : `google-antigravity` SDK または `agy` CLI へ直接プロンプトを投げる
