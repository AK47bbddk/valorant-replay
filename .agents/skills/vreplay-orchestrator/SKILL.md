---
name: vreplay-orchestrator
description: >-
  Orchestrates feature development, bug fixes, subagent implementation,
  automated pytest verification, and code review in the valorant-replay
  repository. Use when planning, dispatching, or managing tasks/tickets for
  valorant-replay.
---

# valorant-replay Orchestrator Workflow

`valorant-replay` において、指示書作成 → 実装・テスト → レビュー → マージを Antigravity (`agy`) 上で一元管理する手順。

## Step 1: タスク指示書の準備 (Task Spec)

1. 新規タスクの場合、`docs/tasks/_TEMPLATE.md` をコピーして `docs/tasks/<name>.md` を作成する。
   - 既存チケット（デスクトップ PC 側の `ticket_*.md` など）がある場合はそれを読み込む。
2. 仕様確認のために `PROMPT.md` の**該当する節だけ**を読む（全文は読まない）。
3. 「触ってよいファイル」「変更内容」「通すテスト」を明確に定義する。

## Step 2: 実装の委任 (Implementation Dispatch)

状況に応じて以下のいずれかで実装担当に委任する：

- **パターン A（Plan / Execute モード）**:
  - プランを確定して実行モデル（Execution Model）に引き継ぎ、単一タスクを順番に実装・テストする。
- **パターン B（並列サブエージェント / Swarm）**:
  - 複数タスクを並列処理する場合、または親コンテキストを汚さず独立作業させる場合は、`invoke_subagent`（`TypeName: "self"`, `Model: "flash"`, `Workspace: "share"`）または `swarm`（`workspace: "share"`）を使う。
  - ブランチ `task/<name>` を切り、指示書の「触ってよいファイル」のみ編集して `python -m pytest -x -q <テスト>` を実行させる。
- **パターン C（CLI / Python SDK 自動実行）**:
  - スクリプト経由で一括実行・テスト・レビューパック生成・通知まで回す場合：
    ```powershell
    python tools/agy_orchestrator.py run-task <name> --engine agy
    ```

## Step 3: コードレビュー (Review Dispatch)

1. `vreplay-review` スキル（またはレビュー用サブエージェント、`TypeName: "research"` 等）を使い、`main..HEAD` の差分と指示書を突き合わせてレビューする。
2. または以下のコマンドでレビューパックを生成し、必要に応じて SDK 経由で自動レビューを実行する：
   ```powershell
   python tools/agy_orchestrator.py review <name> --range main..HEAD
   ```
3. 判定が「直してからマージ」または「やり直し」の場合は、指摘事項を指示書末尾の「レビュー結果・修正履歴」に追記し、Step 2 へ戻す。

## Step 4: マージ・ドキュメント同期・通知 (Merge & Notify)

1. レビュー判定が「マージしてよい」であることを確認する。
2. 仕様変更がある場合は `PROMPT.md` の該当節および `README.md` が更新されていることを確認する。
3. 対象テストが最終状態で通ることを確認したうえで、`main` へマージする（リモートへの `git push` はユーザーが求めたときのみ）。
4. 必要に応じて `vreplay.notify.send`（`notify.url` / `VREPLAY_NOTIFY_URL`）で完了通知を送る。
