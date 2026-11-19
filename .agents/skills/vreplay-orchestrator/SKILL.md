---
name: vreplay-orchestrator
description: >-
  Orchestrates feature development, bug fixes, subagent/swarm implementation,
  automated pytest verification, cross-model code review, and escalation in the
  valorant-replay repository. Use when planning, dispatching, or managing
  tasks/tickets for valorant-replay.
---

# valorant-replay Orchestrator Workflow (Neon + Swarm)

`valorant-replay` において、自宅デスクトップ PC を単一正本とし、**指示書作成 → 実装・テスト → 別モデルレビュー（最大 3 周）→ 相談票エスカレーション → 人間確認・マージ** を一元管理する手順。

## Step 1: タスク指示書の準備 (Task Spec)

1. `docs/tasks/_TEMPLATE.md` を元に `docs/tasks/<name>.md` を作成する。
2. 仕様確認のために `PROMPT.md` の**該当する節だけ**を読む（全文は読まない）。
3. 「触ってよいファイル」「変更内容」「通すテスト」「受け入れ条件」を明記し、ユーザーの GO を確認する。

## Step 2: 実装と別モデルレビューのループ (Max 3 Rounds)

規模に応じて使い分ける：

- **標準（複数モジュール／精度検証を伴うタスク）— Swarm 運用**：
  - `swarm` の `add` で **実装ワーカー**（`workspace: "share"`、高速モデル）と **レビューワーカー**（`workspace: "inherit"`、**実装とは別のモデル** 例：Claude Sonnet 等）を起動する。
  - 実装ワーカーはブランチ `task/<name>` で「触ってよいファイル」のみを変更し、`python -m pytest -x -q <対象テスト>` を通してコミットする。
  - レビューワーカーは `git diff main...HEAD` と指示書を読み、`vreplay-review` 形式で判定する（コード編集禁止）。
- **軽量（単発の小修正）— サブエージェント / OpenCode**：
  - `invoke_subagent`（`TypeName: "self"`, `Workspace: "share"`）または OpenCode（OpenRouter 枠）で実装し、別セッション／別モデルでレビューする。

### ループ規則
1. レビュー判定が **「直してからマージ」** なら、「必ず直すこと」だけを実装担当に戻す（**最大 3 周**）。
2. 以下のいずれかに該当したら **即停止** し、指示書末尾に「**相談票**」を書いて上位モデル（Claude Opus / Claude Code / Muse）へエスカレーションする：
   - 3 周しても「必ず直すこと」が残る／同じ問題で堂々巡りする
   - 仕様や定義の判断が必要（エリア境界・スモーク分類定義など）
   - 「触ってよいファイル」の外を変える必要がある
   - 環境都合でテストが動かない

## Step 3: 完了レポートと人間確認・マージ (Report & Merge)

1. レビュー判定が **「マージしてよい」** になったら、以下をまとめた完了レポートをユーザーに提示して停止する：
   - 変えたファイル一覧
   - 実行したテストと結果
   - レビュー結果・迷った点（および出力プレビューがあればそのパス）
2. ユーザーの最終承認を得てから `main` へマージし、必要に応じて `PROMPT.md` / `README.md` を同期する。
3. `git push` はユーザーが求めたときのみ実行し、マージ済みの作業ブランチ・worktree は削除して `main` をクリーンに保つ。
