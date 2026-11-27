# 開発バックログ（自律開発キュー）

このファイルは、`/teamwork` や `/goal` などの自律エージェントが **次に実装すべきタスクを上から順に拾うためのキュー** です。

## 運用ルール（エージェント向け）

1. ステータスが `TODO` のうち、**最も優先度（P0 → P1 → P2）が高いもの** を 1 つ選ぶ。
2. 選択時にステータスを `IN_PROGRESS` に更新する（または実装完了時に `PR_OPEN` へ更新する）。
3. 指示書（`docs/tasks/<id>.md`）がまだ無い場合は `docs/tasks/_TEMPLATE.md` から作成する。
4. ブランチ `task/<id>` を切り、実装・`pytest`・別モデルレビュー（最大 3 周）を行う。
5. レビューで「マージしてよい」が出たら `task/<id>` を `git push` し、`.github/pull_request_template.md` に沿って **GitHub Pull Request を自動作成** する（`main` への直接マージは禁止）。
6. PR 作成後、ステータスを `PR_OPEN`（PR 番号付き）に更新し、**ユーザーのチェックを待たずに次の `TODO` タスクへ進むか、コンテキストの競合を避けるため待機・報告する**。
   - 依存関係がない別タスクは別ブランチで先行着手してよい。
   - 同じファイルを触るタスクは前の PR がマージされるまで待つ。

---

## P0（最優先・直近で実装するタスク）

| ID | 状態 | タイトル | 指示書 | 主な対象ファイル | 完了条件 / 確認方法 |
|---|---|---|---|---|---|
| `agent-dot-accuracy` | `DONE` | ミニマップ・エージェント点検知の高精度化（2段階 MLP+2D CNN ＆ `dot_mask` カバー率改善） | [agent-dot-accuracy.md](tasks/agent-dot-accuracy.md) | `vreplay/agent_dots.py`, `vreplay/minimap.py`, `vreplay/tactics.py`, `vreplay/data/agent_dot_model.npz`, `tools/train_agent_dot.py` | PR #6 マージ済み（P=0.929, R=0.933, F1=0.931） |
| `smoke-detection-accuracy` | `PR_OPEN` | 定半径（`R≈9.0px@H=304`）ドームスモーク検知器とクリーンミニマップ構造事前知識による高精度化 | [smoke-detection-accuracy.md](tasks/smoke-detection-accuracy.md) | `vreplay/smoke_detector.py`, `vreplay/data/smoke_model.npz`, `tools/train_smoke_detector.py`, `vreplay/smokes.py`, `tests/test_smokes.py` | 7-fold LOMO-CV・ブラインド評価・オフライン `pytest` を満たすこと |

---

## P1（自律改善ループの基盤）

| ID | 状態 | タイトル | 指示書 | 主な対象ファイル | 完了条件 / 確認方法 |
|---|---|---|---|---|---|
| `pr-preview-artifacts` | `TODO` | PR レビュー用のサンプル出力（HTML/ミニマップ画像要約）自動生成スクリプト | 未作成（着手時に作成） | `tools/pr_preview.py`, `tests/test_pr_preview.py` | コード diff を読まなくても変更の見た目を PR 上で確認できるサマリーを生成できる |

---

## P2（精度・機能改善バックログ）

| ID | 状態 | タイトル | 指示書 | 主な対象ファイル | 完了条件 / 確認方法 |
|---|---|---|---|---|---|
| `opening-smoke-refine` | `TODO` | スモーク検知結果を使った初動分類（Fake / Split / Default）の精度改善 | 未作成 | `vreplay/opening_playbook.py`, `tests/test_opening_playbook.py` | `smoke-detection` マージ後、`eval_all` で初動分類スコアが向上すること |
| `playbook-v5-smoke-summary` | `TODO` | プレイブック（v5 / Notion）への初動スモーク傾向サマリー追加 | 未作成 | `vreplay/playbook_v5.py`, `vreplay/notion_playbook.py` | 既存テストが通り、スモークの有無で出力が壊れないこと |

---

## 完了済み（直近のアーカイブ）

| ID | 状態 | タイトル | PR / コミット |
|---|---|---|---|
| `backlog-pr-ci` | `DONE` | バックログ・PR 運用・CI 基盤の整備 | PR #2 (`bd2d887`) |
| `smoke-detection` | `DONE` | ミニマップからのスモーク位置検知と初動分類への連携 | PR #3 (`f00cd1f`) |
| `eval-all` | `DONE` | ブラインドデータセット統合評価ツール `tools/eval_all.py` の整備 | PR #4 (`960192e`) |
| `playable-minimap-region` | `DONE` | ミニマップ存在判定・プレイ可能床マスク導入による誤検知抑止とブラインド評価改善 | PR #5 (`2217e61`) |
