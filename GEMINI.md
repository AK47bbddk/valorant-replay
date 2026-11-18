# valorant-replay Antigravity (`agy`) ワークスペースルール

このリポジトリは Google Antigravity (`agy`) をオーケストレーターとして開発・検証・レビューを一貫して回す。
既存の `AGENTS.md` / `docs/WORKFLOW.md`（OpenCode 運用）とも互換性を保つこと。

## 1. オーケストレーションの役割分担

| 役割 | 担当 | 主な責務 |
|---|---|---|
| **Orchestrator** | Planning Model (`Argon`) | 仕様確認（`PROMPT.md` は必要な節だけ読む）、指示書 `docs/tasks/<名前>.md`（または `ticket_*.md`）の作成、サブエージェント/実行モデルへの指示出し、レビュー結果に基づくマージ判断 |
| **Implementer** | Execution Model / Subagent (`flash`) / OpenCode | ブランチ `task/<名前>`（またはワークツリー `vr-<名前>`）上で指示書の「触ってよいファイル」のみを変更し、`python -m pytest -x -q <対象テスト>` を通す |
| **Reviewer** | Review Subagent (`.agents/skills/vreplay-review`) | `tools/review_pack.py` の基準（`## 判定` / `## 必ず直すこと` / `## できれば直すこと` / `## 確認したこと`）に沿って差分を検査する |

## 2. コーディング・安全ルール（`AGENTS.md` 準拠）

- **秘密情報の厳禁**: `NOTION_TOKEN`、YouTube cookies、各種 API キー（`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`）、`VREPLAY_NOTIFY_URL`、`DISCORD_BOT_TOKEN` を読む・書く・コミットしない。
- **テストデータ・知識ベースの保護**: `tests/fixtures/` の画像、および `config/kb/` 配下の YAML を指示書なしに書き換えない。
- **変更範囲の遵守**: 指示書の「触ってよいファイル」以外を変更しない。指示書にないリファクタリングや依存パッケージ追加を行わない。範囲が足りなければ作業を止め、指示書末尾の「質問・ブロッカー」に理由を書く。
- **テストの誠実性**: テストを削除したりアサーションを緩めたりして通さない。`pytest` がネットワークなしで通る状態を維持する。
- **ドキュメント同期**: 仕様が変わったら `PROMPT.md` の該当の節と `README.md` も合わせて更新する。
- **コミットとマージ**:
  - コミットコマンド: `git -c user.name="AK47bbddk" -c user.email="aokki123@gmail.com" commit -m "<日本語で何を変えたか>"`
  - レビューで「マージしてよい」が出るまで `main` へのマージおよび `git push` は行わない。

## 3. ドメイン固有の不変条件（`vreplay` ピットフォール）

- **ミニマップ・座標系**:
  - `valorant-api` のコールアウト座標変換（`x` と `y` の入れ替え、および Sunset 用の `tactics.CALLOUT_FIX`）を壊さない。
  - TactiVal 座標変換（`tactival.MAP_ROT` と `exact_calib()`）の正規化 0..1 座標とボード座標（1920x1080 内 `(448, 28)` に 1024x1024 配置）を取り違えない。
- **ウェアハウス（SQLite）**:
  - Google Drive 上（`/content/drive` 配下）の `warehouse.sqlite3` では WAL モードを使わない。
- **部分失敗の許容**:
  - 通知（`vreplay/notify.py`）や LLM 要約（`vreplay/llm.py`）の通信失敗・タイムアウトでパイプライン本体を停止させない。
  - 動画解析の一部が失敗しても、VLR 由来のメタデータ等で出力を継続できる構造を維持する。
