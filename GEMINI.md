# valorant-replay Antigravity (`agy`) ワークスペースルール

このリポジトリは Google Antigravity (`agy`) をオーケストレーターとして開発・検証・レビューを一貫して回す。
既存の `AGENTS.md` / `docs/WORKFLOW.md`（OpenCode 運用）とも互換性を保つこと。

## 1. オーケストレーションの役割分担（デスクトップ PC 一元管理・Swarm 運用）

- **単一正本（Single Source of Truth）**: コード編集・テスト・ブランチ操作は **自宅デスクトップ PC のみ** で行う。ノート PC（SSH）やスマホ（Remote Control / Discord Bot）はデスクトップ PC を操作するクライアントとして使う。
- **1 回の起動 = 1 タスク = 1 ブランチ（`task/<名前>`）** を原則とする。

| 役割 | 担当 | 主な責務 | してはいけないこと |
|---|---|---|---|
| **司令塔 (Orchestrator)** | **Antigravity (`Gemini 4 Neon`)** | `docs/BACKLOG.md` からのタスク選定、仕様確認（`PROMPT.md` は必要な節だけ読む）、指示書作成、Swarm 割り振り、`task/<名前>` の PR 作成 | ユーザー確認なしの `main` マージ・`main` への直接 `git push` |
| **実装担当 (Implementer)** | **Swarm ワーカー**（軽量・高速モデル）または **OpenCode**（OpenRouter 枠） | `workspace: "share"`（独立 worktree）と `task/<名前>` ブランチで指示書の「触ってよいファイル」のみ実装し、`python -m pytest -x -q <対象テスト>` を通してコミット | `main` の変更、`push`、指示書外の編集 |
| **レビュー担当 (Reviewer)** | **Swarm の別モデルワーカー**（例：Claude Sonnet 等、実装と異なるモデル）または **Claude Code** | `git diff main...HEAD` と指示書のみを読み、`.agents/skills/vreplay-review` の基準（`## 判定` / `## 必ず直すこと` / `## できれば直すこと` / `## 確認したこと`）で厳格判定 | ファイル編集、コミット |
| **上位相談役 (Escalation)** | **Claude（Opus / Claude Code）** / **上位モデル** / **Muse** | 3 周ループで解決しない難題・仕様判断の「相談票」にテキストで回答 | リポジトリファイルの直接編集 |

### 自動改善ループと停止条件
1. 実装 → テスト → **実装と別モデルのレビュー** を **最大 3 周** まで回す。
2. 「マージしてよい」かつ `pytest` が通ったら、`task/<名前>` を push して **GitHub Pull Request を自動作成** し、ユーザーにチェックを依頼する（`main` へのマージはユーザーが GitHub 上で行う）。
3. **3 周しても「必ず直すこと」が消えない／同じ問題で詰まる／仕様判断が必要／触ってよいファイル外の変更が必要** になった時点で即停止し、指示書末尾に「相談票」を書いて上位モデルへエスカレーションする。

## 2. コーディング・安全ルール（`AGENTS.md` 準拠）

- **秘密情報の厳禁**: `NOTION_TOKEN`、YouTube cookies、各種 API キー（`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`）、`VREPLAY_NOTIFY_URL`、`DISCORD_BOT_TOKEN` を読む・書く・コミットしない。
- **テストデータ・知識ベースの保護**: `tests/fixtures/` の画像、および `config/kb/` 配下の YAML を指示書なしに書き換えない。
- **変更範囲の遵守**: 指示書の「触ってよいファイル」以外を変更しない。指示書にないリファクタリングや依存パッケージ追加を行わない。範囲が足りなければ作業を止め、指示書末尾の「質問・ブロッカー」に理由を書く。
- **テストの誠実性**: テストを削除したりアサーションを緩めたりして通さない。`pytest` がネットワークなしで通る状態を維持する。
- **ドキュメント同期**: 仕様が変わったら `PROMPT.md` の該当の節と `README.md` も合わせて更新する。
- **コミットとマージ**:
  - コミットコマンド: `git -c user.name="AK47bbddk" -c user.email="aokki123@gmail.com" commit -m "<日本語で何を変えたか>"`
  - レビューで「マージしてよい」が出るまで `task/<名前>` の PR 作成は行わない。また、ユーザー確認なしで `main` へのマージおよび `main` への直接 `git push` は行わない。

## 3. ドメイン固有の不変条件（`vreplay` ピットフォール）

- **ミニマップ・座標系**:
  - `valorant-api` のコールアウト座標変換（`x` と `y` の入れ替え、および Sunset 用の `tactics.CALLOUT_FIX`）を壊さない。
  - TactiVal 座標変換（`tactival.MAP_ROT` と `exact_calib()`）の正規化 0..1 座標とボード座標（1920x1080 内 `(448, 28)` に 1024x1024 配置）を取り違えない。
- **ウェアハウス（SQLite）**:
  - Google Drive 上（`/content/drive` 配下）の `warehouse.sqlite3` では WAL モードを使わない。
- **部分失敗の許容**:
  - 通知（`vreplay/notify.py`）や LLM 要約（`vreplay/llm.py`）の通信失敗・タイムアウトでパイプライン本体を停止させない。
  - 動画解析の一部が失敗しても、VLR 由来のメタデータ等で出力を継続できる構造を維持する。
