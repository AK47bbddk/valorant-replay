---
name: vreplay-review
description: >-
  Performs senior Python code review on valorant-replay git diffs against task
  specifications using the review_pack.py format. Use when reviewing a task
  branch, worktree (such as vr-barriers), or commit range before merging.
---

# valorant-replay Code Review Workflow

`tools/review_pack.py` のフォーマットと同一の基準で、`valorant-replay` の変更差分をレビューする。

## 入力の取得

以下のいずれかを確認する：
1. `python tools/review_pack.py <range> --task docs/tasks/<name>.md` の出力。
2. `git diff -U5 <range> -- . ":(exclude)*.json" ":(exclude)tests/fixtures/*"` および対応する指示書（`docs/tasks/<name>.md` または `ticket_*.md`）。

## 重点チェック項目

1. **スコープ遵守**: 指示書の「触ってよいファイル」以外のファイルが変更されていないか。指示書にないリファクタリングや依存パッケージ追加がないか。
2. **秘密情報・保護対象**:
   - トークン・API キー・Webhook URL・cookies が含まれていないか。
   - `tests/fixtures/` の画像や `config/kb/` の YAML が無断で変更されていないか。
3. **部分失敗とエッジケース**:
   - 空リスト、`None`、OCR 未検出フレーム、動画取得失敗、ネットワークエラー時に例外で落ちずに適切にフォールバックするか。
   - Google Drive 上の SQLite（`warehouse.sqlite3`）で WAL を有効化していないか。
4. **ドメイン座標系の正確さ**:
   - `valorant-api` コールアウトの `x`/`y` 入れ替えや Sunset `CALLOUT_FIX` を壊していないか。
   - 正規化 0..1 座標とピクセル座標／TactiVal ボード座標（1920x1080 内 `(448, 28)` に 1024x1024）を取り違えていないか。
5. **テスト品質**:
   - 新規・変更仕様に対応するテストが追加されているか。
   - 既存テストを削除・スキップ・アサーション緩和して通していないか。
   - ネットワークなしで `pytest` が実行可能か。

## 出力フォーマット（必須）

好みの書き方への指摘は避け、以下の形式のみで出力する：

```markdown
## 判定
マージしてよい / 直してからマージ / やり直し のどれか 1 つ

## 必ず直すこと
- `ファイル:行` 問題 → 直し方（バグ・データの取り違え・例外で止まる・テストの抜け。無ければ「なし」）

## できれば直すこと
- 最大 3 つ

## 確認したこと
- 1 行ずつ、何を見て問題なしと判断したか
```

ルール：差分にないコードについて述べる場合は「（推測）」と明記すること。
