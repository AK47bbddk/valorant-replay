# コーディング担当（Antigravity Subagent / OpenCode）への約束

このリポジトリは **Antigravity (`agy`) オーケストレーション**（`GEMINI.md` および `.agents/skills/`）を軸に、指示書作成 → 実装・テスト → レビュー → マージで回す（従来の OpenCode 運用とも互換）。手順の全体は `docs/WORKFLOW.md`。

- **ワークスペースルール**: `GEMINI.md`
- **スキル**:
  - `.agents/skills/vreplay-orchestrator/SKILL.md`（全体の進行・タスク投入）
  - `.agents/skills/vreplay-review/SKILL.md`（差分レビュー）
  - `.agents/skills/vreplay-remote-sync/SKILL.md`（ノート PC / 自宅 PC / Discord Bot 連携）

## やること

1. 渡された指示書 `docs/tasks/<名前>.md` を読む。仕様の詳細が要るときだけ `PROMPT.md` の該当の節を読む（全部は読まない）。
2. ブランチ `task/<名前>` を切る（`git switch -c task/<名前>`）。
3. 指示書の「触ってよいファイル」だけを変える。足りなければ止めて、理由を `docs/tasks/<名前>.md` の末尾の「質問」に書く。
4. 指示書の「通すテスト」を実行する：`python -m pytest -x -q <テスト>`。通るまで直す。テストを消したり緩めたりして通さない。
5. 仕様が変わったら `PROMPT.md` の該当の節と README も直す。
6. コミット：`git -c user.name="AK47bbddk" -c user.email="aokki123@gmail.com" commit -m "<日本語で何を変えたか>"`。push はしない（main にも入れない）。
7. レビュー用のファイルを作る：
   `python tools/review_pack.py main..HEAD --task docs/tasks/<名前>.md --out "G:/マイドライブ/valorant-replay/_review/review_<名前>.txt"`
8. 最後に 3 行で報告：変えたファイル・テストの結果・迷ったところ。

## しないこと

- 秘密情報（Notion トークン・cookies・API キー）を読む・書く・コミットする。
- `tests/fixtures/` の画像、`config/kb/` の YAML を勝手に書き換える。
- 指示書に無いリファクタリング、依存パッケージの追加。
