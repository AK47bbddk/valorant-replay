# 開発の回し方（モデルの分担）

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
   Claude に「review_<名前> をレビューに出して」と言う → Claude が Gemini（Argon）に読ませて結果を受け取る。
4. 「必ず直すこと」があれば、Claude が指摘を指示書の末尾に足し、2 に戻る（OpenCode に直させる）。
5. 「マージしてよい」なら、Claude がテストを確かめて main に入れ、PROMPT.md（プロジェクトの要約も）を直す。

## 節約のきまり

- Colab の見張りは Claude にさせない。セルの最後に通知（終わった・失敗した）を出し、Claude は終わってからまとめるだけ。
- 1 つの話題は 1 つのチャット。区切りで `HANDOFF` を残して新しいチャットへ。
- ログは末尾とエラーだけ貼る。スクショは必要なときだけ。
- 放送の画像を見て原因を探すデバッグと設計の判断は Opus が担当（安いモデルだとやり直しが増える）。
