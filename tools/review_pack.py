"""Gemini（アプリ）に貼るレビュー依頼文を作る。

使い方: python tools/review_pack.py [範囲 (既定 HEAD~1..HEAD)] [--task docs/tasks/xxx.md] [--out ファイル]
--out を省くと標準出力に出す。ファイルの先頭にレビューの依頼文が入るので、Gemini には「ドライブの review_<名前> を開いて中の指示どおりに」とだけ送ればよい。
"""
import subprocess, sys

MAX = 60000  # Gemini アプリの入力に収める目安（文字）

def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout

def main():
    args = sys.argv[1:]
    task, out = "", None
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]
    if "--task" in args:
        i = args.index("--task"); task = open(args[i + 1], encoding="utf-8").read(); del args[i:i + 2]
    rng = args[0] if args else "HEAD~1..HEAD"
    log = git("log", "--format=- %h %s", rng)
    diff = git("diff", "-U5", rng, "--", ".", ":(exclude)*.json", ":(exclude)tests/fixtures/*")
    cut = ""
    if len(diff) > MAX:
        diff, cut = diff[:MAX], f"\n（差分が長いので先頭 {MAX} 文字だけ。残りは別に送る）\n"
    text = (f"""あなたは Python のシニアレビュアーです。VALORANT の試合動画を解析するリポジトリ valorant-replay の変更をレビューしてください。

出力の形（日本語・この形だけ）:
## 判定
マージしてよい / 直してからマージ / やり直し のどれか 1 つ
## 必ず直すこと
- `ファイル:行` 問題 → 直し方（バグ・データの取り違え・例外で止まる・テストの抜け。無ければ「なし」）
## できれば直すこと
- 最大 3 つ
## 確認したこと
- 1 行ずつ、何を見て問題なしと判断したか

ルール: 差分に無いコードについては推測と書く。好みの書き方の指摘はしない。

{"## 作業指示書" + chr(10) + task if task else ""}
## コミット
{log}
## 差分{cut}
```diff
{diff}```""")
    if out:
        from pathlib import Path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        print(f"書き出し: {out}（{len(text)} 文字）")
    else:
        print(text)

main()
