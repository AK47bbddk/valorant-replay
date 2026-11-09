# colab-notify

## 目的
Colab の長い解析（`vreplay batch` / `run` / `playbook`）の進み具合を人や Claude が見張らなくて済むよう、終わったとき・失敗したときにスマホへ通知する。

## 触ってよいファイル
- vreplay/notify.py（新規）
- vreplay/__main__.py
- vreplay/batch.py
- config/default.yaml
- tests/test_notify.py（新規）
- README.md（「通知」の節を足す）
- PROMPT.md（「非機能要件」に 2〜3 行）

## やること
1. `vreplay/notify.py` に `send(cfg, title: str, body: str, level: str = "info") -> bool` を作る。
   - 送り先は `cfg["notify"]["url"]`、無ければ環境変数 `VREPLAY_NOTIFY_URL`。どちらも無ければ何もせず False。
   - URL が `https://ntfy.sh/` などの ntfy なら：本文を POST、ヘッダー `Title`（日本語はそのまま UTF-8 で可。だめなら RFC 2047 にする）と `Priority`（error のとき 4、それ以外 3）。
   - URL に `discord.com/api/webhooks/` を含むなら：JSON `{"content": "**title**\nbody"}`（2000 文字で切る）を POST。
   - 通信は `urllib.request`、タイムアウト 10 秒。どんな例外でも本体を止めず、ログに 1 行出して False。
2. `vreplay/__main__.py`：`batch`・`run`・`playbook` の実行を包み、
   - 正常終了で `send(cfg, "vreplay batch 完了", "<完了数>/<対象数> マップ・失敗 <n>・<所要時間>")`（数は各コマンドが返す／ログから数えられる範囲で。取れなければ所要時間だけ）。
   - 例外で `send(cfg, "vreplay batch 失敗", "<例外の型>: <メッセージ> の末尾 300 文字", level="error")` を送ってから例外を上げ直す。
   - `--no-notify` で送らない。
3. `vreplay/batch.py`：同じマップで失敗が 3 回続いた・360p しか取れないマップが 3 つ続いた、のように「放っておくと全部失敗する」状態になったら途中でも 1 回だけ `level="error"` で送る（同じ理由では 1 回だけ）。
4. `config/default.yaml` に `notify: {url: null}` を足す（コメントで ntfy と Discord の例）。
5. テスト：`urllib.request.urlopen` を monkeypatch して、ntfy と Discord それぞれの URL・ヘッダー・本文、URL 無しで False、通信の例外で False と、`--no-notify` で呼ばれないことを確かめる。

## 通すテスト
- `python -m pytest -x -q tests/test_notify.py`
- `python -m pytest -x -q tests/test_batch.py tests/test_pot_server.py`（あれば）

## 終わりの条件
- 通知の URL を入れずに今までどおり動く（何も送らない・止まらない）。
- ntfy の URL を入れると、batch の終わりと失敗でスマホに 1 通ずつ届く。

## 質問
