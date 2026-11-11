# agents-precision（ミニマップのエージェント照合：間違った検出を出さない＋確かめる道具を足す）

※ Claude が 2026-11-11 に作成。青木の指摘「Colab の解析でミニマップのキャラ読み取りがうまくいっておらず、解析の精度が落ちている」への 1 本目。

## 目的
`agents.match` が、エージェントのいない場所（床だけ・塗りつぶしの丸・スキルの印）や別の顔に名前を付けている。
この間違いが追跡（tracks）→ 配置（setups）→ 手・プレイブックまで流れている。
まず「分からないものは出さない」に寄せ、あわせて実データで照合の良し悪しを数えられる道具（フレームの書き出しと再生）を足す。
見落とし（再現率）を上げる作業は、この道具で数字が出てから 2 本目の指示書で行う。

## 分かっていること（Claude が確認済み）
Drive の `_tactics/agents_check.jpg` 3 枚（2026-11-06 と 10-26 の解析ぶん、計 30 タイル）を見た結果：

| 見えた間違い | 例（タイルの表示） | コード上の理由 |
|---|---|---|
| 床だけの場所に名前が付く | `Fade 0.52` `Sage 0.49` `Chamber 0.57` `Neon 0.51` | しきい値 `thr=0.45` が低い。顔の中身があるかを見ていない |
| 暗い赤の塗りつぶしの丸に名前が付く | `Omen 0.50` `Chamber 0.48` | `_ring_kernel` は「輪 − 0.5×内側」なので、塗りつぶしの丸でも 0.5 になり、候補のしきい値 0.15 を通る |
| スキルの印・薄くなったアイコンに名前が付く | `Sage 0.47` `Phoenix 0.49` | 同上（顔の中身を見ていない） |
| きれいなアイコンに別のエージェントの名前が付く | 赤い髪の顔に `Sova 0.56`、黒髪の顔に `Neon 0.65` | ハンガリアン法は余った枠を余った候補に入れる。その候補では別のエージェントの方が似ていても確かめていない |
| 重なったアイコン（スポーン・黄色い縁の観戦中の選手のそば）に高い点で名前が付く | `Sage 0.72` `Fade 0.70` `Omen 0.70` `Omen 0.60` | 重なりを区別していない |

- 30 タイルのうち、点が 0.58 未満のものは 11 個で、そのうち 10 個が上の間違いだった（正しそうなのは `Yoru 0.51` の 1 個）。
- `agents_check.jpg` は「いちばん多く見つかった 1 枚」だけなので、間違いの多い 1 枚が選ばれやすい。全体の割合は分からない（だから下の 4・5 で数えられるようにする）。
- 以上は確認用の画像とコードを読んだ結果。実データで照合を動かして確かめてはいない（しきい値の数字は出発点）。

## 触ってよいファイル
- vreplay/agents.py
- vreplay/tactics.py（`_agent_positions`・`_agent_tracks` と、その 2 つを呼ぶ所だけ）
- vreplay/track.py（`best_chain` の `min_score` の既定値だけ）
- tools/agents_eval.py（新規）
- config/default.yaml（`tactics:` の下）
- tests/test_agents.py、tests/test_agents_eval.py（新規）
- PROMPT.md（「ミニマップの配置とキルの正確さ」の「エージェントの照合」）

## やること

### 1. `agents.match` に「出さない」ための確かめを足す
引数を足す：`margin: float = 0.06`、`min_face_std: float = 12.0`、`crowded_margin: float = 0.10`、`strict: bool = True`、`debug: list | None = None`。
`thr` の既定値を 0.45 → **0.58** にする。`strict=False` のときは下の a〜d を行わない（今までと同じ動き）。

候補（`find_icons` の結果）ごとに、割り当ての前に次を計算する：

- a. **塗りつぶしの丸を捨てる**：候補の色（warm / cool）のマスクで、中心から半径 0.28D の内側の画素のうち同じ色の割合が 0.5 以上なら捨てる（理由 `filled`）。黄色の縁の候補には使わない。
- b. **顔の中身が無い場所を捨てる**：グレースケールで、中心から半径 0.30D の内側の標準偏差が `min_face_std` 未満なら捨てる（理由 `flat`）。
- c. **重なりの印**：ほかの候補（色は問わない）の中心が 1.0D 以内にあれば `crowded=True`。

割り当て（`_assign`）の後、組 (候補 i, 枠 (t, a)) ごとに：

- d. **差の確かめ**：`second` = 候補 i で、a 以外のエージェントの顔の点の最大（縁がチーム色なら同じチーム t の中、黄色の縁なら 10 人全員の中）。
  `S[i, j] - second < margin`（`crowded` のときは `crowded_margin`）なら捨てる（理由 `margin`）。
  同じチームに照合できるエージェントが 1 人しかいないときは、この確かめを飛ばす。
- 今までの `S[i, j] >= thr` で落ちるものは理由 `thr`、生存人数の上限で落ちるものは理由 `cap`。

返り値の各要素に `"overlap": bool`（＝`crowded`）と `"margin": float`（小数 2 桁）を足す。今あるキーは変えない。
`debug` にリストが渡されたら、捨てた候補を `{"x", "y", "color", "agent"（いちばん似ていたエージェント）, "score", "second", "ring", "reason"}` で足す。

`estimate_icon_frac` の中の `match(..., thr=0.0, ...)` には `strict=False` を付ける（大きさの見積もりは今までどおり）。

### 2. `consensus` と `track.best_chain` をそろえる
- `consensus`：`overlap` が真の検出は、`strong`（0.6 以上）だけでは残さない。ほかのフレームの近くの検出（`sup >= 1`）があるときだけ残す。
- `track.best_chain` の `min_score` の既定値を 0.45 → 0.58。

### 3. 設定から変えられるようにする
`config/default.yaml` の `tactics:` に足す（コメント付き）：

```yaml
  # エージェントの照合の厳しさ。数字は tools/agents_eval.py で実データを数えてから直す
  agent_match: {thr: 0.58, margin: 0.06, crowded_margin: 0.10, min_face_std: 12}
  # 確かめ用：1 マップにつき何枚のミニマップを _tactics/agents_dump/ に書き出すか（0 で書き出さない）
  agents_dump: 0
```

`_agent_positions`・`_agent_tracks` は `cfg["tactics"]["agent_match"]`（無ければ既定値）を `AG.match` に渡す。

### 4. フレームの書き出し（`tactics.agents_dump` が 1 以上のとき）
`_agent_tracks` で、照合する全フレーム（`todo`）から、ラウンドと時刻にまんべんなく散るよう等間隔に N 枚選び、`<out_dir>/_tactics/agents_dump/` に書く。
`_agent_tracks` に `out_dir` が渡っていなければ引数を足す。

- `rNN_tSSS.png`：`AG.match` に渡したミニマップそのまま（PNG・縮めない）。SSS はラウンド開始からの秒（3 桁、負なら 000）。
- `icons/<エージェント名>.png`：照合に使った 64×64 のアイコン。
- `mask.png`・`background.png`：`AG.match` に渡したもの（無ければ書かない）。
- `meta.json`：`{"teams": {"1": [...], "2": [...]}, "icon_frac": 0.058, "params": {...}, "frames": [{"file", "round", "t", "color_team", "alive", "found": [...], "rejected": [...]}]}`。
  `found` は `AG.match` の返り値、`rejected` は `debug` の中身。選んだフレームはキャッシュを使わず、`debug` を付けてもう一度照合する。
- `sheet.jpg`：選んだフレームを 2 倍に拡大して並べた確認用の画像。残した検出は緑の丸＋「名前 点」、捨てた候補は赤の ×＋理由（`filled` / `flat` / `margin` / `thr` / `cap`）。1 枚に 12 フレームまで、あふれたら `sheet_2.jpg`…。

ログに 1 行足す：`エージェントの照合：候補 n・残した n・捨てた（filled n / flat n / margin n / thr n / cap n）`（全フレームの合計。数えるためだけに `debug` を使ってよい）。

### 5. `tools/agents_eval.py`（新規。ネットワークを使わない）
- `replay(dump_dir, **params) -> list[dict]`：`meta.json`・PNG・`icons/`・`mask.png`・`background.png` を読み、フレームごとに `AG.match` をやり直した結果を返す。
- `score(results, labels) -> dict`：`labels` は `{"rNN_tSSS.png": [{"agent", "team", "x", "y", "overlap": bool}]}`。
  検出は、同じ（team, agent）のラベルが 0.6D 以内にあれば正解。
  返り値：`precision`（正解 ÷ 残した検出）・`recall`（正解 ÷ ラベル）・`wrong_id`（0.6D 以内にラベルはあるが別人）・`ghost`（0.6D 以内にラベルが無い）を、全体と `overlap` の真偽別に。
- コマンド：
  - `python tools/agents_eval.py <dump_dir>`：ラベル無しなら、残した数と理由ごとの捨てた数だけ表示。`labels.json` があれば上の数字を表示。
  - `--init-labels`：今の検出から `labels.json` の下書きを作る（人が間違いを直して使う）。すでにあれば上書きしない。
  - `--thr --margin --crowded-margin --min-face-std`：数字を変えてやり直す。

### 6. テスト
`tests/test_agents.py` に足す（今あるテストは変えずに通す）：
1. 暖色の塗りつぶしの丸（直径 D）と、暖色の輪だけで中が一様な丸を置いたミニマップ → 検出 0 個。`debug` に `filled`・`flat` が入る。
2. チーム 1 が `["Sova", "Jett"]`。Jett のアイコンを 1 つ、離れた所に「Jett 0.6＋Sova 0.4 を混ぜた顔」を暖色の縁で置く → 返り値に Sova が無い（理由 `margin`）。Jett は正しい位置で 1 つ。
3. 2 つのアイコンを中心間 0.7D で重ねる → 返り値の検出は `overlap` が真で、名前が付くなら正しい名前（別人の名前は付かない）。
4. `strict=False` なら 1・2 の候補が今までどおり返る。
5. `consensus`：`overlap` が真で点 0.7 の検出が 1 フレームだけ → 残らない。2 フレームで近くにある → 残る。

`tests/test_agents_eval.py`（新規）：
6. 合成のミニマップ 3 枚で `tactics.agents_dump: 3` の書き出しを作り（`_agent_tracks` を直接呼ぶか、書き出しの関数を分けて呼ぶ）、ファイル一式ができること。
7. `replay` の結果が書き出し時の `found` と同じ（エージェント・チーム・位置）。合成の正解ラベルで `score` の precision・recall が 1.0。1 人ぶん別人にしたラベルでは `wrong_id` が 1。

## 通すテスト
- `python -m pytest -x -q tests/test_agents.py tests/test_agents_eval.py tests/test_track.py tests/test_tactics.py`
- `python -m pytest -q`（全体）

## 終わりの条件
- 上のテストがすべて通る。今あるテストを消したり緩めたりしていない。
- `tactics.agents_dump: 0`（既定）では書き出しが一切起きず、処理時間が変わらない。
- Colab で 1 マップを `tactics.agents_dump: 40`・`--force` で解析し直すと `_tactics/agents_dump/sheet.jpg` ができ、床だけの場所・塗りつぶしの丸・スキルの印に緑の丸（残した検出）が付いていない。
- 同じ書き出しに人が直した `labels.json` を置いて `python tools/agents_eval.py <dump_dir>` を実行すると、重なっていないアイコンの precision が 95% 以上（届かなければ数字を報告する。しきい値を勝手に動かして合わせない）。
- recall の数字を報告に書く（目標は置かない。2 本目の指示書の出発点にする）。

## しないこと
- アイコンの取り方（`load_icons`）・候補の拾い方（`find_icons` のしきい値・NMS）・追跡の鎖の作り方は変えない（2 本目で扱う）。
- `tests/fixtures/` の今ある画像を書き換えない。
- 実データが無いまま、しきい値を合成テストに合わせて動かさない。

## 質問
（実装担当が書く。無ければ空のまま）
