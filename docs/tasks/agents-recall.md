# agents-recall（ミニマップのエージェント照合 2 本目：precision 95% を保ったまま見落としを減らす）

※ Claude が 2026-11-12 に作成。[agents-precision](agents-precision.md)（1 本目）の検証結果を受けた 2 本目。
ブランチは `task/agents-precision` から `task/agents-recall` を切る（1 本目はまだ main に入っていないため）。

## 目的
1 本目で「間違った名前を出さない」はできた（precision 97.8%）。代わりに見落としが多い（recall 48.6%、重なりの無いアイコンでは 18.8%）。
本物のアイコンを捨てている確かめ（filled・margin・thr）を直し、候補が出ない場所の拾い方を足して recall を上げる。
**precision 95% 以上は落とさない**。数字はすべて 1 本目と同じ dump＋labels.json で比べる。

## 分かっていること

### 1 本目の検証の数字（LOUD vs Vitality・Sunset・40 フレーム・ラベル 276）
| | precision | recall |
|---|---|---|
| all | 97.8% | 48.6% |
| overlap | 98.2% | 78.3% |
| clear | 96.4% | 18.8% |

- wrong_id 1・ghost 2。目で見て、床・塗りつぶしの丸・スキルの印への誤検出は無し。
- 見落とし 142 件：候補なし 54／thr 42／filled 22（うち本物の Chamber 14）／margin 13（うち本物の Neon 10）／flat・cap 11。
- main との recall の比べ（2026-11-12、窓口が同じ dump＋labels・同じ score で計測。VM 環境）: main は precision [all] 81.0%／[overlap] 83.8%／[clear] 72.2%、recall [all] 64.9%／[overlap] 82.6%／[clear] 47.1%、wrong_id 4・ghost 38（検出 221）。つまり 1 本目で増えた見落としは特に clear（47.1% → 18.8〜21.7%）で、overlap の recall 低下は小さい（82.6% → 78.3%）。main の overlap 組分けは、main が内部で使うのと同じ候補集合から 1 本目と同じ定義で復元した。

### コードを読んで立てた仮説（Claude は実データで確かめていない。段階 0・1 で確かめる）
| 見落とし | 疑っている理由（コード上） |
|---|---|
| clear が 18.8% と低い | `score` は、検出の組分けに**検出側**の `overlap`、ラベルの組分けに**ラベル側**の `overlap` を使っている。`--init-labels` の下書きでは `overlap` は検出の値で、人が足した見落としのラベルは `overlap: false` のままになりやすい。すると「見落とし」が clear に寄り、clear の recall が実際より低く出る |
| 候補なし 54 | 背景（`tactics.py` 432 行、マップ全体から最大 40 枚の中央値）に、毎ラウンドのスポーンや長く守る定位置のアイコンが残る。`find_icons` はその輪の反応を差し引くので、**動かないアイコンほど候補にならない**。ほかに `nms=0.8` での抑え込み・`per_color=12` の上限・床マスク・縁 0.5D の除外・縁の色の範囲外（彩度が低い）も考えられる |
| filled 22（Chamber 14） | `WARM` の色相 0〜12 は、肌・茶色の髪や服にもかかる。Chamber の顔そのものが中心 0.28D で「暖色 50% 以上」になり、塗りつぶしの丸と同じに見えている（暖色のチームの Chamber だけで起きているはず） |
| margin 13（Neon 10） | 2 番目の点の相手（同じチームの似た顔）が、**同じフレームの別の場所ですでにはっきり見つかっている**のに、差の確かめではその相手も数に入れている。1 チームに同じエージェントは 1 人なので、その相手は 2 番目の候補にならない |
| thr 42 | 0.58 の一本の線。本物の点と偽物の点がどこで分かれるかはまだ数えていない |

## 触ってよいファイル
- vreplay/agents.py
- vreplay/tactics.py（`_agent_match_params` と、`AG.match` を呼ぶ 3 か所に新しい設定を渡す所だけ）
- config/default.yaml（`tactics.agent_match` の下）
- tools/agents_eval.py
- tests/test_agents.py、tests/test_agents_eval.py
- PROMPT.md（「ミニマップの配置とキルの正確さ」の「エージェントの照合」）

## 進め方の決まり
- 直すことはすべて **設定のスイッチ**（`tactics.agent_match` の下）で入り切りできるようにし、既定は**切**で作る。全部切なら 1 本目と出力が 1 つも変わらないこと（段階 0 の 3 で確かめる）。
- 1 つ直すごとに `tools/agents_eval.py` で数え、下の「報告の表」に 1 行足す。既定を入にするのは段階 6 で、数字を見てから。
- しきい値は粗い刻み（下に書いた候補の値だけ）で試す。ラベルに合わせて細かく刻まない（40 フレームしかないので、細かく合わせると別の試合で崩れる）。
- labels.json は直さない。ラベルの間違いに気づいたら「質問」に書く（`file`・位置・どう違うか）。

## やること

### 段階 0：数える道具を足す（照合の動きは変えない）
`tools/agents_eval.py` に足す：

1. **重なりの組分けを位置から決め直す**（`--group geom`、既定は `geom`。`--group flag` で 1 本目と同じ組分け）
   - ラベル：同じフレームのほかのラベルの中心が 1.0D 以内にあれば overlap。
   - 検出：正解なら結んだラベルの組。wrong_id なら近くのラベルの組。ghost なら、1.0D 以内にラベルがあれば overlap、無ければ clear。
   - 表示は `geom` と `flag` の両方を並べる（1 本目の数字と比べられるように）。
2. **見落としの理由を数える** `misses(results, labels, dump_dir, **params) -> list[dict]`。結べなかったラベルごとに 1 行：
   - その位置から 0.6D 以内に捨てた候補（`rejected`）があれば：`reason`（filled / flat / margin / thr / cap）、`score`・`second`・`second_agent`、**ラベルのエージェントの点 `true_score`**、`filled_ratio`・`face_std`、`color`、`crowded`。
     - `true_score` を出すため、`AG.match` の `debug` の各要素に `"faces": {エージェント名: 点}`（その候補を照合してよいエージェント全員）と `"second_agent"` を足す（debug だけの追加。返り値は変えない）。
     - 候補のいちばん似たエージェントがラベルと別人なら `id_mismatch: true`（それは thr や margin を緩めても救えない）。
   - 候補が無ければ、`find_icons` と同じ計算をその場でやり直して次のどれかに分ける（優先はこの順）：
     `mask`（`_on_floor` が偽）／`edge`（縁 0.5D の内側）／`bg`（背景を差し引かない輪の反応はしきい値 0.15 以上、差し引くと未満）／`nms`（もっと強い山が `nms×D` 以内にあって消された）／`k`（しきい値は超えたが `per_color` の上限で入らなかった）／`color`（縁 0.42〜0.62D の輪で warm・cool・yellow のどれも 15% 未満）／`weak`（それ以外）。
   - 出力：`<dump_dir>/eval/misses.csv`、理由 × 組（overlap / clear）× 縁の色の表、`misses_sheet.jpg`（見落としの場所を 2 倍で切り抜き、ラベルの名前と理由を書いたタイル。1 枚 24 タイル）。
   - 追加の表：エージェントごとの recall、ラウンド開始からの秒で 0〜15 / 15〜40 / 40〜 に分けた recall。
3. **1 本目の再現** `--baseline`：新しいスイッチを全部切にして照合し、`--group flag` の数字を出す。**1 本目の数字（97.8 / 98.2 / 96.4、48.6 / 78.3 / 18.8、wrong_id 1・ghost 2）とぴったり同じ**になることを確かめ、報告に貼る。違えばそこで止めて報告する。
4. **掃き出し** `--sweep <yaml>`：設定の組を並べた yaml を読み、組ごとに precision・recall（geom の all / overlap / clear）・wrong_id・ghost・1 フレームあたりの照合時間を `eval/sweep.csv` に書く。
5. 段階 0 の `misses` の数が、上の「候補なし 54・thr 42・filled 22・margin 13・flat/cap 11」と合うか確かめる（合わなければ、どちらの数え方か違いを報告）。

### 段階 1：clear の recall が低い理由を切り分ける（報告だけ。コードは段階 0 のもののみ）
次の順に数え、それぞれの結果を報告に書く。

1. `--group geom` で数え直す。clear の recall がどう変わるか。
   - 大きく上がる → 低さの多くはラベルの組分けのずれ。以降は geom の数字で判断する。
   - ほぼ同じ → 本当に重なりの無いアイコンを落としている。2 へ。
2. clear の見落としを理由で分ける（段階 0 の表）。いちばん多い理由がどれか。
3. 背景を疑う確かめ：`replay(..., background=None)` 相当（`--no-background`）で数える。clear の recall と precision の変わり方、増えた ghost の場所（`misses_sheet.jpg` と同じ形で `ghosts_sheet.jpg`）。
4. 時刻で分ける：0〜15 秒（スポーン）の見落としが多いか。多ければ背景の中央値にスポーンのアイコンが残っている可能性が高い。
5. 縁の色で分ける：yellow（観戦中の選手）・warm・cool のどれに偏っているか。
6. エージェントで分ける：Chamber・Neon 以外に recall が極端に低いエージェントがあるか。

段階 2〜5 は、ここで多かった理由から手を付ける（下の順は目安）。

### 段階 2：filled が Chamber を捨てる（スイッチ `filled_rescue`）
1. まず数える：10 人のアイコンを、照合と同じ大きさ D に縮めて灰色（90）の上に置き、`_filled_ratio` を warm・cool で出す（`tpl_fill[エージェント][色]`）。filled で捨てた候補を「ラベルの本物」と「ghost」に分け、`filled_ratio`・`face_std`・`true_score` を並べる。
2. 直す：`_filled_ratio >= 0.5` でも、次をすべて満たせば捨てずに先へ進める（返り値に `"rescued": "filled"` を付ける）。
   - その候補を照合してよいエージェントのうち点がいちばん高い a について、`tpl_fill[a][候補の色] >= tpl_fill_min`（顔そのものがその色を多く含む）
   - `face[i, a] >= filled_rescue`
   - 救った候補も、そのあとの margin・thr・cap の確かめはそのまま受ける
3. 試す値：`filled_rescue` ∈ {0.62, 0.66, 0.70}、`tpl_fill_min` ∈ {0.30, 0.40}。既定（切）は `filled_rescue: null`。
4. 見る所：本物の Chamber 14 件のうち何件戻るか。暗い赤の塗りつぶしの丸に名前が付いていないか（`ghosts_sheet.jpg`）。

### 段階 3：margin が Neon を捨てる（スイッチ `margin_exclusive`、`margin_pair`）
1. まず数える：margin で捨てた本物について、`second_agent` が誰か、その相手が同じフレームで別の場所に検出されているか（残した検出に同じチーム・同じ名前があるか）、その検出の点。
2. 直す A（`margin_exclusive: true`）：差の確かめの `second` から、**同じフレームですでに残した、同じチームのエージェント**を外す。
   - 組を点の高い順に確かめる今の順はそのまま。1 周目で margin で落ちた組は、全員の確かめが終わってから、残した検出を除いて 2 周目でもう一度確かめる（順番で結果が変わらないように）。
   - 黄色の縁の候補は、外すのを「両チームで残したエージェント（チームも一致するもの）」に限る。
   - 2 周目で残った検出には `"margin_pass": 2` を付ける。
3. 直す B（`margin_pair: true`、A と別に試す）：2 人のアイコン同士の似ている度合い `sim(a, b)`（大きさ D、円の内側 0.42 の CCOEFF）を最初に 1 回計算する。必要な差を `margin × clip((1 − sim) / 0.6, 0.5, 1.0)` にする（似た 2 人ほど、本物でも差が小さいため）。crowded のときは今の `crowded_margin` のまま。
4. 試す値：A は入・切だけ。B は `margin` ∈ {0.04, 0.06} との組。
5. 見る所：本物の Neon 10 件のうち何件戻るか。wrong_id が増えていないか（増えたら A・B のどちらで増えたか）。

### 段階 4：thr で落ちる 42 件（スイッチ `gray_floor`）
1. まず数える：thr で捨てた候補の点の分布を、「ラベルの本物（同じエージェント・0.6D 以内）」「ラベルはあるが別人」「ラベル無し」に分けて 0.02 刻みで表にする。`id_mismatch` のものは数に入れるが、救う対象にはしない。
2. 直す（灰色帯）：`gray_floor <= 点 < thr` の組を、次をすべて満たすときだけ残す（返り値に `"tier": "gray"`）。
   - crowded でない・縁がチーム色（黄色ではない）
   - 差 `点 − second >= gray_margin`
   - `face_std >= gray_face_std`
   - 輪の反応 `ring >= gray_ring`
3. `consensus`：`tier == "gray"` の検出は `overlap` と同じ扱い（ほかのフレームの近くの検出 `sup >= 1` があるときだけ残す）。`track.best_chain` は今回変えない（gray は追跡では今までどおり `min_score` で落ちる。追跡まで流すかは、この 2 本目の数字を見て 3 本目で決める）。
4. 試す値：`gray_floor` ∈ {0.48, 0.52}、`gray_margin` ∈ {0.10, 0.14}、`gray_face_std: 18`、`gray_ring: 0.30`。
5. ついでに試す（スイッチ `face_inner`、既定 0.42）：`_face_score` の円の内側を 0.36 にして縁の色を外す。thr の見落としが減るか、偽物の点も上がるかを見る。

### 段階 5：候補が出ない 54 件（段階 0 の内訳で多いものから）
- `bg` が多いとき（スイッチ `bg_mode: subtract | flag`、既定 `subtract`）：
  `flag` では、山は**差し引かない**輪の反応から拾い、各候補に背景の反応 `bg = Rb(x, y)` を記録する。`bg >= 0.15` の候補（動かない模様と重なる候補）は、`face[i, a] >= thr + 0.04` かつ `face_std >= gray_face_std` のときだけ残す（設置位置の赤い円やUIの色は顔が無いので落ちる）。gray 帯には入れない。
  山の数が増えるので、`flag` のときは `per_color` を 16 にする。
- `nms` が多いとき（スイッチ `cand_nms`、既定 0.8）：{0.6, 0.7} を試す。
- `k` が多いとき（スイッチ `cand_per_color`、既定 12）：{16, 20} を試す。
- `color`・`weak` が多いとき（スイッチ `face_search: true`）：輪の候補とは別に、顔から候補を探す。
  そのフレームでまだ残していない（チーム, エージェント）ごとに、床マスクの中で顔のテンプレートを照合し、いちばん高い場所を 1 つ取る。次をすべて満たせば候補に足す（`"source": "face"`）：
  点が `thr + 0.06` 以上・`face_std >= gray_face_std`・その場所の縁の輪にそのチームの色が 15% 以上・残した検出から 0.6D 以上離れている。
  足した候補も margin・cap の確かめは受ける。1 フレームあたりの時間の増え方を報告する。
- `mask`・`edge`：数だけ報告（今回は直さない）。

### 段階 6：組み合わせて既定を決める
1. 段階 2〜5 のスイッチを 1 つずつ入れた結果（ほかは切）を報告の表に並べる。
2. precision を落とさずに recall が上がったものだけを組み合わせて数える。
3. 既定にする条件：geom の precision が all・overlap・clear のすべてで 95% 以上、wrong_id＋ghost が合わせて 6 以下、1 フレームの照合時間の増え方が 30% 以内。満たす組の中で all の recall がいちばん高いもの。
4. その組を `config/default.yaml` の `agent_match` の既定にする（コメントに、決めた日と数字を 1 行書く）。
5. PROMPT.md の「エージェントの照合」に、足したスイッチと既定、決めた理由の数字を書く。

## 設定（`config/default.yaml` の `tactics.agent_match` に足す。下は「全部切」の値）
```yaml
  agent_match:
    thr: 0.58
    margin: 0.06
    crowded_margin: 0.10
    min_face_std: 12
    # 2 本目（agents-recall）。null / false / 今までの値 なら 1 本目と同じ動き
    filled_rescue: null      # 例 0.66。顔そのものが縁と同じ色を含むエージェント（Chamber など）を filled で捨てない
    tpl_fill_min: 0.35
    margin_exclusive: false  # 同じフレームで残したエージェントを 2 番目の相手から外す
    margin_pair: false       # 似た 2 人ほど必要な差を小さくする
    gray_floor: null         # 例 0.52。thr 未満でも条件付きで残す下限
    gray_margin: 0.12
    gray_face_std: 18
    gray_ring: 0.30
    face_inner: 0.42
    bg_mode: subtract        # subtract | flag
    cand_nms: 0.8
    cand_per_color: 12
    face_search: false
```
`_agent_match_params` はこれらをすべて `AG.match` に渡す（無いキーは今の既定）。`estimate_icon_frac` の `strict=False` の呼び出しには何も足さない。

## テスト
`tests/test_agents.py` に足す（今あるテストは変えずに通す）：
1. 全部切なら、1 本目のテストの合成画像すべてで返り値が今と同じ（キーも値も）。
2. 顔の中心が暖色のアイコン（合成）を暖色の縁で置く：`filled_rescue: 0.66` なら見つかり `rescued == "filled"`。暖色の塗りつぶしの丸は、`filled_rescue` を入れても検出 0 個。
3. チーム 1 が `["Neon", "X"]`（X は Neon と似せた合成の顔）。X をはっきり 1 か所、離れた所に Neon を置き、Neon の点の差が margin 未満になるようにする：`margin_exclusive: true` なら Neon が残る（`margin_pass == 2`）。X がフレームにいないときは捨てられる（理由 `margin`）。
4. 点 0.53 前後・重なり無し・差 0.15 の候補：`gray_floor: 0.52` なら `tier == "gray"` で残る。同じ候補を 0.7D で重ねると残らない。`consensus` で gray が 1 フレームだけなら残らず、2 フレームで近くにあれば残る。
5. 背景にも同じ場所に同じアイコンがある：`bg_mode: subtract` では見つからず、`flag` では見つかる。背景にある「顔の無い暖色の輪」は、どちらでも見つからない。
6. 縁の彩度を落としたアイコン：`face_search: false` では見つからず、`true` では `source == "face"` で見つかる。床だけのミニマップでは `face_search: true` でも検出 0 個。

`tests/test_agents_eval.py` に足す：
7. 合成の書き出しで、重なっているのに `overlap: false` のラベル → `--group geom` では overlap に数えられる。
8. `misses` が、合成で作った `bg`・`nms`・`thr`・`filled` の見落としをその理由に分ける。`true_score` がラベルのエージェントの点になっている。
9. `--baseline` が、全部切の `replay` と `score(..., group="flag")` に一致する。
10. `--sweep` の yaml 2 組で `sweep.csv` が 2 行になる。

## 通すテスト
- `python -m pytest -x -q tests/test_agents.py tests/test_agents_eval.py tests/test_track.py tests/test_tactics.py`
- `python -m pytest -q`（全体）

## 終わりの条件
- 上のテストがすべて通る。今あるテストを消したり緩めたりしていない。
- `--baseline` が 1 本目の数字とぴったり同じ（報告に出力を貼る）。
- 段階 1 の切り分けの結果（6 項目）を報告に書いている。
- 下の「報告の表」が埋まっている。
- 新しい既定で、geom の precision が all・overlap・clear すべて 95% 以上。flag の組分けの数字も並べて書く。
- recall（all・geom）の目安は 70% 以上。届かなければ、残った見落としの理由の内訳を書いて止める（しきい値を細かく刻んで合わせない）。
- Colab で同じマップを `agents_dump: 40`・`--force` で解析し直し、`sheet.jpg` で、床・塗りつぶしの丸・スキルの印に緑の丸が付いていない（1 本目と同じ目視の確認）。`agents_dump` は確認後に 0 へ戻す。

## 報告の表（この形で埋める）
| 設定 | P all | P overlap | P clear | R all | R overlap | R clear | wrong_id | ghost | ms/フレーム |
|---|---|---|---|---|---|---|---|---|---|
| 1 本目（flag の組分け） | 97.8 | 98.2 | 96.4 | 48.6 | 78.3 | 18.8 | 1 | 2 | |
| 1 本目（geom の組分け） | | | | | | | | | |
| filled_rescue 0.66 | | | | | | | | | |
| margin_exclusive | | | | | | | | | |
| margin_pair | | | | | | | | | |
| gray_floor 0.52 | | | | | | | | | |
| face_inner 0.36 | | | | | | | | | |
| bg_mode flag | | | | | | | | | |
| cand_nms / per_color | | | | | | | | | |
| face_search | | | | | | | | | |
| 決めた既定の組 | | | | | | | | | |

あわせて、決めた既定の組での見落としの理由の内訳（段階 0 の表）を書く。

## しないこと
- labels.json・`tests/fixtures/` の今ある画像を書き換えない。
- アイコンの取り方（`load_icons`）・`track.best_chain`・追跡の鎖の作り方は変えない。
- 背景の作り方（`tactics.py` 432 行の中央値）は変えない（`bg_mode` で照合の側だけ変える。作り方を変えるかは 3 本目）。
- 1 本目の確かめ（filled・flat・margin・thr・cap）を消さない。救うのはスイッチの条件を満たすときだけ。

## できれば（必須ではない）
- 別の試合・別のマップで dump を 1 つ作り、`--init-labels` の下書きを人が直したら、既定の組をそこでも数える（40 フレーム 1 マップに合わせすぎていないかの確認）。

## 質問
（実装担当が書く。無ければ空のまま）
