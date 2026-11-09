# playbook-v4-rebuild（プレイブック v4 を 1 から作り直す）

## 目的
`claude/playbook-v4` ブランチの v4 は、マップの図を**自作**していた（追跡の点をぼかした「床」・追跡の点の中央値に置いたラベル・半径 7 の丸の色塗り）。
そのため形が崩れ、TactiVal・Valoplant・valorant-api で見るマップと違う絵になっていた。
v4 を main から作り直す。図は**公式のマップ画像（valorant-api の displayIcon＝TactiVal のマップの元画像）**の上にだけ描き、エリアの色塗りはそのマップの実際の通路の形に沿わせる。
判定・文・並びの方針は `docs/PLAYBOOK_v4.md`（下の 0. で持ってくる）のまま。

## 今の v4（claude/playbook-v4）で起きていたこと（同じことをしない）
| 問題 | 原因（旧コード） | 今回の決まり |
|---|---|---|
| マップの形が崩れる・データの少ない通路が欠ける | `floor_uri()`：全ラウンドの追跡点のヒートマップを 2 値化して「床」にした | 下地は公式画像だけ（A1）。自作の下地は作らない |
| エリア名・色の丸が通路からずれる | `MapCtx.center`：追跡点の中央値をエリアの位置にした | エリアの位置は公式画像の上のエリアの形の重心（A5） |
| 色塗りがエリアの形と合わず隣と重なる | `Fig.svg(hi=…)`：中心に半径 7 の丸 | エリアの多角形を塗る（A4・A6） |
| 向きが TactiVal と違う（Split が 90 度回っている） | valorant-api の向きのまま | TactiVal と同じ向きに回す（A2） |
| 1 ファイルに判定・文・描画が混ざる（805 行） | `playbook_v4.py` | 図・判定・冊子を 3 つのモジュールに分ける |

## 触ってよいファイル
- 新規 `vreplay/mapfig.py`（図：下地・座標・エリアの形・SVG）
- 新規 `vreplay/gamemodel.py`（ラウンド × 局面 → ノードの判定。描画しない）
- 新規 `vreplay/playbook_v4.py`（冊子の HTML を組み立てる）
- 新規 `scripts/playbook_v4.py`（report.json の glob から 1 冊作る）
- `vreplay/__main__.py`（`vreplay playbook --v4` を足すだけ）
- 新規 `tests/test_mapfig.py`・`tests/test_gamemodel.py`・`tests/test_playbook_v4.py`
- `tests/deckfx.py`（合成ラウンドの関数を足すだけ。既存の関数は変えない）
- `PROMPT.md`（プレイブックの節に v4 の 3 モジュールを 3〜5 行で）
- 0. で持ってくるファイル（中身は変えない）

`vreplay/deck.py`・`tactics.py`・`tactival.py` は**読んで関数を呼ぶだけ**（変えない）。

## やること

### 0. 準備
```bash
git fetch origin
git switch -c task/playbook-v4-rebuild origin/claude/playbook-v4-rebuild-task   # この指示書が入っている（中身は main と同じ＋指示書）
git checkout origin/claude/playbook-v4 -- config/kb/game_model.yaml config/kb/maps/split.yaml config/kb/maps/sunset.yaml docs/PLAYBOOK_v4.md
```
`claude/playbook-v4` の `vreplay/playbook_v4.py`・`vreplay/gamemodel.py`・テストは**持ってこない・コピーしない**（1 から書く）。

### A. `vreplay/mapfig.py`（図）

**A1 下地は公式画像だけ**
- `base_image(cfg, map_name) -> np.ndarray | None`：valorant-api の displayIcon（`tactics.load_map(map_name, cfg)["icon"]` の URL）を `<output.dir>/_cache/map_icons/<norm>.png` にキャッシュして **RGBA・1024×1024** で返す（`deck._icon` と同じ取り方。ただしアルファを残す。`deck._icon` は変えない）。
- 取れないときだけ、その試合の `uv_base`（放送のミニマップを同じ座標に写した画像、`records()` の `r["uv_base"]`）を使う。アルファが無いので「白に近い画素以外」を不透明として扱う。
- **どちらも無ければ図を出さない**（スライドには「マップ画像がありません（オフライン）」の 1 行）。点の分布や コールアウトから下地を作らない。

**A2 座標と向き**
- 入力の位置はすべて `records()` の `agents`・`tracks`・`uv` の (u, v)＝valorant-api の displayIcon の 0..1（Sunset は `tactics.CALLOUT_FIX` 済み）。そのまま使う。
- 表示は TactiVal と同じ向き：`k = tactival.MAP_ROT[map]`（90 度単位）。`to_view(u, v, k) -> (x, y)`（0..1）と、同じ回転を画像にかける `rotate_image(img, k)` を作る。
  回転の向きは `tactival.exact_calib()` と同じ（`(u, v)` を k 回 `(v, 1 − u)` に送る）。
- 設定 `playbook.map_orientation`：`tactival`（既定）／`api`（k = 0）。

**A3 整合チェック（下地と点がずれていないか）**
- `check_alignment(base, callouts, rows) -> dict`：コールアウト（スポーン以外）の点と、全ラウンドの追跡点のうち、下地の不透明な画素（アルファ > 0、半径 6px 以内）に乗っている割合。
- コールアウト 90% 未満か追跡点 80% 未満なら、ログに出して `uv_base` に切り替える。両方ダメなら公式画像のまま使い、付録に「マップ画像との整合 NN%」を出す。

**A4 エリアの形（色塗りの単位）**
- `area_regions(base, callouts, kb_map) -> dict[str, list[polygon]]`：
  1. 通路のマスク＝下地のアルファ > 0（`uv_base` のときは A1 の「不透明」）。
  2. マスクの画素を**いちばん近いコールアウト**に割り当てる（ボロノイ。256×256 で計算して 1024 に拡大してよい）。コールアウトから 0.08（0..1 の距離）より遠い画素はどこにも入れない。
  3. コールアウトの名前は `tactics.area_name(superRegionName, regionName)`（`records()` のエリア名と同じ呼び方になる）。スポーンは除く。
  4. `config/kb/maps/<map>.yaml` の `areas[].callouts` にあるものは、その `name` に**まとめる**（例 Split の MID＝Mid Top・Mid Mail・Mid Vent・Mid Bottom）。kb に無いコールアウトは単独のエリア。
  5. `cv2.findContours` で多角形に（`approxPolyDP` の eps 1.5px、面積 40px² 未満の欠片は捨てる）。
- 結果を `<output.dir>/_cache/map_areas/<norm>_<k>.json` にキャッシュ（下地の画像の sha1 をキーに含める）。
- `adjacent(regions) -> set[tuple[str, str]]`：多角形を 3px 膨らませて重なる組＝隣どうし。

**A5 ラベルの位置**
- エリアの名前は、そのエリアのいちばん大きい多角形の重心。重心が多角形の外なら、距離変換の最大の点。追跡点の中央値は使わない。
- 名前どうしが 3% より近いときは、小さいエリアの方を下へ 3% ずらす。

**A6 色塗り**
- Map Prio：GREEN `#197A4B`／YELLOW `#8A6D00`／RED `#C7322B`／BLUE `#0017C1`。多角形を塗り 0.35・縁 1px（同じ色）。凡例を図の下に。
- バイタル：縁 `#0017C1` 2.5px（塗りなし）＋名前を太字。バイタルにつながるエリア：縁 `#8A6D00` 1.5px の破線。
- 色だけで区別しない：凡例と、右の文の見出しにも色名を書く。

**A7 駒（エージェントの点）**
- `agents.load_icons(cfg, names)` のアイコンがあれば、丸く切り抜いて置く（直径＝図の幅の 4.5%、チーム色の縁：守り `#0017C1`・攻め `#C7322B`）。無ければ同じ大きさの丸に頭 2 文字。
- 2 つの駒が 3% より近いときは、互いに離す（最大 2% まで動かす。元の位置へ細い線を引く）。
- 1:40 → 1:30 → 1:20 の動きは、同じ駒の位置を破線の矢印でつなぎ、1:40 を濃く・1:20 を薄く。

**A8 SVG の出し方**
- `Figure(map_name, cfg)`：下地・回転・エリアを 1 回だけ作って持つ。`fig.svg(layers, *, prio=None, vital=None, connected=None, labels=None, alt="") -> str`。
- 下地は冊子の先頭の 1 つの `<svg><defs>` に `<image id="map-<norm>">`（data URI・JPEG 85・1024px）として 1 回だけ入れ、各図は `<use href="#map-<norm>">` で参照する（図ごとに data URI を埋めない）。
- `viewBox` は不透明な画素の外接矩形＋余白 3%（マップが枠いっぱいに出る）。`preserveAspectRatio="xMidYMid meet"`、CSS は `aspect-ratio:1/1; width:min(100%,460px)`。
- 文字：Noto Sans JP 700、黒字に白のふち、表示で 12px 以上。`role="img"` と `aria-label` を必ず付ける。

### B. `vreplay/gamemodel.py`（判定）
- 純粋な関数だけ（ファイル・ネットワーク・描画なし）。表は `docs/PLAYBOOK_v4.md` の「判定のルール」、名前・しきい値は `config/kb/game_model.yaml`（`params`）から。
- `MapModel`：`area_of(u, v)`（A4 の多角形で点 → エリア。どれにも入らなければ `records()` の付けたエリア名）、`half(area)`（攻め側半分か：エリアの重心が攻めのスポーンに近いか、`tactics.load_map()["spawns"]`）、`is_vital`・`is_choke`（kb）、`connected()`。
  - つながるエリア＝**A4 で隣どうし**のエリアのうち、全チームの追跡でバイタルと 5 秒以内に行き来した回数が多い順に最大 6（隣でないエリアは数えない）。
- `setup_at_140(r) -> dict[agent, (u, v, area)] | None`：`agents["1:40"]`。無ければ `tracks` の e = −3〜+3 秒のいちばん遅い点。3 人未満なら `None`。
- `classify(r, opp, model) -> {"setup", "buildup", "vital", "defense"|"attacking", "retake"|"postplant", "extreme"}`（当てはまらない局面は `None`。複数当てはまれば表の上を優先）。キルの場所は倒された人の `tracks` のキル時刻にいちばん近い点。
- `def_formation(r, model) -> (形, 性格, 場所|None)`（PLAYBOOK_v4.md の 3 節。形は `tactics.shape_code`）。

### C. `vreplay/playbook_v4.py`（冊子）
- `build(reports, team, map_name, out, cfg, log=print) -> Path`。並びは `docs/PLAYBOOK_v4.md` の 6 節。図はすべて `mapfig.Figure` 経由（このファイルで画像を作らない・座標を変換しない）。
- スライドごとに 1 関数（`cover`・`vital_map`・`prio_overview`・`area_card`・`formation`・`midround`・`retake`・`lean_buildup`・`vital_phase`・`attacking`・`postplant`・`extreme`・`pistol`・`appendix`）。入力はデータの dict、出力は HTML の文字列。
- 文の決まりは PLAYBOOK_v4.md の 5 節（データ由来の語を 1 つ以上・`GENERIC` の禁止文・攻守で別の動詞・守りに「取りに行く」なし）。
- 見た目：16:9、図は左 52%・右に文（エージェント 5 行）。デジタル庁デザインシステム（Noto Sans JP・`#0017C1`・コントラスト 4.5:1・フォーカス黄＋黒）。既存の冊子の CSS（`deck.py` の CSS）を読み込んで足すだけ。
- `vreplay playbook --v4 --team <T> --map <M> --out <dir>` と `scripts/playbook_v4.py` の両方から呼べる。

## テスト（ネットワークなし）
`tests/deckfx.py` に合成マップを足す：512×512 の RGBA に「A サイト・MID・B サイト」の通路を矩形で描き、コールアウト 8 点を置いたもの（`base_image` を monkeypatch で差し替える）。
- mapfig
  - `to_view` と `rotate_image` が同じ向き：k = 0..3 で、画像上の目印の画素と `to_view` した点の画素が一致する
  - 下地が無いとき図を出さず、自作の下地も作らない（`<image`・`<use` が無く「マップ画像がありません」が出る）
  - `area_regions`：MID の 4 コールアウトが kb で 1 エリアにまとまる／多角形がマスクの外に出ない／隣どうしが `adjacent` に入る
  - ラベルの位置がそのエリアの多角形の中にある
  - 冊子 1 冊の HTML で、下地の data URI がちょうど 1 回
  - 追跡点の半分を通路の外に置いた合成データで `check_alignment` が切り替えを返す
- gamemodel：PLAYBOOK_v4.md の受け入れ条件どおり（1:40 で型が決まる・開幕プッシュ／平たく守ってローテ／カウンター配置／スタック／1-3-1・リテイク 3 種・攻めの早く深く狭い／フリーズ／Joker／ダイレクトプレー、1 ラウンドずつ合成）
- playbook_v4：2 チームぶんの合成データで本編の文の重なり 10% 以下・`GENERIC` 0・守りの行に「取りに行く」なし・守りの形の図の下に「1:40」

## 通すテスト
- `python -m pytest -x -q tests/test_mapfig.py tests/test_gamemodel.py tests/test_playbook_v4.py`
- 最後に `python -m pytest -q`（既存のテストも全部）

## 終わりの条件
- `grep -n "floor\|GaussianBlur\|center\[" vreplay/playbook_v4.py vreplay/mapfig.py` で、自作の下地・追跡点の中央値のラベルが無い
- 合成データで作った冊子を `python scripts/shot_slides.py <html> shots/ バイタル Map エリア 1:40` で撮り、マップのスライド（バイタルの地図・Map Prio・エリアカード 1 枚・守りの形 1 枚）の PNG 4 枚をレビュー用に `_review/` に置く。通路の形が合成マップと同じで、色がエリアの形に沿っている
- 上の「通すテスト」が全部通る
- Colab での本番 2 冊（PRX Split・LOUD Sunset → Drive `playbook_v4_test/`）は**ユーザーが**作る。手順を報告の最後に 2 行で書く

## やらないこと
- 下地を点・ヒートマップ・コールアウトから作る
- `deck.py`・`tactics.py`・`tactival.py`・`config/kb/` の中身を変える
- TactiVal の JSON の書き出し（今回は図だけ。v4 の型を TactiVal に出すのは別の指示書）
- 判定に LLM を使う（言い回しの整えだけ）

## 質問
（実装担当が書く。無ければ空のまま）
