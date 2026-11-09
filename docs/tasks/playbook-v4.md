# タスク：プレイブック v4（ゲームモデル・1:40 の配置・エリアカード）

方針は `docs/PLAYBOOK_v4.md`（Project の PLAYBOOK_v4.md と同じもの）。v3 の実装の上に足す。
最後に PRX Split と LOUD Sunset の 2 冊を作り直して、Drive の `playbook_v4_test/` に置く。

## 触るファイル

- `vreplay/deck.py`：配置の時刻、型の分け方、エリアカード、文の作り方、並び
- `vreplay/howto.py`：`agent_lines` の動詞を攻守で分ける
- 新規 `vreplay/gamemodel.py`：ラウンド × 局面 → ノードの判定（純粋な関数だけ。描画はしない）
- 新規 `config/kb/game_model.yaml`（下に中身）
- `config/kb/maps/split.yaml`・`sunset.yaml`：`vital` は `vital.patch` で入れ済み（Split＝A Ramps・MID、Sunset＝A Main・MID。Split に A Ramps のエリアを新設）。`counter` はまだ
- `vreplay/notion_playbook.py`：見出しを v4 の並びに
- `tests/`：下のテストを足す。`python -m pytest -q` が全部通ってからコミット

## 手順

1. **1:40 の配置**
   - `records()` の `agents["1:40"]` を使う関数 `setup_at_140(r)` を作る。無ければ `tracks` の e = −3〜+3 秒のいちばん遅い点。3 人未満なら `None`。
   - 守りの型・攻めの寄りはこれだけで決める。今の 5〜40 秒の平均・1:15 の人数を型の決定に使っている所（`defaults`・`setup_clusters`・v3 の配置スライド）を置き換える。
   - 図は 1:40・1:30・1:20 の 3 コマ（`AGENT_MARKS` に `("1:20", 20)` を足す。アイコン照合の側にも同じ節目が要るなら `tactics.MARKS` も）。
2. **ゲームモデルの判定**（`gamemodel.py`）
   - `classify(r) -> {"buildup": str|None, "vital": str|None, "attack"/"defense": str|None, "retake"/"postplant": str|None, "extreme": str|None}`
   - 判定は PLAYBOOK_v4.md の表のとおり。しきい値は `game_model.yaml` の `params` から読む。
   - キルの場所は、倒された人の `tracks` のキル時刻にいちばん近い点のエリア。
   - 当てはまらない局面は `None`。1 局面に複数当てはまったら表の上にあるものを優先。
3. **守りの形**：`def_formation(r)` → `("1-3-1", "開幕プッシュ", "Mid Mail")` のような (形, 性格, 場所)。名前は「形 性格（場所）」。`names.yaml` があれば置き換え。
4. **エリアカード**：`area_cards(rows, side, kb_map)` → エリアごとの dict（1:40 の人数分布・担当・守り方 or 取り方・来たときのノードと勝敗・見本）。スライドは 1 エリア 1 枚。
5. **文**：
   - `GENERIC` のリスト（禁止文）を deck.py に置き、本編を作ったあとに含まれていたらその文を落とす（ログに出す）。
   - 動詞の辞書を `VERBS = {"defense": {...}, "attack": {...}}` に分ける。
   - 「v3_test で 2 冊とも同じだった文」は全部 `GENERIC` へ（PRX Split／LOUD Sunset の HTML から本編の文を取り出して共通のものを入れる）。
6. **並び**：PLAYBOOK_v4.md の 6 節。Objectives はゲームモデルの要約に置き換える。
7. **2 冊を作る**：Colab で
   `vreplay playbook --team "Paper Rex" --map Split --out playbook_v4_test` と `--team LOUD --map Sunset`（v3_test を作ったときと同じ引数・同じ試合）。
   Drive の `valorant-replay/playbook_v4_test/` に `PRX_Split_v4_test.html`・`LOUD_Sunset_v4_test.html`。

## config/kb/game_model.yaml

```yaml
# コーチのゲームモデル（xmind）。tags の略号は意味が確定していないので表示だけ。
attack:
  buildup:
    - {id: fast_deep, name: 早く深く狭いビルドアップ, tags: [DC], text: バイタルエリアをコントロールしにいく}
    - {id: bu_reaction, name: ビルドアップリアクション, tags: [DC], text: 少ない人数で持つ場所に圧を掛けられたら、反対側のバイタルをコントロールしに行く}
    - {id: bu_counter, name: ビルドアップカウンター, tags: [PrP], text: 敵が圧を掛けてきた場所にファイトしに行く}
    - {id: hold, name: 正しい戦略の模索（ホールド）, tags: [PoC], text: ホールドして敵のアクションを待つ, next: [ホールドリアクション, ホールドカウンター]}
    - {id: freeze, name: ネガティブフィードバック（フリーズ）, tags: [BP], text: 早く深く狭いビルドアップ中にファーストキルを取られたら止まる}
  vital:
    - {id: v_reaction, name: バイタルリアクション, tags: [DC], text: 主力がいないバイタルで敵が動いたら、近いサイトを取りに行く}
    - {id: v_counter, name: バイタルカウンター, tags: [PrP], text: 主力のいるバイタルに圧を掛けられたらファイトする}
    - {id: v_hold, name: 正しい戦略の模索（ホールド／ローテーション）, tags: [PoC], text: ホールドして待つか、別のバイタルを取りに行く, next: [ホールドリアクション, ホールドカウンター, joker]}
    - {id: v_occupy, name: バイタル占拠, tags: [BTP], text: バイタルを取ったあとスキルや撃ち合いで圧を掛ける}
  attacking:
    - {id: site_occupy, name: サイト占拠, tags: [DC, PrP], text: デュエリストが奥まで飛び込み、全員でカバーする}
    - {id: direct, name: ダイレクトプレー, tags: [DC, PrP], text: バイタル占拠中にキルが出たらサイトに飛び込む}
    - {id: challenge, name: チャレンジ, tags: [PoC], text: 主力がコンタクトでサイトに入る}
    - {id: joker, name: Joker, tags: [BP], text: 待つ裏取りを攻める裏取りに切り替え、敵サイトの近くまで単独で圧を掛ける}
  postplant:
    - {id: choke_counter, name: チョークカウンター, tags: [PrP], text: 敵が入ってくるチョークでカウンター}
    - {id: boom, name: 爆破狙い, tags: [BTP], text: サイトに 1〜2 人残して他は退く。時間を稼いで爆破, next: [スキルリアクション]}
defense:
  buildup:
    - {id: fast_deep, name: 早く深いビルドアップ（開幕プッシュ）, tags: [DC], text: バイタルエリアをコントロールしにいく}
    - {id: bu_counter, name: ビルドアップカウンター, tags: [PrP], text: 敵が圧を掛けてきた場所にファイトしに行く}
    - {id: hold, name: 正しい戦略の模索（ホールド）, tags: [PoC], text: ホールドして敵のアクションを待つ}
  vital:
    - {id: v_reaction, name: バイタルリアクション, tags: [DC], text: 主力がいないバイタルで敵が動いたらプッシュする}
    - {id: v_occupy, name: バイタル占拠, tags: [BTP], text: 取ったバイタルに留まって敵のアクションを待つ}
    - {id: v_counter, name: バイタルカウンター, tags: [PrP], text: 主力のいるバイタルに圧を掛けられたらファイトする}
    - {id: solo, name: 単独守備, tags: [BTP], text: 1 人がサイトに残り、他はローテする}
  defense:
    - {id: reaction_def, name: リアクションディフェンス, tags: [DC], text: 敵が人数の少ないサイトに行ったら、主力が裏を取りに行く}
    - {id: choke_def, name: カウンターチョークディフェンス, tags: [PrP], text: 主力のいるサイトに来たら全力でチョークで倒す}
    - {id: back_def, name: バックディフェンス, tags: [BTP], text: サイトの裏で時間を稼ぎ、ローテの味方と一緒に倒す}
    - {id: retake_def, name: リテイクディフェンス, tags: [BTP], text: リテイクすると決めて守る。時間は稼ぐ}
  retake:
    - {id: fast, name: ファストローテ → ファストリテイク, tags: [DC]}
    - {id: surround, name: 囲い込みリテイク, tags: [DC]}
    - {id: slow, name: ゆっくりセットアップしてリテイク, tags: [BTP]}
    - {id: reaction, name: リアクションリテイク, tags: [DC], text: スプリットになるので、どちらが先にアクションを掛けるか意識する}
extreme:
  - {id: up2, name: 人数有利（+2）, text: 距離を詰めてプレイする}
  - {id: down1, name: 人数不利（−1）, text: 2 つに分かれて片方がチャレンジ、片方がコンタクト（ミニゲーム）}
  - {id: anti_eco, name: マネー有利（アンチエコ）, text: 普段取らない戦略を取る（CT に抜ける、いつも取るバイタルをホールドにする）}
params:
  fast_deep_by: 15        # 開始から何秒までにバイタルへ 2 人
  push_by: 12             # 守りの開幕プッシュ：何秒までに攻め側半分
  reaction_win: 15
  trade_win: 5
  freeze_still: 10
  fast_retake_win: 12
  slow_retake_after: 15
  surround_min_entries: 2
```

## テスト（ネットワークなし）

- `tests/deckfx.py` の合成ラウンドで：1:40 の点と 1:15 の点をわざと違う場所に置き、型が 1:40 で決まること
- 開幕プッシュ・平たく守ってローテ・カウンター配置・スタック・1-3-1 がそれぞれ判定されること（1 ラウンドずつ合成）
- リテイク 3 種（ファスト・囲い込み・ゆっくり）の判定
- 攻めのノード（早く深く狭い・フリーズ・Joker・ダイレクトプレー）の判定
- 2 チームぶんの合成データで deck を作り、本編の文の重なりが 10% 以下・`GENERIC` の文が 0
- 守りの行に「取りに行く」が無い

## やらないこと

- 略号（DC など）の意味で判定を変えない（表示だけ）
- 判定に LLM を使わない（言い回しの整えだけ）
