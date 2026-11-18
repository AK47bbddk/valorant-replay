# タスク指示書：さらなる検知精度向上・スモーク検知・初動配置分類強化＆30件ブラインドテスト作成（opening-playbook）

- **タスクID / ブランチ**: `task/opening-playbook`
- **作成日**: 2026-11-18
- **参照すべき `PROMPT.md` の節**: ミニマップの配置とキルの正確さ（エージェントの照合）・プレイブック・TactiVal・知識ベース（`config/kb/`）・バリア座標（`vreplay/barriers.py`）

## 背景・目的
1. **さらなる検知精度向上**:
   - `agents-precision`〜`agents-recall4` の改善（`filled`/`flat`/`margin` 棄却、`filled_rescue`、`margin_exclusive`、`gray_floor`、`face_inner`、`gray_w`、`self_norm`、`face_search`、`cap_over`、`prior` 時系列統合）に加え、**黄色リング（観戦中）の縁色判定・低リング候補の逆色棄却/ミラーマッチ補正・異色間の同一位置重複抑止（NMS）・バリアマスク（`team_masks`）統合**により、3つの実データ評価セット（Sunset / 他5マップ / Skye・Brimstone編）すべてで Precision 95% 以上を達成する。
   - `vreplay/tactics.py` の `detect_dots` でもスモークの中空リング・大円を選手アイコンの重なり（`k=2, 3`）として誤計上しないようフィルタリングする。
2. **スモークのアビリティ検知追加**:
   - ミニマップ上の**ドーム型スモーク**（Omen / Brimstone / Clove / Astra / Jett 等の暗い円形ドーム、および Viper / Harbor / Cypher 等のリング付きオーブ）と**壁型スモーク・カーテン**（Viper トキシックカーテン / Harbor ハイタイド / Sage 壁 / Neon ファストレーン等）を検出する `AG.detect_smokes()` を実装し、`tactics.py` のラウンド解析結果（`rt["smokes"]`・コールアウト/ゾーン付与）および初動分類に統合する。
3. **初動配置の分類機能向上**:
   - `tactics.py` の `_agent_setups` で複数節目（`1:15`・`1:30`・`1:40`）からエージェント別の持ち場（`who` / `who_area`）を補完統合し、スモーク検知結果も付与する。
   - `opening_playbook.py` の `_auto_tag_round` / `classify_report_rounds` / `classify_round_record` を強化し、エージェント別の立ち位置（オペ枠・センチネル枠・索敵枠・モク枠・エントリー枠）、検知されたスモーク位置（MID膨らみモク・Aメインモク・Bメインモク）、ローテーション（Bフェイク→A、Aフェイク→B）、スプリット/セット/ラッシュを高精度に自動分類する。
4. **ユーザー向け30件のブラインドテスト作成**:
   - `scripts/blind_test.py` を強化し、**① 開幕位置・エージェント検知の正確性、② 初動配置・戦術の分類（4択＆自動分類との答え合わせ）、③ スモーク・アビリティ検知の正確性、④ サイト読み取り**（＋検知オーバーレイ画像の切替機能）を備えたインタラクティブ HTML テスト（30問、`blind_test_30.html` および `playbook_JSON/generated/blind_test_30.html`）を生成する。
   - あわせて自動テスト（`pytest`）にも検知精度・スモーク検知・初動分類・ブラインドテストの検証ケースを30件以上整備する。

## 触ってよいファイル
- `docs/tasks/opening-playbook.md`（本指示書）
- `vreplay/agents.py`
- `vreplay/tactics.py`
- `vreplay/track.py`
- `vreplay/barriers.py`
- `vreplay/opening_playbook.py`
- `vreplay/__main__.py`
- `config/default.yaml`
- `scripts/opening_playbook.py`
- `scripts/blind_test.py`
- `tools/agents_eval.py`
- `tools/agents_blind_eval.py`
- `tools/agents_check_html.py`
- `tests/test_agents.py`
- `tests/test_agents_eval.py`
- `tests/test_barriers.py`
- `tests/test_opening_playbook.py`
- `tests/test_blind_test.py`
- `playbook_JSON/generated/` および `blind_test_30.html`
- `PROMPT.md`, `README.md`

※ `config/kb/` の既存 YAML および `tests/fixtures/` の画像は書き換えない。

## 通すテスト
```powershell
python -m pytest -x -q --basetemp=.pytest_tmp tests/test_agents.py tests/test_agents_eval.py tests/test_barriers.py tests/test_opening_playbook.py tests/test_blind_test.py
```

## 受け入れ条件
- [ ] 指定したテストがすべてネットワークなしで通る
- [ ] 実データのブラインド評価3セットすべてで Precision 95% 以上を達成する（Sunset 99.1%, 5マップ 97.9%, Skye/Brimstone編 95.4%以上）
- [ ] ドーム型スモーク・壁型スモークが `AG.detect_smokes` および `tactics.py` で検出され、初動分類に反映される
- [ ] 初動配置分類がエージェント位置・スモーク・フェイク・スプリットを識別できる
- [ ] ユーザー向けの30件ブラインドテスト HTML（`blind_test_30.html` / `playbook_JSON/generated/blind_test_30.html`）が生成され、そのままブラウザで開いて30問解答・採点できる

## 質問・ブロッカー
（なし）

## レビュー結果・修正履歴
（実装後に記録）
