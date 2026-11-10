# b-boba-override（B Boba の座標だけ個別に上書きする）

※窓口 (Muse) が 2026-11-10 に作成。青木の決定は「案1（個別の上書き）だけ」。座標補正 CALLOUT_FIX 全体の作り直しや、床外地点の汎用スナップはやらない。

## 背景（切り分け済み）
- Sunset の B Boba は、補正後 (0.230, 0.205) では displayIcon の床の外（最寄りの床まで 0.044）。原因は CALLOUT_FIX（4点アフィン）の北西での残差。元座標 (0.126, 0.344) は床の上にある。
- B Boba は B Site の領域アンカーなので、正しい位置に置き直す。

## やること
1. `vreplay/tactics.py` の `load_map` で、CALLOUT_FIX の後に適用する個別上書きを持てるようにする（例: `CALLOUT_OVERRIDE = {"sunset": {"B Boba": (x, y)}}`）。対象は Sunset の B Boba 1件だけ。
2. 上書き座標は窓口が別途渡す確定値を使う（この指示書の「確定値」欄を窓口が埋めてから着手すること）。
3. テスト: 上書き後の B Boba が displayIcon の床の上（または最寄りの床まで 0.01 以内）にあり、他のコールアウトが変わらないこと。既存の対象テスト（test_tactics / test_mapfig / test_playbook_v4）を通すこと。

## 確定値
- B Boba = (0.1256, 0.3440)（候補A＝公式データの元座標。2026-11-10 に青木が拡大図で確認して決定）

## 触ってよいファイル
- `vreplay/tactics.py`、`tests/test_tactics.py`
- 触らないもの: CALLOUT_FIX の係数、他のマップ・他のコールアウト、kb の YAML
