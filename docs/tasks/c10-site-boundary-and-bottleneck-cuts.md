# c10-site-boundary-and-bottleneck-cuts（C10 を B の境界にし、境界はボトルネックで垂直に切る）

※窓口 (Muse) が 2026-11-11 に作成。青木の決定: C10 は B 案（C10 より北は B Site）。一般則: 境界は「他と比べて細くなっている部分で、移動方向に垂直な直線」にする。斜めの距離境界は不可。

## 1. Sunset の kb（config/kb/maps/sunset.yaml）
- C10 (0.1134, 0.5867) のペアを [B Main, B Site] に戻す。
- B Site の callouts に "B Main" を加え、B Main の callouts は ["B Lobby"] のみにする（C10 より北の通路はサイト側という青木のモデル）。
- A Elbow の縦帯にチョークを1件追加する: [0.918, 0.345, A Site, A Alley]（窓口の実測: Elbow→Alley の通路幅プロファイルで縦帯の幅 0.031 が続く位置。A Site と A Alley の境界をここで横に切る）。

## 2. mapfig の切れ目
- C10 と追加した Elbow のチョークは、円ではなく断面バリアで切ること（既存の _barrier_mask を使う。円で局所分離できても、領域がチョークを回り込む場合はバリアにする。特に C10 は西端の細い通路を伝う回り込みを止める）。
- 受け入れ条件（テストにする）:
  - B Site の領域が、西側（u < 0.17）で C10 の緯度（v = 0.5867）より南に出ないこと。
  - A Site と A Alley の境界が、追加チョークの位置で通路を横切る直線になっていること（チョーク両側 0.04 のプローブが別区画で、北側プローブのエリアが A Alley か A Link、南側が A Site）。
  - 既存の不変条件（局所分離・サイト側への入口画素の混入なし）は全て維持する。除外はしない。

## 3. その他
- 既存テスト（test_kb / test_playbook_v4 / test_gamemodel / test_tactival / test_meta / test_deck / test_mapfig / test_tactics）を全て通す。kb の変更で落ちるテストは、変更後のモデルに合わせて直してよい（理由を報告に書く）。
- Sunset の領域図を _review/ に再作成し、PROMPT.md の記述を実態に合わせる。
- 触ってよいファイル: config/kb/maps/sunset.yaml、vreplay/mapfig.py、vreplay/kb.py（必要最小限）、tests/、PROMPT.md。push・マージはしない。
