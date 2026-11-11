# boundary-walls-sunset（青木が引いた境界線10本を kb のデータにして、その線で切る）

※窓口 (Muse) が 2026-11-11 に作成。青木が Tactival で Sunset の境界線の例として線分10本を提示した（変換済み座標は複製内 .review_tmp/sunset_walls.json、重ね図は窓口の output/sunset_walls_overlay.png）。全て通路のボトルネックに壁から壁まで引かれている。

## 線分の対応（displayIcon 座標 [u1, v1, u2, v2]）
1. [0.0853, 0.5763, 0.1525, 0.5763] B Main 通路（C10 の境界線。チョーク点より少し北）
2. [0.2429, 0.4884, 0.2441, 0.5348] B サイト前面の縦線（C12）
3. [0.3797, 0.4408, 0.3821, 0.4786] Mid | B Market（C11）
4. [0.7461, 0.5996, 0.7461, 0.6631] A Lobby → A Main（C14）
5. [0.6741, 0.4725, 0.7535, 0.4737] A Main | A Site（C15）
6. [0.8915, 0.2783, 0.8927, 0.3553] A サイト北東ドアの縦線（C16＋エルボー対策。出入口の高さ全体を塞ぐ）
7. [0.4713, 0.3601, 0.5446, 0.3601] Mid | Mid Top（C13）
8. [0.3198, 0.2490, 0.3198, 0.2868] B Market 北の通路
9. [0.7828, 0.2355, 0.8451, 0.2355] Mid Top | A Link
10. [0.6777, 0.3382, 0.7388, 0.3382] Mid Top 南東の通路

## やること
1. kb（config/kb/maps/sunset.yaml）に `walls` として上記10本を入れる（[u1, v1, u2, v2] のリスト）。
2. mapfig: 区画分けの前に、walls の線分を歩行マスクへ切れ目として焼き込む（256グリッドで太さ2〜3px、線分の両端は壁に届いている前提で延長しない）。チョークの円・バリアは、近くに wall があるチョークでは使わない（wall を優先）。wall が無いチョークは従来どおり。
3. 前タスク（c10-site-boundary-and-bottleneck-cuts）で追加したエルボーのチョーク [0.918, 0.345, A Site, A Alley] は削除する（線分6がその役割を果たす）。C10 のペア [B Main, B Site] とコールアウトの付け替え（B Site に B Main、B Main は B Lobby のみ）は維持する。
4. 受け入れ条件（テストにする）:
   - 各 wall の両側 0.03 の点（線分の中点から法線方向）が、想定のエリア対になること（1: B Main/B Site、2: B Main か B Market / B Site、3: Mid/B Market、4: A Main/A Main（同一エリア内の線）、5: A Main/A Site、6: A Site / A Alley か A Link、7: Mid/Mid Top、8: B Site 側 / B Market、9: Mid Top/A Link、10: Mid Top / A Main 側）。
   - B Site の領域が線分1より南（西側 u<0.17）にでないこと。A Site の領域が線分6より東に出ないこと。
   - 既存の不変条件とテストが全て通ること（落ちる場合はモデル変更に合わせて理由つきで直す）。
5. Sunset の領域図を _review/ に再作成し、PROMPT.md を実態に合わせる。レビューパックを作る。push・マージはしない。

## 触ってよいファイル
- config/kb/maps/sunset.yaml、vreplay/mapfig.py、vreplay/kb.py（walls の読み取りで必要な場合のみ）、tests/、PROMPT.md
