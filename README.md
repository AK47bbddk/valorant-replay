# valorant-replay

前日（JST）に行われた VALORANT のプロ試合を、**翌朝 Notion のマップページを開くだけでふり返れる**ようにするパイプラインです。
重い処理は Google Colab の GPU で動かし、コードはこのリポジトリから `git pull` して使います。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AK47bbddk/valorant-replay/blob/main/colab.ipynb)

## 何ができるか

```
vlr.gg 結果一覧 ──► 試合ページ（マップ・ラウンド勝敗・エコ・選手・マップ別VOD）
        │                              │
        │                              ▼
        │                  Notion「大会ログくん」「マップ詳細」に upsert
        ▼
YouTube の該当マップ区間だけ取得（yt-dlp）
        │
        ▼  2fps でデコード、上部 HUD のタイマーとスコアを OCR（EasyOCR / GPU）
ラウンド分割 ＋ リプレイ・ポーズ除外
        │
        ▼  ラウンドごとに
開始時ミニマップ ／ ミニマップ GIF ／ 購入パネル ＋ エコ区分 ／ キルログ ／ ULT 使用（β）／ 該当秒への YouTube リンク
        │
        ▼
Notion「マップ詳細」ページ本文 ＋ Drive に HTML レポート
```

各マップページの中身：

- 概要（VOD・VLR へのリンク、スコア、解析の信頼度）
- ラウンド一覧表（勝者・勝ち方・攻守・両チームの購入段階・キル数・ラウンド時間）
- ラウンドごとのトグル見出し
  - ▶ このラウンドを見る（YouTube の該当秒）、両チームの購入段階と所持金
  - ラウンド開始時のミニマップ、ラウンド中のミニマップ GIF（4 倍速）
  - 購入フェーズ終了時のプレイヤーパネル（武器・アーマー・ULT ポイントが写る）
  - キルログ（時刻・キラー → 被キル、VLR の選手名に補正）とキルフィードの切り抜き
  - ULT 使用（β）

## 初回セットアップ（1 回だけ）

1. **Notion のトークン**
   [My integrations](https://www.notion.so/profile/integrations) で内部インテグレーションを作り、シークレットをコピー。
   Notion の「Valo」ページ右上 `…` → 接続 → そのインテグレーションを追加（配下の 2 つの DB に書けるようになる）。
2. **Colab のシークレット**：ノートブック左の 🔑 に `NOTION_TOKEN` を登録し、ノートブックからのアクセスを許可。
3. **GPU**：ランタイム → ランタイムのタイプを変更 → T4 GPU。
4. （任意）設定の上書き：Drive の `MyDrive/valorant-replay/config.yaml` に変えたい項目だけ書く。例：

   ```yaml
   filters:
     event_include: ["VCT", "Challengers 2026: Japan"]
     teams: ["ZETA DIVISION", "DetonatioN FocusMe"]
   ```

## 毎朝の使い方

`colab.ipynb` を開いて 2 つのセルを実行するだけです。

```python
# セル1：最新コードを取得
!git -C /content/valorant-replay pull -q 2>/dev/null || git clone -q https://github.com/AK47bbddk/valorant-replay /content/valorant-replay
%cd /content/valorant-replay
!pip -q install -r requirements.txt

# セル2：実行
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
os.environ['NOTION_TOKEN'] = userdata.get('NOTION_TOKEN')
os.environ['VREPLAY_CONFIG'] = '/content/drive/MyDrive/valorant-replay/config.yaml'
!python -m vreplay run
```

オプション：

| コマンド | 内容 |
|---|---|
| `run --date 2026-10-21` | 日付（JST）を指定 |
| `run --match <VLRのURL>` | 1 試合だけ |
| `run --no-video` | VLR のデータだけ（GPU 不要・数十秒） |
| `run --force` | 作成済みのマップページも作り直す |
| `calibrate --match <URL> --map 1 --at 600` | HUD の ROI を描いた確認画像を出す |

## HUD の位置がずれるとき（最初に 1 回確認推奨）

放送レイアウトは大会や年で変わります。タイマーとスコアは OCR で位置を自動検出しますが、
ミニマップ・キルフィード・プレイヤーパネルは `config/default.yaml` の比率座標で切り抜いています。

```
!python -m vreplay calibrate --match https://www.vlr.gg/xxxxx --map 1 --at 600
```

Drive の `valorant-replay/calibrate/rois.jpg` に枠が描かれた画像が出るので、ずれていれば
`config.yaml` の `hud.roi` を直してください（値は画面幅・高さに対する 0〜1 の比率）。

## YouTube に弾かれるとき

Colab の IP から「Sign in to confirm you're not a bot」と言われることがあります。
ブラウザ拡張などで `cookies.txt`（Netscape 形式）を書き出して Drive に置き、`config.yaml` に

```yaml
video:
  cookies: /content/drive/MyDrive/valorant-replay/cookies.txt
```

と書いてください。cookies はリポジトリには絶対に入れないでください。

## 精度について

- **ラウンド分割・リプレイ除外**：スコア（左＋右）が確定的に増えた瞬間をラウンド終了、タイマーが 1:40 付近に跳んだ瞬間を開始とし、
  タイマーの巻き戻り・停止・古いスコアの表示をリプレイ／ポーズとして捨てています。HUD が読めればかなり安定します。
- **キルログ**：キルフィード領域の OCR で「左の名前 → 右の名前」の行を拾い、VLR の選手名に寄せています。
- **ULT 使用（β）**：パネルの ULT アイコンの光り方の変化で判定しているため、放送レイアウトによっては調整（`hud.ult_in_row`, `hud.ult_ready_threshold`）が必要です。ULT ポイントそのものは購入パネルの画像で確認してください。
- 解析が一部失敗しても、VLR 由来のデータとスクリーンショットだけでページは作られます。ページ冒頭の灰色の行に読み取り率が出ます。

## 開発

```
pip install -r requirements.txt pytest
pytest -q        # ネットワーク・GPU なしで通る（合成データ）
```

仕様は [`PROMPT.md`](PROMPT.md) にまとめてあります（AI エージェントに改修を頼むときはこれを渡す）。

| ファイル | 役割 |
|---|---|
| `vreplay/vlr.py` | vlr.gg の結果一覧・試合ページ・エコ・VOD のパース |
| `vreplay/notion.py` | Notion API（upsert、画像アップロード、ページ本文の生成） |
| `vreplay/video.py` | yt-dlp で区間ダウンロード、ffmpeg でフレーム供給 |
| `vreplay/hud.py` | OCR、タイマー／スコアの解釈、ROI 自動キャリブレーション |
| `vreplay/rounds.py` | ラウンド分割とリプレイ除外 |
| `vreplay/extract.py` | ミニマップ GIF、購入パネル、キルログ、ULT |
| `vreplay/pipeline.py` | 全体の流れ |
| `vreplay/report.py` | HTML レポート |
