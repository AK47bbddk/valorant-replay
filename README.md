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
!pip -q install -e '/content/valorant-replay[video]'
!test -x ~/.deno/bin/deno || (curl -fsSL https://deno.land/install.sh | sh -s -- -y > /dev/null 2>&1)
%cd /content/valorant-replay

# セル2：実行
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
os.environ['NOTION_TOKEN'] = userdata.get('NOTION_TOKEN')
os.environ['VREPLAY_CONFIG'] = '/content/drive/MyDrive/valorant-replay/config.yaml'
!vreplay run
```

オプション：

| コマンド | 内容 |
|---|---|
| `run --date 2026-10-21` | 日付（JST）を指定 |
| `run --match <VLRのURL>` | 1 試合だけ |
| `run --no-video` | VLR のデータだけ（GPU 不要・数十秒） |
| `run --force` | 作成済みのマップページも作り直す |
| `run --upsert-only` | Notion の試合・マップ行だけ更新（本文・動画解析なし） |
| `calibrate --match <URL> --map 1 --at 600` | HUD の ROI を描いた確認画像を出す |
| `fetch` | 動画の該当区間だけを Drive に保存（自宅 PC 用、下の「方法 B」） |

## 試合リストの自動更新（GitHub Actions）

`.github/workflows/daily-notion.yml` が毎朝 5:23（JST）に前日の試合を VLR から取り、Notion の「大会ログくん」「マップ詳細」の**行だけ**を更新します（GPU 不要、本文は書かない）。
朝 Colab を実行すると、その行のマップページ本文に動画解析が埋まります。

有効にするには、GitHub のリポジトリ → Settings → Secrets and variables → Actions に `NOTION_TOKEN` を登録してください。
Actions タブの「daily-notion」→「Run workflow」で日付を指定した手動実行もできます。

## HUD の位置がずれるとき（最初に 1 回確認推奨）

放送レイアウトは大会や年で変わります。タイマーとスコアは OCR で位置を自動検出しますが、
ミニマップ・キルフィード・プレイヤーパネルは `config/default.yaml` の比率座標で切り抜いています。

```
!vreplay calibrate --match https://www.vlr.gg/xxxxx --map 1 --at 600
```

Drive の `valorant-replay/calibrate/rois.jpg` に枠が描かれた画像が出るので、ずれていれば
`config.yaml` の `hud.roi` を直してください（値は画面幅・高さに対する 0〜1 の比率）。

## YouTube に弾かれるとき（「Sign in to confirm you're not a bot」）

Colab は Google のデータセンターの IP なので、YouTube にボット判定されやすいです。
yt-dlp の YouTube OAuth ログインは YouTube 側の変更で使えなくなっているため（yt-dlp 公式 wiki）、次のどちらかで回避します。

### 方法 A：自分の YouTube の cookies を渡す（手軽・Colab だけで完結）

1. ブラウザで**シークレットウィンドウ**を開き、YouTube にログインする（BAN のリスクを避けるならサブアカウント推奨）。
2. 同じタブで `https://www.youtube.com/robots.txt` を開く（シークレットのタブはこれ 1 つだけにする）。
3. 拡張機能「Get cookies.txt LOCALLY」などで youtube.com の cookies を **Netscape 形式**で書き出す。
4. シークレットウィンドウを閉じる（開いたままだと cookies が更新されて無効になる）。
5. 書き出したファイルを Drive の `MyDrive/valorant-replay/cookies.txt` に置く。
   （または中身を Colab の 🔑 に `YT_COOKIES` として登録）

cookies は数週間〜数か月で切れます。またボット判定が出たら書き出し直してください。

### 方法 B：動画だけ自宅 PC で落とす（いちばん確実）

家庭回線ならボット判定はほぼ出ません。自宅の Windows PC で `vreplay fetch` を動かすと、
前日の対象マップの区間だけを Google ドライブ（パソコン版）の `valorant-replay/videos` に保存し、
Colab の `vreplay run` は動画をダウンロードせずにそこから読みます。

```powershell
# 初回だけ
winget install Git.Git Python.Python.3.12 Gyan.FFmpeg DenoLand.Deno
cd C:\Users\aokki\programing
git clone https://github.com/AK47bbddk/valorant-replay
py -m pip install -e ".\valorant-replay[fetch]"

# 実行（scripts/fetch_windows.ps1 の $Drive をマイドライブのパスに合わせる）
powershell -ExecutionPolicy Bypass -File .\valorant-replay\scripts\fetch_windows.ps1
```

タスクスケジューラで毎朝（GitHub Actions の 5:23 より後、例えば 5:45）に登録すれば、
起きた時点で動画が Drive に揃っていて、Colab の実行は解析だけになります。
1 マップあたり 1080p で 0.5〜1.5 GB 程度です。容量が気になるときは `video.max_height: 720` に下げてください。

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
