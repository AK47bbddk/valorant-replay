# valorant-replay

前日（JST）に行われた VALORANT のプロ試合を、**翌朝 Notion のマップページを開くだけでふり返れる**ようにするパイプラインです。
重い処理は Google Colab の CPU（並列）と GPU（デコード）で動かし、コードはこのリポジトリから `git pull` して使います。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AK47bbddk/valorant-replay/blob/main/colab.ipynb)

## 何ができるか

```
vlr.gg 結果一覧 ──► 試合ページ（マップ・ラウンド勝敗・エコ・選手・マップ別VOD）
        │                              │
        │                              ▼
        │                  Notion「大会ログくん」「マップ詳細」に upsert
        ▼
YouTube の該当マップ区間だけ取得（yt-dlp、次のマップは裏で先にダウンロード）
        │
        ▼  キャリブレーション：放送レイアウト（Riot / CN）を自動選択、カード位置は ULT アイコンの丸から、
        │  キルフィードは選手名から自動検出
        ▼  並列スキャン（CPU 数ぶんのプロセス）：2fps でデコード、HUD を OCR（RapidOCR・見た目キャッシュ）
ライブ判定（スコアバーの見た目）でリプレイ・選手カメラ・解説席を除外 → ラウンド分割（スコア＋タイマー）
        │
        ▼  ラウンドごとに（スレッドで並列）
開始時ミニマップ ／ ミニマップ GIF ／ 購入パネル・各選手の武器 ／ キルログ（武器つき）／ ULT 使用 ／ YouTube リンク
        │
        ▼
Notion「マップ詳細」ページ本文 ＋ Drive に HTML レポート
```

各マップページの中身：

- 概要（VOD・VLR へのリンク、スコア、解析の信頼度と所要時間）
- **ラウンド一覧表**（1 ラウンド 1 行）

  | R | スコア | 勝者 | 勝ち方 | VIT 購入・武器 | LOUD 購入・武器 | 先制キル | キル | ULT 使用 | 時間 |
  |---|---|---|---|---|---|---|---|---|---|
  | R6 ▶ | 1-5 | LOUD | 💀全滅（攻撃） | フル 21.3k<br>ファントム×3・ヴァンダル・オペレーター | ハーフ 11.6k<br>ヴァンダル×2・シェリフ×2 | Derke→tkzin（VIT） | 3-5 | Darker（Waylay） | 1:05 |

- ラウンドごとのトグル見出し
  - ▶ このラウンドを見る（YouTube の該当秒）、両チームの購入段階・残り資金・各選手の武器
  - ラウンド開始時のミニマップ、ラウンド中のミニマップ GIF（4 倍速・リプレイ除外）
  - 購入フェーズ終了時のプレイヤーパネル
  - キルログ（時刻・キラー［武器］→ 被キル）とキルフィードの切り抜き
  - ULT（開始時に溜まっていた選手、使用した選手と時刻）

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
| `calibrate --match <URL> --map 1 --at 600` | レイアウト・カード・キルフィードの検出結果を画像と表で確認 |
| `fetch` | 動画の該当区間だけを Drive に保存（自宅 PC 用、下の「方法 B」） |

## 試合リストの自動更新（GitHub Actions）

`.github/workflows/daily-notion.yml` が毎朝 5:23（JST）に前日の試合を VLR から取り、Notion の「大会ログくん」「マップ詳細」の**行だけ**を更新します（GPU 不要、本文は書かない）。
朝 Colab を実行すると、その行のマップページ本文に動画解析が埋まります。

有効にするには、GitHub のリポジトリ → Settings → Secrets and variables → Actions に `NOTION_TOKEN` を登録してください。
Actions タブの「daily-notion」→「Run workflow」で日付を指定した手動実行もできます。

## 検出結果を確かめたいとき

放送レイアウト（Riot 共通 / VCT CN）・プレイヤーカードの位置・キルフィードの位置は自動で決めます。
結果を目で確かめたいときは次を実行してください。

```
!vreplay calibrate --match https://www.vlr.gg/xxxxx --map 1 --at 600
```

Drive の `valorant-replay/calibrate/` に、枠を描いた `rois.jpg` とカードの判定を描いた `panels.jpg` が出て、
セルにはカードごとの「名前・生存・ULT・武器」が表で出ます。

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

実際の放送（VCT China 2026 Threshold と Champions 2026 Shanghai）から切り出した HUD で確かめています（`tests/`）。

- **リプレイ・非ライブの除外**：画面上部のスコアバーの見た目を、タイマーとスコアが読めたフレームから作ったお手本と比べます。
  リプレイは別のミニバー、選手カメラや解説席はバーが無いので外れます（実データで、ライブとの差 2〜4 に対しリプレイは 24 以上）。
- **ラウンド分割**：ライブのフレームのスコア合計＋1 をラウンド番号とし、確定した増加だけを採用します。
  開始は購入フェーズ（0:30→0:00）のあと 1:40 付近に跳んだ瞬間で、映っていなければ残り時間から逆算します（実データで誤差 1 秒以内）。
  タイマーが止まった区間（テクニカルポーズ）はライブ区間から除きます。
- **武器**：カードの白いシルエットを、valorant-api.com の武器アイコンと HUD から切り出した見本に重ねて判別します。
  実データでファントム・ヴァンダル・シェリフ・ゴースト・クラシックを確認済み。自信がないときは「不明」と表示します。
- **ULT**：アイコン外周の点線が全部つながって光ると「溜まっている」。生存したまま光った輪が消えたら「使用」です。
  倒されたカードは灰色になるので判定から外します。
- **選手名**：カードの名前欄を読み、VLR の出場選手に寄せます（実データで 20 人中 20 人一致）。
- 解析が一部失敗しても、VLR 由来のデータとスクリーンショットだけでページは作られます。ページ冒頭の灰色の行に読み取り率が出ます。

## 速さについて

- マップ区間を CPU 数ぶんの時間チャンクに分けて別プロセスで同時に解析します（Colab の CPU を全部使う）。
- HUD の数字は同じ見た目が何度も出るので、見た目が一致したら前の結果を使い回し、実際の OCR は 1 割程度に減らしています。
- GPU は動画のデコード（NVDEC）に使います（ffmpeg が対応していれば自動。ログに「デコード GPU」と出ます）。
- 解析中に次のマップの動画を裏でダウンロードします。
- ログに「キャリブレーション／スキャン／ラウンドごとの素材」の所要時間が出ます。

## 開発

```
pip install -r requirements.txt pytest
pytest -q        # ネットワーク・GPU なしで通る（実放送の HUD 切り出し＋合成データ）
```

仕様は [`PROMPT.md`](PROMPT.md) にまとめてあります（AI エージェントに改修を頼むときはこれを渡す）。

| ファイル | 役割 |
|---|---|
| `vreplay/vlr.py` | vlr.gg の結果一覧・試合ページ・エコ・VOD のパース |
| `vreplay/notion.py` | Notion API（upsert、画像アップロード、ページ本文の生成） |
| `vreplay/video.py` | yt-dlp で区間ダウンロード、ffmpeg でフレーム供給 |
| `vreplay/ocr.py` | RapidOCR のラッパーと見た目キャッシュ |
| `vreplay/layout.py` | 放送レイアウトのプリセット、タイマー／スコアの解釈、カード位置の検出 |
| `vreplay/scan.py` | 並列スキャン（デコード → HUD の OCR → カードの状態） |
| `vreplay/rounds.py` | ライブ判定（リプレイ除外）とラウンド分割 |
| `vreplay/weapons.py` | 武器シルエットの判別 |
| `vreplay/extract.py` | ミニマップ GIF、購入パネル、キルログ、ULT |
| `vreplay/summary.py` | ラウンド一覧表のセルの組み立て |
| `vreplay/pipeline.py` | 全体の流れ |
| `vreplay/report.py` | HTML レポート |
