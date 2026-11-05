# TactiVal の座標合わせ

**Split はすでに合わせてあります**（`vreplay/data/tactival_calib.json` に同梱。4 点・ずれ 36px・回転 92.9 度）。
ほかのマップを使うときだけ、下の手順で 1 回合わせてください。

TactiVal（作戦ボードのアプリ）のマップはベクター画像で、valorant-api の座標とは原点も倍率も違います。
マップごとに 1 回だけ、次の手順で合わせてください。

1. このフォルダの `calib-<マップ>.tactival.json` を TactiVal で読み込む
2. 4 体の駒を、それぞれ正しい場所へドラッグする

   | エージェント | 置く場所 |
   |---|---|
   | Jett | 攻撃側スポーン |
   | Sage | 防衛側スポーン |
   | Omen | A サイト（設置点のあたり） |
   | Viper | B サイト |
   | Cypher | C サイト（3 サイトのマップだけ） |

3. JSON で書き出して、次を 1 回実行する

   ```bash
   vreplay tactival --calibrate <書き出した.json> --map Sunset
   ```

倍率と原点が `output.dir/_cache/tactival_calib.json` に記録され、以降 `vreplay playbook` が
型ごとの TactiVal ボードを `playbook/tactival/` に書き出します。

- 3 体以上あれば**アフィン変換（回転・鏡映・縦横の倍率）**で合わせます。TactiVal のマップは
  valorant-api の画像に対して回っていることがあり（**Split は 90 度**）、倍率と原点だけでは合いません。
  2 体だけのときは回転を見られないので、必ず 3 体以上置いてください。
- 駒の `x`/`y` はアイコンの左上なので、中身では真ん中に直してから計算しています。
- ここに置いてあるボードの初期位置は推定値です。正しい場所にドラッグしてから書き出してください。
