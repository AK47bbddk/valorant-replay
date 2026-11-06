#!/usr/bin/env bash
# YouTube の PO トークンを作るサーバー（bgutil-ytdlp-pot-provider）を入れて、裏で起動する。
#
# YouTube は PO トークンの無い取得に 360p（形式 18）しか返さなくなった（1080p は 403 で切られる）。
# 360p ではキルログ・カード・ミニマップの縁が読めないので、Colab ではセル1でこれを動かす。
# yt-dlp のプラグイン（pip）がサーバー（127.0.0.1:4416）から自動でトークンをもらう。
# 2 回目以降はサーバーが動いていれば何もしない。
set -e
if curl -fs -o /dev/null http://127.0.0.1:4416/ping; then
  exit 0
fi
pip -q install -U bgutil-ytdlp-pot-provider
V=$(pip show bgutil-ytdlp-pot-provider | awk '/^Version/{print $2}')   # サーバーはプラグインと同じ版にする
D=/content/bgutil-ytdlp-pot-provider
if [ ! -f "$D/server/build/main.js" ] || [ "$(cat "$D/.version" 2>/dev/null)" != "$V" ]; then
  rm -rf "$D"
  git -c advice.detachedHead=false clone -q --single-branch --branch "$V" \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$D"
  (cd "$D/server" && npm ci --silent > /dev/null 2>&1 && npx tsc)
  echo "$V" > "$D/.version"
fi
nohup node "$D/server/build/main.js" > /content/pot_server.log 2>&1 &
for _ in $(seq 20); do
  curl -fs -o /dev/null http://127.0.0.1:4416/ping && { echo "PO トークンのサーバーを起動しました（v$V）"; exit 0; }
  sleep 1
done
echo "PO トークンのサーバーが起動しませんでした（/content/pot_server.log を確認）。360p しか取れないことがあります" >&2
