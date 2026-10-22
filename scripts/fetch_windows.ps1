# 自宅の Windows PC で、前日の試合動画（該当マップ区間だけ）を Google ドライブに保存する。
# Colab はクラウドの IP なので YouTube にボット判定されやすい。家庭回線で落として Drive 経由で渡す。
#
# 初回だけ（PowerShell）:
#   winget install Git.Git Python.Python.3.12 Gyan.FFmpeg DenoLand.Deno
#   cd C:\Users\aokki\programing
#   git clone https://github.com/AK47bbddk/valorant-replay
#   py -m pip install -e ".\valorant-replay[fetch]"
# 毎朝の自動実行：タスクスケジューラで「powershell -ExecutionPolicy Bypass -File <このファイル>」を登録

$Repo  = "C:\Users\aokki\programing\valorant-replay"
# Google ドライブ（パソコン版）のマイドライブ配下。ドライブ文字は環境に合わせて変える
$Drive = "G:\マイドライブ\valorant-replay"

$env:VREPLAY_OUTPUT_DIR = $Drive
Set-Location $Repo
git pull -q
py -m vreplay fetch @args *>> (Join-Path $Drive "fetch.log")
