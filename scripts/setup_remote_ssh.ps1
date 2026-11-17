<#
.SYNOPSIS
    ノート PC 用の SSH 鍵（id_ed25519）を生成し、公開鍵とデスクトップ PC 接続設定の案内を表示する。
.DESCRIPTION
    既存の ~/.ssh/id_ed25519 があれば上書きせず、そのまま公開鍵を表示・クリップボードへコピーします。
#>
[CmdletBinding()]
param(
    [string]$Comment = "laptop-valorant-replay",
    [string]$DesktopHostAlias = "desktop-pc",
    [string]$DesktopHostName = "<デスクトップPCのTailscaleまたはローカルIP>",
    [string]$DesktopUser = $env:USERNAME
)

$ErrorActionPreference = "Stop"

$sshDir = Join-Path $HOME ".ssh"
$privKey = Join-Path $sshDir "id_ed25519"
$pubKey = Join-Path $sshDir "id_ed25519.pub"

if (-not (Test-Path $sshDir)) {
    New-Item -ItemType Directory -Path $sshDir -Force | Out-Null
}

if (Test-Path $privKey) {
    Write-Host "[INFO] 既存の SSH 秘密鍵が見つかりました（上書きしません）: $privKey" -ForegroundColor Cyan
} else {
    Write-Host "[INFO] 新しい Ed25519 SSH 鍵を生成します: $privKey" -ForegroundColor Green
    & ssh-keygen -t ed25519 -C $Comment -N '""' -f $privKey
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen に失敗しました (終了コード: $LASTEXITCODE)"
    }
}

if (-not (Test-Path $pubKey)) {
    throw "公開鍵が見つかりません: $pubKey"
}

$pubKeyText = (Get-Content -Path $pubKey -Raw).Trim()

try {
    Set-Clipboard -Value $pubKeyText
    $copiedMsg = "（クリップボードにコピーしました）"
} catch {
    $copiedMsg = "（クリップボードへの自動コピーはスキップされました）"
}

Write-Host ""
Write-Host "=== 1. 公開鍵 ($pubKey) $copiedMsg ===" -ForegroundColor Yellow
Write-Host $pubKeyText
Write-Host ""

Write-Host "=== 2. デスクトップ PC 側での登録方法 ===" -ForegroundColor Yellow
Write-Host "[Windows OpenSSH Server（標準ユーザー）の場合]" -ForegroundColor Cyan
Write-Host "  `$auth = `"`$env:USERPROFILE\.ssh\authorized_keys`""
Write-Host "  New-Item -ItemType Directory -Force -Path (Split-Path `$auth) | Out-Null"
Write-Host "  Add-Content -Path `$auth -Value '$pubKeyText'"
Write-Host ""
Write-Host "[Windows OpenSSH Server（管理者ユーザー）の場合]" -ForegroundColor Cyan
Write-Host "  管理者 PowerShell で C:\ProgramData\ssh\administrators_authorized_keys に上記1行を追記してください。"
Write-Host ""
Write-Host "[WSL / Linux の場合]" -ForegroundColor Cyan
Write-Host "  mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$pubKeyText' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
Write-Host ""

Write-Host "=== 3. ノート PC 側 ~/.ssh/config の設定例 ===" -ForegroundColor Yellow
$configExample = @"
Host $DesktopHostAlias
    HostName $DesktopHostName
    User $DesktopUser
    IdentityFile $privKey
    IdentitiesOnly yes
"@
Write-Host $configExample
Write-Host ""
Write-Host "設定後、以下のコマンドで接続と vr-barriers ワークツリーを確認できます:" -ForegroundColor Green
Write-Host "  ssh $DesktopHostAlias"
Write-Host "  ssh $DesktopHostAlias `"git -C C:\Users\$DesktopUser\valorant-replay worktree list`""
