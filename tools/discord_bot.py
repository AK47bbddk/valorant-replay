"""自宅デスクトップ PC で常駐させ、外出先から Discord 経由で valorant-replay を操作する Bot。

環境変数（リポジトリやコードにハードコードしないこと）:
  DISCORD_BOT_TOKEN        必須: Discord Bot のトークン
  DISCORD_ALLOWED_USER_IDS 必須: 実行を許可する Discord ユーザー ID（カンマ区切り）
  VREPLAY_WORKTREE_DIR     任意: 既定の対象ワークツリー（例: vr-barriers のパス）

使い方:
  pip install -U discord.py
  python tools/discord_bot.py [--worktree PATH] [--prefix "!vr "]
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORCHESTRATOR = ROOT / "tools" / "agy_orchestrator.py"


def _parse_allowed_ids(raw: str) -> set[int]:
    ids: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


async def _exec_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    text = (stdout or b"").decode("utf-8", errors="replace").strip()
    return proc.returncode or 0, text


def _find_cli(name: str) -> list[str] | None:
    for cand in (name, f"{name}.cmd", f"{name}.exe", f"{name}.bat", f"{name}.ps1"):
        found = shutil.which(cand)
        if found:
            if found.lower().endswith(".ps1"):
                return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", found]
            return [found]
    return None


async def _run_prompt_impl(prompt_text: str, worktree: Path) -> str:
    try:
        from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig

        config = LocalAgentConfig(
            system_instructions=(
                "あなたは valorant-replay の開発アシスタントです。"
                "GEMINI.md / AGENTS.md に従い、要点を日本語で簡潔に返してください。"
            ),
            capabilities=CapabilitiesConfig(),
        )
        prev_cwd = Path.cwd()
        os.chdir(worktree)
        chunks: list[str] = []
        try:
            async with Agent(config) as agent:
                resp = await agent.chat(prompt_text)
                async for token in resp:
                    chunks.append(token)
        finally:
            os.chdir(prev_cwd)
        return "".join(chunks).strip() or "(応答が空でした)"
    except ImportError:
        agy_cmd = _find_cli("agy")
        if not agy_cmd:
            return (
                "google-antigravity SDK も `agy` CLI も見つかりませんでした。"
                "自宅 PC にどちらかを導入してください。"
            )
        rc, out = await _exec_subprocess([*agy_cmd, "-p", prompt_text], cwd=worktree)
        return f"[exit={rc}]\n{out}" if out else f"[exit={rc}] (出力なし)"


def run_bot(worktree: Path, prefix: str) -> int:
    try:
        import discord
        from discord.ext import commands
    except ImportError:
        print(
            "discord.py がインストールされていません。自宅 PC で `pip install -U discord.py` を実行してください。",
            file=sys.stderr,
        )
        return 1

    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        print("環境変数 DISCORD_BOT_TOKEN が設定されていません。", file=sys.stderr)
        return 1

    allowed_ids = _parse_allowed_ids(os.environ.get("DISCORD_ALLOWED_USER_IDS", ""))
    if not allowed_ids:
        print(
            "[warn] DISCORD_ALLOWED_USER_IDS が未設定です。セキュリティ保護のため全コマンドが拒否されます。",
            file=sys.stderr,
        )

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(
        command_prefix=commands.when_mentioned_or(prefix),
        intents=intents,
        help_command=commands.DefaultHelpCommand(no_category="コマンド"),
    )

    async def _send_output(ctx: commands.Context, title: str, text: str) -> None:
        body = text or "(出力なし)"
        header = f"**{title}**\n"
        if len(header) + len(body) + 10 <= 1950:
            await ctx.reply(f"{header}```\n{body}\n```", mention_author=False)
            return
        buf = io.BytesIO(body.encode("utf-8"))
        preview = body[-800:]
        await ctx.reply(
            f"{header}出力が長いためファイルで添付します（末尾抜粋）：\n```\n{preview}\n```",
            file=discord.File(buf, filename="output.txt"),
            mention_author=False,
        )

    @bot.check
    async def _check_allowed_user(ctx: commands.Context) -> bool:
        if ctx.author.id not in allowed_ids:
            await ctx.reply(
                "許可されていないユーザーです。`DISCORD_ALLOWED_USER_IDS` にユーザー ID を設定してください。",
                mention_author=False,
            )
            return False
        return True

    @bot.event
    async def on_ready() -> None:
        print(f"Discord Bot 起動完了: {bot.user} / worktree={worktree}")

    @bot.command(name="status", help="ブランチ・worktree・タスク/チケット一覧を表示")
    async def cmd_status(ctx: commands.Context) -> None:
        rc, out = await _exec_subprocess(
            [sys.executable, str(ORCHESTRATOR), "status", "--worktree", str(worktree)],
            cwd=worktree,
        )
        await _send_output(ctx, f"Status (exit={rc})", out)

    @bot.command(name="test", help="pytest を実行（例: !vr test tests/test_reasons.py）")
    async def cmd_test(ctx: commands.Context, *, args_str: str = "") -> None:
        extra = shlex.split(args_str) if args_str.strip() else []
        await ctx.reply(f"テストを実行します: `pytest -x -q {' '.join(extra)}`", mention_author=False)
        rc, out = await _exec_subprocess(
            [sys.executable, "-m", "pytest", "-x", "-q", *extra],
            cwd=worktree,
        )
        await _send_output(ctx, f"Pytest 結果 (exit={rc})", out)

    @bot.command(name="review", help="差分レビューパックを生成・判定（例: !vr review barrier_v2 main..HEAD）")
    async def cmd_review(ctx: commands.Context, name: str, diff_range: str = "main..HEAD") -> None:
        await ctx.reply(f"レビュー準備中: `{name}` (`{diff_range}`)", mention_author=False)
        cmd = [
            sys.executable,
            str(ORCHESTRATOR),
            "review",
            name,
            "--range",
            diff_range,
            "--worktree",
            str(worktree),
            "--no-notify",
        ]
        try:
            import google.antigravity  # noqa: F401
            cmd.append("--use-sdk")
        except ImportError:
            pass
        rc, out = await _exec_subprocess(cmd, cwd=worktree)
        await _send_output(ctx, f"Review 結果 ({name}, exit={rc})", out)

    @bot.command(name="task", help="タスクを実装→テスト→レビューパックまで実行（例: !vr task barrier_v2 agy）")
    async def cmd_task(ctx: commands.Context, name: str, engine: str = "agy") -> None:
        await ctx.reply(f"タスク実行を開始します: `{name}` (engine=`{engine}`)", mention_author=False)
        rc, out = await _exec_subprocess(
            [
                sys.executable,
                str(ORCHESTRATOR),
                "run-task",
                name,
                "--engine",
                engine,
                "--worktree",
                str(worktree),
            ],
            cwd=worktree,
        )
        await _send_output(ctx, f"Task 完了 ({name}, exit={rc})", out)

    @bot.command(name="prompt", help="Antigravity (SDK / agy CLI) に自由記述で指示を出す")
    async def cmd_prompt(ctx: commands.Context, *, text: str) -> None:
        await ctx.reply("エージェントへ送信中…", mention_author=False)
        try:
            result = await _run_prompt_impl(text, worktree=worktree)
        except Exception as e:
            result = f"実行エラー ({type(e).__name__}): {e}"
        await _send_output(ctx, "Agent 応答", result)

    bot.run(token)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="discord_bot",
        description="valorant-replay 遠隔オーケストレーション用 Discord Bot",
    )
    ap.add_argument(
        "--worktree",
        default=os.environ.get("VREPLAY_WORKTREE_DIR") or str(ROOT),
        help="作業対象のワークツリーディレクトリ（既定: VREPLAY_WORKTREE_DIR またはリポジトリルート）",
    )
    ap.add_argument("--prefix", default="!vr ", help="コマンドプレフィックス（既定: '!vr '）")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    worktree = Path(args.worktree).expanduser().resolve()
    if not worktree.exists():
        print(f"対象ディレクトリが見つかりません: {worktree}", file=sys.stderr)
        return 1
    return run_bot(worktree=worktree, prefix=args.prefix)


if __name__ == "__main__":
    raise SystemExit(main())
