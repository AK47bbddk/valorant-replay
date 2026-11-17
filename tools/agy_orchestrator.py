"""Antigravity (`agy`) を軸に valorant-replay の開発・テスト・レビューを統括する CLI。

使い方:
  python tools/agy_orchestrator.py status [--worktree PATH]
  python tools/agy_orchestrator.py new-task <name> [--title ...] [--files ...] [--tests ...]
  python tools/agy_orchestrator.py review <name> [--range main..HEAD] [--out FILE] [--use-sdk]
  python tools/agy_orchestrator.py run-task <name> [--engine agy|sdk|opencode] [--worktree PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_worktree(worktree: str | None) -> Path:
    raw = worktree or os.environ.get("VREPLAY_WORKTREE_DIR") or str(ROOT)
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"対象ディレクトリが見つかりません: {path}")
    return path


def _find_task_file(root: Path, name: str) -> Path | None:
    """タスク指示書（docs/tasks/<name>.md や ticket_<name>.md）を探す。"""
    p = Path(name)
    if p.is_file():
        return p.resolve()
    stem = name[:-3] if name.endswith(".md") else name
    candidates = [
        root / "docs" / "tasks" / f"{stem}.md",
        root / f"ticket_{stem}.md",
        root / f"{stem}.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _find_cli(name: str) -> list[str] | None:
    for cand in (name, f"{name}.cmd", f"{name}.exe", f"{name}.bat", f"{name}.ps1"):
        found = shutil.which(cand)
        if found:
            if found.lower().endswith(".ps1"):
                return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", found]
            return [found]
    return None


def _notify(root: Path, title: str, body: str, level: str = "info", no_notify: bool = False) -> None:
    """vreplay.notify を使って通知を送る（未設定・通信失敗でも本体を止めない）。"""
    if no_notify:
        return
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from vreplay import config, notify

        try:
            cfg = config.load()
        except BaseException:
            cfg = {}
        notify.send(cfg, title, body, level=level)
    except Exception as e:
        print(f"[warn] 通知の送信をスキップしました ({type(e).__name__}: {e})", file=sys.stderr)


async def _run_sdk_async(
    prompt: str,
    system_instructions: str,
    read_only: bool,
    cwd: Path,
) -> str:
    try:
        from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
    except ImportError as e:
        raise SystemExit(
            "google-antigravity Python SDK が見つかりません。"
            "`pip install google-antigravity` で導入するか、`--engine agy` / `--engine opencode` を使用してください。"
        ) from e

    kwargs: dict = {"system_instructions": system_instructions}
    if not read_only:
        kwargs["capabilities"] = CapabilitiesConfig()

    config = LocalAgentConfig(**kwargs)
    prev_cwd = Path.cwd()
    chunks: list[str] = []
    os.chdir(cwd)
    try:
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            async for token in response:
                chunks.append(token)
    finally:
        os.chdir(prev_cwd)
    return "".join(chunks).strip()


def run_sdk(
    prompt: str,
    system_instructions: str,
    read_only: bool = True,
    cwd: Path = ROOT,
) -> str:
    return asyncio.run(_run_sdk_async(prompt, system_instructions, read_only=read_only, cwd=cwd))


def _dispatch_implementation(
    engine: str,
    prompt: str,
    cwd: Path,
    model: str | None = None,
) -> tuple[int, str, str]:
    if engine == "sdk":
        sys_inst = (
            "あなたは valorant-replay の実装担当エージェントです。"
            "リポジトリ内の GEMINI.md と AGENTS.md のルールを厳守し、"
            "指示書の「触ってよいファイル」だけを変更してテストを通してください。"
        )
        out = run_sdk(prompt, sys_inst, read_only=False, cwd=cwd)
        return 0, out, ""

    if engine == "agy":
        base_cmd = _find_cli("agy")
        if not base_cmd:
            raise SystemExit(
                "`agy` CLI が PATH に見つかりません。Antigravity CLI を導入するか、`--engine sdk` / `--engine opencode` を指定してください。"
            )
        cmd = [*base_cmd]
        if model:
            cmd.extend(["--model", model])
        cmd.extend(["-p", prompt])
        proc = _run_cmd(cmd, cwd=cwd)
        return proc.returncode, proc.stdout, proc.stderr

    if engine == "opencode":
        base_cmd = _find_cli("opencode")
        if not base_cmd:
            raise SystemExit("`opencode` CLI が PATH に見つかりません。")
        proc = _run_cmd([*base_cmd, "run", prompt], cwd=cwd)
        return proc.returncode, proc.stdout, proc.stderr

    raise SystemExit(f"未知のエンジンです: {engine}")


def cmd_new_task(args: argparse.Namespace) -> int:
    wt = _resolve_worktree(args.worktree)
    stem = args.name[:-3] if args.name.endswith(".md") else args.name
    task_path = wt / "docs" / "tasks" / f"{stem}.md"
    if task_path.exists() and not args.overwrite:
        raise SystemExit(f"すでに存在します: {task_path}（上書きする場合は --overwrite）")

    tmpl_path = wt / "docs" / "tasks" / "_TEMPLATE.md"
    if tmpl_path.is_file():
        content = tmpl_path.read_text(encoding="utf-8")
    else:
        content = (
            "# タスク指示書：<タスク名>\n\n"
            "- **タスクID / ブランチ**: `task/<名前>`\n"
            "- **作成日**: YYYY-MM-DD\n\n"
            "## 触ってよいファイル\n- `vreplay/xxx.py`\n\n"
            "## 通すテスト\n```powershell\npython -m pytest -x -q\n```\n"
        )

    title = args.title or stem
    content = content.replace("<タスク名>", title)
    content = content.replace("task/<名前>", f"task/{stem}")
    content = content.replace("vr-<名前>", f"vr-{stem}")
    content = content.replace("YYYY-MM-DD", date.today().isoformat())

    if args.files:
        files_md = "\n".join(f"- `{f}`" for f in args.files)
        content = content.replace("- `vreplay/xxx.py`\n- `tests/test_xxx.py`", files_md)
        content = content.replace("- `vreplay/xxx.py`", files_md)

    if args.tests:
        content = content.replace("python -m pytest -x -q tests/test_xxx.py", args.tests)

    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(content, encoding="utf-8")
    print(f"タスク指示書を作成しました: {task_path}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    wt = _resolve_worktree(args.worktree)
    review_pack_py = wt / "tools" / "review_pack.py"
    if not review_pack_py.is_file():
        raise SystemExit(f"review_pack.py が見つかりません: {review_pack_py}")

    cmd = [sys.executable, str(review_pack_py), args.range]
    task_file = _find_task_file(wt, args.name)
    if task_file:
        cmd.extend(["--task", str(task_file)])
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = wt / out_path
        cmd.extend(["--out", str(out_path)])
    else:
        out_path = None

    proc = _run_cmd(cmd, cwd=wt)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    if out_path and out_path.is_file():
        print(proc.stdout.strip())
        pack_text = out_path.read_text(encoding="utf-8")
    else:
        pack_text = proc.stdout
        if not args.use_sdk:
            print(pack_text)

    if args.use_sdk:
        skill_path = wt / ".agents" / "skills" / "vreplay-review" / "SKILL.md"
        skill_inst = skill_path.read_text(encoding="utf-8") if skill_path.is_file() else ""
        sys_inst = (
            "あなたは valorant-replay のシニア Python レビュアーです。"
            "以下のスキル基準に厳密に従い、日本語でレビュー結果を出力してください。\n\n"
            + skill_inst
        )
        review_result = run_sdk(pack_text, sys_inst, read_only=True, cwd=wt)
        print(review_result)
        _notify(
            wt,
            f"vreplay review 完了 ({args.name})",
            review_result[:1500],
            level="info",
            no_notify=args.no_notify,
        )
    return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    wt = _resolve_worktree(args.worktree)
    task_file = _find_task_file(wt, args.name)
    if not task_file:
        raise SystemExit(
            f"タスク指示書が見つかりません: {args.name} "
            f"(docs/tasks/{args.name}.md または ticket_{args.name}.md)"
        )

    rel_task = task_file.relative_to(wt) if task_file.is_relative_to(wt) else task_file
    impl_prompt = (
        f"タスク指示書 `{rel_task}` を読み、リポジトリの GEMINI.md と AGENTS.md の手順に従って実装してください。\n"
        f"- 指示書の「触ってよいファイル」のみを変更すること。\n"
        f"- テストを実行し、全件パスさせてローカルコミットまで行うこと（main マージ・push はしない）。\n"
        f"- 最後に変更ファイル・テスト結果・迷った点を簡潔に報告すること。"
    )

    print(f"[1/3] 実装実行 (engine={args.engine}, worktree={wt}, task={rel_task})")
    try:
        rc, out, err = _dispatch_implementation(args.engine, impl_prompt, cwd=wt, model=args.model)
    except SystemExit as e:
        _notify(wt, f"vreplay run-task 失敗 ({args.name})", str(e), level="error", no_notify=args.no_notify)
        raise
    except Exception as e:
        _notify(
            wt,
            f"vreplay run-task 失敗 ({args.name})",
            f"{type(e).__name__}: {e}",
            level="error",
            no_notify=args.no_notify,
        )
        raise

    if out:
        print(out.strip())
    if err:
        print(err.strip(), file=sys.stderr)
    if rc != 0:
        msg = f"実装エンジン ({args.engine}) が終了コード {rc} で失敗しました。\n{err[-500:]}"
        _notify(wt, f"vreplay run-task 失敗 ({args.name})", msg, level="error", no_notify=args.no_notify)
        return rc

    print(f"[2/3] テスト検証 ({args.test_cmd})")
    test_argv = shlex.split(args.test_cmd)
    if test_argv and test_argv[0] in ("python", "python3"):
        test_argv[0] = sys.executable
    t_proc = _run_cmd(test_argv, cwd=wt)
    if t_proc.stdout:
        print(t_proc.stdout.strip())
    if t_proc.stderr:
        print(t_proc.stderr.strip(), file=sys.stderr)
    if t_proc.returncode != 0:
        tail = (t_proc.stdout + "\n" + t_proc.stderr).strip()[-600:]
        _notify(
            wt,
            f"vreplay テスト失敗 ({args.name})",
            f"終了コード {t_proc.returncode}\n{tail}",
            level="error",
            no_notify=args.no_notify,
        )
        return t_proc.returncode

    review_out = wt / "_work" / "reviews" / f"review_{args.name}.txt"
    if not args.skip_review_pack:
        print(f"[3/3] レビューパック生成 ({review_out})")
        r_proc = _run_cmd(
            [
                sys.executable,
                str(wt / "tools" / "review_pack.py"),
                args.range,
                "--task",
                str(task_file),
                "--out",
                str(review_out),
            ],
            cwd=wt,
        )
        if r_proc.stdout:
            print(r_proc.stdout.strip())
        if r_proc.returncode != 0 and r_proc.stderr:
            print(f"[warn] review_pack.py: {r_proc.stderr.strip()}", file=sys.stderr)

    _notify(
        wt,
        f"vreplay run-task 完了 ({args.name})",
        f"engine={args.engine} / テスト通過 / worktree={wt.name}",
        level="info",
        no_notify=args.no_notify,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    wt = _resolve_worktree(args.worktree)
    print(f"Worktree : {wt}")

    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=wt)
    if branch.returncode == 0:
        print(f"Branch   : {branch.stdout.strip()}")
    else:
        print("Branch   : (Git リポジトリではありません)")
        return 1

    wt_list = _run_cmd(["git", "worktree", "list"], cwd=wt)
    if wt_list.returncode == 0 and wt_list.stdout.strip():
        print("\n[Worktrees]")
        print(wt_list.stdout.strip())

    st = _run_cmd(["git", "status", "-sb"], cwd=wt)
    if st.returncode == 0 and st.stdout.strip():
        print("\n[Git Status]")
        print(st.stdout.strip())

    log = _run_cmd(["git", "log", "-n", "5", "--oneline"], cwd=wt)
    if log.returncode == 0 and log.stdout.strip():
        print("\n[Recent Commits]")
        print(log.stdout.strip())

    tasks_dir = wt / "docs" / "tasks"
    task_files = sorted(p.name for p in tasks_dir.glob("*.md") if p.name != "_TEMPLATE.md") if tasks_dir.is_dir() else []
    ticket_files = sorted(p.name for p in wt.glob("ticket_*.md"))
    print("\n[Tasks & Tickets]")
    print("  docs/tasks :", ", ".join(task_files) if task_files else "(なし)")
    print("  ticket_*   :", ", ".join(ticket_files) if ticket_files else "(なし)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agy_orchestrator",
        description="valorant-replay 用 Antigravity (agy) オーケストレーション CLI",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new-task", help="docs/tasks/<name>.md をテンプレートから作成する")
    p_new.add_argument("name", help="タスク名（例: barrier_v2）")
    p_new.add_argument("--title", help="見出しタイトル")
    p_new.add_argument("--files", nargs="*", help="触ってよいファイルの一覧")
    p_new.add_argument("--tests", help="通すテストコマンド")
    p_new.add_argument("--worktree", help="対象リポジトリ/ワークツリーパス")
    p_new.add_argument("--overwrite", action="store_true", help="既存ファイルを上書きする")

    p_rev = sub.add_parser("review", help="review_pack.py の生成と（任意で）SDK 自動レビューを行う")
    p_rev.add_argument("name", help="タスク名または指示書ファイル名")
    p_rev.add_argument("--range", default="main..HEAD", help="git diff 範囲（既定: main..HEAD）")
    p_rev.add_argument("--out", help="出力ファイルパス（省略時は標準出力）")
    p_rev.add_argument("--worktree", help="対象リポジトリ/ワークツリーパス")
    p_rev.add_argument("--use-sdk", action="store_true", help="google-antigravity SDK（読み取り専用）でレビューを実行")
    p_rev.add_argument("--no-notify", action="store_true", help="通知を送らない")

    p_run = sub.add_parser("run-task", help="実装 → pytest 検証 → レビューパック生成 → 通知を一貫実行する")
    p_run.add_argument("name", help="タスク名（docs/tasks/<name>.md または ticket_<name>.md）")
    p_run.add_argument("--engine", choices=["sdk", "agy", "opencode"], default="agy", help="実行エンジン")
    p_run.add_argument("--model", help="agy エンジン使用時のモデル名")
    p_run.add_argument("--worktree", help="対象リポジトリ/ワークツリーパス（例: vr-barriers）")
    p_run.add_argument("--test-cmd", default=f"{sys.executable} -m pytest -x -q", help="検証テストコマンド")
    p_run.add_argument("--range", default="main..HEAD", help="レビューパック用の git diff 範囲")
    p_run.add_argument("--skip-review-pack", action="store_true", help="レビューパック生成をスキップする")
    p_run.add_argument("--no-notify", action="store_true", help="通知を送らない")

    p_st = sub.add_parser("status", help="ブランチ・ワークツリー・タスク/チケット一覧を確認する")
    p_st.add_argument("--worktree", help="対象リポジトリ/ワークツリーパス")

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "new-task":
        return cmd_new_task(args)
    if args.cmd == "review":
        return cmd_review(args)
    if args.cmd == "run-task":
        return cmd_run_task(args)
    if args.cmd == "status":
        return cmd_status(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
