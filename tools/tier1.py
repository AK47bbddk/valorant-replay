"""2026 VCT Tier 1 の全試合を解析するための道具（試合リスト・解析済みとの照合・DB への登録・バッチ実行）。

  # 1) vlr.gg の大会ページから全試合を並べ、解析済み（ウェアハウス・report.json）と照合して CSV/JSON に出す
  python tools/tier1.py list --warehouse G:/マイドライブ/valorant-replay/warehouse.sqlite3 \
      --reports G:/マイドライブ/valorant-replay --by-map data/by_map --out _review/tier1_2026_matches

  # 2) 残りの試合をウェアハウスに登録（試合ページを読むだけ・動画なし）し、大会ごとの ID ファイルを作る
  python tools/tier1.py seed --list _review/tier1_2026_matches.json --order importance [--dry-run]

  # 3) 大会ごとに vreplay batch --ids-file を順に回す（--dry-run なら対象を数えるだけ）
  python tools/tier1.py run --list _review/tier1_2026_matches.json --order importance --dry-run

Tier 1 の定義：vlr.gg の VCT 2026 ハブ（/vct-2026）に並ぶ大会のうち、
Champions・Masters・4 地域（Americas / EMEA / Pacific / China）の Kickoff・Stage 1・Stage 2
（プレイオフは各 Stage の大会ページに含まれる）。Qualifier・Threshold・Ascension・Challengers・
Game Changers・翌年（VCT 2027）の予選は含めない。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "https://www.vlr.gg"
HUB = "/vct-2026"
JST = ZoneInfo("Asia/Tokyo")
# 大会ページの試合一覧の日付・時刻は「閲覧者の地域」の時間帯で出る（試合ページの data-utc-ts とは違う）。
# 日本の PC から読むと JST（2026-11-14 に、DB の試合ページ由来の日時 54 試合と照合してずれ 0 を確認）。
# Colab など海外から読むときは --display-tz で合わせる。正確な日時は seed で試合ページから DB に入る
DISPLAY_TZ = "Asia/Tokyo"
EXCLUDE_WORDS = ("qualifier", "threshold", "ascension", "challengers", "game changers", "vct 2027",
                 "academy", "showmatch")
REGIONS = ("Americas", "EMEA", "Pacific", "China")
# 重要度順（importance）での並び。小さいほど先
CATEGORY_RANK = {"champions": 0, "masters": 1, "stage2": 2, "stage1": 3, "kickoff": 4}
FIELDS = ["event", "category", "region", "event_id", "phase", "series", "date_jst", "dt_jst",
          "team1", "team2", "score1", "score2", "status", "maps_played", "has_vod", "match_id",
          "url", "analyzed_maps", "failed_maps", "remaining_maps", "state", "duplicate"]


def _t(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


# ---------------------------------------------------------------- 対象の定義
def classify_event(name: str) -> dict | None:
    """大会名 → {category, region}。Tier 1 でなければ None。"""
    n = name.strip()
    low = n.lower()
    if "2026" not in low or any(w in low for w in EXCLUDE_WORDS):
        return None
    if re.search(r"\bchampions\b", low) and "valorant" in low:
        return {"category": "champions", "region": "International"}
    if re.search(r"\bmasters\b", low):
        return {"category": "masters", "region": "International"}
    m = re.match(r"vct 2026:\s*(americas|emea|pacific|china)\s+(kickoff|stage 1|stage 2)\s*$", low)
    if not m:
        return None
    region = next(r for r in REGIONS if r.lower() == m.group(1))
    return {"category": m.group(2).replace(" ", ""), "region": region}


def parse_hub(soup) -> list[dict]:
    """VCT 2026 ハブのページ → Tier 1 の大会 [{id, name, url, category, region, dates}]。"""
    out, seen = [], set()
    for a in soup.select('a[href^="/event/"]'):
        m = re.match(r"/event/(\d+)/", a.get("href", ""))
        if not m or m.group(1) in seen:
            continue
        title = a.select_one(".event-item-title")
        name = _t(title) if title else _t(a).split(" completed")[0].split(" ongoing")[0].split(" upcoming")[0]
        c = classify_event(name)
        if not c:
            continue
        seen.add(m.group(1))
        dates = _t(a.select_one(".event-item-desc-item.mod-dates")).replace("Dates", "").strip()
        out.append({"id": m.group(1), "name": name, "url": BASE + a["href"], "dates": dates, **c})
    return out


# ---------------------------------------------------------------- 大会ページの試合一覧
def _dt_jst(day_label: str, time_text: str, tz: ZoneInfo = JST) -> tuple[str, str | None]:
    """見出しの日付と時刻（tz で表示されている）→ (JST の日付, JST の日時 ISO)。時刻が TBD なら日付は見出しのまま。"""
    day = datetime.strptime(re.sub(r"\b(Today|Yesterday)\b", "", day_label).strip(), "%a, %B %d, %Y")
    try:
        t = datetime.strptime(time_text.strip().upper(), "%I:%M %p")
    except ValueError:
        return day.date().isoformat(), None
    dt = day.replace(hour=t.hour, minute=t.minute, tzinfo=tz).astimezone(JST)
    return dt.date().isoformat(), dt.isoformat()


def parse_event_matches(soup, event: dict, tz: ZoneInfo = JST) -> list[dict]:
    rows = []
    for label in soup.select("div.wf-label.mod-large"):
        card = label.find_next_sibling("div", class_="wf-card")
        for a in (card.select("a.match-item") if card else []):
            mid = re.match(r"/(\d+)/", a.get("href", ""))
            if not mid:
                continue
            teams = [_t(x) for x in a.select(".match-item-vs-team-name .text-of")]
            scores = [_t(x) for x in a.select(".match-item-vs-team-score")]
            ev = a.select_one(".match-item-event")
            series = _t(ev.select_one(".match-item-event-series")) if ev else ""
            phase = _t(ev).replace(series, "", 1).strip() if ev else ""
            status = _t(a.select_one(".ml-status")) or "?"
            vod = " ".join(_t(v) for v in a.select(".match-item-vod")).lower()
            try:
                date_jst, dt_jst = _dt_jst(_t(label), _t(a.select_one(".match-item-time")), tz)
            except ValueError:
                date_jst, dt_jst = "", None
            s1, s2 = (int(s) if s.isdigit() else None for s in (scores + ["", ""])[:2])
            done = status.lower() == "completed"
            rows.append({
                "event": event["name"], "category": event["category"], "region": event["region"],
                "event_id": event["id"], "phase": phase, "series": series,
                "date_jst": date_jst, "dt_jst": dt_jst,
                "team1": teams[0] if teams else "", "team2": teams[1] if len(teams) > 1 else "",
                "score1": s1, "score2": s2, "status": status,
                "maps_played": (s1 + s2) if done and s1 is not None and s2 is not None else None,
                "has_vod": "youtube" in vod, "match_id": mid.group(1),
                "url": BASE + a["href"].split("?")[0],
            })
    return rows


def fetch_all(events: list[dict], log=print, tz: ZoneInfo = JST) -> list[dict]:
    from vreplay import vlr
    rows = []
    for ev in events:
        soup = vlr.fetch(f"/event/matches/{ev['id']}/?series_id=all&group=all")
        got = parse_event_matches(soup, ev, tz)
        log(f"  {ev['name']}: {len(got)} 試合")
        rows += got
    return rows


# ---------------------------------------------------------------- 解析済みとの照合
def _ro_copy(path: Path) -> Path:
    """Drive 上の SQLite を直接開かない（同期中のファイルを壊さない）。一時フォルダに複製して読む。"""
    dst = Path(tempfile.mkdtemp(prefix="tier1_wh_")) / "wh.sqlite3"
    shutil.copyfile(path, dst)
    return dst


def analyzed_from_warehouse(path: Path) -> tuple[set, set]:
    """(解析済みの (試合ID, マップ順), 失敗の (試合ID, マップ順))。"""
    tmp = _ro_copy(path)
    try:
        c = sqlite3.connect(f"file:{tmp.as_posix()}?mode=ro", uri=True)
        done = {(str(a), int(b)) for a, b in c.execute(
            "SELECT match_id, map_order FROM maps WHERE status='done' AND video_ok=1")}
        failed = {(str(a), int(b)) for a, b in c.execute("SELECT match_id, map_order FROM maps WHERE status='failed'")}
        c.close()
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return done, failed


def _video_ok_tail(p: Path, n: int = 32768) -> bool | None:
    """report.json の video_ok は末尾近く（timings の手前）にある。1 件約 1MB を Drive から丸ごと読むと
    遅いので、末尾だけ読む。見つからなければ None（そのときは全体を読む）。"""
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        m = re.findall(rb'"video_ok":\s*(true|false)', f.read())
    return (m[-1] == b"true") if m else None


def analyzed_from_reports(root: Path) -> set:
    """output.dir/<日付>/<試合>_<順>_<マップ>/report.json から video_ok のマップを集める（フォルダ名で試合と順を取る）。"""
    out = set()
    for p in root.glob("20??-??-??/*_*_*/report.json"):
        m = re.match(r"(\d+)_(\d+)_", p.parent.name)
        if not m:
            continue
        ok = _video_ok_tail(p)
        if ok is None:
            try:
                ok = bool(json.loads(p.read_text(encoding="utf-8")).get("video_ok"))
            except Exception:
                continue
        if ok:
            out.add((m.group(1), int(m.group(2))))
    return out


def analyzed_from_by_map(root: Path) -> set:
    """data/by_map/<マップ>/<n>.json（中身の match_id・order）から video_ok のマップを集める。"""
    out = set()
    for p in root.glob("*/*.json"):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(rep, dict) and rep.get("match_id") and rep.get("video_ok"):
            out.add((str(rep["match_id"]), int(rep.get("order") or 0)))
    return out


def reconcile(rows: list[dict], done: set, failed: set) -> list[dict]:
    seen: dict[str, int] = {}
    for r in rows:
        mid = r["match_id"]
        r["duplicate"] = mid in seen
        seen[mid] = seen.get(mid, 0) + 1
        ana = sorted(o for (m, o) in done if m == mid)
        r["analyzed_maps"] = len(ana)
        r["failed_maps"] = len({o for (m, o) in failed if m == mid} - set(ana))
        mp = r.get("maps_played")
        if r["status"].lower() != "completed":
            r["remaining_maps"], r["state"] = None, "upcoming"
        elif mp is None:
            r["remaining_maps"], r["state"] = None, "unknown"
        else:
            r["remaining_maps"] = max(0, mp - len(ana))
            r["state"] = ("done" if r["remaining_maps"] == 0 else "partial" if ana else
                          "todo" if r["has_vod"] else "todo_novod")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_ev: dict[str, dict] = {}
    for r in rows:
        if r["duplicate"]:
            continue
        e = by_ev.setdefault(r["event"], {"event": r["event"], "category": r["category"], "region": r["region"],
                                          "matches": 0, "completed": 0, "maps": 0, "matches_done": 0,
                                          "maps_analyzed": 0, "maps_remaining": 0, "matches_no_vod": 0,
                                          "first": r["date_jst"], "last": r["date_jst"]})
        e["matches"] += 1
        e["first"], e["last"] = min(e["first"], r["date_jst"]), max(e["last"], r["date_jst"])
        if r["status"].lower() == "completed":
            e["completed"] += 1
            e["maps"] += r["maps_played"] or 0
            e["maps_analyzed"] += min(r["analyzed_maps"], r["maps_played"] or 0)
            e["maps_remaining"] += r["remaining_maps"] or 0
            e["matches_done"] += r["state"] == "done"
            e["matches_no_vod"] += not r["has_vod"]
    tot = {k: sum(e[k] for e in by_ev.values()) for k in
           ("matches", "completed", "maps", "matches_done", "maps_analyzed", "maps_remaining", "matches_no_vod")}
    return {"events": list(by_ev.values()), "total": tot}


def write_outputs(rows: list[dict], events: list[dict], out: Path, sources: list[str]) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    doc = {"generated_at": datetime.now(JST).isoformat(timespec="seconds"),
           "definition": __doc__.split("Tier 1 の定義：")[1].strip(),
           "analyzed_sources": sources, "events": events, "summary": summ, "matches": rows}
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return summ


# ---------------------------------------------------------------- 実行の順番
def order_events(events: list[dict], order: str) -> list[dict]:
    """newest：終わりの新しい大会から／importance：Champions → Masters → Stage 2 → Stage 1 → Kickoff
    （同じ格なら新しい方から）／oldest：古い方から。"""
    last = {e["event"]: e["last"] for e in events}
    if order == "newest":
        return sorted(events, key=lambda e: last[e["event"]], reverse=True)
    if order == "oldest":
        return sorted(events, key=lambda e: last[e["event"]])
    if order == "importance":
        return sorted(events, key=lambda e: (CATEGORY_RANK.get(e["category"], 9),
                                             "".join(chr(0x10FFFF - ord(ch)) for ch in last[e["event"]])))
    raise SystemExit(f"--order は newest / importance / oldest のどれかです：{order}")


def todo_rows(doc: dict, events: list[str] | None = None) -> list[dict]:
    want = {e.lower() for e in events or []}
    return [r for r in doc["matches"]
            if not r["duplicate"] and r["state"] in ("todo", "partial")
            and (not want or r["event"].lower() in want or r["event_id"] in want)]


def plan(doc: dict, order: str, events: list[str] | None = None) -> list[tuple[dict, list[dict]]]:
    todo = todo_rows(doc, events)
    evs = [e for e in doc["summary"]["events"] if any(r["event"] == e["event"] for r in todo)]
    return [(e, [r for r in todo if r["event"] == e["event"]]) for e in order_events(evs, order)]


def ids_path(dir_: Path, ev: dict) -> Path:
    return dir_ / f"ids_{ev['category']}_{re.sub(r'[^A-Za-z0-9]+', '_', ev['event']).strip('_')}.txt"


# ---------------------------------------------------------------- コマンド
def cmd_list(a) -> None:
    from vreplay import vlr
    if a.event_ids:
        events = [{"id": i, "name": n, "url": f"{BASE}/event/{i}", "dates": "", **classify_event(n)}
                  for i, n in (x.split("=", 1) for x in a.event_ids)]
    else:
        events = parse_hub(vlr.fetch(HUB))
    print(f"Tier 1 の大会 {len(events)} 件")
    rows = fetch_all(events, tz=ZoneInfo(a.display_tz))
    done, failed, sources = set(), set(), []
    for w in a.warehouse or []:
        d, f = analyzed_from_warehouse(Path(w))
        done |= d
        failed |= f
        sources.append(f"warehouse:{w}（done {len(d)}・failed {len(f)}）")
    for root in a.reports or []:
        d = analyzed_from_reports(Path(root))
        done |= d
        sources.append(f"reports:{root}（{len(d)} マップ）")
    for root in a.by_map or []:
        d = analyzed_from_by_map(Path(root))
        done |= d
        sources.append(f"by_map:{root}（{len(d)} マップ）")
    rows = reconcile(rows, done, failed)
    summ = write_outputs(rows, events, Path(a.out), sources)
    for e in summ["events"]:
        print(f"  {e['event']}: {e['matches']} 試合・{e['maps']} マップ（解析済み {e['maps_analyzed']}・"
              f"残り {e['maps_remaining']}・VOD なし {e['matches_no_vod']} 試合）")
    t = summ["total"]
    print(f"合計 {t['matches']} 試合・{t['maps']} マップ／解析済み {t['maps_analyzed']}／残り {t['maps_remaining']}")
    print(f"→ {Path(a.out).with_suffix('.csv')} / .json")


def cmd_seed(a) -> list[Path]:
    """残りの試合をウェアハウスに登録（status='meta'）し、大会ごとの ID ファイルを書く。動画は取らない。"""
    from vreplay import config, db, vlr
    doc = json.loads(Path(a.list).read_text(encoding="utf-8"))
    steps = plan(doc, a.order, a.event)
    out_dir = Path(a.ids_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = None
    if not a.dry_run:
        cfg = config.load(a.config)
        if a.warehouse:
            cfg.setdefault("warehouse", {})["path"] = a.warehouse
        conn = db.open_warehouse(cfg)
    files, n_new = [], 0
    for ev, rows in steps:
        p = ids_path(out_dir, ev)
        p.write_text("".join(r["match_id"] + "\n" for r in rows), encoding="utf-8")
        files.append(p)
        print(f"● {ev['event']}: {len(rows)} 試合・残り {sum(r['remaining_maps'] for r in rows)} マップ → {p.name}")
        for r in rows[: a.limit or None]:
            if conn is None:
                continue
            have = conn.execute("SELECT COUNT(*) FROM maps WHERE match_id=?", (r["match_id"],)).fetchone()[0]
            if have >= (r["maps_played"] or 0) and not a.refresh:
                continue
            try:
                db.upsert_match(conn, vlr.parse_match(r["url"]), source="tier1")
                conn.commit()
                n_new += 1
            except Exception as e:
                print(f"  × {r['url']}: {e}")
    if conn is not None:
        conn.commit()
        conn.close()
        print(f"ウェアハウスに {n_new} 試合を登録しました")
    else:
        print("（--dry-run：ウェアハウスには書きません。ID ファイルだけ作りました）")
    return files


def cmd_run(a) -> int:
    """大会ごとに vreplay batch --ids-file を順に回し、進捗を JSONL に残す。"""
    doc = json.loads(Path(a.list).read_text(encoding="utf-8"))
    steps = plan(doc, a.order, a.event)
    out_dir = Path(a.ids_dir)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = log_dir / "tier1_progress.jsonl"
    rc_all = 0
    for ev, rows in steps:
        p = ids_path(out_dir, ev)
        if not p.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(r["match_id"] + "\n" for r in rows), encoding="utf-8")
        cmd = [sys.executable, "-m", "vreplay", "batch", "--ids-file", str(p)]
        if a.config:
            cmd += ["--config", a.config]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if a.dry_run:
            cmd += ["--dry-run", "--no-notify"]
        print(f"● {ev['event']}（残り {sum(r['remaining_maps'] for r in rows)} マップ）: {' '.join(cmd)}")
        t0 = datetime.now(JST)
        log_file = log_dir / f"{p.stem}.log"
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {t0.isoformat(timespec='seconds')} {' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            last = ""
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
                if "batch 完了" in line or "動画解析の対象" in line or "処理待ち" in line:
                    last = line.strip()
            rc = proc.wait()
        rec = {"event": ev["event"], "ids_file": str(p), "dry_run": a.dry_run, "started": t0.isoformat(timespec="seconds"),
               "finished": datetime.now(JST).isoformat(timespec="seconds"), "returncode": rc, "summary": last,
               "log": str(log_file)}
        with open(progress, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rc:
            rc_all = rc
            print(f"  × 終了コード {rc}（{log_file}）")
            if not a.keep_going:
                break
    return rc_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tier1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="vlr.gg から Tier 1 の全試合を並べ、解析済みと照合する")
    ls.add_argument("--warehouse", action="append", help="解析済みの照合に使う warehouse.sqlite3（読むだけ・複数可）")
    ls.add_argument("--reports", action="append", help="output.dir（<日付>/<試合>_<順>_<マップ>/report.json・複数可）")
    ls.add_argument("--by-map", action="append", help="data/by_map（<マップ>/<n>.json・複数可）")
    ls.add_argument("--event-ids", nargs="*", help="ハブを使わず大会を指定（例 2766='Valorant Champions 2026'）")
    ls.add_argument("--display-tz", default=DISPLAY_TZ,
                    help="vlr.gg の一覧が表示する時間帯（閲覧者の地域。日本からなら Asia/Tokyo）")
    ls.add_argument("--out", default=str(ROOT / "_review" / "tier1_2026_matches"))
    for name, fn in (("seed", None), ("run", None)):
        p = sub.add_parser(name)
        p.add_argument("--list", default=str(ROOT / "_review" / "tier1_2026_matches.json"))
        p.add_argument("--order", default="importance", choices=["importance", "newest", "oldest"])
        p.add_argument("--event", nargs="*", help="この大会だけ（大会名か vlr の大会 ID）")
        p.add_argument("--ids-dir", default=str(ROOT / "_review" / "tier1_ids"))
        p.add_argument("--limit", type=int, help="seed：1 大会あたり登録する試合数／run：batch に渡すマップ数の上限")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--config")
    sub.choices["seed"].add_argument("--warehouse", help="登録先（既定は config の warehouse.path）")
    sub.choices["seed"].add_argument("--refresh", action="store_true", help="登録済みの試合も試合ページを読み直す")
    sub.choices["run"].add_argument("--log-dir", default=str(ROOT / "_review" / "tier1_logs"))
    sub.choices["run"].add_argument("--keep-going", action="store_true", help="ある大会で失敗しても次の大会へ進む")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        cmd_list(a)
    elif a.cmd == "seed":
        cmd_seed(a)
    else:
        sys.exit(cmd_run(a))


if __name__ == "__main__":
    main()
