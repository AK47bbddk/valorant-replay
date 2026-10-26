"""大量の試合をためる SQLite ウェアハウス。

VLR から取ったメタデータ（勝敗・購入・構成・選手）と、動画解析の結果
（キル・ULT・サイト・位置ヒートマップ）を 1 か所に保存し、
チーム別／構成別のプレイブック集計に使う。

SQLite なので Drive に置いても使えるが、複数プロセスから同時に書くと壊れやすい。
`batch --shard` で並列に回すときは、DB も出力先ごとに分ける。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import vlr

ATTACK, DEFENSE = "攻撃", "防衛"
OPP = {ATTACK: DEFENSE, DEFENSE: ATTACK, "防御": ATTACK}


def _canon(side: str | None) -> str | None:
    """VLR の表記ゆれ（防衛/防御/防衞）を DB の表記にそろえる。"""
    if not side:
        return None
    if side == ATTACK or "攻" in side:
        return ATTACK
    if side in (DEFENSE, "防御", "防衞") or "防" in side or "衛" in side or "衞" in side:
        return DEFENSE
    return None

SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS matches (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  team1 TEXT, team2 TEXT, tag1 TEXT, tag2 TEXT,
  event TEXT, stage TEXT,
  dt_jst TEXT,
  map_count TEXT,
  source TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS maps (
  match_id TEXT NOT NULL,
  map_order INTEGER NOT NULL,
  map_name TEXT,
  game_id TEXT,
  score1 INTEGER, score2 INTEGER, total_rounds INTEGER,
  vod_id TEXT, vod_t INTEGER, vod_url TEXT,
  status TEXT NOT NULL DEFAULT 'meta',   -- meta / done / failed
  video_ok INTEGER DEFAULT 0,
  analysis_note TEXT,
  report_json TEXT,
  perf_json TEXT,
  tactics_json TEXT,
  updated_at TEXT,
  PRIMARY KEY (match_id, map_order),
  FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS players (
  match_id TEXT NOT NULL,
  map_order INTEGER NOT NULL,
  team INTEGER NOT NULL,
  name TEXT NOT NULL,
  agent TEXT,
  PRIMARY KEY (match_id, map_order, team, name)
);

CREATE TABLE IF NOT EXISTS rounds (
  match_id TEXT NOT NULL,
  map_order INTEGER NOT NULL,
  n INTEGER NOT NULL,
  winner INTEGER,
  side TEXT,
  side1 TEXT,
  side2 TEXT,
  how TEXT,
  buy1 TEXT, buy2 TEXT,
  bank1 REAL, bank2 REAL,
  loadout1 REAL, loadout2 REAL,
  duration TEXT,
  first_kill_team INTEGER,
  first_kill_player TEXT,
  first_kill_weapon TEXT,
  kills_json TEXT,
  ults_json TEXT,
  ult_ready_json TEXT,
  weapons_json TEXT,
  notes_json TEXT,
  site TEXT,
  site_conf REAL,
  tactics_json TEXT,
  PRIMARY KEY (match_id, map_order, n),
  FOREIGN KEY (match_id, map_order) REFERENCES maps(match_id, map_order) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS comps (
  match_id TEXT NOT NULL,
  map_order INTEGER NOT NULL,
  team INTEGER NOT NULL,
  comp_key TEXT NOT NULL,
  agents_json TEXT,
  PRIMARY KEY (match_id, map_order, team)
);

CREATE TABLE IF NOT EXISTS failures (
  url TEXT PRIMARY KEY,
  error TEXT,
  updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_maps_status ON maps(status);
CREATE INDEX IF NOT EXISTS idx_rounds_map ON rounds(match_id, map_order);
CREATE INDEX IF NOT EXISTS idx_comps_key ON comps(comp_key);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(dt_jst);
"""


# プレイブック用に後から足した列（ラウンドの開始・設置の時刻、設置サイト、勝因、タイマーの節目ごとの配置など）
ROUND_EXTRA = [("t_drop", "REAL"), ("plant_t", "REAL"), ("plant_site", "TEXT"), ("yt_url", "TEXT"),
               ("eco1", "TEXT"), ("eco2", "TEXT"), ("reason_main", "TEXT"), ("reason_json", "TEXT"),
               ("timeline_json", "TEXT"), ("kill_sources_json", "TEXT"),
               ("setups_json", "TEXT"), ("first_fight", "TEXT")]


class _SyncedConnection(sqlite3.Connection):
    """Google Drive 上の DB をローカルの複製で開き、commit・close のたびに Drive へ書き戻す接続。

    Drive（/content/drive）の上で SQLite を直接書くと、Drive の同期と SQLite のロック・ジャーナルが
    かみ合わず「database disk image is malformed」で壊れることがある。ローカル（/tmp）の複製を使い、
    書き戻しは一時ファイルに丸ごとコピーしてから名前を差し替える（途中で切れても元のファイルは壊れない）。"""
    remote: Path | None = None
    _dirty = False

    def commit(self):
        super().commit()
        self._dirty = True
        self._push(min_interval=600.0)   # DB 全体のコピーなので 10 分に 1 回まで（close で必ず書き戻す）

    def _push(self, min_interval: float = 0.0):
        import time
        if self.remote is None or not self._dirty:
            return
        now = time.time()
        if min_interval and now - getattr(self, "_last_push", 0.0) < min_interval:
            return
        tmp = self.remote.with_name(self.remote.name + ".tmp")
        try:
            dst = sqlite3.connect(str(tmp))
            with dst:
                self.backup(dst)
            dst.close()
            tmp.replace(self.remote)
            self._dirty, self._last_push = False, now
        except Exception as e:
            print(f"  ウェアハウスを Drive に書き戻せませんでした（次の commit でやり直します）: {e}")

    def close(self):
        try:
            if self.in_transaction:
                super().commit()
                self._dirty = True
            self._push()
        finally:
            super().close()


def _ok(path: Path) -> bool:
    try:
        c = sqlite3.connect(str(path))
        r = c.execute("PRAGMA quick_check").fetchone()
        c.close()
        return bool(r) and r[0] == "ok"
    except sqlite3.DatabaseError:
        return False


def connect(path: str | Path, rebuild_root: str | Path | None = None, log=print) -> sqlite3.Connection:
    """DB を開く。壊れていたら別名で退避して作り直し、rebuild_root があればその下の report.json から入れ直す。
    Drive 上のファイルはローカルの複製で開き、commit・close で書き戻す（_SyncedConnection）。"""
    import shutil
    import tempfile
    from datetime import datetime as _dt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    on_drive = "/drive/" in str(path.resolve())
    rebuilt = False
    if path.exists() and path.stat().st_size and not _ok(path):
        bad = path.with_name(f"{path.stem}.broken-{_dt.now():%Y%m%d-%H%M%S}{path.suffix}")
        path.replace(bad)
        for ext in ("-journal", "-wal", "-shm"):
            Path(str(path) + ext).unlink(missing_ok=True)
        log(f"ウェアハウスが壊れていたので {bad.name} に退避し、作り直します")
        rebuilt = True
    if on_drive:
        local = Path(tempfile.gettempdir()) / "vreplay_wh" / path.name
        local.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("", "-journal", "-wal", "-shm"):
            Path(str(local) + ext).unlink(missing_ok=True)
        if path.exists():
            shutil.copyfile(path, local)
        conn = sqlite3.connect(str(local), timeout=60, factory=_SyncedConnection, check_same_thread=False)
        conn.remote = path
        conn._dirty = not path.exists()
    else:
        conn = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL" if not on_drive else "PRAGMA journal_mode=DELETE")
    except sqlite3.DatabaseError:
        conn.execute("PRAGMA journal_mode=DELETE")
    conn.executescript(SCHEMA)
    _migrate(conn)
    if rebuilt and rebuild_root is not None:
        n = sync_reports(conn, Path(rebuild_root), log=log)
        log(f"  report.json {n} 件から作り直しました")
    return conn


def open_warehouse(cfg: dict, log=print) -> sqlite3.Connection:
    """ウェアハウスを開く（壊れていたら output.dir の report.json から作り直す）。"""
    return connect(warehouse_path(cfg), rebuild_root=Path(cfg["output"]["dir"]), log=log)


def _migrate(conn: sqlite3.Connection) -> None:
    """既存 DB に後から増えた列を足す（CREATE TABLE IF NOT EXISTS では増えないため）。"""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(maps)")}
    if "tactics_json" not in cols:
        conn.execute("ALTER TABLE maps ADD COLUMN tactics_json TEXT")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(rounds)")}
    for name, typ in ROUND_EXTRA:
        if name not in cols:
            conn.execute(f"ALTER TABLE rounds ADD COLUMN {name} {typ}")


def warehouse_path(cfg: dict) -> Path:
    return Path((cfg.get("warehouse") or {}).get("path") or Path(cfg["output"]["dir"]) / "warehouse.sqlite3")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _opp(side: str | None) -> str | None:
    s = _canon(side)
    return OPP.get(s) if s else None


def _side1_for_round(n: int, start: str | None) -> str | None:
    """チーム1の n ラウンド目の側。12 ラウンドごとに交代、OT は 1 ラウンドごと。"""
    if not start:
        return None
    if n <= 12:
        return start
    if n <= 24:
        return _opp(start)
    return start if (n - 25) % 2 == 0 else _opp(start)


def infer_start_side(md: vlr.MapData) -> str | None:
    votes: dict[str, int] = {}
    for r in md.rounds:
        if not r.winner or not r.side:
            continue
        need = _canon(r.side if r.winner == 1 else _opp(r.side))
        for cand in (ATTACK, DEFENSE):
            if _side1_for_round(r.n, cand) == need:
                votes[cand] = votes.get(cand, 0) + 1
    return max(votes, key=votes.get) if votes else None


def side1_by_round(md: vlr.MapData) -> dict[int, str | None]:
    """VLR の各ラウンドの勝者側から、チーム1が攻めか守りかを復元する。"""
    start = infer_start_side(md)
    return {r.n: _side1_for_round(r.n, start) for r in md.rounds}


def _comp(players: list[vlr.Player], team: int) -> tuple[str, list[str]] | None:
    agents = sorted(p.agent for p in players if p.team == team and p.agent)
    return ("|".join(agents), agents) if agents else None


def upsert_match(conn: sqlite3.Connection, m: vlr.Match, source: str = "vlr") -> None:
    now = _now()
    conn.execute(
        """INSERT INTO matches(id,url,team1,team2,tag1,tag2,event,stage,dt_jst,map_count,source,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
             url=excluded.url, team1=excluded.team1, team2=excluded.team2, tag1=excluded.tag1,
             tag2=excluded.tag2, event=excluded.event, stage=excluded.stage, dt_jst=excluded.dt_jst,
             map_count=excluded.map_count, source=excluded.source, updated_at=excluded.updated_at""",
        (m.id, m.url, m.team1, m.team2, m.tag1, m.tag2, m.event, m.stage,
         m.dt_jst.isoformat() if m.dt_jst else None, m.map_count, source, now))
    for md in m.maps:
        start = infer_start_side(md)
        conn.execute(
            """INSERT INTO maps(match_id,map_order,map_name,game_id,score1,score2,total_rounds,
                                vod_id,vod_t,vod_url,status,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?, 'meta', ?)
               ON CONFLICT(match_id,map_order) DO UPDATE SET
                 map_name=excluded.map_name, game_id=excluded.game_id, score1=excluded.score1,
                 score2=excluded.score2, total_rounds=excluded.total_rounds, vod_id=excluded.vod_id,
                 vod_t=excluded.vod_t, vod_url=excluded.vod_url, updated_at=excluded.updated_at""",
            (m.id, md.order, md.name, md.game_id, md.score1, md.score2, md.total_rounds,
             md.vod_id, md.vod_t, md.vod_url, now))
        conn.execute("DELETE FROM players WHERE match_id=? AND map_order=?", (m.id, md.order))
        for p in md.players:
            conn.execute("INSERT OR REPLACE INTO players(match_id,map_order,team,name,agent) VALUES(?,?,?,?,?)",
                         (m.id, md.order, p.team, p.name, p.agent))
        econ = {e.n: e for e in md.econ}
        for r in md.rounds:
            e = econ.get(r.n)
            s1 = _side1_for_round(r.n, start)
            conn.execute(
                """INSERT INTO rounds(match_id,map_order,n,winner,side,side1,side2,how,
                                      buy1,buy2,bank1,bank2,loadout1,loadout2)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(match_id,map_order,n) DO UPDATE SET
                     winner=excluded.winner, side=excluded.side, side1=excluded.side1, side2=excluded.side2,
                     how=excluded.how, buy1=excluded.buy1, buy2=excluded.buy2, bank1=excluded.bank1,
                     bank2=excluded.bank2, loadout1=excluded.loadout1, loadout2=excluded.loadout2""",
                (m.id, md.order, r.n, r.winner, _canon(r.side), s1, _opp(s1), r.how,
                 e.buy1 if e else None, e.buy2 if e else None, e.bank1 if e else None,
                 e.bank2 if e else None, e.loadout1 if e else None, e.loadout2 if e else None))
        for team in (1, 2):
            c = _comp(md.players, team)
            if c:
                conn.execute("INSERT OR REPLACE INTO comps(match_id,map_order,team,comp_key,agents_json) VALUES(?,?,?,?,?)",
                             (m.id, md.order, team, c[0], json.dumps(c[1], ensure_ascii=False)))


def _team_no(rep: dict, name: str | None) -> int | None:
    if not name:
        return None
    if name == rep.get("team1"):
        return 1
    if name == rep.get("team2"):
        return 2
    return None


def ingest_report(conn: sqlite3.Connection, rep: dict, status: str | None = None) -> None:
    """pipeline の report.json を DB に反映する（動画解析の結果）。"""
    mid, order = str(rep["match_id"]), int(rep["order"])
    now = _now()
    videook = bool(rep.get("video_ok"))
    st = status or ("done" if videook else "failed")
    # 既存レポートだけを取り込む場合でも外部キーが通るように、まず試合とマップの行を用意する
    conn.execute(
        """INSERT OR IGNORE INTO matches(id,url,team1,team2,tag1,tag2,event,stage,dt_jst,map_count,source,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, rep.get("vlr_url") or "", rep.get("team1"), rep.get("team2"), rep.get("tag1"), rep.get("tag2"),
         rep.get("event"), rep.get("stage"), rep.get("date"), None, "report", now))
    conn.execute(
        """INSERT OR IGNORE INTO maps(match_id,map_order,map_name,score1,score2,total_rounds,status,updated_at)
           VALUES(?,?,?,?,?,?, 'meta', ?)""",
        (mid, order, rep.get("map"), rep.get("score1"), rep.get("score2"),
         (rep.get("score1") or 0) + (rep.get("score2") or 0), now))
    conn.execute(
        """UPDATE maps SET status=?, video_ok=?, analysis_note=?, report_json=?, perf_json=?,
                          tactics_json=COALESCE(?, tactics_json), updated_at=?
           WHERE match_id=? AND map_order=?""",
        (st, int(videook), rep.get("analysis_note"), json.dumps(rep, ensure_ascii=False),
         json.dumps({"timings": rep.get("timings"), "perf_note": rep.get("perf_note")}, ensure_ascii=False),
         json.dumps(rep.get("tactics"), ensure_ascii=False) if rep.get("tactics") else None,
         now, mid, order))
    for rr in rep.get("rounds") or []:
        kills = rr.get("kills") or []
        fk = kills[0] if kills else {}
        s1 = _canon(rr.get("side1"))
        conn.execute(
            """INSERT INTO rounds(match_id,map_order,n,winner,side,side1,side2,how,buy1,buy2,bank1,bank2,
                                  duration,first_kill_team,first_kill_player,first_kill_weapon,
                                  kills_json,ults_json,ult_ready_json,weapons_json,notes_json,
                                  site,site_conf,tactics_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(match_id,map_order,n) DO UPDATE SET
                 side1=COALESCE(excluded.side1, side1), side2=COALESCE(excluded.side2, side2),
                 duration=excluded.duration, first_kill_team=excluded.first_kill_team,
                 first_kill_player=excluded.first_kill_player, first_kill_weapon=excluded.first_kill_weapon,
                 kills_json=excluded.kills_json, ults_json=excluded.ults_json,
                 ult_ready_json=excluded.ult_ready_json, weapons_json=excluded.weapons_json,
                 notes_json=excluded.notes_json, site=COALESCE(excluded.site, site),
                 site_conf=COALESCE(excluded.site_conf, site_conf),
                 tactics_json=COALESCE(excluded.tactics_json, tactics_json)""",
            (mid, order, rr.get("n"), rr.get("winner_no"), _canon(rr.get("side")),
             s1, _opp(s1), rr.get("how"),
             rr.get("buy1"), rr.get("buy2"), rr.get("bank1"), rr.get("bank2"), rr.get("duration"),
             _team_no(rep, fk.get("killer_team")), fk.get("killer"), fk.get("weapon"),
             json.dumps(kills, ensure_ascii=False), json.dumps(rr.get("ults") or [], ensure_ascii=False),
             json.dumps(rr.get("ult_ready") or [], ensure_ascii=False),
             json.dumps(rr.get("weapons") or {}, ensure_ascii=False),
             json.dumps(rr.get("notes") or [], ensure_ascii=False),
             (rr.get("tactics") or {}).get("site"), (rr.get("tactics") or {}).get("site_conf"),
             json.dumps(rr.get("tactics"), ensure_ascii=False) if rr.get("tactics") else None))
        tac = rr.get("tactics") or {}
        reason = rr.get("reason") or {}
        conn.execute(
            """UPDATE rounds SET t_drop=?, plant_t=?, plant_site=?, yt_url=?, eco1=COALESCE(?, eco1),
                                 eco2=COALESCE(?, eco2), reason_main=?, reason_json=?, timeline_json=?,
                                 kill_sources_json=?, setups_json=?, first_fight=?
               WHERE match_id=? AND map_order=? AND n=?""",
            (rr.get("t_drop"), rr.get("plant_t"), tac.get("plant_site"), rr.get("yt_url"),
             rr.get("eco1"), rr.get("eco2"), reason.get("main"),
             json.dumps(reason, ensure_ascii=False) if reason else None,
             json.dumps(tac.get("timeline"), ensure_ascii=False) if tac.get("timeline") else None,
             json.dumps(rr.get("kill_sources"), ensure_ascii=False) if rr.get("kill_sources") else None,
             json.dumps(tac.get("setups"), ensure_ascii=False) if tac.get("setups") else None,
             tac.get("first_fight"), mid, order, rr.get("n")))
    for p in rep.get("players") or []:
        team = p.get("team")
        if team in (1, 2):
            conn.execute("INSERT OR REPLACE INTO players(match_id,map_order,team,name,agent) VALUES(?,?,?,?,?)",
                         (mid, order, team, p.get("name"), p.get("agent")))
    for team in (1, 2):
        agents = sorted(p.get("agent") for p in rep.get("players") or []
                        if p.get("team") == team and p.get("agent"))
        if agents:
            conn.execute("INSERT OR REPLACE INTO comps(match_id,map_order,team,comp_key,agents_json) VALUES(?,?,?,?,?)",
                         (mid, order, team, "|".join(agents), json.dumps(agents, ensure_ascii=False)))


def load_match(conn: sqlite3.Connection, match_id: str) -> vlr.Match | None:
    row = conn.execute("SELECT * FROM matches WHERE id=?", (str(match_id),)).fetchone()
    if not row:
        return None
    dt = None
    if row["dt_jst"]:
        try:
            dt = datetime.fromisoformat(row["dt_jst"])
        except ValueError:
            pass
    m = vlr.Match(id=row["id"], url=row["url"], team1=row["team1"], team2=row["team2"],
                  event=row["event"] or "", stage=row["stage"] or "", dt_jst=dt,
                  tag1=row["tag1"] or "", tag2=row["tag2"] or "")
    for mr in conn.execute("SELECT * FROM maps WHERE match_id=? ORDER BY map_order", (str(match_id),)):
        md = vlr.MapData(order=mr["map_order"], name=mr["map_name"] or "", game_id=mr["game_id"] or "",
                         score1=mr["score1"] or 0, score2=mr["score2"] or 0,
                         vod_url=mr["vod_url"], vod_id=mr["vod_id"], vod_t=mr["vod_t"])
        md.players = [vlr.Player(r["name"], r["agent"] or "", r["team"])
                      for r in conn.execute("SELECT * FROM players WHERE match_id=? AND map_order=? ORDER BY team,name",
                                            (str(match_id), mr["map_order"]))]
        for rr in conn.execute("SELECT * FROM rounds WHERE match_id=? AND map_order=? ORDER BY n",
                               (str(match_id), mr["map_order"])):
            md.rounds.append(vlr.Round(rr["n"], rr["winner"], rr["side"], rr["how"]))
            md.econ.append(vlr.Econ(rr["n"], rr["buy1"] or "", rr["buy2"] or "", rr["bank1"], rr["bank2"],
                                    rr["loadout1"], rr["loadout2"]))
        m.maps.append(md)
    return m


def pending_maps(conn: sqlite3.Connection, retry_failed: bool = True, match_id: str | None = None,
                 limit: int | None = None) -> list[sqlite3.Row]:
    statuses = "('meta','failed')" if retry_failed else "('meta')"
    sql = ("SELECT match_id, map_order, map_name, vod_id, vod_t FROM maps "
           f"WHERE status IN {statuses} AND vod_id IS NOT NULL AND vod_t IS NOT NULL")
    args: list = []
    if match_id:
        sql += " AND match_id=?"
        args.append(str(match_id))
    sql += " ORDER BY match_id, map_order"
    if limit:
        sql += " LIMIT ?"
        args.append(int(limit))
    return conn.execute(sql, args).fetchall()


def mark_failed(conn: sqlite3.Connection, match_id: str, map_order: int, error: str) -> None:
    conn.execute("UPDATE maps SET status='failed', analysis_note=?, updated_at=? WHERE match_id=? AND map_order=?",
                 (str(error)[:1000], _now(), str(match_id), int(map_order)))


def stats(conn: sqlite3.Connection) -> dict:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0] or 0)
    return {
        "matches": one("SELECT COUNT(*) FROM matches"),
        "maps": one("SELECT COUNT(*) FROM maps"),
        "maps_done": one("SELECT COUNT(*) FROM maps WHERE status='done'"),
        "maps_meta": one("SELECT COUNT(*) FROM maps WHERE status='meta'"),
        "maps_failed": one("SELECT COUNT(*) FROM maps WHERE status='failed'"),
        "rounds": one("SELECT COUNT(*) FROM rounds"),
        "rounds_with_kills": one("SELECT COUNT(*) FROM rounds WHERE kills_json IS NOT NULL"),
        "rounds_with_site": one("SELECT COUNT(*) FROM rounds WHERE site IS NOT NULL"),
        "rounds_with_plant_site": one("SELECT COUNT(*) FROM rounds WHERE plant_site IS NOT NULL"),
    }


def sync_reports(conn: sqlite3.Connection, root: Path, log=print) -> int:
    """既存の report.json をまとめて DB に取り込む。"""
    n = 0
    for p in sorted(Path(root).glob("**/report.json")):
        try:
            rep = json.loads(p.read_text(encoding="utf-8"))
            if "match_id" not in rep:
                continue
            ingest_report(conn, rep)
            n += 1
            if n % 50 == 0:
                conn.commit()
                log(f"  {n} 件取り込み…")
        except Exception as e:
            log(f"  スキップ {p}: {e}")
    conn.commit()
    return n
