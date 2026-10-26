"""Notion API：既存の「大会ログくん」「マップ詳細」に upsert し、マップページ本文へ分析を書き込む。"""
from __future__ import annotations

import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .vlr import WIN_TYPES, Match, classify_region, classify_tier

API = "https://api.notion.com/v1"
VERSION = "2025-09-03"


class Notion:
    """Notion API の薄いラッパー。複数スレッドから使える。

    Notion の上限は平均 3 リクエスト/秒。これまでは「応答のあとに 0.34 秒待つ」だったので、画像の送信
    （数 MB）の時間がそのまま積み上がっていた。今は「リクエストを始める間隔」を 0.34 秒以上にそろえるだけにして、
    送信中のファイルは並行して流す。
    """
    MIN_INTERVAL = 0.34

    def __init__(self, token: str):
        self.token = token
        self._local = threading.local()
        self._lock = threading.Lock()
        self._next = 0.0

    @property
    def s(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"Authorization": f"Bearer {self.token}", "Notion-Version": VERSION})
            self._local.s = s
        return s

    def _pace(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.MIN_INTERVAL
        if wait > 0:
            time.sleep(wait)

    def req(self, method: str, path: str, json: dict | None = None, files=None) -> dict:
        for i in range(6):
            self._pace()
            r = self.s.request(method, API + path, json=json, files=files, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 2 * (i + 1))))
                if files:                                   # 送り直すときはファイルを先頭から
                    for f in files.values():
                        try:
                            f[1].seek(0)
                        except Exception:
                            pass
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Notion {method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json()
        raise RuntimeError(f"Notion {method} {path}: retries exhausted")

    # ---- 読み書き
    def query(self, ds: str, flt: dict) -> list[dict]:
        out, cursor = [], None
        while True:
            body = {"filter": flt, "page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            r = self.req("POST", f"/data_sources/{ds}/query", body)
            out += r.get("results", [])
            if not r.get("has_more"):
                return out
            cursor = r["next_cursor"]

    def create_page(self, ds: str, props: dict) -> dict:
        return self.req("POST", "/pages", {"parent": {"type": "data_source_id", "data_source_id": ds},
                                           "properties": props})

    def update_page(self, pid: str, props: dict) -> dict:
        return self.req("PATCH", f"/pages/{pid}", {"properties": props})

    def children(self, bid: str) -> list[dict]:
        out, cursor = [], None
        while True:
            q = "?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
            r = self.req("GET", f"/blocks/{bid}/children{q}")
            out += r.get("results", [])
            if not r.get("has_more"):
                return out
            cursor = r["next_cursor"]

    def append(self, bid: str, blocks: list[dict], batch: int = 20):
        for i in range(0, len(blocks), batch):
            self.req("PATCH", f"/blocks/{bid}/children", {"children": blocks[i:i + batch]})

    def clear(self, bid: str):
        for b in self.children(bid):
            self.req("DELETE", f"/blocks/{b['id']}")

    def upload(self, path: str | Path) -> str | None:
        """File Upload API（単一パート）。返り値は image ブロックに使う file_upload ID。"""
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        fu = self.req("POST", "/file_uploads", {"mode": "single_part", "filename": p.name, "content_type": ctype})
        with p.open("rb") as fh:
            self.req("POST", f"/file_uploads/{fu['id']}/send", files={"file": (p.name, fh, ctype)})
        return fu["id"]


# ---------------------------------------------------------------- プロパティ
def _rt(text: str, link: str | None = None, bold=False, color: str | None = None) -> dict:
    t = {"type": "text", "text": {"content": str(text)[:2000]}}
    if link:
        t["text"]["link"] = {"url": link}
    ann = {}
    if bold:
        ann["bold"] = True
    if color:
        ann["color"] = color
    if ann:
        t["annotations"] = ann
    return t


def P_title(t): return {"title": [_rt(t)]}
def P_text(t): return {"rich_text": [_rt(t)] if t else []}
def P_url(u): return {"url": u or None}
def P_num(x): return {"number": x}
def P_date(d): return {"date": {"start": d} if d else None}
def P_select(n): return {"select": {"name": n.replace(",", " ")[:100]} if n else None}
def P_multi(ns): return {"multi_select": [{"name": n.replace(",", " ")[:100]} for n in ns if n]}
def P_rel(ids): return {"relation": [{"id": i} for i in ids]}


# ---------------------------------------------------------------- upsert
def upsert_match(nt: Notion, cfg: dict, m: Match) -> tuple[str, dict[int, dict]]:
    """試合行とマップ行を作成/更新。返り値: (試合ページID, {マップ順: マップページ})"""
    ds_match, ds_map = cfg["notion"]["match_data_source"], cfg["notion"]["map_data_source"]
    vod0 = next((x.vod_url.split("?")[0] for x in m.maps if x.vod_url), None)
    props = {
        "名前": P_title(m.title), "試合URL": P_url(m.url), "動画URL": P_url(vod0),
        "試合日": P_date(m.dt_jst.date().isoformat() if m.dt_jst else None),
        "ステージ": P_text(m.stage), "マップカウント": P_text(m.map_count),
        "トーナメント": P_select(m.event), "Tier": P_select(classify_tier(m.event)),
        "チーム": P_multi([m.team1, m.team2]),
    }
    region = classify_region(m.event)
    if region:
        props["地域"] = P_select(region)
    hit = nt.query(ds_match, {"property": "試合URL", "url": {"equals": m.url}})
    page = nt.update_page(hit[0]["id"], props) if hit else nt.create_page(ds_match, props)
    mid = page["id"]

    existing = {}
    for p in nt.query(ds_map, {"property": "試合", "relation": {"contains": mid}}):
        order = (p["properties"].get("マップ順") or {}).get("number")
        if order is not None:
            existing[int(order)] = p
    maps = {}
    for md in m.maps:
        winner = m.team1 if md.score1 > md.score2 else m.team2 if md.score2 > md.score1 else None
        mp = {
            "マップ名": P_title(f"{m.title} {md.name}"), "マップ順": P_num(md.order),
            "動画URL": P_url(md.vod_url), "勝者": P_select(winner),
            "チーム1ラウンド": P_num(md.score1), "チーム2ラウンド": P_num(md.score2),
            "合計ラウンド": P_num(md.total_rounds), "試合": P_rel([mid]),
        }
        maps[md.order] = (nt.update_page(existing[md.order]["id"], mp) if md.order in existing
                          else nt.create_page(ds_map, mp))
    nt.update_page(mid, {"マップ": P_rel([p["id"] for p in maps.values()])})
    return mid, maps


# ---------------------------------------------------------------- ページ本文
def _para(*rts) -> dict:
    return {"type": "paragraph", "paragraph": {"rich_text": list(rts)}}


def _h(level: int, text: str, children: list | None = None) -> dict:
    key = f"heading_{level}"
    b = {"type": key, key: {"rich_text": [_rt(text)]}}
    if children is not None:
        b[key]["is_toggleable"] = True
        b[key]["children"] = children
    return b


def _bullet(*rts) -> dict:
    return {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": list(rts)}}


def _img(fid: str | None, caption: str = "") -> dict | None:
    if not fid:
        return None
    return {"type": "image", "image": {"type": "file_upload", "file_upload": {"id": fid},
                                       "caption": [_rt(caption)] if caption else []}}


def _table(rows: list[list]) -> dict:
    """セルは文字列か rich_text のリスト。"""
    def cell(c):
        return c if isinstance(c, list) else [_rt(c)]
    return {"type": "table", "table": {
        "table_width": len(rows[0]), "has_column_header": True, "has_row_header": True,
        "children": [{"type": "table_row", "table_row": {"cells": [cell(c) for c in r]}} for r in rows]}}


def build_blocks(rep: dict, uploads: dict[str, str | None]) -> list[dict]:
    """rep: pipeline の map レポート（dict）。uploads: ローカルパス → file_upload ID"""
    from . import summary as S
    t1, t2 = rep["tag1"], rep["tag2"]
    head = [_rt("▶ VOD", link=rep.get("vod_url")) if rep.get("vod_url") else _rt("VOD なし"),
            _rt("　|　"), _rt("VLR", link=rep["vlr_url"]),
            _rt(f"　|　{rep['team1']} {rep['score1']} - {rep['score2']} {rep['team2']}　|　生成 {rep['generated_at']}")]
    blocks = [{"type": "callout", "callout": {"rich_text": head, "icon": {"type": "emoji", "emoji": "🎬"}}}]
    if rep.get("analysis_note"):
        blocks.append(_para(_rt(rep["analysis_note"], color="gray")))
    if rep.get("perf_note"):
        blocks.append(_para(_rt(rep["perf_note"], color="gray")))

    rows = [["R", "スコア", "勝者", "勝ち方", "勝因", f"{t1} 購入・武器", f"{t2} 購入・武器", "先制キル", "キル",
             "ULT 使用", "時間"]]
    for r in rep["rounds"]:
        rcell = [_rt(f"R{r['n']} ▶", link=r["yt_url"])] if r.get("yt_url") else f"R{r['n']}"
        win = S.tag(rep, r.get("winner"))
        rows.append([rcell, r.get("score") or "-", [_rt(win, bold=True, color="blue" if r.get("winner_no") == 1 else "red")],
                     S.how(r), S.reason_cell(r), S.buy_cell(r, "1"), S.buy_cell(r, "2"), S.first_kill(rep, r),
                     S.kill_count(rep, r), S.ult_used(r, rep), r.get("duration") or "-"])
    blocks += flow_blocks(rep)
    blocks += [_h(2, "ラウンド一覧"), _table(rows), _h(2, "ラウンドごと")]
    for r in rep["rounds"]:
        blocks += round_blocks(rep, r, uploads)
    return blocks


def flow_blocks(rep: dict) -> list[dict]:
    """前半・後半の組み立て（ピストル → アンチエコ → … の並び）と、同じ初動の形からの攻め先。"""
    from . import flow as F
    out = []
    fl = F.flow(rep)
    if fl:
        out.append(_h(2, "ラウンドの組み立て"))
        for it in fl:
            rts = [_rt(f"{it['tag']}　", bold=True), _rt(f"{it['half']}（{it['side'] or '?'}）")]
            for i, (n, lab, w) in enumerate(it["items"]):
                rts.append(_rt(" → " if i else "　"))
                rts.append(_rt(f"R{n} {lab}", color="blue" if w else "red"))
            out.append(_bullet(*rts))
        out.append(_para(_rt("青＝勝ち・赤＝負け。ULT の数・オペ（自分が持つ）・対オペ（相手が持つ）を付けています。"
                             "ボーナスはピストルと 2R を取った 3R の持ち越し。", color="gray")))
    br_ = [g for g in F.branches(rep) if len(g["rounds"]) >= 2]
    if br_:
        out.append(_h(3, "同じ初動の形からの攻め先（攻撃側）"))
        for g in br_:
            out.append(_bullet(_rt(F.branch_text(g))))
    return out


def _columns(*cols: list) -> dict | None:
    """横並び（column_list）。中身の無い列は省く。1 列だけなら None（呼び出し側で普通に置く）。"""
    cols = [[b for b in c if b] for c in cols]
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return None
    return {"type": "column_list", "column_list": {
        "children": [{"type": "column", "column": {"children": c}} for c in cols]}}


def round_blocks(rep: dict, r: dict, uploads: dict[str, str | None]) -> list[dict]:
    """1 ラウンドぶん。見出し → 勝因（色つきの囲み）→ 開始時ミニマップと GIF を横並び → 購入・武器 → キル・ULT。
    折りたたまず、スクロールするだけで 1 ラウンドずつ全部が見えるようにする（API の入れ子の制限で、
    横並びはページの直下に置く）。"""
    from . import summary as S
    t1, t2 = rep["tag1"], rep["tag2"]
    a = r.get("assets") or {}
    out = []
    head = [_rt(f"R{r['n']}　{S.tag(rep, r.get('winner'))} 勝利　{r.get('score') or ''}　{S.how(r)}")]
    if r.get("yt_url"):
        head.append(_rt("　▶", link=r["yt_url"]))
    out.append({"type": "heading_3", "heading_3": {"rich_text": head}})

    rs = r.get("reason") or {}
    if rs.get("tags"):
        rt = [_rt("勝因：", bold=True), _rt("・".join(rs["tags"][:3]), bold=True)]
        for line in rs.get("lines") or []:
            rt.append(_rt("\n" + line))
        if rs.get("caveat"):
            rt.append(_rt("\n" + rs["caveat"], color="gray"))
        extra = []
        if r.get("plant_clock"):
            extra.append(f"設置 {r['plant_clock']} 経過")
        if r.get("duration"):
            extra.append(f"ラウンド時間 {r['duration']}")
        if S.setup_line(rep, r):
            extra.append("初動：" + S.setup_line(rep, r))
        if S.caster_line(r):
            extra.append("解説：" + S.caster_line(r))
        if extra:
            rt.append(_rt("\n" + "　".join(extra), color="gray"))
        out.append({"type": "callout", "callout": {"rich_text": rt,
                                                   "icon": {"type": "emoji", "emoji": S.reason_icon(r)}}})

    start = _img(uploads.get(a.get("minimap_start")), "開始時のミニマップ")
    gif = _img(uploads.get(a.get("minimap_gif")), f"ラウンド中（{rep.get('gif_speed', 4)}倍速・リプレイ除外）")
    board = _img(uploads.get(a.get("board")), "配置図（開始直後 1:40・アイコン照合）")
    cl = _columns(*[[b] for b in (start, gif, board) if b])
    out += [cl] if cl else [b for b in (start, gif, board) if b]

    for no, tg in (("1", t1), ("2", t2)):
        bank = r.get(f"bank{no}")
        out.append(_bullet(_rt(f"{tg}　", bold=True),
                           _rt(f"{r.get(f'buy{no}') or '-'}" + (f"（残り {bank}k）" if bank is not None else "")),
                           _rt(("　" + S.player_weapons(r, no)) if S.player_weapons(r, no) else "")))
    kills = r.get("kills") or []
    if kills:
        rt = [_rt("キル　", bold=True)]
        for i, k in enumerate(kills):
            w = f"［{k['weapon_ja']}］" if k.get("weapon_ja") else ""
            if k.get("cause") == "spike":
                rt += [_rt(("\n" if i else "") + f"{k.get('timer') or '--:--'} "), _rt("💥 スパイクの爆発", bold=True),
                       _rt(f" → {k.get('victim') or '?'}")]
                continue
            src = {"card": "カードで確認", "implied": "全滅から推定"}.get(k.get("source"), "キラーは読めず")
            who = k.get("killer") or f"（{S.tag(rep, k.get('killer_team'))}・{src}）"
            rt += [_rt(("\n" if i else "") + f"{k.get('timer') or '--:--'} "),
                   _rt(who, bold=True, color="blue" if k.get("killer_team") == rep["team1"] else "red"),
                   _rt(f"{w} → {k.get('victim') or '?'}")]
        out.append(_para(*rt))
    if r.get("ults") or r.get("ult_ready"):
        bits = []
        if r.get("ult_ready"):
            bits.append("開始時に溜まっていた：" + "、".join(r["ult_ready"]))
        if r.get("ults"):
            bits.append("使用：" + "、".join(f"{u.get('timer') or '--:--'} {u.get('player') or '?'}"
                                           + (f"（{u['agent']}）" if u.get("agent") else "") for u in r["ults"]))
        out.append(_para(_rt("ULT　", bold=True), _rt("　".join(bits))))
    buy = _img(uploads.get(a.get("buy_panels")), "購入時の装備表")
    strip = _img(uploads.get(a.get("kill_strip")), "キルフィード")
    cl = _columns([buy], [strip])
    out += [cl] if cl else [b for b in (buy, strip) if b]
    for n in r.get("notes") or []:
        out.append(_para(_rt(n, color="gray")))
    out.append({"type": "divider", "divider": {}})
    return out


def write_map_page(nt: Notion, page_id: str, rep: dict, force: bool = False) -> bool:
    if nt.children(page_id):
        if not force:
            return False
        nt.clear(page_id)
    paths = list(dict.fromkeys(p for r in rep["rounds"] for p in (r.get("assets") or {}).values() if p))

    def up(path):
        try:
            return path, nt.upload(path)
        except RuntimeError as e:  # 1 枚の失敗でページ全体を止めない
            print("  upload failed:", path, e)
            return path, None

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="Notion-画像") as ex:  # 送信は並行、リクエストの開始は 3 回/秒に抑える
        uploads: dict[str, str | None] = dict(ex.map(up, paths))
    nt.append(page_id, build_blocks(rep, uploads), batch=10)
    return True


def how_ja(how: str | None) -> str | None:
    return WIN_TYPES.get(how or "", how)
