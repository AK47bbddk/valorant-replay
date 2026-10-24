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

    rows = [["R", "スコア", "勝者", "勝ち方", f"{t1} 購入・武器", f"{t2} 購入・武器", "先制キル", "キル", "ULT 使用", "時間"]]
    for r in rep["rounds"]:
        rcell = [_rt(f"R{r['n']} ▶", link=r["yt_url"])] if r.get("yt_url") else f"R{r['n']}"
        win = S.tag(rep, r.get("winner"))
        rows.append([rcell, r.get("score") or "-", [_rt(win, bold=True, color="blue" if r.get("winner_no") == 1 else "red")],
                     S.how(r), S.buy_cell(r, "1"), S.buy_cell(r, "2"), S.first_kill(rep, r),
                     S.kill_count(rep, r), S.ult_used(r, rep), r.get("duration") or "-"])
    blocks += [_h(2, "ラウンド一覧"), _table(rows), _h(2, "ラウンド詳細")]

    for r in rep["rounds"]:
        a = r.get("assets") or {}
        kids = []
        line = []
        if r.get("yt_url"):
            line += [_rt("▶ このラウンドを見る", link=r["yt_url"]), _rt(f"　ラウンド時間 {r.get('duration') or '-'}")]
        if line:
            kids.append(_para(*line))
        for no, tg in (("1", t1), ("2", t2)):
            bank = r.get(f"bank{no}")
            kids.append(_bullet(_rt(f"{tg}：", bold=True), _rt(f"{r.get(f'buy{no}') or '-'}"
                                                              + (f"（残り {bank}k）" if bank is not None else "")),
                                _rt(("　" + S.player_weapons(r, no)) if S.player_weapons(r, no) else "")))
        kids.append(_img(uploads.get(a.get("minimap_start")), "ラウンド開始時のミニマップ"))
        kids.append(_img(uploads.get(a.get("minimap_gif")), f"ラウンド中のミニマップ（{rep.get('gif_speed', 4)}倍速・リプレイ除外）"))
        kids.append(_img(uploads.get(a.get("buy_panels")), "購入フェーズ終了時のパネル（武器・アーマー・ULT）"))
        if r.get("kills"):
            kids.append(_para(_rt("キルログ", bold=True)))
            for k in r["kills"]:
                w = f"［{k['weapon_ja']}］" if k.get("weapon_ja") else ""
                kids.append(_bullet(_rt(f"{k.get('timer') or '--:--'}  "),
                                    _rt(k.get("killer") or "?", bold=True),
                                    _rt(f"（{S.tag(rep, k.get('killer_team'))}）{w} → {k.get('victim') or '?'}")))
            kids.append(_img(uploads.get(a.get("kill_strip")), "キルフィードの切り抜き"))
        if r.get("ults") or r.get("ult_ready"):
            kids.append(_para(_rt("ULT", bold=True)))
            if r.get("ult_ready"):
                kids.append(_bullet(_rt("開始時に溜まっていた：" + "、".join(r["ult_ready"]))))
            for u in r.get("ults") or []:
                kids.append(_bullet(_rt(f"{u.get('timer') or '--:--'}  {u.get('player') or '?'}"
                                        + (f"（{u['agent']}）" if u.get("agent") else "")
                                        + f" が使用（{S.tag(rep, u.get('team'))}）")))
        for n in r.get("notes") or []:
            kids.append(_para(_rt(n, color="gray")))
        title = f"R{r['n']}　{S.tag(rep, r.get('winner'))} 勝利　{S.how(r)}　{r.get('score') or ''}"
        blocks.append(_h(3, title, [k for k in kids if k]))
    return blocks


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

    with ThreadPoolExecutor(max_workers=4) as ex:      # 送信は並行、リクエストの開始は 3 回/秒に抑える
        uploads: dict[str, str | None] = dict(ex.map(up, paths))
    nt.append(page_id, build_blocks(rep, uploads), batch=10)
    return True


def how_ja(how: str | None) -> str | None:
    return WIN_TYPES.get(how or "", how)
