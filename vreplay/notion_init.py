"""Notion の試合・マップ用データベースを作成する一回限りの補助コマンド。

既存DBを参照して親ページとスキーマを確認し、必要なら新しい2DBを作る。
トークンは環境変数 NOTION_TOKEN から読む（ログには出さない）。
"""
from __future__ import annotations

import argparse
import json
import os

from .notion import Notion

MATCH_TITLE = "大会ログくん（新）"
MAP_TITLE = "マップ詳細（新）"


def clean_id(value: str | None) -> str:
    if not value:
        return ""
    s = value.strip().split("?")[0].rstrip("/").split("/")[-1]
    s = s.replace("-", "")
    return s


def rich_text(text: str) -> list[dict]:
    return [{"type": "text", "text": {"content": text}}]


def _ds_id(db: dict) -> str | None:
    ds = db.get("data_sources") or []
    return ds[0].get("id") if ds else None


def inspect(nt: Notion, ref_id: str) -> dict:
    me = nt.req("GET", "/users/me")
    db = nt.req("GET", f"/databases/{ref_id}")
    ds_id = _ds_id(db)
    ds = nt.req("GET", f"/data_sources/{ds_id}") if ds_id else {}
    return {
        "bot": me.get("name"),
        "database": {
            "id": db.get("id"),
            "title": "".join(x.get("plain_text", "") for x in db.get("title") or []),
            "parent": db.get("parent"),
            "data_sources": db.get("data_sources") or [],
        },
        "data_source": {
            "id": ds.get("id"),
            "properties": ds.get("properties") or {},
        },
    }


def inspect_data_sources(nt: Notion, ids: list[str]) -> dict:
    out = {}
    for raw in ids:
        ds_id = clean_id(raw)
        if not ds_id:
            continue
        ds = nt.req("GET", f"/data_sources/{ds_id}")
        out[ds_id] = {
            "id": ds.get("id"),
            "title": "".join(x.get("plain_text", "") for x in ds.get("title") or []),
            "parent": ds.get("parent"),
            "database_parent": ds.get("database_parent"),
            "properties": ds.get("properties") or {},
        }
    return {"bot": nt.req("GET", "/users/me").get("name"), "data_sources": out}


def _match_props() -> dict:
    return {
        "名前": {"title": {}},
        "試合URL": {"url": {}},
        "動画URL": {"url": {}},
        "試合日": {"date": {}},
        "ステージ": {"rich_text": {}},
        "マップカウント": {"rich_text": {}},
        "トーナメント": {"select": {"options": []}},
        "Tier": {"select": {"options": [
            {"name": n} for n in ("Tier1", "Tier2", "GC", "Unknown")
        ]}},
        "地域": {"select": {"options": []}},
        "チーム": {"multi_select": {"options": []}},
    }


def _map_props(match_ds: str) -> dict:
    return {
        "マップ名": {"title": {}},
        "マップ順": {"number": {"format": "number"}},
        "動画URL": {"url": {}},
        "勝者": {"select": {"options": []}},
        "チーム1ラウンド": {"number": {"format": "number"}},
        "チーム2ラウンド": {"number": {"format": "number"}},
        "合計ラウンド": {"number": {"format": "number"}},
        "試合": {"relation": {
            "data_source_id": match_ds,
            "dual_property": {"synced_property_name": "マップ"},
        }},
    }


def _create(nt: Notion, parent_id: str, title: str, props: dict) -> tuple[str, str]:
    body = {
        "parent": {"type": "page_id", "page_id": parent_id},
        "title": rich_text(title),
        "initial_data_source": {
            "title": rich_text(title),
            "properties": props,
        },
    }
    db = nt.req("POST", "/databases", body)
    db_id = db["id"]
    ds_id = _ds_id(db)
    if not ds_id:
        db = nt.req("GET", f"/databases/{db_id}")
        ds_id = _ds_id(db)
    if not ds_id:
        raise RuntimeError(f"database {db_id} has no data source")
    return db_id, ds_id


def create(nt: Notion, parent_id: str, ref_id: str) -> dict:
    match_db, match_ds = _create(nt, parent_id, MATCH_TITLE, _match_props())
    map_db, map_ds = _create(nt, parent_id, MAP_TITLE, _map_props(match_ds))
    return {
        "reference_database_id": ref_id,
        "parent_page_id": parent_id,
        "match_database_id": match_db,
        "match_data_source_id": match_ds,
        "map_database_id": map_db,
        "map_data_source_id": map_ds,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-db", default="37cd54a7c90b812eb284cdeca3e749be")
    ap.add_argument("--data-sources", default="")
    ap.add_argument("--parent-page", default="")
    ap.add_argument("--mode", choices=("inspect", "create"), default="inspect")
    a = ap.parse_args(argv)

    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        raise SystemExit("NOTION_TOKEN is empty")
    nt = Notion(token)

    if a.data_sources:
        print(json.dumps(inspect_data_sources(nt, a.data_sources.split(",")), ensure_ascii=False, indent=2))
        return 0

    ref_id = clean_id(a.reference_db)
    info = inspect(nt, ref_id)

    if a.mode == "inspect":
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    parent = clean_id(a.parent_page) or clean_id((info["database"].get("parent") or {}).get("page_id"))
    if not parent:
        raise SystemExit("parent page id is not available; pass --parent-page")
    out = create(nt, parent, ref_id)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
