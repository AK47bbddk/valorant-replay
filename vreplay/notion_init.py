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
MATCH_DS = "34cd54a7-c90b-80fd-9f31-000b2682704c"
MAP_DS = "34cd54a7-c90b-8180-a2df-000b5f0d3378"


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
        props = {}
        for name, prop in (ds.get("properties") or {}).items():
            item = {"type": prop.get("type")}
            if prop.get("type") == "relation":
                rel = prop.get("relation") or {}
                item["relation"] = {
                    "data_source_id": rel.get("data_source_id"),
                    "database_id": rel.get("database_id"),
                    "dual": bool(rel.get("dual_property")),
                }
            props[name] = item
        out[ds_id] = {
            "id": ds.get("id"),
            "title": "".join(x.get("plain_text", "") for x in ds.get("title") or []),
            "parent": ds.get("parent"),
            "database_parent": ds.get("database_parent"),
            "properties": props,
        }
    return {"bot": nt.req("GET", "/users/me").get("name"), "data_sources": out}


def _clone_props(props: dict) -> dict:
    out = {}
    for name, prop in props.items():
        t = prop.get("type")
        if t == "relation":
            continue
        if t == "title":
            out[name] = {"title": {}}
        elif t == "rich_text":
            out[name] = {"rich_text": {}}
        elif t == "url":
            out[name] = {"url": {}}
        elif t == "date":
            out[name] = {"date": {}}
        elif t == "number":
            out[name] = {"number": {"format": (prop.get("number") or {}).get("format", "number")}}
        elif t == "multi_select":
            out[name] = {"multi_select": {"options": []}}
        elif t == "select":
            opts = []
            for o in (prop.get("select") or {}).get("options") or []:
                item = {"name": o["name"]}
                if o.get("color"):
                    item["color"] = o["color"]
                opts.append(item)
            out[name] = {"select": {"options": opts[:100]}}
        elif t == "checkbox":
            out[name] = {"checkbox": {}}
        else:
            raise RuntimeError(f"unsupported property type: {name} ({t})")
    return out


def _find_existing(nt: Notion, title: str) -> tuple[str, str] | None:
    res = nt.req("POST", "/search", {"query": title, "page_size": 100})
    for obj in res.get("results", []):
        if obj.get("object") != "database":
            continue
        got = "".join(x.get("plain_text", "") for x in obj.get("title") or [])
        if got != title:
            continue
        db_id = obj["id"]
        ds_id = _ds_id(obj)
        if not ds_id:
            obj = nt.req("GET", f"/databases/{db_id}")
            ds_id = _ds_id(obj)
        if ds_id:
            return db_id, ds_id
    return None


def _create_or_reuse(nt: Notion, parent_id: str, title: str, props: dict) -> tuple[str, str]:
    found = _find_existing(nt, title)
    if found:
        return found
    return _create(nt, parent_id, title, props)


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


def create(nt: Notion, parent_id: str, ref_id: str, match_src: str, map_src: str) -> dict:
    match_src = nt.req("GET", f"/data_sources/{clean_id(match_src)}")
    map_src = nt.req("GET", f"/data_sources/{clean_id(map_src)}")
    match_props = _clone_props(match_src.get("properties") or {})
    map_props = _clone_props(map_src.get("properties") or {})

    match_db, match_ds = _create_or_reuse(nt, parent_id, MATCH_TITLE, match_props)
    map_props["試合"] = {"relation": {
        "data_source_id": match_ds,
        "dual_property": {"synced_property_name": "マップ"},
    }}
    map_db, map_ds = _create_or_reuse(nt, parent_id, MAP_TITLE, map_props)

    match_now = nt.req("GET", f"/data_sources/{match_ds}")
    rel = (match_now.get("properties") or {}).get("マップ") or {}
    if (rel.get("relation") or {}).get("data_source_id") != map_ds:
        nt.req("PATCH", f"/data_sources/{map_ds}", {"properties": {
            "試合": {"relation": {
                "data_source_id": match_ds,
                "dual_property": {"synced_property_name": "マップ"},
            }},
        }})

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
    ap.add_argument("--match-ds", default=MATCH_DS)
    ap.add_argument("--map-ds", default=MAP_DS)
    ap.add_argument("--parent-page", default="")
    ap.add_argument("--mode", choices=("inspect", "create"), default="inspect")
    a = ap.parse_args(argv)

    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        raise SystemExit("NOTION_TOKEN is empty")
    nt = Notion(token)

    if a.data_sources:
        print(json.dumps(inspect_data_sources(nt, a.data_sources.split(",")), ensure_ascii=True, indent=2))
        return 0

    ref_id = clean_id(a.reference_db)
    info = inspect(nt, ref_id)

    if a.mode == "inspect":
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    parent = clean_id(a.parent_page) or clean_id((info["database"].get("parent") or {}).get("page_id"))
    if not parent:
        raise SystemExit("parent page id is not available; pass --parent-page")
    out = create(nt, parent, ref_id, a.match_ds, a.map_ds)
    print(json.dumps(out, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
