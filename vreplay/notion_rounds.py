"""全ラウンドを Notion のデータベース（ラウンドDB）にする。1 行 = 1 ラウンド × 1 チーム。

Notion のフィルターで「ULT を 2 個以上使ったラウンド」「G2 の攻撃のフェイク」「対オペのフルバイで勝ったラウンド」
のように検索できるよう、ラウンドの種類・ULT・オペ・初動の形・攻め先・勝因をプロパティにする。

- データベースは親ページ（notion.rounds_parent か playbook.notion_parent）の下に 1 つだけ作り、
  ID は output.dir/_cache/notion_rounds.json に覚える（config の notion.rounds_data_source でも指定できる）
- 行はキー「試合ID:マップ順:R:チーム番号」で上書きするので、何度流しても重複しない
- report.json だけから作れるので、解析済みのマップは `vreplay rounds` でまとめて入れられる
"""
from __future__ import annotations

import json
from pathlib import Path

from . import flow as F
from .notion import Notion, _rt

TYPES = ["ピストル", "アンチエコ", "ボーナス", "フルバイ", "対フォース", "フォース", "ハーフ", "エコ", "OT", "?"]
PROPS = {
    "名前": {"title": {}},
    "キー": {"rich_text": {}},
    "チーム": {"select": {}},
    "相手": {"select": {}},
    "マップ": {"select": {}},
    "大会": {"select": {}},
    "ステージ": {"rich_text": {}},
    "日付": {"date": {}},
    "R": {"number": {}},
    "前後半": {"select": {"options": [{"name": n} for n in ("前半", "後半", "OT")]}},
    "半分の何R目": {"number": {}},
    "攻守": {"select": {"options": [{"name": "攻撃", "color": "red"}, {"name": "防衛", "color": "blue"}]}},
    "結果": {"select": {"options": [{"name": "勝ち", "color": "green"}, {"name": "負け", "color": "gray"}]}},
    "勝ち方": {"select": {}},
    "スコア": {"rich_text": {}},
    "種類": {"select": {"options": [{"name": n} for n in TYPES]}},
    "組み立て": {"rich_text": {}},
    "ULT数": {"number": {}},
    "ULTラウンド": {"checkbox": {}},
    "ULT": {"multi_select": {}},
    "オペ": {"checkbox": {}},
    "対オペ": {"checkbox": {}},
    "購入": {"select": {}},
    "残金k": {"number": {}},
    "決め手": {"multi_select": {}},
    "初動の形": {"select": {}},
    "形": {"rich_text": {}},
    "攻め先": {"select": {}},
    "分岐": {"select": {}},
    "設置": {"select": {}},
    "先制": {"select": {"options": [{"name": "取った"}, {"name": "取られた"}]}},
    "構成": {"multi_select": {}},
    "動画": {"url": {}},
    "マップページ": {"url": {}},
}


def _sel(v) -> dict:
    v = str(v or "").replace(",", " ").strip()[:100]
    return {"select": {"name": v} if v else None}


def _multi(vs) -> dict:
    return {"multi_select": [{"name": str(v).replace(",", " ")[:100]} for v in dict.fromkeys(vs) if v][:25]}


def _txt(v) -> dict:
    return {"rich_text": [_rt(str(v))] if v else []}


def rows(rep: dict, map_url: str | None = None) -> list[tuple[str, dict]]:
    """report.json 1 マップぶん → [(キー, プロパティ)]"""
    types = F.all_types(rep)
    names = {1: rep.get("team1"), 2: rep.get("team2")}
    tags = {no: rep.get(f"tag{no}") or names[no] for no in (1, 2)}
    comps = {no: [p.get("agent") for p in rep.get("players") or [] if p.get("team") == no and p.get("agent")]
             for no in (1, 2)}
    out = []
    for rr in rep.get("rounds") or []:
        n = int(rr.get("n") or 0)
        if not n or rr.get("winner_no") not in (1, 2):
            continue
        kills = sorted((k for k in rr.get("kills") or [] if k.get("killer_team")), key=lambda k: float(k.get("t") or 0))
        rt = rr.get("tactics") or {}
        reason = rr.get("reason") or {}
        for no in (1, 2):
            t = types.get((n, no)) or {}
            b = F.branch(rep, rr, no) or {}
            su = (rt.get("setups") or {}).get(str(no)) or {}
            fk = kills[0] if kills else None
            key = f"{rep.get('match_id')}:{rep.get('order')}:{n}:{no}"
            bank = rr.get(f"bank{no}")
            branch_kind = None
            if b.get("tag"):
                branch_kind = ("フェイク" if b["tag"].startswith("フェイク") else "MIDから" if b["tag"].startswith("MID")
                               else "分散から" if b["tag"].startswith("分散") else "直行")
            props = {
                "名前": {"title": [_rt(f"{tags[no]} vs {tags[3 - no]} {rep.get('map')} R{n}"
                                      f"（{t.get('label') or '?'}・{'勝' if t.get('win') else '負'}）")]},
                "キー": _txt(key),
                "チーム": _sel(names[no]), "相手": _sel(names[3 - no]), "マップ": _sel(rep.get("map")),
                "大会": _sel(rep.get("event")), "ステージ": _txt(rep.get("stage")),
                "日付": {"date": {"start": str(rep.get("date"))[:10]} if rep.get("date") else None},
                "R": {"number": n}, "前後半": _sel(t.get("half")), "半分の何R目": {"number": t.get("idx")},
                "攻守": _sel(t.get("side")), "結果": _sel("勝ち" if t.get("win") else "負け"),
                "勝ち方": _sel(rr.get("how_ja")), "スコア": _txt(rr.get("score")),
                "種類": _sel(t.get("type")), "組み立て": _txt(t.get("label")),
                "ULT数": {"number": len(t.get("ults") or [])}, "ULTラウンド": {"checkbox": bool(t.get("ult_round"))},
                "ULT": _multi(t.get("ults") or []),
                "オペ": {"checkbox": "オペ" in (t.get("mods") or [])},
                "対オペ": {"checkbox": "対オペ" in (t.get("mods") or [])},
                "購入": _sel((rr.get(f"buy{no}") or "").split(" ")[0]),
                "残金k": {"number": float(bank) if isinstance(bank, (int, float)) else None},
                "決め手": _multi(reason.get("tags") or []),
                "初動の形": _sel(su.get("name")), "形": _txt(su.get("shape")),
                "攻め先": _sel(b.get("final")), "分岐": _sel(branch_kind),
                "設置": _sel(rt.get("plant_site")),
                "先制": _sel(None if fk is None else "取った" if fk.get("killer_team") == names[no] else "取られた"),
                "構成": _multi(comps[no]),
                "動画": {"url": rr.get("yt_url") or None},
                "マップページ": {"url": map_url or None},
            }
            out.append((key, props))
    return out


def _cache_path(cfg: dict) -> Path:
    return Path(cfg["output"]["dir"]) / "_cache" / "notion_rounds.json"


def ensure_db(nt: Notion, cfg: dict, parent: str | None, log=print) -> str | None:
    """ラウンドDB の data source ID。無ければ parent の下に作る。足りないプロパティは足す。"""
    ds = (cfg.get("notion") or {}).get("rounds_data_source")
    cp = _cache_path(cfg)
    if not ds and cp.exists():
        try:
            ds = json.loads(cp.read_text("utf-8")).get("data_source")
        except Exception:
            ds = None
    if ds:
        try:
            cur = nt.req("GET", f"/data_sources/{ds}")
            missing = {k: v for k, v in PROPS.items() if k not in (cur.get("properties") or {})}
            if missing:
                nt.req("PATCH", f"/data_sources/{ds}", {"properties": missing})
            return ds
        except Exception as e:
            log(f"  ラウンドDB（{ds}）を開けませんでした：{e}")
            if (cfg.get("notion") or {}).get("rounds_data_source"):
                return None
    if not parent:
        log("  ラウンドDB を作る親ページがありません（config の notion.rounds_parent か --notion <URL>）")
        return None
    r = nt.req("POST", "/databases", {
        "parent": {"type": "page_id", "page_id": parent}, "is_inline": False,
        "title": [_rt("ラウンドDB")],
        "icon": {"type": "emoji", "emoji": "🗂️"},
        "initial_data_source": {"properties": PROPS}})
    ds = r["data_sources"][0]["id"]
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"data_source": ds, "database": r.get("id"), "url": r.get("url")}), "utf-8")
    log(f"  ラウンドDB を作りました：{r.get('url', '')}")
    return ds


def sync_map(nt: Notion, ds: str, rep: dict, map_url: str | None = None, log=print) -> int:
    """1 マップぶんの行を作る・上書きする。返り値は書いた行数。"""
    rs = rows(rep, map_url)
    if not rs:
        return 0
    prefix = f"{rep.get('match_id')}:{rep.get('order')}:"
    have = {}
    for pg in nt.query(ds, {"property": "キー", "rich_text": {"starts_with": prefix}}):
        k = "".join(t.get("plain_text", "") for t in (pg.get("properties", {}).get("キー", {}).get("rich_text") or []))
        have[k] = pg["id"]
    for key, props in rs:
        if key in have:
            nt.update_page(have[key], props)
        else:
            nt.create_page(ds, props)
    return len(rs)


def parent_of(cfg: dict, override: str | None = None) -> str | None:
    from .notion_playbook import page_id
    for s in (override, (cfg.get("notion") or {}).get("rounds_parent"), (cfg.get("playbook") or {}).get("notion_parent")):
        pid = page_id(str(s or ""))
        if pid:
            return pid
    return None


def backfill(nt: Notion, cfg: dict, parent: str | None, root: Path, since: str | None = None,
             until: str | None = None, log=print) -> int:
    """output.dir の report.json をまとめてラウンドDB に入れる。"""
    ds = ensure_db(nt, cfg, parent, log)
    if not ds:
        return 0
    n_maps = n_rows = 0
    for p in sorted(Path(root).glob("**/report.json")):
        try:
            rep = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        d = str(rep.get("date") or "")[:10]
        if "match_id" not in rep or (since and d < since) or (until and d > until):
            continue
        try:
            k = sync_map(nt, ds, rep, log=log)
        except Exception as e:
            log(f"  ✗ {p.parent.name}: {e}")
            continue
        n_maps += 1
        n_rows += k
        log(f"  {d} {rep.get('tag1')} vs {rep.get('tag2')} {rep.get('map')}: {k} 行")
    log(f"ラウンドDB：{n_maps} マップ・{n_rows} 行")
    return n_rows
