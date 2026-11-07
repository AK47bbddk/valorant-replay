"""マップ・構成の知識ベース（プレイブックの「マップのメタ」「じゃんけん」「構成の特徴」の元）。

3 つの出どころを重ねる（後のものが前を上書き）：

1. リポジトリ同梱の YAML（`config/kb/`）
   - `concepts.yaml` … チョークポイントの守り方（壊す・返す・殺す）、じゃんけん構造、ターン、エントリーの種類など
   - `agents.yaml`   … エージェントの役割と特徴（フラッシュ・索敵・ドライに強い…）
   - `comps.yaml`    … 構成ごとのメモ（強み・弱み・今のメタへの刺さり方・アンチピック）
   - `maps/<map>.yaml` … エリア（評価・ドライで進めるか）、チョークポイント、攻守の「手」と判定条件・相性
2. Obsidian のメモ（Valo-wiki の content フォルダ。`kb.vault`）
   - `maps/<Map>.md` の「評価：サイト距離(近)、テイク難易度(中)…」と「強い動き（セオリー）」
   - `agents/<role>/<Agent>.md` のフォルダ名から役割、`meta/Current_Meta.md` の Tier
3. 手で足す YAML（`<output.dir>/kb/` に同じ形で置く。Drive で編集できる）

手（hands）の判定条件（`when`）は vreplay.hands が読む。書ける条件は hands.py の先頭を参照。
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent / "config" / "kb"

_CACHE: dict = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _merge(a, b):
    """dict は再帰的に、hands・areas などの名前つきリストは name で重ねる。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _merge(a.get(k), v) if k in a else copy.deepcopy(v)
        return out
    if isinstance(a, list) and isinstance(b, list) and all(isinstance(x, dict) and "name" in x for x in a + b):
        out = [copy.deepcopy(x) for x in a]
        idx = {x["name"]: i for i, x in enumerate(out)}
        for x in b:
            if x["name"] in idx:
                out[idx[x["name"]]] = _merge(out[idx[x["name"]]], x)
            else:
                out.append(copy.deepcopy(x))
        return out
    return copy.deepcopy(b) if b is not None else copy.deepcopy(a)


def _read_yaml(p: Path) -> dict:
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------- Obsidian（Valo-wiki）
EVAL_KEYS = {"サイト距離": "distance", "テイク難易度": "take", "分岐": "branch", "ロング": "long", "ホールド": "hold"}


def parse_vault_map(text: str) -> dict:
    """マップのページ（maps/Split.md など）から、エリアの評価と「強い動き」を拾う。"""
    areas, theory = [], {"attack": [], "defense": []}
    cur = None
    sec = None
    side = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            sec = s[3:]
            side = None
            cur = None
            continue
        if s.startswith("### "):
            head = re.sub(r"^\d+\.\s*", "", s[4:]).strip()
            if sec and ("強い動き" in sec or "セオリー" in sec):
                side = "attack" if "攻" in head else "defense" if ("防" in head or "守" in head) else None
                continue
            cur = {"name": head}
            areas.append(cur)
            continue
        m = re.match(r"-\s*\*\*評価\*\*\s*[:：]\s*(.+)", s)
        if m and cur is not None:
            ev = {}
            for part in re.split(r"[、,]", m.group(1)):
                mm = re.match(r"\s*(\S+?)\((.+?)\)", part)
                if not mm:
                    continue
                for ja, key in EVAL_KEYS.items():
                    if mm.group(1).startswith(ja):
                        ev[key] = mm.group(2)
            if ev:
                cur["eval"] = ev
            continue
        m = re.match(r"-\s*\*\*(戦術考察|攻撃|防衛|防衛側の回答|攻めの崩し)\*\*\s*[:：]\s*(.+)", s)
        if m and cur is not None:
            cur.setdefault("notes", []).append(f"{m.group(1)}：{_strip_links(m.group(2))}")
            continue
        m = re.match(r"-\s*\*\*(.+?)\*\*\s*[:：]\s*(.+)", s)
        if m and side:
            theory[side].append({"name": _strip_links(m.group(1)), "detail": _strip_links(m.group(2))})
    areas = [a for a in areas if a.get("eval") or a.get("notes")]
    return {"areas": areas, "theory": theory}


def _strip_links(s: str) -> str:
    return re.sub(r"\[\[([^\]|]+)(\|([^\]]+))?\]\]", lambda m: m.group(3) or m.group(1), s or "").strip()


def parse_vault(root: str | Path) -> dict:
    """Obsidian の content フォルダ全体から、マップ・役割・Tier。"""
    root = Path(root)
    out: dict = {"maps": {}, "agents": {}, "tiers": {}, "pages": []}
    if not root.exists():
        return out
    for p in sorted((root / "maps").glob("*.md")):
        try:
            d = parse_vault_map(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        d["page"] = p.stem
        out["maps"][_norm(p.stem)] = d
    for role in ("duelist", "initiator", "controller", "sentinel"):
        for p in sorted((root / "agents" / role).glob("*.md")):
            out["agents"][p.stem] = {"role": role}
    cm = root / "meta" / "Current_Meta.md"
    if cm.exists():
        tier = None
        for line in cm.read_text(encoding="utf-8").splitlines():
            m = re.match(r"#+\s*([SABCD])\s*Tier", line.strip())
            if m:
                tier = m.group(1)
                continue
            if line.startswith("## "):
                tier = None
            if tier:
                for a in re.findall(r"\[\[([^\]|]+)", line):
                    out["tiers"].setdefault(a, tier)
    out["pages"] = sorted(p.stem for p in root.rglob("*.md") if "raw" not in p.parts)
    return out


# ---------------------------------------------------------------- 読み込み
class KB:
    def __init__(self, concepts: dict, agents: dict, comps: list, maps: dict, vault: dict | None = None):
        self.concepts = concepts
        self.agents = agents
        self.comps = comps
        self.maps = maps
        self.vault = vault or {}

    def map(self, name: str | None) -> dict:
        """マップの知識（YAML ＋ Obsidian のエリア評価・セオリー）。無ければ空の形。"""
        key = _norm(name)
        m = copy.deepcopy(self.maps.get(key) or {"map": name, "areas": [], "hands": {"attack": [], "defense": []}})
        v = (self.vault.get("maps") or {}).get(key)
        if v:
            # Obsidian の評価は、YAML に同じ名前のエリアが無いときだけ足す（YAML に書いた方を優先）
            have = {_norm(a.get("name")) for a in m.get("areas") or []}
            for a in v.get("areas") or []:
                if _norm(a["name"]) in have:
                    for a2 in m["areas"]:
                        if _norm(a2.get("name")) == _norm(a["name"]):
                            a2.setdefault("eval", a.get("eval"))
                            a2.setdefault("notes", a.get("notes"))
                    continue
                m.setdefault("areas", []).append({**a, "source": "obsidian"})
            m["theory"] = v.get("theory")
            m["page"] = v.get("page")
        m.setdefault("hands", {}).setdefault("attack", [])
        m["hands"].setdefault("defense", [])
        return m

    def role(self, agent: str | None) -> str | None:
        a = self.agent(agent)
        return a.get("role") if a else None

    def agent(self, agent: str | None) -> dict:
        if not agent:
            return {}
        k = _norm(agent)
        for name, d in self.agents.items():
            if _norm(name) == k or k in [_norm(x) for x in d.get("aliases") or []]:
                out = dict(d)
                out["name"] = name
                t = (self.vault.get("tiers") or {}).get(name)
                if t:
                    out["tier"] = t
                return out
        v = (self.vault.get("agents") or {}).get(agent)
        return {"name": agent, **v} if v else {}

    def comp_notes(self, comp: list[str] | tuple, map_name: str | None = None) -> list[dict]:
        """構成メモのうち、その構成に当てはまるもの（must の全員がいる。map が書いてあれば一致するもの）。"""
        have = {_norm(a) for a in comp or []}
        out = []
        for c in self.comps:
            must = [_norm(a) for a in c.get("must") or []]
            if not must or not all(a in have for a in must):
                continue
            maps = [_norm(x) for x in c.get("maps") or []]
            if maps and _norm(map_name) not in maps:
                continue
            out.append(c)
        # 当てはまる人数の多いもの（＝細かいもの）から
        out.sort(key=lambda c: -len(c.get("must") or []))
        return out


def load(cfg: dict | None = None, refresh: bool = False) -> KB:
    cfg = cfg or {}
    kcfg = cfg.get("kb") or {}
    out_dir = Path((cfg.get("output") or {}).get("dir") or ".")
    dirs = [ROOT]
    for d in (kcfg.get("dir"), out_dir / "kb"):
        if d and Path(d).exists() and Path(d) != ROOT:
            dirs.append(Path(d))
    vault = kcfg.get("vault") or ""
    if not vault:
        for cand in (out_dir / "kb" / "vault", Path("/content/Valo-wiki/content")):
            if cand.exists():
                vault = str(cand)
                break
    key = (tuple(str(d) for d in dirs), vault)
    if not refresh and key in _CACHE:
        return _CACHE[key]
    concepts, agents, comps, maps = {}, {}, [], {}
    for d in dirs:
        concepts = _merge(concepts, _read_yaml(d / "concepts.yaml"))
        agents = _merge(agents, (_read_yaml(d / "agents.yaml").get("agents") or {}))
        new = _read_yaml(d / "comps.yaml").get("comps") or []
        comps = _merge(comps, new) if comps and new else (comps or new)
        for p in sorted((d / "maps").glob("*.yaml")) if (d / "maps").exists() else []:
            y = _read_yaml(p)
            k = _norm(y.get("map") or p.stem)
            maps[k] = _merge(maps.get(k) or {}, y)
    kb = KB(concepts, agents, comps, maps, parse_vault(vault) if vault else None)
    _CACHE[key] = kb
    return kb
