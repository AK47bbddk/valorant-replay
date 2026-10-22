"""vlr.gg のスクレイピング：前日の試合一覧、試合ごとのマップ・ラウンド・エコ・選手・VOD。"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

BASE = "https://www.vlr.gg"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
VLR_TZ = ZoneInfo("America/New_York")  # cookie なしの vlr.gg は米国東部時間で表示する
JST = ZoneInfo("Asia/Tokyo")
TZ_ABBR = {
    "EDT": "America/New_York", "EST": "America/New_York",
    "PDT": "America/Los_Angeles", "PST": "America/Los_Angeles",
    "CEST": "Europe/Berlin", "CET": "Europe/Berlin", "BST": "Europe/London",
    "JST": "Asia/Tokyo", "KST": "Asia/Seoul", "UTC": "UTC", "GMT": "UTC",
}
WIN_TYPES = {"elim": "全滅", "defuse": "解除", "boom": "爆破", "time": "時間切れ"}
BUY_LEVEL = {"": "eco", "$": "semi-eco", "$$": "semi-buy", "$$$": "full"}

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})


def fetch(url: str, tries: int = 3) -> BeautifulSoup:
    if url.startswith("/"):
        url = BASE + url
    for i in range(tries):
        try:
            r = _session.get(url, timeout=30)
            if r.status_code == 200:
                time.sleep(0.8)  # vlr.gg への負荷を抑える
                return BeautifulSoup(r.text, "lxml")
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"VLR fetch failed: {url}")


# ---------------------------------------------------------------- データ構造
@dataclass
class Round:
    n: int
    winner: int | None = None      # 1 or 2
    side: str | None = None        # 勝ったチームの攻守 "攻撃" / "防衛"
    how: str | None = None         # elim / defuse / boom / time


@dataclass
class Econ:
    n: int
    buy1: str = ""
    buy2: str = ""
    bank1: float | None = None     # 千クレジット
    bank2: float | None = None


@dataclass
class Player:
    name: str
    agent: str
    team: int


@dataclass
class MapData:
    order: int
    name: str
    game_id: str
    score1: int = 0
    score2: int = 0
    rounds: list[Round] = field(default_factory=list)
    econ: list[Econ] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    vod_url: str | None = None
    vod_id: str | None = None
    vod_t: int | None = None

    @property
    def total_rounds(self) -> int:
        return self.score1 + self.score2


@dataclass
class Match:
    id: str
    url: str
    team1: str
    team2: str
    event: str = ""
    stage: str = ""
    dt_jst: datetime | None = None
    maps: list[MapData] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.team1} vs {self.team2}"

    @property
    def map_count(self) -> str:
        w1 = sum(1 for m in self.maps if m.score1 > m.score2)
        w2 = sum(1 for m in self.maps if m.score2 > m.score1)
        return f"{w1}-{w2}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dt_jst"] = self.dt_jst.isoformat() if self.dt_jst else None
        return d


# ---------------------------------------------------------------- 補助
def _t(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el else ""


def youtube_parts(url: str) -> tuple[str | None, int | None]:
    """YouTube URL → (動画ID, 開始秒)。"""
    if not url:
        return None, None
    u = urlparse(url)
    q = parse_qs(u.query)
    vid = None
    if "youtu.be" in u.netloc:
        vid = u.path.strip("/").split("/")[0]
    elif "youtube.com" in u.netloc:
        vid = (q.get("v") or [None])[0]
        if not vid and "/live/" in u.path:
            vid = u.path.split("/live/")[1].split("/")[0]
    t = (q.get("t") or q.get("start") or [None])[0]
    secs = None
    if t:
        m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?", t)
        if m and any(m.groups()):
            h, mi, s = (int(x or 0) for x in m.groups())
            secs = h * 3600 + mi * 60 + s
    return vid, secs


def classify_tier(event: str) -> str:
    e = event.lower()
    if "game changers" in e:
        return "GC"
    if "challengers" in e or "ascension" in e:
        return "Tier2"
    if e.startswith("vct") or "masters" in e or "champions" in e:
        return "Tier1"
    return "Unknown"


REGIONS = ["Pacific", "Americas", "EMEA", "China", "Japan", "Korea", "Brazil", "North America",
           "LATAM North", "LATAM South", "Oceania", "Southeast Asia", "Türkiye", "France", "DACH",
           "Spain", "South Asia"]


def classify_region(event: str) -> str | None:
    if any(k in event for k in ("Masters", "Champions", "Esports World Cup")) and "Qualifier" not in event:
        return "International"
    for r in sorted(REGIONS, key=len, reverse=True):
        if r.lower() in event.lower():
            return r
    return None


# ---------------------------------------------------------------- 結果一覧
def list_results(target: date, pages: int = 4) -> list[dict]:
    """JST の target 日に行われた試合（VLR の results から）。"""
    out = []
    lo = datetime.combine(target, datetime.min.time(), JST)
    hi = lo + timedelta(days=1)
    for page in range(1, pages + 1):
        soup = fetch(f"/matches/results/?page={page}")
        oldest = None
        for label in soup.select("div.wf-label.mod-large"):
            dtext = re.sub(r"\b(Today|Yesterday)\b", "", _t(label)).strip()
            try:
                day = datetime.strptime(dtext, "%a, %B %d, %Y").date()
            except ValueError:
                continue
            card = label.find_next_sibling("div", class_="wf-card")
            if not card:
                continue
            for a in card.select("a.match-item, a.wf-module-item"):
                tm = _t(a.select_one(".match-item-time"))
                try:
                    clock = datetime.strptime(tm, "%I:%M %p").time()
                except ValueError:
                    clock = datetime.min.time()
                dt = datetime.combine(day, clock, VLR_TZ).astimezone(JST)
                oldest = dt if oldest is None or dt < oldest else oldest
                if not (lo <= dt < hi):
                    continue
                teams = [_t(x) for x in a.select(".match-item-vs-team-name .text-of")]
                ev = a.select_one(".match-item-event")
                series = _t(ev.select_one(".match-item-event-series")) if ev else ""
                event = _t(ev).replace(series, "").strip() if ev else ""
                out.append({
                    "url": BASE + a.get("href", ""),
                    "team1": teams[0] if teams else "", "team2": teams[1] if len(teams) > 1 else "",
                    "event": event, "stage": series, "dt_jst": dt,
                })
        if oldest is not None and oldest < lo:
            break
    return out


def filter_results(items: list[dict], f: dict) -> list[dict]:
    inc = [s.lower() for s in f.get("event_include") or []]
    exc = [s.lower() for s in f.get("event_exclude") or []]
    teams = {s.lower() for s in f.get("teams") or []}
    out = []
    for it in items:
        ev = it["event"].lower()
        if inc and not any(s in ev for s in inc):
            continue
        if any(s in ev for s in exc):
            continue
        if teams and not ({it["team1"].lower(), it["team2"].lower()} & teams):
            continue
        out.append(it)
    return out[: f.get("max_matches") or None]


# ---------------------------------------------------------------- 試合ページ
def _parse_rounds(game) -> list[Round]:
    rounds = []
    for col in game.select("div.vlr-rounds-row-col"):
        num = col.select_one(".rnd-num")
        if not num or not _t(num).isdigit():
            continue
        r = Round(n=int(_t(num)))
        sqs = col.select("div.rnd-sq")
        for i, sq in enumerate(sqs[:2]):
            cls = sq.get("class", [])
            if "mod-win" in cls:
                r.winner = i + 1
                r.side = "攻撃" if "mod-t" in cls else "防衛" if "mod-ct" in cls else None
                img = sq.find("img")
                if img and img.get("src"):
                    r.how = re.sub(r"\.\w+$", "", img["src"].rsplit("/", 1)[-1])
        if r.winner:
            rounds.append(r)
    return rounds


def _parse_players(game) -> list[Player]:
    players = []
    for ti, table in enumerate(game.select("table.wf-table-inset.mod-overview")[:2]):
        for tr in table.select("tbody tr"):
            name = _t(tr.select_one("td.mod-player .text-of")) or _t(tr.select_one("td.mod-player a div"))
            img = tr.select_one("td.mod-agents img")
            agent = (img.get("title") or img.get("alt") or "") if img else ""
            if name:
                players.append(Player(name=name, agent=agent, team=ti + 1))
    return players


def _parse_econ(game) -> list[Econ]:
    tables = game.select("table.wf-table-inset.mod-econ")
    if not tables:
        return []
    t = tables[-1]
    banks = []
    for b in t.select("div.bank"):
        try:
            banks.append(float(_t(b).replace(",", "")))
        except ValueError:
            banks.append(None)
    sqs = [BUY_LEVEL.get(_t(s).replace(" ", ""), "") for s in t.select("div.rnd-sq")]
    n = min(len(banks), len(sqs)) // 2
    return [Econ(n=i + 1, buy1=sqs[2 * i], buy2=sqs[2 * i + 1], bank1=banks[2 * i], bank2=banks[2 * i + 1])
            for i in range(n)]


def _parse_vods(soup) -> list[tuple[str, str]]:
    """[(ラベル, URL)]"""
    box = soup.select(".match-vods a[href]") or soup.select(".match-streams-bets-container a[href]")
    links = [(_t(a), a["href"]) for a in box if "youtu" in a.get("href", "")]
    if not links:
        links = [(_t(a), a["href"]) for a in soup.select("a[href*='youtu']")]
    return links


def _match_date(soup) -> datetime | None:
    els = soup.select(".match-header-date .moment-tz-convert")
    if len(els) >= 2:
        dtext = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", _t(els[0]))
        ttext = _t(els[1])
        m = re.match(r"(\d{1,2}:\d{2}\s*[AP]M)\s*([A-Z]{2,4})?", ttext)
        year = (els[0].get("data-utc-ts") or "")[:4] or str(datetime.now().year)
        if m:
            try:
                tz = ZoneInfo(TZ_ABBR.get(m.group(2) or "", "America/New_York"))
                d = datetime.strptime(f"{dtext} {year} {m.group(1)}", "%A, %B %d %Y %I:%M %p")
                return d.replace(tzinfo=tz).astimezone(JST)
            except ValueError:
                pass
    ts = els[0].get("data-utc-ts") if els else None
    if ts:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=VLR_TZ).astimezone(JST)
        except ValueError:
            pass
    return None


def parse_match(url: str, soup: BeautifulSoup | None = None, econ_soup: BeautifulSoup | None = None) -> Match:
    mid = re.search(r"vlr\.gg/(\d+)", url).group(1)
    soup = soup or fetch(url)
    names = [_t(x) for x in soup.select(".match-header-link-name .wf-title-med")]
    ev_box = soup.select_one(".match-header-event")
    stage = _t(soup.select_one(".match-header-event-series"))
    event = ""
    if ev_box:
        first = ev_box.find("div").find("div") if ev_box.find("div") else None
        event = _t(first) if first else _t(ev_box).replace(stage, "").strip()
    m = Match(id=mid, url=url.split("?")[0], team1=names[0] if names else "?",
              team2=names[1] if len(names) > 1 else "?", event=event, stage=stage, dt_jst=_match_date(soup))

    games = [g for g in soup.select("div.vm-stats-game") if g.get("data-game-id") not in (None, "all")]
    econ_games = {}
    if econ_soup is None and games:
        try:
            econ_soup = fetch(m.url + "/?game=all&tab=economy")
        except RuntimeError:
            econ_soup = None
    if econ_soup is not None:
        econ_games = {g.get("data-game-id"): g for g in econ_soup.select("div.vm-stats-game")}

    for i, g in enumerate(games):
        header = g.select_one(".vm-stats-game-header")
        mp = header.select_one(".map") if header else None
        name_el = mp.select_one("span") if mp else None
        name = (name_el.find(string=True, recursive=False) or _t(name_el)) if name_el else ""
        name = re.sub(r"\bPICK\b", "", str(name)).strip()
        scores = [_t(s) for s in header.select(".team .score")] if header else []
        md = MapData(order=i + 1, name=name, game_id=g["data-game-id"],
                     score1=int(scores[0]) if scores and scores[0].isdigit() else 0,
                     score2=int(scores[1]) if len(scores) > 1 and scores[1].isdigit() else 0)
        md.rounds = _parse_rounds(g)
        md.players = _parse_players(g)
        eg = econ_games.get(md.game_id)
        md.econ = _parse_econ(eg) if eg is not None else []
        if md.rounds and not (md.score1 or md.score2):
            md.score1 = sum(r.winner == 1 for r in md.rounds)
            md.score2 = sum(r.winner == 2 for r in md.rounds)
        m.maps.append(md)

    # VOD：ラベル先頭の数字（"1Abyss" / "Map 1"）かマップ名で対応づける
    vods = _parse_vods(soup)
    for label, href in vods:
        num = re.match(r"\s*(?:Map\s*)?(\d+)", label)
        target = None
        if num:
            k = int(num.group(1))
            target = next((x for x in m.maps if x.order == k), None)
        if target is None:
            target = next((x for x in m.maps if x.name and x.name.lower() in label.lower()), None)
        if target is not None and not target.vod_url:
            target.vod_url = href
            target.vod_id, target.vod_t = youtube_parts(href)
    # 「Full match」だけの場合は全マップ同じ動画（開始秒なし）
    if vods and not any(x.vod_url for x in m.maps):
        href = vods[0][1]
        for x in m.maps:
            x.vod_url = href
            x.vod_id, x.vod_t = youtube_parts(href)
    return m
