"""YouTube の動画から、必要な時間帯のバイト範囲だけを並列に取ってくる。

yt-dlp の --download-sections は内部で ffmpeg が 1 本の長い HTTP 接続で読むため、YouTube に実時間程度
（1 MB/s 前後）へ絞られる。YouTube の DASH 用 mp4 は「ftyp・moov（初期化）・sidx（各断片の位置と長さ）・
moof/mdat の断片…」という作りなので、
  1) 先頭を少し読んで sidx から「どのバイト範囲がどの時刻か」を知り、
  2) 必要な時刻の断片だけを 8MB ずつ並列に Range 要求で取り、
  3) 初期化部分の後ろにつなげて 1 本の mp4 にし、ffmpeg で時刻を 0 始まりに直す
ことで、絞られずに速く取れる（yt-dlp 本体も通常のダウンロードでは同じく分割して取っている）。
"""
from __future__ import annotations

import re
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

CHUNK = 8 * 1024 * 1024


def _boxes(buf: bytes, pos: int = 0):
    while pos + 8 <= len(buf):
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            if pos + 16 > len(buf):
                return
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hdr = 16
        elif size == 0:
            size = len(buf) - pos
        if size < hdr:
            return
        yield pos, typ.decode("latin-1"), size, hdr
        pos += size


def parse_sidx(buf: bytes, pos: int, hdr: int) -> dict:
    p = pos + hdr
    version = buf[p]
    p += 4
    _ref_id, timescale = struct.unpack(">II", buf[p:p + 8])
    p += 8
    if version == 0:
        ept, first = struct.unpack(">II", buf[p:p + 8])
        p += 8
    else:
        ept, first = struct.unpack(">QQ", buf[p:p + 16])
        p += 16
    _, count = struct.unpack(">HH", buf[p:p + 4])
    p += 4
    refs = []
    for _ in range(count):
        a, dur, _sap = struct.unpack(">III", buf[p:p + 12])
        p += 12
        if a >> 31:
            raise ValueError("階層になった sidx には対応していません")
        refs.append((a & 0x7FFFFFFF, dur))
    return {"timescale": timescale, "ept": ept, "first": first, "refs": refs}


def plan(head: bytes) -> dict:
    """先頭のバイト列から、初期化部分の長さと、各断片の (開始バイト, 長さ, 開始秒, 終了秒)。"""
    init_end, sidx = None, None
    for pos, typ, size, hdr in _boxes(head):
        if typ == "moov":
            init_end = pos + size
        if typ == "sidx":
            if pos + size > len(head):
                raise EOFError(pos + size)
            sidx = parse_sidx(head, pos, hdr)
            anchor = pos + size + sidx["first"]
            break
        if typ in ("moof", "mdat"):
            break
    if init_end is None or sidx is None:
        raise ValueError("moov か sidx が見つかりません（DASH 形式ではない）")
    ts = sidx["timescale"]
    t = sidx["ept"] / ts
    b = anchor
    segs = []
    for size, dur in sidx["refs"]:
        segs.append((b, size, t, t + dur / ts))
        b += size
        t += dur / ts
    return {"init_end": init_end, "segments": segs}


class Getter:
    """1 つの動画 URL からバイト範囲を取る。スレッドごとに接続を持つ。

    yt-dlp を同じ設定（cookies・クライアント・JS ランタイム・なりすまし）でプロセス内に作り、その urlopen で取る。
    以前は requests で直接取っていたが、yt-dlp が使う接続と違うためか 403 になった（Colab のログ）。
    取り方は 2 通り：Range ヘッダー（header）と、YouTube の再生プレーヤーと同じ &range=a-b（query）。
    probe() で小さな範囲を取って、通る方を選ぶ。
    """

    def __init__(self, url: str, headers: dict | None = None, ydl_opts: dict | None = None, cookies=None):
        import threading
        self.url, self.headers, self.opts, self.cookies = url, dict(headers or {}), ydl_opts, cookies
        self.mode = "header"
        self._tl = threading.local()

    def _client(self):
        c = getattr(self._tl, "c", None)
        if c is None:
            if self.opts is not None:
                import yt_dlp
                c = yt_dlp.YoutubeDL({**self.opts, "quiet": True, "no_warnings": True})
            else:
                c = requests.Session()
                if self.cookies is not None:
                    c.cookies = self.cookies
            self._tl.c = c
        return c

    def _once(self, a: int, b: int, mode: str) -> tuple[int, bytes]:
        h = dict(self.headers)
        url = self.url
        if mode == "header":
            h["Range"] = f"bytes={a}-{b}"
        else:
            url = url + ("&" if "?" in url else "?") + f"range={a}-{b}"
        c = self._client()
        if self.opts is not None:
            from yt_dlp.networking import Request
            try:
                with c.urlopen(Request(url, headers=h)) as r:
                    return int(getattr(r, "status", 200)), r.read()
            except Exception as e:
                st = getattr(getattr(e, "response", None), "status", None) or getattr(e, "status", None)
                if st:
                    return int(st), b""
                raise
        r = c.get(url, headers=h, timeout=60)
        return r.status_code, r.content

    def fetch(self, url: str, tries: int = 4) -> bytes:
        """URL を丸ごと取る（HLS のプレイリストと断片用）。"""
        last = None
        c = self._client()
        for _ in range(tries):
            try:
                if self.opts is not None:
                    from yt_dlp.networking import Request
                    with c.urlopen(Request(url, headers=dict(self.headers))) as r:
                        st, data = int(getattr(r, "status", 200)), r.read()
                else:
                    rr = c.get(url, headers=dict(self.headers), timeout=60)
                    st, data = rr.status_code, rr.content
                if st == 200 and data:
                    return data
                last = f"HTTP {st}・{len(data)} バイト"
            except Exception as e:
                st_ = getattr(getattr(e, "response", None), "status", None) or getattr(e, "status", None)
                last = f"HTTP {st_}" if st_ else str(e)[:200]
        raise RuntimeError(f"取れませんでした（{last}）：{url[:80]}")

    def get(self, a: int, b: int, tries: int = 4, exact: bool = True) -> bytes:
        last = None
        for _ in range(tries):
            try:
                st, data = self._once(a, b, self.mode)
                if st in (200, 206) and (len(data) == b - a + 1 or (not exact and data)):
                    return data
                last = f"HTTP {st}・{len(data)} バイト"
            except Exception as e:
                last = str(e)[:200]
        raise RuntimeError(f"範囲 {a}-{b} を取れませんでした（{self.mode}：{last}）")

    def probe(self) -> str:
        """小さな範囲（1KB）を両方の取り方で試し、通った方を使う。どちらもだめなら理由を返して例外。"""
        why = []
        for mode in ("header", "query"):
            try:
                st, data = self._once(0, 1023, mode)
                if st in (200, 206) and len(data) >= 1024:
                    self.mode = mode
                    return mode
                why.append(f"{mode}: HTTP {st}・{len(data)} バイト")
            except Exception as e:
                why.append(f"{mode}: {str(e)[:120]}")
        raise RuntimeError("範囲の取得を拒否されました（" + " / ".join(why) + "）")


def fetch_section(getter: "Getter", start: float, end: float, out: Path, threads: int = 8,
                  margin: float = 1.0) -> tuple[float, float]:
    """[start, end] 秒を含む断片だけを取って out（mp4）に書く。返り値 (取った区間の開始秒, 終了秒)。"""
    n = 512 * 1024
    while True:
        head = getter.get(0, n - 1, exact=False)
        try:
            p = plan(head)
            break
        except EOFError as e:
            if n >= 16 * 1024 * 1024:
                raise
            n = max(n * 2, int(e.args[0]) + 1024)
    segs = [s for s in p["segments"] if s[3] > start - margin and s[2] < end]
    if not segs:
        raise ValueError("指定の時間帯が動画に含まれていません")
    a, b = segs[0][0], segs[-1][0] + segs[-1][1] - 1
    init = head[:p["init_end"]] if p["init_end"] <= len(head) else getter.get(0, p["init_end"] - 1)
    tmp = out.with_suffix(".frag.mp4")
    with open(tmp, "wb") as fh:
        fh.write(init)
        fh.truncate(len(init) + (b - a + 1))
    ranges = [(x, min(b, x + CHUNK - 1)) for x in range(a, b + 1, CHUNK)]

    def one(rg):
        data = getter.get(rg[0], rg[1])
        with open(tmp, "r+b") as fh:
            fh.seek(len(init) + rg[0] - a)
            fh.write(data)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, ranges))
    # 断片の時刻は元の動画の先頭からなので、0 始まりの普通の mp4 に直す（ffmpeg の -ss が速く効くように）
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c", "copy", "-avoid_negative_ts", "make_zero",
                        "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"つなぎ直しに失敗：{r.stderr.strip()[-300:]}")
    return segs[0][2], segs[-1][3]


def parse_m3u8(text: str, base: str) -> dict:
    """HLS のメディアプレイリストから、初期化部分（EXT-X-MAP）と各断片の (URL, 開始秒, 終了秒)。"""
    from urllib.parse import urljoin
    segs, t, dur, init = [], 0.0, None, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MAP"):
            m = re.search(r'URI="([^"]+)"', line)
            if m:
                init = urljoin(base, m.group(1))
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#"):
            d = dur if dur is not None else 0.0
            segs.append((urljoin(base, line), t, t + d))
            t += d
            dur = None
    if not segs:
        raise ValueError("プレイリストに断片がありません（マスタープレイリストかもしれません）")
    return {"init": init, "segments": segs}


def fetch_hls(getter: "Getter", playlist_url: str, start: float, end: float, out: Path, threads: int = 16,
              margin: float = 1.0) -> tuple[float, float]:
    """HLS（m3u8）の、[start, end] を含む断片だけを並列に取って 1 本の mp4 にする。返り値 (取った区間の開始秒, 終了秒)。
    YouTube の tv・web_safari などのクライアントは、PO トークンの要らない HLS を返す（DASH の URL は 403 になる）。"""
    text = getter.fetch(playlist_url).decode("utf-8", "replace")
    p = parse_m3u8(text, playlist_url)
    segs = [s for s in p["segments"] if s[2] > start - margin and s[1] < end]
    if not segs:
        raise ValueError("指定の時間帯が動画に含まれていません")
    datas: list[bytes | None] = [None] * len(segs)

    def one(i):
        datas[i] = getter.fetch(segs[i][0])

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, range(len(segs))))
    ext = ".mp4" if p["init"] else ".ts"
    tmp = out.with_suffix(".frag" + ext)
    with open(tmp, "wb") as fh:
        if p["init"]:
            fh.write(getter.fetch(p["init"]))
        for d in datas:
            fh.write(d)
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-map", "0:v:0", "-c", "copy", "-an",
                        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(out)],
                       capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"つなぎ直しに失敗：{r.stderr.strip()[-300:]}")
    return segs[0][1], segs[-1][2]


def ydl_options(fmt: str, cookies: str = "", client: str | None = None, deno: str | None = None) -> dict:
    """セクション取得（yt-dlp のコマンド）と同じ設定を、プロセス内の yt-dlp 用に作る。"""
    import yt_dlp
    args = ["--no-warnings", "--remote-components", "ejs:github", "-f", fmt]
    if cookies and Path(cookies).exists():
        args += ["--cookies", cookies]
    if client:
        args += ["--extractor-args", f"youtube:player_client={client}"]
    if deno:
        args += ["--js-runtimes", f"deno:{deno}"]
    return yt_dlp.parse_options(args).ydl_opts


def format_info(video_id: str, fmt: str, cookies: str = "", client: str | None = None, env=None) -> dict:
    """プロセス内の yt-dlp で、選んだ形式の直接 URL・ヘッダー・設定を得る（ダウンロードはしない）。"""
    import shutil
    import yt_dlp
    deno = shutil.which("deno", path=(env or {}).get("PATH")) if env else shutil.which("deno")
    opts = ydl_options(fmt, cookies, client, deno)
    class _Quiet:                                    # エラーは例外で受け取るので、画面には出さない
        def debug(self, msg):
            pass

        warning = info = error = debug

    with yt_dlp.YoutubeDL({**opts, "quiet": True, "no_warnings": True, "ignoreerrors": False,
                           "logger": _Quiet()}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    if not info:
        raise RuntimeError("形式の情報を取れませんでした")
    f = info
    if "requested_formats" in info:
        f = next((x for x in info["requested_formats"] if x.get("vcodec") not in (None, "none")), info)
    proto = str(f.get("protocol") or "")
    if not f.get("url") or not (proto in ("https", "http") or proto.startswith("m3u8")):
        raise RuntimeError(f"範囲取得に向かない形式です（{proto}）")
    return {"url": f["url"], "headers": f.get("http_headers") or {}, "format_id": f.get("format_id"),
            "height": f.get("height"), "ext": f.get("ext"), "opts": opts,
            "kind": "hls" if proto.startswith("m3u8") else "dash"}
