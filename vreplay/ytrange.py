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

import json
import struct
import subprocess
import sys
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


def _get(sess, url: str, a: int, b: int, headers: dict, tries: int = 4, exact: bool = True) -> bytes:
    last = None
    for _ in range(tries):
        try:
            r = sess.get(url, headers={**headers, "Range": f"bytes={a}-{b}"}, timeout=60)
            if r.status_code in (200, 206) and (len(r.content) == b - a + 1 or (not exact and r.content)):
                return r.content
            last = f"HTTP {r.status_code}・{len(r.content)} バイト"
        except requests.RequestException as e:
            last = str(e)
    raise RuntimeError(f"範囲 {a}-{b} を取れませんでした（{last}）")


def fetch_section(url: str, headers: dict, start: float, end: float, out: Path, threads: int = 8,
                  margin: float = 1.0, cookies=None) -> tuple[float, float]:
    """[start, end] 秒を含む断片だけを取って out（mp4）に書く。返り値 (取った区間の開始秒, 終了秒)。"""
    sess = requests.Session()
    if cookies is not None:
        sess.cookies = cookies
    n = 512 * 1024
    while True:
        head = _get(sess, url, 0, n - 1, headers, exact=False)
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
    init = head[:p["init_end"]] if p["init_end"] <= len(head) else _get(sess, url, 0, p["init_end"] - 1, headers)
    tmp = out.with_suffix(".frag.mp4")
    with open(tmp, "wb") as fh:
        fh.write(init)
        fh.truncate(len(init) + (b - a + 1))
    ranges = [(x, min(b, x + CHUNK - 1)) for x in range(a, b + 1, CHUNK)]

    def one(rg):
        data = _get(sess, url, rg[0], rg[1], headers)
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


def format_info(video_id: str, fmt: str, cookies: str = "", client: str | None = None, env=None) -> dict:
    """yt-dlp で、選んだ形式の直接 URL と、取得に必要なヘッダーを得る（ダウンロードはしない）。"""
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "-j", "--remote-components", "ejs:github", "-f", fmt,
           f"https://www.youtube.com/watch?v={video_id}"]
    if cookies and Path(cookies).exists():
        cmd[3:3] = ["--cookies", cookies]
    if client:
        cmd[3:3] = ["--extractor-args", f"youtube:player_client={client}"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-500:])
    info = json.loads(r.stdout.splitlines()[-1])
    f = info
    if "requested_formats" in info:
        f = next((x for x in info["requested_formats"] if x.get("vcodec") not in (None, "none")), info)
    if f.get("protocol") not in ("https", "http") or not f.get("url"):
        raise RuntimeError(f"範囲取得に向かない形式です（{f.get('protocol')}）")
    return {"url": f["url"], "headers": f.get("http_headers") or {}, "format_id": f.get("format_id"),
            "height": f.get("height"), "ext": f.get("ext")}
