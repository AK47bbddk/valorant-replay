"""処理時間とマシンのリソース（CPU・GPU・メモリ）を測って、ログ・HTML・JSON に出す。

Colab では nvidia-smi から GPU 使用率・VRAM を、psutil から CPU・RAM を 1 秒ごとに読む。
psutil や nvidia-smi が無くても動く（その項目だけ表示しない）。
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:  # pragma: no cover - psutil が無い環境でも動かす
    psutil = None

_LOCK = threading.RLock()
_SAMPLES: list[dict] = []
_ENTRIES: list[dict] = []
_RUNNING = False
_THREAD: threading.Thread | None = None
_STARTED_AT = 0.0
_GPU_STATE: bool | None = None          # None=未確認 / False=なし・読めない / True=あり
_GPU_NAME: str | None = None
_PROC = None
_DEC_FIELD: bool | None = None          # nvidia-smi で utilization.decoder（NVDEC の使用率）を読めるか
_CPU_SEEN: dict = {}                    # (種類, pid/tid, 作成時刻) → これまでに使った CPU 秒

STAGE_JA = {
    "matches": "試合の選択",
    "download": "動画の取得",
    "calibrate": "キャリブレーション",
    "scan": "HUD スキャン",
    "tactics": "サイト・位置の解析",
    "assets": "ラウンド素材",
    "notion": "Notion 書き込み",
    "html": "HTML レポート",
}
STAGE_ORDER = list(STAGE_JA)


# ---------------------------------------------------------------- サンプリング
def reset():
    global _SAMPLES, _ENTRIES, _STARTED_AT
    with _LOCK:
        _SAMPLES = []
        _ENTRIES = []
        _STARTED_AT = time.time()
        _CPU_SEEN.clear()


def _gpu_name() -> str:
    global _GPU_NAME
    if _GPU_NAME is not None:
        return _GPU_NAME
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=3)
        _GPU_NAME = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:
        _GPU_NAME = ""
    return _GPU_NAME


def _gpu_snapshot() -> dict:
    global _GPU_STATE
    if _GPU_STATE is False:
        return {}
    if _GPU_STATE is None and not shutil.which("nvidia-smi"):
        _GPU_STATE = False
        return {}
    global _DEC_FIELD
    try:
        fields = "utilization.gpu,memory.used,memory.total,temperature.gpu"
        r = None
        if _DEC_FIELD is not False:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={fields},utilization.decoder",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            if r.returncode != 0 or not r.stdout.strip():
                _DEC_FIELD, r = False, None              # 古いドライバーは decoder を読めない
            else:
                _DEC_FIELD = True
        if r is None:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
        if r.returncode != 0 or not r.stdout.strip():
            _GPU_STATE = False
            return {}
        vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]

        def num(i):
            try:
                return float(vals[i])
            except (IndexError, ValueError):
                return None

        _GPU_STATE = True
        out = {"gpu_util": num(0), "gpu_mem_used_mb": num(1), "gpu_mem_total_mb": num(2), "gpu_temp_c": num(3)}
        if _DEC_FIELD:
            out["dec_util"] = num(4)
        return out
    except Exception:
        _GPU_STATE = False
        return {}


# ---------------------------------------------------------------- CPU の使い道
def _thread_kind(name: str) -> str:
    if name.startswith("取得"):
        return "動画の取得"
    if name.startswith("素材"):
        return "ラウンド素材（装備表・名前・キルフィード・GIF）"
    if name.startswith("Notion"):
        return "Notion への書き込み"
    if name.startswith("vreplay-perf"):
        return "測定"
    return "ほかの解析（キャリブレーション・ミニマップ・ラウンド分割）"


def _proc_kind(p, main_pid: int) -> str:
    name = (p.name() or "").lower()
    if "ffmpeg" in name or "ffprobe" in name:
        return "ffmpeg（メインから：装備表・1 枚取り・確認）" if p.ppid() == main_pid else "ffmpeg（スキャンのデコード）"
    if "deno" in name or "aria2" in name:
        return "動画の取得"
    try:
        cmd = " ".join(p.cmdline())
    except Exception:
        cmd = ""
    if "yt_dlp" in cmd or "yt-dlp" in cmd:
        return "動画の取得"
    if "multiprocessing" in cmd or "python" in name:
        return "スキャンの作業（切り抜き・カード・OCR の前後処理）"
    return "その他"


def _cpu_tree():
    """このプロセス（スレッドごと）と子プロセスが使った CPU 秒を、種類ごとに覚える。"""
    if not psutil or _PROC is None:
        return
    try:
        names = {t.native_id: t.name for t in threading.enumerate()}
        for th in _PROC.threads():
            nm = names.get(th.id)
            if nm is None:
                continue
            _CPU_SEEN[(_thread_kind(nm), "t", th.id, nm)] = th.user_time + th.system_time
        for p in _PROC.children(recursive=True):
            try:
                with p.oneshot():
                    ct = p.cpu_times()
                    kind = _proc_kind(p, _PROC.pid)
                    _CPU_SEEN[(kind, "p", p.pid, p.create_time())] = ct.user + ct.system
            except Exception:
                continue
    except Exception:
        pass


def cpu_breakdown() -> list[tuple[str, float]]:
    """[(種類, CPU 秒)]（多い順）。終わったプロセス・スレッドは最後に見た値（最大 3 秒ぶん少なめ）。"""
    _cpu_tree()
    tot: dict[str, float] = {}
    for (kind, *_), sec in list(_CPU_SEEN.items()):
        tot[kind] = tot.get(kind, 0.0) + sec
    return sorted(((k, v) for k, v in tot.items() if v >= 0.5), key=lambda kv: -kv[1])


def sample() -> dict:
    """今この瞬間の CPU・RAM・GPU・空きディスク。"""
    global _PROC
    s: dict = {"t": time.time()}
    if psutil:
        try:
            s["cpu"] = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            s["ram_used_mb"] = vm.used / 1e6
            s["ram_total_mb"] = vm.total / 1e6
        except Exception:
            pass
        try:
            if _PROC is None:
                _PROC = psutil.Process()
            s["proc_cpu"] = float(_PROC.cpu_percent(interval=None))
        except Exception:
            pass
    else:
        try:
            s["cpu"] = min(100.0, os.getloadavg()[0] / (os.cpu_count() or 1) * 100)
        except Exception:
            pass
    try:
        s["disk_free_gb"] = shutil.disk_usage(os.getcwd()).free / 1e9
    except Exception:
        pass
    s.update(_gpu_snapshot())
    return s


def start(interval: float = 1.0):
    """1 秒ごとのサンプリングを裏で始める（二重起動はしない）。"""
    global _RUNNING, _THREAD, _PROC
    if _RUNNING:
        return
    reset()
    if psutil:
        try:
            psutil.cpu_percent(interval=None)
            if _PROC is None:
                _PROC = psutil.Process()
            _PROC.cpu_percent(interval=None)
        except Exception:
            pass

    def loop():
        k = 0
        while _RUNNING:
            s = sample()
            with _LOCK:
                _SAMPLES.append(s)
            if k % 3 == 0:
                _cpu_tree()
            k += 1
            time.sleep(interval)

    _RUNNING = True
    _THREAD = threading.Thread(target=loop, name="vreplay-perf", daemon=True)
    _THREAD.start()


def stop():
    global _RUNNING, _THREAD
    _RUNNING = False
    t, _THREAD = _THREAD, None
    if t and t.is_alive():
        t.join(timeout=2.5)


def elapsed() -> float:
    return time.time() - _STARTED_AT if _STARTED_AT else 0.0


def _take_sample():
    s = sample()
    with _LOCK:
        _SAMPLES.append(s)


# ---------------------------------------------------------------- 集計
def _nums(samples: list[dict], key: str) -> list[float]:
    return [float(s[key]) for s in samples if s.get(key) is not None]


def summarize_samples(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    out: dict = {"samples": len(samples)}
    cpu, pcpu = _nums(samples, "cpu"), _nums(samples, "proc_cpu")
    ram, ram_total = _nums(samples, "ram_used_mb"), _nums(samples, "ram_total_mb")
    gpu, vram, vram_total = (_nums(samples, "gpu_util"), _nums(samples, "gpu_mem_used_mb"),
                             _nums(samples, "gpu_mem_total_mb"))
    temp, disk = _nums(samples, "gpu_temp_c"), _nums(samples, "disk_free_gb")
    if cpu:
        out["cpu_avg"] = round(statistics.fmean(cpu), 1)
        out["cpu_max"] = round(max(cpu), 1)
    if pcpu:
        out["proc_cpu_avg"] = round(statistics.fmean(pcpu), 1)
        out["proc_cpu_max"] = round(max(pcpu), 1)
    if ram:
        out["ram_peak_gb"] = round(max(ram) / 1024, 2)
        out["ram_avg_gb"] = round(statistics.fmean(ram) / 1024, 2)
    if ram_total:
        out["ram_total_gb"] = round(max(ram_total) / 1024, 2)
    if gpu:
        out["gpu_avg"] = round(statistics.fmean(gpu), 1)
        out["gpu_max"] = round(max(gpu), 1)
    dec = _nums(samples, "dec_util")
    if dec:
        out["dec_avg"] = round(statistics.fmean(dec), 1)
        out["dec_max"] = round(max(dec), 1)
    if vram:
        out["vram_peak_gb"] = round(max(vram) / 1024, 2)
    if vram_total:
        out["vram_total_gb"] = round(max(vram_total) / 1024, 2)
    if temp:
        out["gpu_temp_max_c"] = round(max(temp), 1)
    if disk:
        out["disk_free_gb"] = round(min(disk), 1)
    return out


def _merge_resources(entries: list[dict]) -> dict:
    """複数ステージのリソースを、秒数で重みづけしてまとめる。"""
    res = [e.get("resources") for e in entries if e.get("resources")]
    if not res:
        return {}

    def wavg(key: str):
        pairs = [(e["resources"][key], e["seconds"]) for e in entries
                 if e.get("resources") and e["resources"].get(key) is not None]
        if not pairs:
            return None
        total = sum(w for _, w in pairs) or 1.0
        return sum(v * w for v, w in pairs) / total

    def peak(key: str):
        vals = [r[key] for r in res if r.get(key) is not None]
        return max(vals) if vals else None

    out: dict = {}
    for key in ("cpu_avg", "proc_cpu_avg", "gpu_avg", "dec_avg"):
        v = wavg(key)
        if v is not None:
            out[key] = round(v, 1)
    for key in ("cpu_max", "proc_cpu_max", "gpu_max", "dec_max", "gpu_temp_max_c", "ram_peak_gb",
                "vram_peak_gb", "ram_total_gb", "vram_total_gb"):
        v = peak(key)
        if v is not None:
            out[key] = v
    disk = [r["disk_free_gb"] for r in res if r.get("disk_free_gb") is not None]
    if disk:
        out["disk_free_gb"] = round(min(disk), 1)
    return out


@contextmanager
def stage(name: str, tag: str = "", label: str = ""):
    """`with perf.stage("download", tag=..., label=...):` で 1 区間を測る。

    yield される dict に bytes などを足すと、そのまま記録に残る。
    """
    t0 = time.time()
    _take_sample()
    entry = {"stage": name, "tag": tag, "label": label or STAGE_JA.get(name, name), "start": t0}
    try:
        yield entry
    finally:
        t1 = time.time()
        _take_sample()
        with _LOCK:
            window = [s for s in _SAMPLES if t0 <= s.get("t", 0) <= t1]
        entry["seconds"] = max(0.0, t1 - t0)
        entry["resources"] = summarize_samples(window)
        with _LOCK:
            _ENTRIES.append(entry)


def entries() -> list[dict]:
    with _LOCK:
        return list(_ENTRIES)


def for_tag(tag: str) -> list[dict]:
    """1 マップぶんの記録（動画取得 → 解析 → 素材）。Notion 書き込みは含めない。"""
    return [{"stage": e["stage"], "label": e.get("label") or STAGE_JA.get(e["stage"], e["stage"]),
             "seconds": round(e.get("seconds", 0.0), 2), "resources": e.get("resources")}
            for e in entries() if e.get("tag") == tag]


def build_summary(entries_: list[dict] | None = None, total_sec: float | None = None) -> dict:
    items = list(entries() if entries_ is None else entries_)
    names = STAGE_ORDER + sorted({e["stage"] for e in items} - set(STAGE_ORDER))
    stages = []
    for name in names:
        group = [e for e in items if e["stage"] == name]
        if not group:
            continue
        row = _merge_resources(group)
        row.update(stage=name, label=STAGE_JA.get(name, name), count=len(group),
                   seconds=round(sum(e.get("seconds", 0.0) for e in group), 2))
        stages.append(row)
    total = float(total_sec if total_sec is not None else elapsed())
    for row in stages:
        row["share"] = round(row["seconds"] / total, 3) if total > 0 else 0.0
    system = {"cpu_count": os.cpu_count(), "python": ".".join(map(str, sys.version_info[:3]))}
    if psutil:
        try:
            system["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:
            pass
    name = _gpu_name()
    if name:
        system["gpu_name"] = name
    return {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sec": round(total, 2), "system": system, "stages": stages,
            "cpu_use": [{"kind": k, "cpu_sec": round(v, 1)} for k, v in cpu_breakdown()],
            "advice": advice(stages, total), "entries": items}


def advice(stages: list[dict], total: float) -> list[str]:
    by = {s["stage"]: s for s in stages}
    dl, scan = by.get("download"), by.get("scan")
    out = []
    if dl and total > 0 and dl["seconds"] / total >= 0.25:
        out.append(f"動画の取得が全体の {dl['seconds'] / total:.0%} を占めています。"
                   "YouTube は 1 本の接続を 1 MB/s 前後に絞るので、config の video.download_parts（既定 16）を増やす、"
                   "または自宅 PC の `vreplay fetch` で先に落とすと短縮できます（取得は解析と並行して進みます）。")
    if scan and scan.get("cpu_avg") is not None and scan["seconds"] >= 30 and scan["cpu_avg"] < 50:
        out.append(f"スキャン中の CPU は平均 {scan['cpu_avg']:.0f}% で余っています。"
                   "video.fps を下げるか、video.workers を調整すると短縮できます。")
    if scan and scan.get("gpu_avg") is not None and scan["gpu_avg"] < 20:
        out.append("GPU の負荷は低めです。OCR も GPU で動かすには onnxruntime-gpu を入れてください"
                   "（ノートブックのセル1で T4 のときは自動で入ります）。")
    return out


def fmt_seconds(sec: float | None) -> str:
    if sec is None:
        return "-"
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f"{sec} 秒"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m} 分 {s:02d} 秒"
    h, m = divmod(m, 60)
    return f"{h} 時間 {m:02d} 分"


def fmt_resources(r: dict | None) -> str:
    if not r:
        return ""
    bits = []
    if r.get("cpu_avg") is not None:
        bits.append(f"CPU 平均 {r['cpu_avg']:.0f}%（最大 {r.get('cpu_max', 0):.0f}%）")
    if r.get("gpu_avg") is not None:
        s = f"GPU 平均 {r['gpu_avg']:.0f}%（最大 {r.get('gpu_max', 0):.0f}%）"
        if r.get("dec_avg") is not None:
            s += f"・デコーダ 平均 {r['dec_avg']:.0f}%"
        if r.get("vram_peak_gb") is not None:
            s += f"・VRAM 最大 {r['vram_peak_gb']:.1f}GB"
        bits.append(s)
    elif _GPU_STATE is False:
        bits.append("GPU なし")
    if r.get("ram_peak_gb") is not None:
        total = f"/{r['ram_total_gb']:.1f}GB" if r.get("ram_total_gb") else ""
        bits.append(f"RAM 最大 {r['ram_peak_gb']:.1f}{total}")
    return "｜".join(bits)


def note(entries_: list[dict], total_sec: float | None = None) -> str:
    """Notion / HTML に 1 行で出す「処理時間とリソース」。"""
    rows = [e for e in entries_ if e["stage"] in ("download", "calibrate", "scan", "tactics", "assets")]
    if not rows:
        return ""
    parts = [f"{STAGE_JA.get(e['stage'], e['stage'])} {fmt_seconds(e.get('seconds'))}" for e in rows]
    r = _merge_resources(rows)
    tail = []
    if r.get("cpu_max") is not None:
        tail.append(f"CPU 最大 {r['cpu_max']:.0f}%")
    if r.get("gpu_max") is not None:
        tail.append(f"GPU 最大 {r['gpu_max']:.0f}%")
    if r.get("ram_peak_gb") is not None:
        tail.append(f"RAM 最大 {r['ram_peak_gb']:.1f}GB")
    if total_sec:
        parts.append(f"合計 {fmt_seconds(total_sec)}")
    return "処理時間：" + "／".join(parts) + (f"（{'・'.join(tail)}）" if tail else "")


def print_summary(summary: dict, log=print):
    log("")
    log(f"━━ 処理時間とリソース（合計 {fmt_seconds(summary.get('total_sec'))}）")
    for s in summary.get("stages") or []:
        share = f"{s.get('share', 0):.0%}"
        res = fmt_resources(s)
        log(f"  {s['label']}：{fmt_seconds(s['seconds'])}（{share}）" + (f"｜{res}" if res else ""))
    if any(s["stage"] == "download" and s.get("count", 1) > 1 for s in summary.get("stages") or []):
        log("  ※ 動画の取得は複数マップを同時に進めるため、時間は合計です（実時間より長くなることがあります）")
    system = summary.get("system") or {}
    line = f"  CPU {system.get('cpu_count', '?')} コア"
    if system.get("ram_total_gb"):
        line += f"／RAM {system['ram_total_gb']}GB"
    if system.get("gpu_name"):
        line += f"／GPU {system['gpu_name']}"
    log(line)
    use = summary.get("cpu_use") or []
    if use:
        tot = sum(u["cpu_sec"] for u in use)
        cap = (system.get("cpu_count") or 1) * max(float(summary.get("total_sec") or 0), 1.0)
        log(f"  CPU の使い道（コア時間の合計 {fmt_seconds(tot)}・使える量の {tot / cap:.0%}）：")
        for u in use:
            log(f"    {u['kind']}：{fmt_seconds(u['cpu_sec'])}（{u['cpu_sec'] / tot:.0%}）")
    for hint in summary.get("advice") or []:
        log(f"  ヒント: {hint}")


def write_json(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
