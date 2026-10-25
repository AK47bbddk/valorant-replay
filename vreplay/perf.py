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
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
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
        return {"gpu_util": num(0), "gpu_mem_used_mb": num(1),
                "gpu_mem_total_mb": num(2), "gpu_temp_c": num(3)}
    except Exception:
        _GPU_STATE = False
        return {}


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
        while _RUNNING:
            s = sample()
            with _LOCK:
                _SAMPLES.append(s)
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
    for key in ("cpu_avg", "proc_cpu_avg", "gpu_avg"):
        v = wavg(key)
        if v is not None:
            out[key] = round(v, 1)
    for key in ("cpu_max", "proc_cpu_max", "gpu_max", "gpu_temp_max_c", "ram_peak_gb",
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
    for hint in summary.get("advice") or []:
        log(f"  ヒント: {hint}")


def write_json(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path
