import time

from vreplay import perf


def test_summarize_and_format():
    samples = [
        {"t": 1, "cpu": 20, "ram_used_mb": 2048, "ram_total_mb": 8192,
         "gpu_util": 5, "gpu_mem_used_mb": 512, "gpu_mem_total_mb": 15360},
        {"t": 2, "cpu": 80, "ram_used_mb": 4096, "ram_total_mb": 8192,
         "gpu_util": 25, "gpu_mem_used_mb": 1024, "gpu_mem_total_mb": 15360},
    ]
    r = perf.summarize_samples(samples)
    assert r["cpu_avg"] == 50 and r["cpu_max"] == 80
    assert r["gpu_avg"] == 15 and r["vram_peak_gb"] == 1.0
    assert r["ram_peak_gb"] == 4.0
    text = perf.fmt_resources(r)
    assert "CPU 平均 50%" in text and "GPU 平均 15%" in text and "RAM 最大 4.0" in text


def test_build_summary_shares_and_advice(monkeypatch):
    monkeypatch.setattr(perf, "_gpu_name", lambda: "")
    entries = [
        {"stage": "download", "label": "動画の取得", "seconds": 60,
         "resources": {"cpu_avg": 10, "cpu_max": 20, "ram_peak_gb": 2.0}},
        {"stage": "scan", "label": "HUD スキャン", "seconds": 180,
         "resources": {"cpu_avg": 70, "cpu_max": 95, "ram_peak_gb": 3.0}},
    ]
    s = perf.build_summary(entries, total_sec=200)
    assert [x["stage"] for x in s["stages"]] == ["download", "scan"]
    assert s["stages"][0]["share"] == 0.3
    assert s["stages"][1]["cpu_max"] == 95
    assert any("動画の取得" in a for a in s["advice"])


def test_stage_records_and_note(monkeypatch):
    monkeypatch.setattr(perf, "_gpu_name", lambda: "")
    perf.start(interval=0.05)
    try:
        with perf.stage("download", tag="1:1", label="動画の取得") as st:
            time.sleep(0.02)
            st["bytes"] = 123
    finally:
        perf.stop()
    rows = perf.for_tag("1:1")
    assert rows and rows[0]["stage"] == "download" and rows[0]["seconds"] >= 0
    text = perf.note(rows)
    assert text.startswith("処理時間：動画の取得")
    assert "秒" in text
