import json
import sys
from datetime import datetime
from email.header import decode_header, make_header

import pytest

from vreplay import notify


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b""


def _capture(monkeypatch, exc=None):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req, timeout))
        if exc is not None:
            raise exc
        return _Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    return calls


def _cfg(url):
    return {"notify": {"url": url}}


def test_send_ntfy(monkeypatch):
    monkeypatch.delenv("VREPLAY_NOTIFY_URL", raising=False)
    calls = _capture(monkeypatch)
    assert notify.send(_cfg("https://ntfy.sh/mytopic"), "vreplay batch 完了", "本文", level="error") is True
    req, timeout = calls[0]
    assert req.full_url == "https://ntfy.sh/mytopic"
    assert req.get_method() == "POST"
    assert req.data == "本文".encode("utf-8")
    assert timeout == 10
    title = str(make_header(decode_header(req.get_header("Title"))))
    assert title == "vreplay batch 完了"
    assert req.get_header("Priority") == "4"

    assert notify.send(_cfg("https://ntfy.sh/mytopic"), "done", "body") is True
    assert calls[1][0].get_header("Priority") == "3"


def test_long_japanese_title_has_no_newline(monkeypatch):
    monkeypatch.delenv("VREPLAY_NOTIFY_URL", raising=False)
    calls = _capture(monkeypatch)
    title = "vreplay batch 完了 " + "あ" * 300
    assert notify.send(_cfg("https://ntfy.sh/mytopic"), title, "本文") is True
    header = calls[0][0].get_header("Title")
    assert "\n" not in header
    assert "\r" not in header
    assert str(make_header(decode_header(header))) == title


def test_send_discord(monkeypatch):
    monkeypatch.delenv("VREPLAY_NOTIFY_URL", raising=False)
    calls = _capture(monkeypatch)
    url = "https://discord.com/api/webhooks/123/token"
    assert notify.send(_cfg(url), "完了", "本文") is True
    req, timeout = calls[0]
    assert req.full_url == url
    assert req.get_method() == "POST"
    assert timeout == 10
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data.decode("utf-8")) == {"content": "**完了**\n本文"}


def test_send_uses_env_url(monkeypatch):
    monkeypatch.setenv("VREPLAY_NOTIFY_URL", "https://ntfy.sh/env-topic")
    calls = _capture(monkeypatch)
    assert notify.send(_cfg(None), "t", "b") is True
    assert calls[0][0].full_url == "https://ntfy.sh/env-topic"


def test_send_without_url_is_false(monkeypatch):
    monkeypatch.delenv("VREPLAY_NOTIFY_URL", raising=False)

    def boom(*a, **k):
        raise AssertionError("URL が無いのに通信した")

    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    assert notify.send(_cfg(None), "t", "b") is False
    assert notify.send({}, "t", "b") is False


def test_send_network_error_is_false(monkeypatch):
    monkeypatch.delenv("VREPLAY_NOTIFY_URL", raising=False)
    _capture(monkeypatch, exc=OSError("通信できません"))
    assert notify.send(_cfg("https://ntfy.sh/mytopic"), "t", "b") is False


def test_no_notify_flag_does_not_send(monkeypatch, tmp_path):
    from vreplay import __main__ as M
    from vreplay import batch

    calls = []
    monkeypatch.setenv("VREPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(notify, "send", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(batch, "run", lambda *a, **k: {"batch": {"total": 0, "done": 0, "failed": 0}})
    monkeypatch.setattr(sys, "argv", ["vreplay", "batch", "--no-notify"])
    M.main()
    assert calls == []


def test_main_sends_completion_and_error(monkeypatch, tmp_path):
    from vreplay import __main__ as M
    from vreplay import batch

    calls = []
    monkeypatch.setenv("VREPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(notify, "send", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["vreplay", "batch"])
    monkeypatch.setattr(batch, "run", lambda *a, **k: {"batch": {"total": 2, "done": 1, "failed": 1}})
    M.main()
    assert len(calls) == 1
    assert calls[0][0][1] == "vreplay batch 完了"
    assert "1/2 マップ" in calls[0][0][2]

    calls.clear()

    def fail(*a, **k):
        raise RuntimeError("だめだった")

    monkeypatch.setattr(batch, "run", fail)
    with pytest.raises(RuntimeError):
        M.main()
    assert len(calls) == 1
    assert calls[0][0][1] == "vreplay batch 失敗"
    assert "RuntimeError: だめだった" in calls[0][0][2]
    assert calls[0][1]["level"] == "error"


def test_main_sends_error_on_nonzero_system_exit(monkeypatch, tmp_path):
    from vreplay import __main__ as M
    from vreplay import batch

    calls = []
    monkeypatch.setenv("VREPLAY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(notify, "send", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(sys, "argv", ["vreplay", "batch"])

    def exit1(*a, **k):
        raise SystemExit(1)

    monkeypatch.setattr(batch, "run", exit1)
    with pytest.raises(SystemExit):
        M.main()
    assert len(calls) == 1
    assert calls[0][0][1] == "vreplay batch 失敗"
    assert calls[0][1]["level"] == "error"

    calls.clear()

    def exit_none(*a, **k):
        raise SystemExit(None)

    monkeypatch.setattr(batch, "run", exit_none)
    with pytest.raises(SystemExit):
        M.main()
    assert calls == []


def _batch_low_res(monkeypatch, tmp_path):
    from vreplay import batch, config, db, vlr
    from vreplay.pipeline import LowResolution

    monkeypatch.setenv("VREPLAY_OUTPUT_DIR", str(tmp_path))
    cfg = config.load()
    cfg["output"]["dir"] = str(tmp_path)
    cfg["warehouse"]["path"] = str(tmp_path / "w.sqlite3")
    cfg["notion"]["enabled"] = False
    cfg["video"]["keep_video"] = False
    m = vlr.Match(id="1", url="https://www.vlr.gg/1/a-vs-b", team1="A", team2="B",
                  event="VCT 2026", stage="Group", dt_jst=datetime(2026, 1, 2, 12))
    for k, name in enumerate(["Ascent", "Bind", "Haven"], start=1):
        m.maps.append(vlr.MapData(order=k, name=name, game_id=str(k), score1=13, score2=10,
                                  rounds=[vlr.Round(1, 1, "攻撃", "elim")], vod_id=f"v{k}", vod_t=100,
                                  vod_url=f"https://youtu.be/v{k}?t=100"))
    conn = db.connect(db.warehouse_path(cfg))
    db.upsert_match(conn, m)
    conn.commit()
    conn.close()
    monkeypatch.setattr(vlr, "parse_match", lambda url: m)
    monkeypatch.setattr(batch, "ensure_pot_server", lambda log=None: "up")

    def low(*a, **k):
        raise LowResolution("360p の動画しか取れませんでした")

    monkeypatch.setattr(batch, "download", low)
    return cfg, batch


def test_batch_alerts_once_when_low_resolution_repeats(monkeypatch, tmp_path):
    cfg, batch = _batch_low_res(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(notify, "send", lambda *a, **k: calls.append((a, k)))
    batch.run(cfg, log=lambda *a: None)
    assert len(calls) == 1
    assert calls[0][1]["level"] == "error"
    assert "低画質" in calls[0][0][1]


def test_batch_no_notify_suppresses_mid_run_alert(monkeypatch, tmp_path):
    cfg, batch = _batch_low_res(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(notify, "send", lambda *a, **k: calls.append((a, k)))
    batch.run(cfg, log=lambda *a: None, notify_enabled=False)
    assert calls == []



def test_send_sets_user_agent(monkeypatch):
    """Discord は Python 既定の User-Agent を 403 で弾くので、必ず付ける。"""
    from vreplay import notify
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    def fake(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake)
    assert notify.send({"notify": {"url": "https://discord.com/api/webhooks/1/x"}}, "t", "b") is True
    assert seen["ua"] and "Python-urllib" not in seen["ua"]
