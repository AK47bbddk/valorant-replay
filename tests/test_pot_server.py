from vreplay import video


def test_ensure_pot_server_without_colab_does_not_run_subprocess(monkeypatch):
    """ping が失敗し、Colab（/content）でもなければサーバーを起動しようとしない。"""

    def no_ping(*a, **k):
        raise OSError("接続できません")

    calls = []
    monkeypatch.setattr(video.urllib.request, "urlopen", no_ping)
    monkeypatch.setattr(video.subprocess, "run", lambda *a, **k: calls.append(a))
    real_exists = video.Path.exists

    def fake_exists(self):
        return False if str(self).replace("\\", "/").rstrip("/").endswith("/content") else real_exists(self)
    monkeypatch.setattr(video.Path, "exists", fake_exists)
    assert video.ensure_pot_server(log=lambda *a: None) == ""
    assert calls == []


def test_ensure_pot_server_reports_up_when_running(monkeypatch):
    """動いていれば "up"（取り直しの合図の "restarted" ではない）。"""

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(video.urllib.request, "urlopen", lambda *a, **k: Resp())
    assert video.ensure_pot_server(log=lambda *a: None) == "up"
