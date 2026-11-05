"""LLM の要約：見出しごとの取り出し・途中で切れたときの言い直し・片側だけの切り戻し。"""
from vreplay import llm

GOOD = """## ATK summary
- デフォで MID を取り、2 巡目のアクションで最終形に持っていく（Mid Vent テイクデフォ 8 回・勝率 63%）。
- B は捨てスプリットが通る（B Link 捨て A スプリット 4 回・3 勝）。
## DEF summary
- 守りデフォ（2-1-2）から時間で MID を減らす。
- B ファストリテイクが強い（11/15）。
## 対策メモ
- ドライに対して→スタンや壁でカウンターする。
- セットに対して→分断した後の寄りの速さで勝つ。
"""
CUT = """## ATK summary
- デフォで MID を取り、2 巡目のアクションで最終形に持っていく。
- 遅めのセット 8 回・勝率 100%、設置後勝率 14/18・"""


def test_sections_and_completeness():
    sec = llm.sections(GOOD)
    assert len(sec["ATK"]) == 2 and len(sec["DEF"]) == 2 and len(sec["memo"]) == 2
    assert sec["ATK"][0].startswith("デフォで MID")
    assert llm.complete(GOOD) and not llm.looks_truncated(GOOD)
    # 途中で切れている：最後の行は捨て、DEF が無いので「完成していない」
    assert llm.looks_truncated(CUT) and not llm.complete(CUT)
    assert llm.sections(CUT)["ATK"] == ["デフォで MID を取り、2 巡目のアクションで最終形に持っていく。"]
    assert llm.sections(CUT)["DEF"] == []


def test_summarize_retries_once_when_cut_off(tmp_path, monkeypatch):
    calls = []

    def fake_call(name, key, base, model, body, max_tokens, log, retries=1, reasoning=None, note=None):
        calls.append({"note": note, "max_tokens": max_tokens})
        return CUT if note is None else GOOD

    monkeypatch.setattr(llm, "_chain", lambda cfg: [("gemini", "k", "http://x", "gemini-3.8-flash")])
    monkeypatch.setattr(llm, "_call", fake_call)
    cfg = {"output": {"dir": str(tmp_path)}}
    out = llm.summarize(cfg, {"a": 1}, log=lambda *a: None)
    assert out == GOOD.strip() and len(calls) == 2 and calls[1]["note"]
    assert calls[0]["max_tokens"] >= 8192                      # Gemini は考える分も上限に含まれる
    # 2 回目も書き切れなければ、切れたままでも返す（片側だけでも使えるように）
    calls.clear()
    monkeypatch.setattr(llm, "_call", lambda *a, **k: CUT)
    out2 = llm.summarize(cfg, {"a": 2}, log=lambda *a: None)
    assert out2 == CUT.strip()


def test_summary_lines_fall_back_per_side():
    """片方の見出ししか書けていないときは、その側だけ従来の箇条書きに戻す。"""
    from vreplay import notion_playbook as NP

    strong = [{"title": "設置後のリテイクが強い", "lines": []}]
    weak = [{"title": "ピストルを落としやすい", "lines": []}]
    atk = NP._summary_lines(CUT, "ATK", strong, weak)
    # 書けていた 1 行は残し、足りない分だけ従来の箇条書きで補う
    assert "最終形" in str(atk[0]) and any("強み" in str(b) for b in atk)
    dfd = NP._summary_lines(CUT, "DEF", strong, weak)
    assert any("強み" in str(b) for b in dfd) and any("弱み" in str(b) for b in dfd)
    both = NP._summary_lines(GOOD, "DEF", strong, weak)
    assert len(both) == 2 and "ファストリテイク" in str(both[1])
