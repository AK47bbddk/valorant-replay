"""GIF とミニマップ（開幕配置図）だけを見て目視で位置・配置を読み取り、正確率を検証するブラインドテスト HTML を作る。

使い方:
    python scripts/blind_test.py <report.json または検索ディレクトリ> --out blind_test.html --max 20
    python scripts/blind_test.py --demo --out blind_test_demo.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import random
from pathlib import Path


def _resolve_media(raw_path: str | None, base_dir: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists():
        return p
    # Colab (/content/drive/...) 等で保存されたパスをローカルの report.json 相対で探す
    parts = p.as_posix().split("/")
    for i in range(len(parts)):
        cand = base_dir.joinpath(*parts[i:])
        if cand.exists():
            return cand
    cand_name = base_dir / p.name
    if cand_name.exists():
        return cand_name
    for sub in ("_tactics", "assets", "clips"):
        c2 = base_dir / sub / p.name
        if c2.exists():
            return c2
    return None


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        ext = path.suffix.lower()
        mime = "image/gif" if ext == ".gif" else "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def collect_items(target: Path, max_items: int = 20, seed: int = 42, map_filter: str = "") -> list[dict]:
    """report.json（単体またはディレクトリ配下）から GIF＋ミニマップが揃うラウンドを集める。"""
    reports = [target] if target.is_file() else sorted(target.rglob("report.json"))
    items: list[dict] = []
    mf = map_filter.strip().lower()
    for rp in reports:
        try:
            rep = json.loads(rp.read_text("utf-8"))
        except Exception:
            continue
        map_name = str(rep.get("map") or rep.get("map_name") or "?")
        if mf and mf not in map_name.lower():
            continue
        base = rp.parent
        tac_rounds = (rep.get("tactics") or {}).get("rounds") or {}
        team1 = ((rep.get("teams") or [{}])[0] or {}).get("name") or rep.get("team1") or "Team 1"
        team2 = ((rep.get("teams") or [{}, {}])[1] or {}).get("name") or rep.get("team2") or "Team 2"
        for r in rep.get("rounds") or []:
            n = r.get("n")
            rt = r.get("tactics") or tac_rounds.get(str(n)) or {}
            assets = r.get("assets") or {}
            gif_p = _resolve_media(assets.get("minimap_gif"), base)
            board_raw = (rt.get("boards") or {}).get("1:40") or assets.get("minimap_start")
            mm_p = _resolve_media(board_raw, base)
            if gif_p is None or mm_p is None:
                continue
            agents_140 = (rt.get("agents") or {}).get("1:40") or []
            if not agents_140:
                # 実reportには agents が無く、開幕位置は tactics.positions（[n, t, x, y, team, team, zone, callout]）にある
                pos = (rep.get("tactics") or {}).get("positions") or []
                rows = [p for p in pos if isinstance(p, list) and len(p) >= 8 and p[0] == n]
                if rows:
                    t0 = min(p[1] for p in rows)
                    agents_140 = [
                        {"agent": "位置", "team": p[4], "callout": p[7], "zone": p[6]}
                        for p in rows if p[1] == t0
                    ]
            setups = rt.get("setups") or {}
            site = rt.get("plant_site") or rt.get("site")
            if not site:
                # サイト正解が無いラウンドはサイト当ての採点ができないので出題しない
                continue
            items.append({
                "id": f"{base.name}_r{n}",
                "_gif_p": gif_p,
                "_mm_p": mm_p,
                "answer": {
                    "map": map_name,
                    "round": n,
                    "match": f"{team1} vs {team2}",
                    "side1": rt.get("side1") or r.get("side1") or "?",
                    "site": site,
                    "agents_140": [
                        {"agent": a.get("agent"), "team": a.get("team"), "callout": a.get("callout"), "zone": a.get("zone")}
                        for a in agents_140 if isinstance(a, dict)
                    ],
                    "setups": {
                        tm: (su.get("name") or su.get("shape") or "")
                        for tm, su in setups.items() if isinstance(su, dict)
                    },
                },
            })
    rng = random.Random(seed)
    rng.shuffle(items)
    selected = items[:max_items] if max_items > 0 else items
    # Base64埋め込みは選ばれた分だけ行う（全件先にエンコードするとDrive上の大量データでメモリを食うため）
    for it in selected:
        it["gif"] = _data_uri(it.pop("_gif_p"))
        it["minimap"] = _data_uri(it.pop("_mm_p"))
    for it in items:
        it.pop("_gif_p", None)
        it.pop("_mm_p", None)
    return selected


def demo_items() -> list[dict]:
    """動作確認用の最小デモデータ（手元に report.json が無い場合向け）。"""
    import cv2
    import numpy as np

    fx = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out = []
    for i, (fname, mname, site) in enumerate([
        ("minimap_sunset_riot.jpg", "Sunset (Demo)", "A"),
        ("minimap_split_riot.jpg", "Split (Demo)", "B"),
    ], 1):
        p = fx / fname
        if p.exists():
            uri = _data_uri(p)
        else:
            blank = np.zeros((240, 240, 3), np.uint8)
            cv2.putText(blank, f"Demo {i}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ok, buf = cv2.imencode(".jpg", blank)
            uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        out.append({
            "id": f"demo_q{i}",
            "gif": uri,
            "minimap": uri,
            "answer": {
                "map": mname,
                "round": i,
                "match": "Demo A vs Demo B",
                "side1": "攻撃" if i == 1 else "防衛",
                "site": site,
                "agents_140": [{"agent": "Jett", "team": 1, "callout": "Attacker Side Spawn", "zone": "攻撃側"}],
                "setups": {"1": "Demo Setup"},
            },
        })
    return out


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>位置確認ブラインドテスト</title>
<style>
  :root {
    --bg: #0f1419; --card: #182028; --line: #2a3542; --text: #e8eef4;
    --muted: #8fa0b3; --ok: #22c55e; --warn: #f59e0b; --bad: #ef4444; --acc: #38bdf8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 16px 20px 40px; }
  header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line); padding-bottom: 12px; margin-bottom: 18px; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .prog { font-size: 14px; color: var(--muted); }
  .stage { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }
  @media (max-width: 760px) { .stage { grid-template-columns: 1fr; } }
  .pane { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; display: flex; flex-direction: column; align-items: center; }
  .pane-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px; align-self: flex-start; }
  .pane img { max-width: 100%; max-height: 460px; border-radius: 6px; object-fit: contain; background: #000; }
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
  .label { font-size: 13px; color: var(--muted); min-width: 145px; }
  button { background: #232e3a; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 9px 14px; font-size: 14px; cursor: pointer; transition: .12s; }
  button:hover { border-color: var(--acc); }
  button.sel-ok { background: rgba(34,197,94,.2); border-color: var(--ok); color: #bbf7d0; }
  button.sel-warn { background: rgba(245,158,11,.2); border-color: var(--warn); color: #fde68a; }
  button.sel-bad { background: rgba(239,68,68,.2); border-color: var(--bad); color: #fecaca; }
  button.sel-acc { background: rgba(56,189,248,.22); border-color: var(--acc); color: #bae6fd; }
  button.primary { background: var(--acc); color: #07131d; border-color: var(--acc); font-weight: 600; }
  input[type=text] { background: #10161d; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; font-size: 14px; flex: 1; min-width: 200px; }
  .reveal { margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--line); display: none; }
  .reveal.show { display: block; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; background: #232e3a; }
  .tag.ok { background: rgba(34,197,94,.2); color: #86efac; }
  .tag.bad { background: rgba(239,68,68,.2); color: #fca5a5; }
  table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
  th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 16px 0; }
  .stat { background: #121820; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .stat .v { font-size: 26px; font-weight: 700; margin-top: 4px; }
  .hidden { display: none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>位置確認ブラインドテスト</h1>
    <div class="prog" id="prog">0 / 0</div>
  </header>

  <div id="quiz">
    <div class="stage">
      <div class="pane">
        <div class="pane-title">GIF（ラウンド中の動き）</div>
        <img id="img-gif" alt="GIF">
      </div>
      <div class="pane">
        <div class="pane-title">ミニマップ（開幕位置）</div>
        <img id="img-mm" alt="ミニマップ">
      </div>
    </div>

    <div class="panel">
      <div class="row">
        <span class="label">1. 開幕位置の正確性</span>
        <button type="button" data-pos="ok" onclick="pickPos('ok')">[1] 正しい（ズレなし）</button>
        <button type="button" data-pos="warn" onclick="pickPos('warn')">[2] 一部ズレ・不足</button>
        <button type="button" data-pos="bad" onclick="pickPos('bad')">[3] 誤検出（大きくズレ）</button>
      </div>
      <div class="row">
        <span class="label">2. 目視サイト読み取り</span>
        <button type="button" data-site="A" onclick="pickSite('A')">A</button>
        <button type="button" data-site="B" onclick="pickSite('B')">B</button>
        <button type="button" data-site="C" onclick="pickSite('C')">C</button>
        <button type="button" data-site="不明" onclick="pickSite('不明')">不明 / その他</button>
      </div>
      <div class="row">
        <span class="label">3. 目視メモ（任意）</span>
        <input type="text" id="memo" placeholder="気になったズレ・配置など">
        <button type="button" class="primary" id="btn-submit" onclick="submitAnswer()">答え合わせ</button>
        <button type="button" class="primary hidden" id="btn-next" onclick="nextQ()">次の問題 →</button>
      </div>

      <div class="reveal" id="reveal"></div>
    </div>
  </div>

  <div id="summary" class="panel hidden">
    <h2 style="margin:0 0 8px;font-size:17px;">検証結果（正確率サマリー）</h2>
    <div class="stats" id="stats"></div>
    <div class="row">
      <button type="button" class="primary" onclick="downloadLog()">結果を JSON で保存</button>
      <button type="button" onclick="restart()">最初からやり直す</button>
    </div>
    <table>
      <thead><tr><th>#</th><th>試合 / ラウンド</th><th>位置判定</th><th>目視サイト</th><th>正解サイト</th><th>メモ</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<script>
const ITEMS = __ITEMS_JSON__;
let idx = 0, curPos = null, curSite = null, results = [];

function render() {
  if (idx >= ITEMS.length) { showSummary(); return; }
  const it = ITEMS[idx];
  document.getElementById("prog").textContent = `Q${idx + 1} / ${ITEMS.length}`;
  document.getElementById("img-gif").src = it.gif;
  document.getElementById("img-mm").src = it.minimap;
  curPos = null; curSite = null;
  document.getElementById("memo").value = "";
  document.querySelectorAll("[data-pos]").forEach(b => b.className = "");
  document.querySelectorAll("[data-site]").forEach(b => b.className = "");
  document.getElementById("reveal").className = "reveal";
  document.getElementById("reveal").innerHTML = "";
  document.getElementById("btn-submit").classList.remove("hidden");
  document.getElementById("btn-next").classList.add("hidden");
}

function pickPos(v) {
  curPos = v;
  document.querySelectorAll("[data-pos]").forEach(b => {
    b.className = b.dataset.pos === v ? ("sel-" + v) : "";
  });
}

function pickSite(v) {
  curSite = v;
  document.querySelectorAll("[data-site]").forEach(b => {
    b.className = b.dataset.site === v ? "sel-acc" : "";
  });
}

function submitAnswer() {
  if (!curPos) { alert("「1. 開幕位置の正確性」を選択してください。"); return; }
  const it = ITEMS[idx];
  const a = it.answer;
  const siteOk = curSite ? (curSite === a.site) : null;
  results.push({ id: it.id, pos: curPos, site_guess: curSite, site_true: a.site, site_ok: siteOk, memo: document.getElementById("memo").value.trim(), answer: a });

  const agList = (a.agents_140 || []).map(x => `${x.agent}(T${x.team}): ${x.callout || x.zone || "?"}`).join(" / ") || "検出なし";
  const siteBadge = siteOk === null ? "" : (siteOk ? '<span class="tag ok">サイト一致</span>' : `<span class="tag bad">サイト不一致 (正解: ${a.site})</span>`);
  document.getElementById("reveal").innerHTML = `
    <div style="margin-bottom:6px;">
      <span class="tag">${a.map} R${a.round}</span>
      <span class="tag">${a.match}</span>
      <span class="tag">Team1=${a.side1}</span>
      ${siteBadge}
    </div>
    <div style="font-size:13px;color:var(--muted);">1:40 検出: ${agList}</div>
  `;
  document.getElementById("reveal").classList.add("show");
  document.getElementById("btn-submit").classList.add("hidden");
  document.getElementById("btn-next").classList.remove("hidden");
}

function nextQ() { idx++; render(); }

function showSummary() {
  document.getElementById("quiz").classList.add("hidden");
  document.getElementById("summary").classList.remove("hidden");
  document.getElementById("prog").textContent = `完了 (${results.length} 問)`;
  const n = results.length || 1;
  const nOk = results.filter(r => r.pos === "ok").length;
  const nNoBad = results.filter(r => r.pos !== "bad").length;
  const nBad = results.filter(r => r.pos === "bad").length;
  const siteRated = results.filter(r => r.site_ok !== null);
  const siteAcc = siteRated.length ? Math.round(100 * siteRated.filter(r => r.site_ok).length / siteRated.length) + "%" : "—";
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="label">位置 正確率 (ズレなし)</div><div class="v" style="color:var(--ok)">${Math.round(100 * nOk / n)}% <span style="font-size:13px;color:var(--muted)">(${nOk}/${results.length})</span></div></div>
    <div class="stat"><div class="label">大ズレなし率 (OK+一部)</div><div class="v">${Math.round(100 * nNoBad / n)}% <span style="font-size:13px;color:var(--muted)">(${nNoBad}/${results.length})</span></div></div>
    <div class="stat"><div class="label">誤検出率 (大きくズレ)</div><div class="v" style="color:var(--bad)">${Math.round(100 * nBad / n)}% <span style="font-size:13px;color:var(--muted)">(${nBad}/${results.length})</span></div></div>
    <div class="stat"><div class="label">目視サイト一致率</div><div class="v" style="color:var(--acc)">${siteAcc}</div></div>
  `;
  const posLbl = { ok: "正しい", warn: "一部ズレ", bad: "誤検出" };
  document.getElementById("tbody").innerHTML = results.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${r.answer.map} R${r.answer.round} (${r.answer.match})</td>
      <td><span class="tag ${r.pos === 'ok' ? 'ok' : r.pos === 'bad' ? 'bad' : ''}">${posLbl[r.pos]}</span></td>
      <td>${r.site_guess || "—"}</td>
      <td>${r.site_true}</td>
      <td>${r.memo || ""}</td>
    </tr>
  `).join("");
}

function downloadLog() {
  const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "blind_test_result.json";
  a.click();
}

function restart() {
  idx = 0; results = [];
  document.getElementById("summary").classList.add("hidden");
  document.getElementById("quiz").classList.remove("hidden");
  render();
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "1") pickPos("ok");
  else if (e.key === "2") pickPos("warn");
  else if (e.key === "3") pickPos("bad");
  else if (e.key === "Enter") {
    if (!document.getElementById("btn-submit").classList.contains("hidden")) submitAnswer();
    else if (!document.getElementById("btn-next").classList.contains("hidden")) nextQ();
  }
});

if (ITEMS.length) render();
else document.getElementById("quiz").innerHTML = "<div class='panel'>問題データが見つかりませんでした。</div>";
</script>
</body>
</html>
"""


def build_html(items: list[dict], out_path: Path) -> Path:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(HTML_TEMPLATE.replace("__ITEMS_JSON__", payload), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="GIF とミニマップのブラインドテスト HTML を生成")
    ap.add_argument("target", nargs="?", help="report.json またはそれを含むディレクトリ")
    ap.add_argument("--out", default="blind_test.html", help="出力 HTML パス")
    ap.add_argument("--max", type=int, default=20, help="最大出題数（0 で全件）")
    ap.add_argument("--seed", type=int, default=42, help="シャッフルシード")
    ap.add_argument("--map", default="", help="マップ名で絞り込み")
    ap.add_argument("--demo", action="store_true", help="フィクスチャを使ったデモ HTML を生成")
    args = ap.parse_args()

    if args.demo:
        items = demo_items()
    else:
        if not args.target:
            raise SystemExit("target（report.json またはディレクトリ）を指定するか、--demo を付けてください。")
        items = collect_items(Path(args.target), max_items=args.max, seed=args.seed, map_filter=args.map)
    if not items:
        raise SystemExit("GIF とミニマップが揃ったラウンドが見つかりませんでした。")
    out = build_html(items, Path(args.out))
    print(f"ブラインドテスト HTML を作成しました: {out} ({len(items)} 問)")


if __name__ == "__main__":
    main()
