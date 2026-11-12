"""プレイブック v5 の出力フォルダを機械検査する（docs/PLAYBOOK_v5.md 8 節）。

    python scripts/validate_playbook_v5.py <出力フォルダ>

構成ページ（comp_*.html）と index.html を、禁止文字列・空要素・図の枚数・スライドの行数・
manifest との整合で見る。FAIL があれば終了コード 1。検証は緩めない（通らないときは生成器を直す）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

BANNED = ["マップ画像がありません", "場所不明", "None", "形だけ", "推定", "&lt;a ", "正しい戦略の模索", "略号"]
PATTERN_MIN = 3
DEFENSE, ATTACK = "防衛", "攻撃"


def strip_tags(h: str) -> str:
    return re.sub(r"<[^>]+>", "", h)


def body_lines(section: str) -> int:
    s = section
    s = re.sub(r"<figure\b.*?</figure>", "", s, flags=re.S)
    s = re.sub(r"<table\b.*?</table>", "", s, flags=re.S)
    s = re.sub(r"<figcaption\b.*?</figcaption>", "", s, flags=re.S)
    s = re.sub(r"<h[1-6]\b.*?</h[1-6]>", "", s, flags=re.S)
    return len(re.findall(r"<(?:p|li)\b", s))


def is_appendix(section: str) -> bool:
    return "付録" in section.split("</p>", 1)[0]


def check_page(name: str, page: str, manifest_entry: dict, results: list, fail: list) -> None:
    def rec(ok: bool, label: str, detail: str = ""):
        results.append((label, ok, detail))
        if not ok:
            fail.append(f"{name}: {label} {detail}")

    # 1. 禁止文字列
    text = strip_tags(page)
    hits = [(b, text.count(b)) for b in BANNED if b in text]
    if "?" in text:
        hits.append(("?", text.count("?")))
    rec(not hits, "禁止文字列", str(hits) if hits else "")

    # 2. 空の ul・figcaption
    empty_ul = len(re.findall(r"<ul\b[^>]*>\s*</ul>", page))
    empty_fc = len(re.findall(r"<figcaption\b[^>]*>\s*</figcaption>", page))
    rec(empty_ul == 0 and empty_fc == 0, "空要素", f"ul={empty_ul} figcaption={empty_fc}")

    # 3. 図の枚数
    maps = len(re.findall(r'class="(?:mapsvg|mo-map)"', page))
    motions = len(re.findall(r'class="motion"', page))
    vods = len(re.findall(r'class="vod"', page))
    rec(maps >= 5, "マップ図の枚数", f"{maps}")
    rec(motions >= 3, "動く図の枚数", f"{motions}")
    rec(vods >= 5, "VODカードの枚数", f"{vods}")

    # 4. 見出しの5人と見本ラウンドの構成
    agents = set(manifest_entry.get("agents") or [])
    keys = manifest_entry.get("sample_comp_keys") or []
    bad = [k for k in keys if set(str(k).split("|")) != agents]
    head_ok = all(a in page for a in agents) and len(agents) == 5
    rec(not bad and head_ok, "見出しと見本の構成一致",
        f"keys={len(keys)} bad={bad} heading={'ok' if head_ok else 'ng'}")

    # 5. 数合わせ
    m = manifest_entry
    rec(m["used_def"] + m["used_atk"] + m["excluded"] == m["total"], "数合わせ",
        f'{m["used_def"]}+{m["used_atk"]}+{m["excluded"]}={m["total"]}')

    # 6. 型はすべて PATTERN_MIN 以上
    lowp = [p for p in m.get("patterns") or [] if p["n"] < PATTERN_MIN]
    rec(not lowp, "型の成立数", str(lowp) if lowp else f'{len(m.get("patterns") or [])} 型')

    # 7. スライド本文が 5 行以内
    over = []
    for sec in re.findall(r'<section class="slide[^"]*">(.*?)</section>', page, re.S):
        if is_appendix(sec):
            continue
        n = body_lines(sec)
        if n > 5:
            over.append(n)
    rec(not over, "スライド本文の行数", f"over={over}" if over else "")

    # 8. font-size 14px 未満
    small = [x for x in re.findall(r"font-size\s*:\s*([0-9.]+)px", page) if float(x) < 14]
    small += [x for x in re.findall(r'font-size="([0-9.]+)"', page) if float(x) < 14]
    rec(not small, "font-size 14px 以上", str(small[:5]) if small else "")

    # 9. 見本ラベルのチーム照合・重複なし・full5・見本なし表示
    hands = m.get("hands") or []
    teams = set(m.get("teams") or [])
    label_bad, full5_bad, no_sample_bad = [], [], []
    sample_rounds = []
    for h in hands:
        rk = h.get("sample_round")
        if rk:
            sample_rounds.append(rk)
            team = h.get("sample_team")
            label = h.get("sample_label") or ""
            if team not in teams:
                label_bad.append(f"{h.get('name')}:team={team}")
            if not label.startswith(f"{team} vs "):
                label_bad.append(f"{h.get('name')}:label={label}")
            if label and label not in text:
                label_bad.append(f"{h.get('name')}:missing")
            if not h.get("sample_full5"):
                full5_bad.append(h.get("name"))
        elif "見本ラウンドがありません" not in text:
            no_sample_bad.append(h.get("name"))
    rec(not label_bad, "見本ラベルのチーム照合",
        str(label_bad) if label_bad else f"{len(sample_rounds)} 件")
    rec(len(sample_rounds) == len(set(sample_rounds)), "見本ラウンドの重複なし",
        f"{len(sample_rounds)} 件 / 一意 {len(set(sample_rounds))}")
    rec(not full5_bad, "見本は full5", str(full5_bad) if full5_bad else "")
    rec(not no_sample_bad, "見本なしの表示", str(no_sample_bad) if no_sample_bad else "")

    # 10. 基本配置の勝率条件・避けるべき型・守り攻め両側
    placements = m.get("placements") or []
    sides = {p.get("side") for p in placements}
    placement_bad, avoid_bad = [], []
    for p in placements:
        if p.get("name"):
            if not p.get("meets_winrate"):
                placement_bad.append(f"{p.get('side')}:{p.get('name')}")
        elif not p.get("reason"):
            placement_bad.append(f"{p.get('side')}:理由なし")
        if p.get("avoid") and "避けるべき型" not in text:
            avoid_bad.append(p.get("side"))
    hand_sides_ok = "守り / 手札" in text and "攻め / 手札" in text
    rec(not placement_bad, "基本配置の勝率条件",
        str(placement_bad) if placement_bad else f"{len(placements)} 側")
    rec(not avoid_bad, "避けるべき型の明示", str(avoid_bad) if avoid_bad else "")
    rec(DEFENSE in sides and ATTACK in sides and hand_sides_ok, "守り・攻め両側の節",
        f"sides={sorted(sides)} 手札={hand_sides_ok}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args(argv)
    d = Path(a.folder)
    mani_p = d / "manifest.json"
    if not mani_p.exists():
        print(f"FAIL: manifest.json がありません（{d}）")
        return 1
    mani = json.loads(mani_p.read_text("utf-8"))
    index = (d / "index.html")
    index_html = index.read_text("utf-8") if index.exists() else ""
    results, fail = [], []

    print(f"検証: {d}（{mani.get('map')} / {mani.get('env_label')}）")
    pages = mani.get("pages") or []
    if not pages:
        print("FAIL: ページが 1 つもありません")
        return 1
    for m in pages:
        p = d / m["file"]
        print(f"  - {m['file']}（{m['total']}R / 守り{m['used_def']} 攻め{m['used_atk']} 除外{m['excluded']}）")
        ok_idx = f'href="{m["file"]}"' in index_html
        results.append(("index からリンク", ok_idx, m["file"]))
        if not ok_idx:
            fail.append(f'index: {m["file"]} へのリンクがありません')
        if not p.exists():
            fail.append(f'{m["file"]} がありません')
            continue
        check_page(m["file"], p.read_text("utf-8"), m, results, fail)

    # 集計
    ok_n = sum(1 for _l, ok, _d in results if ok)
    print("\n項目ごとの結果")
    per = {}
    for label, ok, detail in results:
        per.setdefault(label, [0, 0])
        per[label][0] += 1
        per[label][1] += int(ok)
    for label, (tot, ok) in sorted(per.items()):
        mark = "PASS" if ok == tot else "FAIL"
        print(f"  {mark} {label}: {ok}/{tot}")
    print(f"\n合計: {ok_n}/{len(results)} PASS")
    if fail:
        print("\nFAIL の詳細")
        for x in fail[:40]:
            print(f"  - {x}")
        return 1
    print("すべて PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
