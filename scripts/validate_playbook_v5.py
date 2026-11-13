"""プレイブック v5 の出力フォルダを機械検査する（docs/PLAYBOOK_v5.md 8 節 + 段階B/段階C）。

    python scripts/validate_playbook_v5.py <出力フォルダ>

構成ページ（comp_*.html）と index.html を、禁止文字列・空要素・図の枚数・スライドの行数・
manifest との整合・段階B/C（サイド別×フェーズ3部構成・上位3型・型ごとの3枚図（1枚1時刻）・
旧ノード表の文言なし・代表ラウンドのVODカードと実プレイGIF・サイド別分母の明示・サイド整合）で見る。
FAIL があれば終了コード 1。データが無い環境で出力フォルダ自体が未生成の場合は SKIP（終了コード 0）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

BANNED = ["マップ画像がありません", "場所不明", "None", "形だけ", "推定", "&lt;a ", "正しい戦略の模索", "略号"]
ATTACK_ONLY_PHRASES = ("深くまで早く入って", "バイタルでキルが出たらサイトへ飛び込む",
                       "デュエリストが奥に入り全員で続く", "1人で裏を取り別の場所")
OLD_NODE_PHRASES = (
    "深くまで早く入って主導権を取る",
    "薄い側に圧をかけられたら逆側を取りに行く",
    "前に出てきた相手に合わせて当たる",
    "構えて待つ",
    "先に1人倒されたら止まって立て直す",
    "空いたバイタルを取って近いサイトへ寄る",
    "バイタルの当たりを返り討ちにする",
    "バイタルを取ったら構えるか別へ回る",
    "取ったバイタルに居座って圧をかける",
    "1人をサイトに残して回る",
    "薄いサイトに来たら裏から挟む",
    "サイトの入口で止める",
    "サイトの裏で時間を稼いで合流する",
    "取り返すと決めて時間を使う",
    "デュエリストが奥に入り全員で続く",
    "バイタルでキルが出たらサイトへ飛び込む",
    "入口で当たってそのまま入る",
    "1人で裏を取り別の場所を揺さぶる",
    "入口に来た相手をカウンター",
    "1〜2人残して退き時間を稼ぐ",
    "早く集まってリテイク",
    "複数の入口から囲い込んでリテイク",
    "そろえてからゆっくりリテイク",
    "寄せずに1人で時間を稼ぐ",
    "人数有利なら距離を詰める",
    "距離を詰める",
    "人数不利なら分かれて片方で当たる",
    "マネー有利なら普段と違う手を打つ",
    "タイミングを見て前に出る",
    "薄くなった側を1人で持ち、厚い側に人数を寄せる",
    "1か所に集めて守る",
    "前に出て間合いを詰める",
    "はまると強い",
)
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


def check_page(name: str, page: str, manifest_entry: dict, results: list, fail: list,
               out_dir: Path | None = None) -> None:
    def rec(ok: bool, label: str, detail: str = ""):
        results.append((label, ok, detail))
        if not ok:
            fail.append(f"{name}: {label} {detail}")

    m = manifest_entry
    phase_sections = m.get("phase_sections") or []
    all_phase_types = [t for sec in phase_sections for t in (sec.get("types") or [])]

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

    # 3. 図の枚数（マップ図・実プレイGIF・VODカード）
    maps = len(re.findall(r'class="(?:mapsvg|mo-map|board-card)"', page))
    gifs = len(re.findall(r'<figure class="play-gif">', page))
    vods = len(re.findall(r'class="vod"', page))
    min_gifs = max(1, len(all_phase_types)) if all_phase_types else 0
    min_vods = max(1, len(all_phase_types)) if all_phase_types else 1
    rec(maps >= 5, "マップ図の枚数", f"{maps}")
    rec(gifs >= min_gifs, "実プレイ例のGIF貼付", f"gifs={gifs} required>={min_gifs}")
    rec(vods >= min_vods, "VODカードの枚数", f"{vods}")

    # 4. 見出しの5人と見本ラウンドの構成
    agents = set(m.get("agents") or [])
    keys = m.get("sample_comp_keys") or []
    bad = [k for k in keys if set(str(k).split("|")) != agents]
    head_ok = all(a in page for a in agents) and len(agents) == 5
    rec(not bad and head_ok, "見出しと見本の構成一致",
        f"keys={len(keys)} bad={bad} heading={'ok' if head_ok else 'ng'}")

    # 5. 数合わせ
    rec(m["used_def"] + m["used_atk"] + m["excluded"] == m["total"], "数合わせ",
        f'{m["used_def"]}+{m["used_atk"]}+{m["excluded"]}={m["total"]}')

    # 6. 基本配置の型はすべて PATTERN_MIN 以上
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
        elif h.get("sample_full5") is False:
            status = h.get("sample_status")
            if status == "no_full5":
                expect = "5人そろった見本ラウンドがありません"
            elif status == "no_motion":
                expect = "動く図にできるラウンドがありませんでした"
            else:
                expect = ""
            if not expect or expect not in text:
                no_sample_bad.append(f"{h.get('name')}:{status}")
    rec(not label_bad, "見本ラベルのチーム照合",
        str(label_bad) if label_bad else f"{len(sample_rounds)} 件")
    side_blocks = m.get("side_blocks") or {}
    if side_blocks:
        max_unique = 0
        for s_key in (ATTACK, DEFENSE):
            s_rounds = [h.get("sample_round") for h in hands if h.get("side") == s_key and h.get("sample_round")]
            s_cls = int((side_blocks.get(s_key) or {}).get("classified") or 0)
            max_unique += min(len(s_rounds), s_cls) if s_cls else len(s_rounds)
    else:
        max_unique = min(len(sample_rounds), int(m.get("flow_classified") or len(sample_rounds)))
    rec(len(set(sample_rounds)) >= max_unique, "見本ラウンドの重複なし",
        f"{len(sample_rounds)} 件 / 一意 {len(set(sample_rounds))}")
    rec(not full5_bad, "見本は full5", str(full5_bad) if full5_bad else "")
    rec(not no_sample_bad, "見本なしの表示（書き分け）", str(no_sample_bad) if no_sample_bad else "")

    # 10. 基本配置の勝率条件・避けるべき型・守り攻め両側
    placements = m.get("placements") or []
    sides = {p.get("side") for p in placements}
    placement_bad, avoid_bad, reason_bad = [], [], []
    for p in placements:
        if p.get("name"):
            if not p.get("meets_winrate"):
                placement_bad.append(f"{p.get('side')}:{p.get('name')}")
        elif not p.get("reason"):
            placement_bad.append(f"{p.get('side')}:理由なし")
        else:
            kind = p.get("reason_kind")
            reason = p.get("reason") or ""
            expect = {"no_rounds": "型の成立に必要な", "distance": "同じ形とみなせる",
                      "winrate": "平均勝率", "no_motion": "動く図にできる"}.get(kind)
            if not expect or expect not in reason:
                reason_bad.append(f"{p.get('side')}:{kind}")
        if p.get("avoid") and "避けるべき型" not in text:
            avoid_bad.append(p.get("side"))
    placement_sides_ok = "守り / 配置" in text and "攻め / 配置" in text
    rec(not placement_bad, "基本配置の勝率条件",
        str(placement_bad) if placement_bad else f"{len(placements)} 側")
    rec(not avoid_bad, "避けるべき型の明示", str(avoid_bad) if avoid_bad else "")
    rec(not reason_bad, "基本配置なし理由の書き分け",
        str(reason_bad) if reason_bad else "")
    rec(DEFENSE in sides and ATTACK in sides and placement_sides_ok, "守り・攻め両側の節",
        f"sides={sorted(sides)} 配置={placement_sides_ok}")

    # 11. 守りの型名に攻め専用の言い回しが残っていない
    atk_bad = [p["name"] for p in m.get("patterns") or []
               if p.get("side") == DEFENSE and any(ph in p["name"] for ph in ATTACK_ONLY_PHRASES)]
    rec(not atk_bad, "守りの型名に攻め専用表現なし", str(atk_bad) if atk_bad else "")

    # 12. [段階B/C/D] サイド別3部構成（攻撃: 開幕・エントリー・ポストプラント / 防衛: 序盤・中盤・終盤）
    side_blocks = m.get("side_blocks") or {}
    expected_sec_titles = {
        ATTACK: ("開幕の型", "エントリーの型", "ポストプラントの形"),
        DEFENSE: ("序盤の型", "中盤の強い動き", "終盤：寄せ方・リテイクの仕方"),
    }
    expected_segs = {
        ATTACK: (["early", "entry", "postplant"], ["開幕", "エントリー", "ポストプラント"]),
        DEFENSE: (["early", "mid", "late"], ["序盤", "中盤", "終盤"]),
    }
    atk_early_vocab = {"Aデフォルト", "MIDデフォルト", "Bデフォルト", "Cデフォルト", "ラッシュ", "詰め待ち"}
    atk_entry_vocab = {"Aセット", "Bセット", "Cセット", "Aスプリット", "Bスプリット", "Cスプリット"}
    def_early_vocab = {"A寄せ", "MID寄せ", "B寄せ", "C寄せ", "自陣待機"}

    three_parts_ok = ATTACK in side_blocks and DEFENSE in side_blocks
    order_bad = []
    action_reaction_bad = []
    top3_bad = []
    def_late_bad = []
    stage_d_bad = []
    for side_key in (ATTACK, DEFENSE):
        blk = side_blocks.get(side_key) or {}
        secs = blk.get("sections") or []
        seg_keys = [s.get("seg") for s in secs]
        seg_labels = [s.get("label") for s in secs]
        exp_keys, exp_labels = expected_segs[side_key]
        if seg_keys != exp_keys or seg_labels != exp_labels:
            three_parts_ok = False
        if side_key == ATTACK and ("mid" in seg_keys or "中盤" in seg_labels):
            stage_d_bad.append("atk_has_mid")
        for sec_t in expected_sec_titles[side_key]:
            if f"{side_key} / {sec_t}" not in text:
                three_parts_ok = False
        for sec in secs:
            tps = sec.get("types") or []
            if sec.get("seg") != "postplant" and len(tps) > 3:
                top3_bad.append(f"{side_key}:{sec.get('label')}:len={len(tps)}")
            ns = [t.get("n", 0) for t in tps]
            if ns != sorted(ns, reverse=True):
                order_bad.append(f"{side_key}:{sec.get('label')}:{ns}")
            if side_key == ATTACK and sec.get("seg") == "postplant":
                per_site: dict[str, int] = {}
                for t in tps:
                    s_name = t.get("site") or (str(t.get("type") or "").split("：", 1)[0])
                    per_site[s_name] = per_site.get(s_name, 0) + 1
                over_site = {k: v for k, v in per_site.items() if v > 2}
                if over_site:
                    stage_d_bad.append(f"postplant_over2:{over_site}")
            for t in tps:
                tp_name = t.get("type")
                if side_key == ATTACK and sec.get("seg") == "early" and tp_name not in atk_early_vocab:
                    stage_d_bad.append(f"atk_early_vocab:{tp_name}")
                if side_key == ATTACK and sec.get("seg") == "entry" and tp_name not in atk_entry_vocab:
                    stage_d_bad.append(f"atk_entry_vocab:{tp_name}")
                if side_key == DEFENSE and sec.get("seg") == "early" and tp_name not in def_early_vocab:
                    stage_d_bad.append(f"def_early_vocab:{tp_name}")
                head = t.get("title") or f"{side_key}・{sec.get('label')}：{tp_name}"
                desc = t.get("description") or ""
                if head not in text:
                    order_bad.append(f"missing:{head}")
                if "アクション：" not in desc or "リアクション：" not in desc or desc not in text:
                    action_reaction_bad.append(head)
                if side_key == DEFENSE and sec.get("seg") == "late":
                    if ("寄せ" not in head and "リテイク" not in head) or ("寄せ" not in desc and "リテイク" not in desc):
                        def_late_bad.append(head)
    rec(three_parts_ok and not order_bad and not action_reaction_bad and not top3_bad and not def_late_bad and not stage_d_bad,
        "サイド別3部構成（段階D語彙・攻撃中盤なし・防衛視点）",
        f"parts_ok={three_parts_ok} top3={top3_bad} order={order_bad} ar={action_reaction_bad} "
        f"def_late={def_late_bad} stage_d={stage_d_bad}")

    # 13. [段階B] 図が型ごとに3枚（1枚1時刻・TactiValボードJSONおよび画像または未生成明示）
    board_bad = []
    for t in all_phase_types:
        b_list = t.get("boards") or []
        exp_n = t.get("expected_boards", 3)
        if not (1 <= len(b_list) <= 3 and len(b_list) == exp_n):
            board_bad.append(f"{t.get('seg_label')}:{t.get('type')}:len={len(b_list)}!=exp={exp_n}")
        for b in b_list:
            if not b.get("single_timestamp") or b.get("steps") != 1:
                board_bad.append(f"{t.get('type')}:{b.get('segment')}:steps={b.get('steps')}")
            bjson = b.get("board_json")
            if not bjson or f'data-board="{bjson}"' not in page:
                board_bad.append(f"{t.get('type')}:{b.get('segment')}:missing_attr")
            elif out_dir is not None:
                bp = out_dir / bjson
                if not bp.is_file():
                    board_bad.append(f"{t.get('type')}:{bjson}:no_file")
                else:
                    try:
                        bdata = json.loads(bp.read_text("utf-8"))
                        steps = bdata.get("steps") or []
                        if len(steps) != 1:
                            board_bad.append(f"{bjson}:steps={len(steps)}")
                        else:
                            els = steps[0].get("elements") or []
                            arrows = [e for e in els if e.get("type") == "arrow"]
                            non_icon = [e for e in els if e.get("type") == "agent"
                                        and not (e.get("data") or {}).get("agentId")]
                            if arrows or non_icon:
                                board_bad.append(f"{bjson}:arrows={len(arrows)},non_icon={len(non_icon)}")
                    except Exception as e:                           # noqa: BLE001
                        board_bad.append(f"{bjson}:json_err={e}")
    for c in m.get("claims") or []:
        b_list = c.get("boards") or []
        if not (1 <= len(b_list) <= 3):
            board_bad.append(f"claim:{c.get('metric')}:len={len(b_list)}")
    rec(not board_bad, "型ごとの3枚図（1枚1時刻）",
        str(board_bad[:5]) if board_bad else f"{len(all_phase_types)} 型")

    # 14. [段階B] 旧ノード表の文言（「距離を詰める」等）が生成ページに出ない
    old_hits = [ph for ph in OLD_NODE_PHRASES if ph in text]
    rec(not old_hits, "旧ノード表の文言なし", str(old_hits[:5]) if old_hits else "")

    # 15. [段階B] 代表ラウンドに VOD カードと実プレイ GIF がある
    rep_vod_bad = []
    rep_gif_bad = []
    for t in all_phase_types:
        reps = t.get("representatives") or []
        if not reps:
            rep_vod_bad.append(f"{t.get('seg_label')}:{t.get('type')}:no_reps")
            continue
        for rp in reps:
            lbl = rp.get("label") or ""
            if not rp.get("has_vod") or not lbl or lbl not in text:
                rep_vod_bad.append(f"{t.get('type')}:{lbl}")
        lead_gif = reps[0].get("gif")
        if not lead_gif or f'src="{lead_gif}"' not in page:
            rep_gif_bad.append(f"{t.get('type')}:missing_gif_tag")
        elif out_dir is not None:
            gp = out_dir / lead_gif
            if not gp.is_file() or gp.stat().st_size == 0:
                rep_gif_bad.append(f"{t.get('type')}:missing_gif_file={lead_gif}")
    rec(not rep_vod_bad, "代表ラウンドのVODカード",
        str(rep_vod_bad[:5]) if rep_vod_bad else f"{len(all_phase_types)} 型")
    rec(not rep_gif_bad, "代表ラウンドのGIF実在",
        str(rep_gif_bad[:5]) if rep_gif_bad else f"{len(all_phase_types)} 型")

    # 16. [段階B/C] 分母（分類できたラウンド数／全ラウンド数＋サイド別分母）がページに明示される
    cls_n = m.get("flow_classified")
    sk_n = m.get("flow_skipped")
    tot_n = m.get("total")
    side_denom_bad = []
    for side_key in (ATTACK, DEFENSE):
        blk = side_blocks.get(side_key) or {}
        s_tot = blk.get("total")
        s_cls = blk.get("classified")
        s_sk = blk.get("skipped")
        if s_tot is None or s_cls is None or s_sk is None or s_cls + s_sk != s_tot:
            side_denom_bad.append(f"{side_key}:sum")
        elif f"{side_key} 分類{s_cls}/{s_tot}R" not in text:
            side_denom_bad.append(f"{side_key}:missing_text")
    denom_ok = (
        cls_n is not None
        and sk_n is not None
        and cls_n + sk_n == tot_n
        and f"フェーズ分類できたラウンド数：{cls_n} / 全 {tot_n} ラウンド" in text
        and not side_denom_bad
    )
    rec(denom_ok, "分母（分類ラウンド数・サイド別）の明示",
        f"classified={cls_n} skipped={sk_n} total={tot_n} side_bad={side_denom_bad}")

    # 17. [段階C] サイド整合チェック（攻撃ブロックの代表例は攻撃のみ、防衛ブロックの代表例は防衛のみ）
    side_mix_bad = []
    for side_key, expect_ja, wrong_ja in ((ATTACK, "攻め", "守り"), (DEFENSE, "守り", "攻め")):
        blk = side_blocks.get(side_key) or {}
        for sec in blk.get("sections") or []:
            for t in sec.get("types") or []:
                if t.get("side") != side_key:
                    side_mix_bad.append(f"type_side:{side_key}:{t.get('type')}={t.get('side')}")
                for rp in t.get("representatives") or []:
                    if rp.get("side") != side_key:
                        side_mix_bad.append(f"rep_side:{side_key}:{rp.get('label')}={rp.get('side')}")
        for sec_html in re.findall(
            rf'<section class="slide[^"]*" data-side="{side_key}">(.*?)</section>', page, re.S
        ):
            for li in re.findall(r"<li>(.*?)</li>", sec_html, re.S):
                li_txt = strip_tags(li)
                if li_txt.startswith("代表 "):
                    if f"（{wrong_ja}）" in li_txt or f"（{expect_ja}）" not in li_txt:
                        side_mix_bad.append(f"html:{side_key}:{li_txt[:40]}")
    rec(not side_mix_bad, "サイド整合（攻撃/防衛の混線なし）",
        str(side_mix_bad[:5]) if side_mix_bad else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--strict", action="store_true", help="フォルダ未存在でも SKIP せず FAIL にする")
    a = ap.parse_args(argv)
    d = Path(a.folder)
    if not d.exists() and not a.strict:
        print(f"SKIP: 出力フォルダがありません（実データが無い環境のためスキップ: {d}）")
        return 0
    mani_p = d / "manifest.json"
    if not mani_p.exists():
        print(f"FAIL: manifest.json がありません（{d}）")
        return 1
    mani = json.loads(mani_p.read_text("utf-8"))
    index = d / "index.html"
    index_html = index.read_text("utf-8") if index.exists() else ""
    results, fail = [], []

    print(f"検証: {d}（{mani.get('map')} / {mani.get('env_label')}）")
    bf = mani.get("barrier_filter")
    if isinstance(bf, dict) and "dropped" in bf:
        print(f"  暫定バリアフィルタ: 除外 {bf.get('dropped', 0)} 件"
              f"（自陣外 {bf.get('outside_home', 0)}・ジャンプ {bf.get('jump', 0)}）")
        results.append(("暫定バリアフィルタ統計", True, str(bf)))
    else:
        results.append(("暫定バリアフィルタ統計", False, "barrier_filter missing"))
        fail.append("manifest.json: barrier_filter がありません")
    pages = mani.get("pages") or []
    if not pages:
        print("FAIL: ページが 1 つもありません")
        return 1
    for m in pages:
        p = d / m["file"]
        print(f"  - {m['file']}（{m['total']}R / 守り{m['used_def']} 攻め{m['used_atk']} "
              f"除外{m['excluded']} / 分類{m.get('flow_classified', 0)}）")
        ok_idx = f'href="{m["file"]}"' in index_html
        results.append(("index からリンク", ok_idx, m["file"]))
        if not ok_idx:
            fail.append(f'index: {m["file"]} へのリンクがありません')
        if not p.exists():
            fail.append(f'{m["file"]} がありません')
            continue
        check_page(m["file"], p.read_text("utf-8"), m, results, fail, out_dir=d)

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
