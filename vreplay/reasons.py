"""ラウンドの勝因を、キルの順番・スパイクの設置・攻守・購入から判定する。

使う材料（どれかが欠けても、分かる範囲で判定する）
- VLR：勝ったチーム、勝ち方（elim 全滅 / boom 爆破 / defuse 解除 / time 時間切れ）、勝った側（攻撃・防衛）、購入段階
- 動画：キルログ（時刻・キラー・被キル・チーム）、スパイク設置の時刻、ラウンド開始の時刻

判定する勝因（重要な順にタグを並べ、先頭を「主な勝因」とする）
- エース（1 人で 5 キル）／クラッチ（最後の 1 人が 2 人以上を相手に勝つ）／数的不利からの逆転
- リテイク（守備側が設置後に取り返す：解除・全滅）
- ポストプラント成功（攻撃側が設置後に守り切る：爆破・全滅）
- ラッシュ（攻撃側が開始 30 秒以内に設置、または 20 秒以内に 2 キル以上で押し切る）
- カウンター（守備側が開始 20 秒以内に先手のキルを取り、設置させずに勝つ）
- 1ピック（勝ったチームが最初のキル）／先制されて逆転
- エコ勝ち（装備で劣る側の勝利）／アンチエコ（装備で勝る側が落とさなかった）
- 時間切れまで守り切り／設置前に全滅（攻撃側）／設置させずに守り勝ち（守備側）
- ピストル（各ハーフの最初のラウンド）
"""
from __future__ import annotations

ATTACK, DEFENSE = "攻撃", "防衛"
BUY_RANK = {"pistol": 0, "eco": 1, "semi-eco": 2, "semi-buy": 3, "full": 4}
PRIORITY = ["エース", "クラッチ", "リテイク", "ポストプラント成功", "ラッシュ", "カウンター", "数的不利から逆転",
            "エコ勝ち", "1ピック", "先制されて逆転", "時間切れまで守り切り", "設置前に全滅",
            "設置させずに守り勝ち", "アンチエコ", "ピストル"]


def _clock(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = max(0, int(round(sec)))
    return f"{sec // 60}:{sec % 60:02d}"


def analyze(r: dict, rep: dict) -> dict:
    """1 ラウンドぶん（report.json の rounds の要素）の勝因。返り値 {"main", "tags": [...], "summary", "lines": [...]}"""
    W = r.get("winner")
    if not W:
        return {}
    L = rep["team2"] if W == rep["team1"] else rep["team1"]
    tagW, tagL = (rep["tag1"], rep["tag2"]) if W == rep["team1"] else (rep["tag2"], rep["tag1"])
    w_side = r.get("side")
    atk = W if w_side == ATTACK else L if w_side == DEFENSE else None
    how = r.get("how")
    t0 = r.get("t_drop")
    kills = [k for k in (r.get("kills") or []) if k.get("killer_team") in (W, L)]
    plant = r.get("plant_t")
    if plant is None and how in ("boom", "defuse"):
        plant = -1.0                                           # 設置はしたが時刻は分からない
    tags: dict[str, str] = {}

    def el(t):                                                # 開始からの経過秒
        return None if t is None or t0 is None or t < 0 else t - t0

    # --- キルの流れ（5v5 から）
    alive = {W: 5, L: 5}
    worst, clutch_at, kw = 0, None, {}
    for k in kills:
        victim_team = L if k["killer_team"] == W else W
        alive[victim_team] = max(0, alive[victim_team] - 1)
        if k["killer_team"] == W:
            kw[k["killer"]] = kw.get(k["killer"], 0) + 1
        diff = alive[L] - alive[W]
        worst = max(worst, diff)
        if alive[W] == 1 and alive[L] >= 2 and clutch_at is None:
            clutch_at = (alive[L], k)
    deaths_W = {k["victim"] for k in kills if k["killer_team"] == L}

    # --- 個人
    best = max(kw.items(), key=lambda kv: kv[1]) if kw else None
    if best and best[1] >= 5:
        tags["エース"] = f"{best[0]} が 5 キル"
    if clutch_at:
        roster = [p["name"] for p in rep.get("players") or []
                  if (p.get("team") == 1 and W == rep["team1"]) or (p.get("team") == 2 and W == rep["team2"])]
        left = [n for n in roster if n not in deaths_W]
        who = left[0] if len(left) == 1 else "最後の 1 人"
        tags["クラッチ"] = f"{who} が 1v{clutch_at[0]} を制する"
    elif worst >= 2:
        tags["数的不利から逆転"] = f"最大 {worst} 人差の不利から勝利"
    if best and 3 <= best[1] < 5 and "エース" not in tags:
        tags.setdefault("_multi", f"{best[0]} が {best[1]} キル")

    # --- 先制
    if kills:
        fk = kills[0]
        when = el(fk.get("t"))
        who = f"{fk.get('killer')}→{fk.get('victim')}（{_clock(when)} 経過）"
        if fk["killer_team"] == W:
            tags["1ピック"] = who
        else:
            tags["先制されて逆転"] = who
        early = [k for k in kills if el(k.get("t")) is not None and el(k["t"]) <= 20]
        if atk == W and len([k for k in early if k["killer_team"] == W]) >= 2:
            tags["ラッシュ"] = f"開始 20 秒で {len([k for k in early if k['killer_team'] == W])} キル"
        if atk == L and fk["killer_team"] == W and when is not None and when <= 20 and plant is None:
            tags["カウンター"] = f"守備側が {_clock(when)} で先手、設置させずに勝利"

    # --- スパイク
    pe = el(plant) if plant and plant > 0 else None
    if atk == W:
        if plant is not None:
            post = [k for k in kills if plant > 0 and k.get("t") and k["t"] >= plant]
            pw = sum(1 for k in post if k["killer_team"] == W)
            detail = ("爆破" if how == "boom" else "設置後に全滅" if how == "elim" else "設置後に勝利")
            if pe is not None:
                detail += f"（設置 {_clock(pe)} 経過" + (f"、設置後 {pw} キル）" if post else "）")
            tags["ポストプラント成功"] = detail
            if pe is not None and pe <= 30:
                tags["ラッシュ"] = tags.get("ラッシュ") or f"開始 {_clock(pe)} で設置"
        elif how == "elim":
            tags["設置前に全滅"] = "設置せずに全員を倒す"
    elif atk == L:
        if how == "defuse" or (plant is not None and how == "elim"):
            post = [k for k in kills if plant and plant > 0 and k.get("t") and k["t"] >= plant]
            pw = sum(1 for k in post if k["killer_team"] == W)
            tags["リテイク"] = ("解除" if how == "defuse" else "設置後に全滅") + (f"（設置後 {pw} キル）" if post else "")
        elif how == "time":
            tags["時間切れまで守り切り"] = "設置させずに時間切れ"
        elif how == "elim" and "カウンター" not in tags:
            tags["設置させずに守り勝ち"] = "設置前に全員を倒す"

    # --- 経済・ピストル
    bw = BUY_RANK.get(r.get("eco1") if W == rep["team1"] else r.get("eco2"))
    bl = BUY_RANK.get(r.get("eco2") if W == rep["team1"] else r.get("eco1"))
    if bw is not None and bl is not None and bw > 0 and bl > 0:
        if bw <= 2 and bl >= 3:
            tags["エコ勝ち"] = "装備で劣る側が勝利"
        elif bw >= 4 and bl <= 2:
            tags["アンチエコ"] = "装備の差を活かして勝利"
    if r.get("n") in (1, 13):
        tags["ピストル"] = "ピストルラウンド"

    multi = tags.pop("_multi", None)
    order = [t for t in PRIORITY if t in tags]
    main = order[0] if order else None
    lines = [f"{t}：{tags[t]}" for t in order]
    if multi:
        lines.append(f"活躍：{multi}")
    summary = f"{tagW} の勝因：" + "・".join(order[:3]) if order else ""
    caveat = None
    expected = 5 if how == "elim" else None
    if expected and sum(1 for k in kills if k["killer_team"] == W) < expected:
        caveat = "キルログが一部しか読めていないため、人数の流れは推定です"
    return {"main": main, "tags": order, "summary": summary, "lines": lines, "caveat": caveat,
            "winner_tag": tagW, "loser_tag": tagL}
