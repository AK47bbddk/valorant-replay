"""ライブ判定とラウンド分割。

1. ライブ判定：画面上部のスコアバーの見た目を、タイマーとスコアが読めたフレームから作った「お手本」と比べる。
   リプレイ・選手カメラ・解説席・CM はスコアバーの形や色が違うので外れる（VCT CN / Riot 放送で確認）。
2. ラウンド番号：ライブのフレームのスコア合計＋1。スコアは減らないので、確定した増加だけを採用する。
3. ラウンド開始：購入フェーズ（タイマー 0:30→0:00）のあと、タイマーが 1:40 付近に跳んだ瞬間。
   見えていなければ、最初に読めた残り時間から逆算する。
4. ラウンド中のポーズ（タイマーが止まったまま）はライブ区間から除く。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median, mean

import numpy as np


@dataclass
class DetRound:
    n: int
    t_first: float               # このラウンドのライブ映像が最初に映った時刻
    t_drop: float                # ラウンド開始（バリア解除）
    t_end: float                 # スコアが更新された時刻（≒ラウンド終了）
    drop_estimated: bool = False
    start_conf: float = 0.0
    start_source: str = ""
    winner_side: str | None = None
    live: list[int] = field(default_factory=list)      # 開始〜終了のライブフレーム（ポーズ除く）
    buy_idx: int | None = None                        # 購入フェーズ終了直前のフレーム
    first_idx: int | None = None                      # 開始直後のフレーム


# ---------------------------------------------------------------- ライブ判定
def liveness(band: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """band: (N, h, w, 3) 上部バンドの縮小画像、conf: タイマーとスコアが読めたフレーム。"""
    N = len(band)
    if N == 0:
        return np.zeros(0, bool), np.zeros(0), "フレームなし"
    if conf.sum() < 8:
        return conf.copy(), np.zeros(N), f"ライブ判定はスコアの読み取りのみ（確実なフレーム {int(conf.sum())} 枚）"
    X = band.reshape(N, -1).astype(np.float32)

    def model(rows):
        C = X[rows]
        ref = np.median(C, axis=0)
        mad = np.median(np.abs(C - ref), axis=0)
        static = mad <= np.percentile(mad, 60)          # 数字など変わる部分を除いた、動かない部分
        return ref, static

    def dist(ref, static):
        return np.abs(X[:, static] - ref[static]).mean(axis=1)

    ref, st = model(conf)
    d = dist(ref, st)
    thr = max(float(np.percentile(d[conf], 90)) * 1.6, 10.0)
    far = conf & (d > thr)
    refs = 1
    if far.sum() >= 12:                                  # 後半でバーの見た目が変わる放送（左右入れ替え等）
        ref2, st2 = model(far)
        d = np.minimum(d, dist(ref2, st2))
        refs = 2
        thr = max(float(np.percentile(d[conf], 90)) * 1.6, 10.0)
    live = d <= thr
    return live, d, f"ライブ判定 {int(live.sum())}/{N} 枚（お手本 {refs} 種）"


# ---------------------------------------------------------------- ラウンド番号
def assign_rounds(live: np.ndarray, sl: list, sr: list, rnd: list, n_rounds: int | None = None,
                  need: int = 2, look: int = 4) -> list[int | None]:
    """ライブフレームごとのラウンド番号（スコア合計＋1、減らない・確定した増加だけ採用）。"""
    N = len(live)
    raw: list[int | None] = [None] * N
    for i in range(N):
        if not live[i]:
            continue
        if sl[i] is not None and sr[i] is not None:
            raw[i] = sl[i] + sr[i] + 1
        elif rnd[i] is not None:
            raw[i] = rnd[i]
    vals = [(i, v) for i, v in enumerate(raw) if v is not None]
    if not vals:
        return [None] * N
    maxn = (n_rounds or 99) + 1
    first = [v for _, v in vals[:7]]
    cur = int(median(first))
    out: list[int | None] = [None] * N
    k = 0
    for i in range(N):
        if not live[i]:
            continue
        while k < len(vals) and vals[k][0] < i:
            k += 1
        if raw[i] is not None and cur < raw[i] <= min(cur + 3, maxn):
            ahead = [v for _, v in vals[k:k + look]]
            if sum(v >= raw[i] for v in ahead) >= min(need, len(ahead)):
                cur = raw[i]
        out[i] = cur
    return out


def round_jumps(t: list[float], live, timer: list, round_sec: int = 100, buy_max: int = 30,
                max_gap: float = 25.0) -> list[int]:
    """ライブのフレームで、購入フェーズの表示（buy_max 秒以下）から試合中の表示（1:25 以上）へ跳んだ位置。
    各ラウンドに 1 回だけ起きるので、スコアが読めなくてもラウンドの数と区切りが分かる。"""
    out, prev = [], None
    for i in range(len(t)):
        if not live[i] or timer[i] is None:
            continue
        v = timer[i]
        if prev is not None and prev[1] <= buy_max + 2 and v >= round_sec - 15 and t[i] - t[prev[0]] <= max_gap:
            out.append(i)
        prev = (i, v)
    return out


def repair_rounds(t: list[float], live, rid: list, timer: list, n_rounds: int | None,
                  round_sec: int = 100, buy_max: int = 30, lead: float = 38.0) -> tuple[list, list[int]]:
    """スコアの読み違いで抜けたラウンドを、タイマーのジャンプから補う。

    スコア合計で決めたラウンド番号が、隣り合う 2 つのジャンプ（＝別々のラウンド）で同じになっていて、
    その前後の番号が抜けていれば、片方をその番号に付け直す。区間は「ジャンプの lead 秒前〜次のジャンプの lead 秒前」
    （購入フェーズ 30 秒＋ラウンド後の数秒）。返り値 (直した rid, 補ったラウンド番号)。
    """
    rid = list(rid)
    js = round_jumps(t, live, timer, round_sec, buy_max)
    present = {r for r in rid if r is not None}
    if len(js) < 2 or not present:
        return rid, []
    top = n_rounds or max(present)
    fixed: list[int] = []
    for _ in range(top):
        present = {r for r in rid if r is not None}
        missing = {n for n in range(1, top + 1) if n not in present}
        if not missing:
            break
        segs = []
        for k, j in enumerate(js):
            a = t[j] - lead
            b = t[js[k + 1]] - lead if k + 1 < len(js) else float("inf")
            idx = [i for i in range(len(t)) if a <= t[i] < b]
            near = [rid[i] for i in idx if live[i] and rid[i] is not None and t[j] <= t[i] < t[j] + 30]
            lab = max(set(near), key=near.count) if near else None
            segs.append((idx, lab))
        changed = False
        for k in range(len(segs)):
            idx, lab = segs[k]
            prev_lab = segs[k - 1][1] if k else 0
            next_lab = segs[k + 1][1] if k + 1 < len(segs) else None
            new = None
            if lab is not None and next_lab == lab and lab - 1 in missing and (prev_lab or 0) < lab - 1:
                new = lab - 1                      # 同じ番号が 2 回：前の方が 1 つ前のラウンド
            elif lab is not None and prev_lab == lab and lab + 1 in missing and (next_lab is None or next_lab > lab + 1):
                new = lab + 1                      # 同じ番号が 2 回：後の方が 1 つ後のラウンド
            elif lab is None and prev_lab and prev_lab + 1 in missing and (next_lab is None or next_lab > prev_lab + 1):
                new = prev_lab + 1                 # 番号が付かなかった区間
            if new is not None:
                for i in idx:
                    if live[i]:
                        rid[i] = new
                fixed.append(new)
                changed = True
                break
        if not changed:
            break
    return rid, sorted(fixed)


# ---------------------------------------------------------------- 分割
def _estimate_drop(pairs: list[tuple[float, float]], round_sec: int) -> tuple[float | None, float]:
    """残り時間の表示から開始時刻を逆算する。返り値 (開始時刻, ばらつき)。

    表示は残り秒の切り上げなので、各フレームの「時刻 −（制限時間 − 表示）」は真の開始時刻以上で、
    その 1 秒以内に入る。表示が止まって見えるフレーム（遅い更新・静止画・映像の遅れ）は後ろにずれるだけなので、
    下側の値を使う。OCR の読み違い 1 つに引っぱられないよう、近くに 2 つ以上そろう最小の値を採る。
    """
    if not pairs:
        return None, 9.0
    est = sorted(t - (round_sec - v) for t, v in pairs)
    need = min(2, len(est))
    lo = next((x for x in est if sum(x <= e <= x + 1.2 for e in est) >= need), est[0])
    near = [e for e in est if lo <= e <= lo + 1.5]
    spread = 9.0 if len(est) < 2 else (1.0 - len(near) / len(est)) * 3.0 + (near[-1] - lo) * 0.5
    return float(lo + 0.3), float(spread)


def _drop_time(ts: list[float], timers: list, round_sec: int, buy_max: int
               ) -> tuple[float | None, bool, int | None, float, str]:
    """(開始時刻, 推定か, 購入フェーズ最後の位置, 信頼度, 根拠)"""
    vals = [(j, t, v) for j, (t, v) in enumerate(zip(ts, timers)) if v is not None]
    if not vals:
        return None, True, None, 0.0, "none"

    def quality(spread: float, n: int) -> float:
        q = 0.95 if spread <= 0.35 else 0.85 if spread <= 0.8 else 0.7 if spread <= 1.5 else 0.5
        return q * min(1.0, 0.65 + n * 0.05)

    # 1:40 付近への跳ね上がり（購入タイマー -> ラウンドタイマー）
    jump_a = None
    for a, (_, _, v) in enumerate(vals):
        prev = vals[a - 1][2] if a else None
        if v >= round_sec - 15 and (a == 0 or prev <= buy_max + 2 or prev < v - 20):
            jump_a = a
            break
    # 購入タイマーはジャンプより前だけを見る（ラウンド終盤の残り秒やスパイクの秒数を購入フェーズと取り違えない）
    pre = vals[:jump_a] if jump_a is not None else vals
    buy = [(j, t, v) for j, t, v in pre if v <= buy_max + 2]
    buy_j = buy[-1][0] if buy else None
    buy_est = buy[-1][1] + buy[-1][2] if buy else None

    if jump_a is not None:
        # ジャンプ直後の数フレームだけを使う（後ろのフレームほど遅れや読み違いが混ざりやすい）
        nxt = [(t, v) for _, t, v in vals[jump_a:jump_a + 5] if buy_max * 0.3 < v <= round_sec + 2]
        jump_est, spread = _estimate_drop(nxt, round_sec)
        if jump_est is not None:
            drop, conf, source = jump_est, quality(spread, len(nxt)), "round-timer"
            if buy_est is not None:
                gap = abs(drop - buy_est)
                if gap <= 2.5:                   # 2つの根拠が一致 -> さらに正確
                    drop = (drop * 2 + buy_est) / 3
                    conf = min(0.98, conf + 0.08)
                    source = "round-timer+buy-timer"
                elif gap > 4.0:                  # 購入タイマーが別画面の可能性
                    conf *= 0.8
            return drop, conf < 0.72, buy_j, conf, source

    first_v = vals[0][2]
    if first_v <= buy_max + 2 and buy:
        # 購入フェーズだけ見えている：0:00 になる時刻を開始とする
        j, t, v = buy[-1]
        return t + v, v > 0.6, j, 0.82 if v <= 0.6 else 0.68, "buy-timer"
    if first_v > buy_max:
        # 途中から映った：残り時間から逆算（複数フレームの中央値）
        nxt = [(t, v) for _, t, v in vals[:6] if buy_max * 0.3 < v <= round_sec + 2]
        est, spread = _estimate_drop(nxt, round_sec)
        if est is not None:
            return est, True, buy_j, quality(spread, len(nxt)), "countdown"
    if buy_est is not None:
        return buy_est, True, buy_j, 0.45, "buy-timer-only"
    return ts[0], True, buy_j, 0.0, "fallback"


def segment(t: list[float], live: np.ndarray, rid: list, timer: list, sl: list, sr: list,
            n_rounds: int | None, round_sec: int = 100, buy_max: int = 30, pause_sec: float = 3.0,
            alive=None, conf=None) -> tuple[list[DetRound], dict]:
    N = len(t)
    stats = {"frames": N, "live": int(np.sum(live)),
             "timer_ok": sum(v is not None for v in timer) / max(1, N),
             "score_ok": sum(a is not None and b is not None for a, b in zip(sl, sr)) / max(1, N)}
    present = sorted({r for r in rid if r is not None})
    if n_rounds:
        present = [r for r in present if r <= n_rounds]
    rounds: list[DetRound] = []
    for n in present:
        idx = [i for i in range(N) if rid[i] == n]
        nxt = [i for i in range(N) if rid[i] is not None and rid[i] > n]
        t_end = t[nxt[0]] if nxt else t[idx[-1]] + 1.0
        ts = [t[i] for i in idx]
        drop, est, buy_j, start_conf, source = _drop_time(ts, [timer[i] for i in idx], round_sec, buy_max)
        if drop is None:
            drop, est, start_conf, source = ts[0], True, 0.0, "fallback"
        drop = max(ts[0] - 60.0, min(drop, t_end))
        r = DetRound(n=n, t_first=ts[0], t_drop=drop, t_end=t_end, drop_estimated=est,
                     start_conf=start_conf, start_source=source)

        def alive_n(i: int) -> int:
            try:
                return int(np.sum(alive[i]))
            except Exception:
                return 0

        def usable(i: int) -> bool:
            return conf is None or bool(conf[i])

        # 購入フェーズの最後ではなく、少し手前の安定したフレームを使う。
        # CN 放送の購入UI -> 試合UI の切り替えアニメーションを避ける。
        buys = [i for i in idx if drop - 12.0 <= t[i] <= drop - 1.2 and usable(i)]
        if buys:
            r.buy_idx = max(buys, key=lambda i: (alive_n(i), -abs(t[i] - (drop - 3.0)), t[i]))
        else:
            r.buy_idx = idx[buy_j] if buy_j is not None else None
        # ライブ区間（ポーズを除く）
        last_v, stall = None, None
        for i in idx:
            if t[i] < drop - 0.01:
                continue
            v = timer[i]
            if v is not None and last_v is not None and abs(v - last_v) < 0.05:
                stall = stall if stall is not None else t[i]
                if t[i] - stall > pause_sec:
                    continue
            else:
                stall = None
            if v is not None:
                last_v = v
            r.live.append(i)
        after = [i for i in r.live if drop + 1.5 <= t[i] <= drop + 8.0 and usable(i)]
        if after:
            r.first_idx = max(after, key=lambda i: (alive_n(i), -abs(t[i] - (drop + 2.5)), -t[i]))
        else:
            after = [i for i in r.live if t[i] >= drop + 1.0]
            r.first_idx = after[0] if after else (r.live[0] if r.live else None)
        # 勝った側：このラウンドと次のラウンドのスコア表示を比べる
        a = [(sl[i], sr[i]) for i in idx if sl[i] is not None and sr[i] is not None]
        b = [(sl[i], sr[i]) for i in nxt[:8] if sl[i] is not None and sr[i] is not None and rid[i] == n + 1]
        if a and b:
            dl = median(x for x, _ in b) - median(x for x, _ in a)
            dr = median(y for _, y in b) - median(y for _, y in a)
            r.winner_side = "L" if dl > dr else "R" if dr > dl else None
        rounds.append(r)
    stats["drop_estimated"] = sum(1 for r in rounds if r.drop_estimated)
    stats["drop_avg_conf"] = round(mean([r.start_conf for r in rounds]), 2) if rounds else 0.0
    return rounds, stats


def orientation(det: list[DetRound], vlr_winners: dict[int, int]) -> dict[int, bool]:
    """ラウンドごとに「画面左がチーム1か」。前半・後半・延長でまとめて多数決（放送で左右が入れ替わる場合に対応）。"""
    def half(n):
        return 0 if n <= 12 else 1 if n <= 24 else 2

    votes = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for r in det:
        w = vlr_winners.get(r.n)
        if w and r.winner_side:
            votes[half(r.n)][0 if (r.winner_side == "L") == (w == 1) else 1] += 1
    total = [sum(v[0] for v in votes.values()), sum(v[1] for v in votes.values())]
    glob = total[0] >= total[1]
    res = {}
    for r in det:
        v = votes[half(r.n)]
        res[r.n] = (v[0] >= v[1]) if sum(v) >= 2 else glob
    return res


def agreement(det: list[DetRound], vlr_winners: dict[int, int], orient: dict[int, bool]) -> float:
    ok = tot = 0
    for r in det:
        w = vlr_winners.get(r.n)
        if w and r.winner_side:
            tot += 1
            ok += ((r.winner_side == "L") == (w == 1)) == orient.get(r.n, True)
    return ok / tot if tot else 0.0
