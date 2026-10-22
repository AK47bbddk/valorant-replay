"""HUD の時系列（タイマー・スコア）からラウンドに分割し、リプレイやポーズを除外する。

考え方
- ラウンド番号 = 左スコア + 右スコア + 1。スコアは単調増加として扱い、確定的に増えた瞬間をラウンド終了とする。
- 購入フェーズ（タイマー ≤ 45 秒）のあと、タイマーが 1:40 付近へ跳んだ瞬間 = バリア解除（ラウンド開始）。
- ライブ判定：そのラウンドのスコアを映していて、タイマーが「最後に採用した値」より巻き戻っておらず、
  長時間止まってもいないサンプルだけを採用する。スコア確定後に古いスコアで流れるリプレイや、
  ラウンド中に挟まるリプレイ（タイマーが戻る）はここで落ちる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


@dataclass
class Sample:
    t: float
    timer: float | None
    sl: int | None
    sr: int | None

    @property
    def idx(self) -> int | None:
        return self.sl + self.sr if self.sl is not None and self.sr is not None else None


@dataclass
class DetRound:
    n: int
    t_start: float
    t_drop: float
    t_end: float
    drop_estimated: bool = False
    winner_side: str | None = None          # "L" / "R"
    live: list[int] = field(default_factory=list)   # 採用したサンプルの添字
    buy_idx: int | None = None              # 購入フェーズ終了直前のサンプル


def _smooth_idx(samples: list[Sample], k: int = 2) -> list[int | None]:
    raw = [s.idx for s in samples]
    out = []
    for i in range(len(raw)):
        win = [v for v in raw[max(0, i - k): i + k + 1] if v is not None]
        out.append(int(median(win)) if raw[i] is not None and win else None)
    return out


def _monotonic(samples: list[Sample], sm: list[int | None], need: int = 4) -> tuple[int | None, dict[int, float]]:
    """確定したスコア合計の推移。返り値: (最初の値, {新しい合計値: 変化した時刻})"""
    valid = [i for i, v in enumerate(sm) if v is not None]
    if not valid:
        return None, {}
    changes: dict[int, float] = {}
    cur = None
    for j, i in enumerate(valid):
        v = sm[i]
        ahead = [sm[x] for x in valid[j: j + need]]
        if cur is None:
            if sum(a == v for a in ahead) >= min(need, len(ahead)) - 1:
                cur = v
            continue
        if cur < v <= cur + 2:
            ok = sum(a is not None and a >= v for a in ahead)
            if ok >= max(1, min(need, len(ahead)) - 1):
                for x in range(cur + 1, v + 1):
                    changes[x] = samples[i].t
                cur = v
    first = next((sm[i] for i in valid), None)
    if changes:
        first = min(changes) - 1
    return first, changes


def _side_of_change(samples: list[Sample], t: float, w: float = 4.0) -> str | None:
    before = [s for s in samples if t - w <= s.t < t and s.sl is not None and s.sr is not None]
    after = [s for s in samples if t <= s.t < t + w and s.sl is not None and s.sr is not None]
    if not before or not after:
        return None
    dl = median(s.sl for s in after) - median(s.sl for s in before)
    dr = median(s.sr for s in after) - median(s.sr for s in before)
    if dl > dr:
        return "L"
    if dr > dl:
        return "R"
    return None


def segment(samples: list[Sample], round_sec: int = 100, buy_max: int = 45,
            pause_sec: float = 3.0) -> tuple[list[DetRound], dict]:
    if not samples:
        return [], {"timer_ok": 0.0, "score_ok": 0.0}
    sm = _smooth_idx(samples)
    first, changes = _monotonic(samples, sm)
    stats = {
        "timer_ok": sum(s.timer is not None for s in samples) / len(samples),
        "score_ok": sum(s.idx is not None for s in samples) / len(samples),
    }
    if first is None:
        return [], stats

    last_live_t = max((s.t for s in samples if s.timer is not None), default=samples[-1].t)
    rounds: list[DetRound] = []
    n = first + 1
    t_start = samples[0].t
    while True:
        t_end = changes.get(n)
        if t_end is None:
            # マップ最終ラウンドなどでスコア変化が見えない：最後のタイマー表示まで
            if t_start >= last_live_t - 10:
                break
            t_end = last_live_t + 2
        ids = [i for i, s in enumerate(samples) if t_start <= s.t < t_end]
        rounds.append(_build_round(samples, sm, ids, n, t_start, t_end, round_sec, buy_max, pause_sec))
        if n not in changes:
            break
        rounds[-1].winner_side = _side_of_change(samples, t_end)
        t_start = t_end
        n += 1
    return rounds, stats


def _build_round(samples, sm, ids, n, t_start, t_end, round_sec, buy_max, pause_sec) -> DetRound:
    want = n - 1
    mine = [i for i in ids if sm[i] in (want, None)]
    # バリア解除：このラウンドのスコアを映したサンプルで、タイマーが 1:40 付近に跳んだ最初の点
    drop_i = next((i for i in mine if samples[i].timer is not None and samples[i].timer >= round_sec - 15
                   and sm[i] == want), None)
    if drop_i is not None:
        t_drop = samples[drop_i].t - (round_sec - samples[drop_i].timer)
        est = False
    else:
        t_drop = t_start + min(30.0, (t_end - t_start) / 3)
        est = True
    r = DetRound(n=n, t_start=t_start, t_drop=max(t_start, t_drop), t_end=t_end, drop_estimated=est)

    buy = [i for i in mine if samples[i].t < r.t_drop and sm[i] == want]
    r.buy_idx = buy[-1] if buy else None

    last_timer, stall_since = None, None
    for i in ids:
        s = samples[i]
        if s.t < r.t_drop:
            continue
        if sm[i] is not None and sm[i] != want:
            continue                      # 別ラウンドのスコア＝リプレイ
        if s.timer is not None:
            if s.timer > round_sec + 1:
                continue
            if last_timer is not None and s.timer > last_timer + 1.5:
                continue                  # タイマーが巻き戻った＝リプレイ
            if last_timer is not None and abs(s.timer - last_timer) < 0.05:
                stall_since = stall_since if stall_since is not None else s.t
                if s.t - stall_since > pause_sec:
                    continue              # 止まっている＝ポーズ
            else:
                stall_since = None
            last_timer = s.timer
        elif last_timer is None:
            continue                      # まだタイマーを一度も確認していない
        r.live.append(i)
    return r


def orientation(det: list[DetRound], vlr_winners: dict[int, int]) -> tuple[bool, float]:
    """画面左がチーム1か。返り値: (team1_left, 一致率)"""
    agree = disagree = 0
    for r in det:
        w = vlr_winners.get(r.n)
        if not w or not r.winner_side:
            continue
        if (r.winner_side == "L") == (w == 1):
            agree += 1
        else:
            disagree += 1
    tot = agree + disagree
    if tot == 0:
        return True, 0.0
    return agree >= disagree, max(agree, disagree) / tot
