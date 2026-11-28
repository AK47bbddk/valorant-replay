"""Pure-Python + NumPy ±2s temporal correction for minimap agent dots and dome smokes.

Provides:
  - ``min_cost_match``: Exact O(N^3) Hungarian (Kuhn-Munkres) max-cardinality,
    min-cost bipartite matching within a distance threshold (zero ``scipy`` dependency).
  - ``correct_agent_dots_sequence``: Conservative 1-frame low-score FP suppression
    followed by ±2s Hungarian overlap/dropout bridging with single-pass low-threshold
    candidate snapping and optional playable-mask / alive-count guards.
  - ``correct_smokes_sequence``: Isolated 1-frame smoke FP suppression and 1-frame
    dropout bridging.
  - ``analyze_round_frames``: End-to-end 1-round frame-sequence helper combining
    exact bundled map-alpha registration, single-pass agent+low detection, smoke
    detection, and ±2s temporal correction.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _dist(a: tuple[float, float], b: tuple[float, float], aspect: float = 1.0) -> float:
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def _hungarian_square(cost: list[list[float]]) -> list[int]:
    """Exact O(N^3) Hungarian algorithm on an N×N cost matrix.

    Returns ``row_to_col`` of length N where ``row_to_col[i] = j``.
    """
    n = len(cost)
    if n == 0:
        return []
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    inf = float("inf")

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        row_i = cost[i - 1]
        while True:
            used[j0] = True
            i0 = p[j0]
            row_i0 = cost[i0 - 1]
            delta = inf
            j1 = 0
            u_i0 = u[i0]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row_i0[j - 1] - u_i0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
        _ = row_i

    row_to_col = [-1] * n
    for j in range(1, n + 1):
        if p[j] > 0:
            row_to_col[p[j] - 1] = j - 1
    return row_to_col


def min_cost_match(
    pts_a: list[tuple[float, float]],
    pts_b: list[tuple[float, float]],
    max_dist: float,
    aspect: float = 1.0,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Exact O(N^3) Hungarian max-cardinality, min-cost bipartite matching.

    Matches points in ``pts_a`` to points in ``pts_b`` with Euclidean distance
    (with ``x`` scaled by ``aspect``) ``<= max_dist``. Maximizes the number of
    matched pairs first, then minimizes total matched distance.

    Returns ``(matches, unmatched_a, unmatched_b)`` where:
      - ``matches``: list of ``(idx_a, idx_b, dist)`` sorted by ``(idx_a, idx_b)``
      - ``unmatched_a``: sorted list of unmatched indices in ``pts_a``
      - ``unmatched_b``: sorted list of unmatched indices in ``pts_b``
    """
    na = len(pts_a)
    nb = len(pts_b)
    md = _finite_float(max_dist, -1.0)
    if na == 0 or nb == 0 or md is None or md < 0.0:
        return [], list(range(na)), list(range(nb))

    asp = _finite_float(aspect, 1.0)
    if asp is None or asp <= 0.0:
        asp = 1.0

    # Transpose when na > nb so that real_rows <= cols; dummy rows have cost 0.0
    # and never distort the max-cardinality / min-cost objective on real rows.
    transposed = na > nb
    if transposed:
        rows_pts, cols_pts = pts_b, pts_a
    else:
        rows_pts, cols_pts = pts_a, pts_b

    nr, nc = len(rows_pts), len(cols_pts)
    big = (md + 1.0) * (nc + 1) * 10.0

    dist_mat: list[list[float]] = []
    cost_mat: list[list[float]] = []
    for i in range(nr):
        rx = _finite_float(rows_pts[i][0])
        ry = _finite_float(rows_pts[i][1])
        d_row: list[float] = []
        c_row: list[float] = []
        for j in range(nc):
            cx = _finite_float(cols_pts[j][0])
            cy = _finite_float(cols_pts[j][1])
            if rx is None or ry is None or cx is None or cy is None:
                d_row.append(float("inf"))
                c_row.append(big)
            else:
                d = math.hypot((rx - cx) * asp, ry - cy)
                d_row.append(d)
                c_row.append(d if d <= md else big)
        dist_mat.append(d_row)
        cost_mat.append(c_row)
    for _ in range(nc - nr):
        cost_mat.append([0.0] * nc)

    row_to_col = _hungarian_square(cost_mat)

    matches: list[tuple[int, int, float]] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i in range(nr):
        j = row_to_col[i]
        if 0 <= j < nc and dist_mat[i][j] <= md:
            d = dist_mat[i][j]
            if transposed:
                ia, ib = j, i
            else:
                ia, ib = i, j
            matches.append((ia, ib, d))
            matched_a.add(ia)
            matched_b.add(ib)

    matches.sort(key=lambda m: (m[0], m[1]))
    unmatched_a = [i for i in range(na) if i not in matched_a]
    unmatched_b = [j for j in range(nb) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


def _clamp_to_mask(
    x: float,
    y: float,
    mask_bool: np.ndarray,
    *,
    max_snap: float = 0.03,
    aspect: float = 1.0,
) -> tuple[float, float] | None:
    """Return ``(x, y)`` if playable in ``mask_bool``, else nearest playable pixel within ``max_snap``."""
    h, w = mask_bool.shape
    px = min(w - 1, max(0, int(round(x * w))))
    py = min(h - 1, max(0, int(round(y * h))))
    if bool(mask_bool[py, px]):
        return float(x), float(y)

    asp = aspect if aspect > 1e-6 else 1.0
    rx = int(math.ceil(max_snap * w / asp)) + 1
    ry = int(math.ceil(max_snap * h)) + 1
    x0 = max(0, px - rx)
    x1 = min(w, px + rx + 1)
    y0 = max(0, py - ry)
    y1 = min(h, py + ry + 1)
    sub = mask_bool[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if ys.size == 0:
        return None
    nx = (xs.astype(np.float64) + x0 + 0.5) / float(w)
    ny = (ys.astype(np.float64) + y0 + 0.5) / float(h)
    d2 = ((nx - x) * asp) ** 2 + (ny - y) ** 2
    best = int(np.argmin(d2))
    if float(d2[best]) <= max_snap * max_snap:
        return float(nx[best]), float(ny[best])
    return None


def _copy_valid_dots(dots: Any) -> list[dict[str, Any]]:
    if not isinstance(dots, list):
        return []
    out: list[dict[str, Any]] = []
    for d in dots:
        if not isinstance(d, dict):
            continue
        c = d.get("color")
        if c not in ("warm", "cool"):
            continue
        x = _finite_float(d.get("x"))
        y = _finite_float(d.get("y"))
        if x is None or y is None:
            continue
        cp = dict(d)
        cp["x"] = x
        cp["y"] = y
        out.append(cp)
    return out


def _copy_valid_smokes(smokes: Any) -> list[dict[str, Any]]:
    if not isinstance(smokes, list):
        return []
    out: list[dict[str, Any]] = []
    for s in smokes:
        if not isinstance(s, dict):
            continue
        x = _finite_float(s.get("x"))
        y = _finite_float(s.get("y"))
        if x is None or y is None:
            continue
        cp = dict(s)
        cp["x"] = x
        cp["y"] = y
        out.append(cp)
    return out


def correct_agent_dots_sequence(
    frames: list[dict[str, Any]],
    *,
    per_color: int = 5,
    aspect: float = 1.0,
    speed: float = 0.065,
    slack: float = 0.04,
    max_dt: float = 3.5,
    snap_radius: float = 0.055,
    suppress_lone_below: float | None = 0.78,
    dot_mask: np.ndarray | None = None,
    alive_counts: list[dict[str, int] | None] | None = None,
) -> list[list[dict[str, Any]]]:
    """Apply ±2s temporal correction to a 1-round sequence of anonymous agent dot detections.

    Never mutates ``frames`` or any input dot dict.
    """
    if not isinstance(frames, list) or not frames:
        return []

    n = len(frames)
    cap_default = max(0, int(per_color))
    asp = _finite_float(aspect, 1.0)
    if asp is None or asp <= 0.0:
        asp = 1.0
    spd = max(0.0, _finite_float(speed, 0.065) or 0.0)
    slk = max(0.0, _finite_float(slack, 0.04) or 0.0)
    mdt = max(0.0, _finite_float(max_dt, 3.5) or 0.0)
    snap_r = max(0.0, _finite_float(snap_radius, 0.055) or 0.0)
    supp_thr = _finite_float(suppress_lone_below) if suppress_lone_below is not None else None

    mask_bool: np.ndarray | None = None
    if isinstance(dot_mask, np.ndarray) and dot_mask.ndim == 2 and dot_mask.size > 0:
        mask_bool = dot_mask.astype(bool)

    ts: list[float] = []
    valid: list[bool] = []
    orig_dots: list[list[dict[str, Any]]] = []
    low_dots: list[list[dict[str, Any]]] = []

    for fr in frames:
        if not isinstance(fr, dict):
            ts.append(0.0)
            valid.append(False)
            orig_dots.append([])
            low_dots.append([])
            continue
        t_val = _finite_float(fr.get("t"))
        v_flag = bool(fr.get("valid", True)) and (t_val is not None)
        ts.append(t_val if t_val is not None else 0.0)
        valid.append(v_flag)
        orig_dots.append(_copy_valid_dots(fr.get("dots")))
        low_dots.append(_copy_valid_dots(fr.get("low_dots")))

    def _max_step(dt: float) -> float:
        return min(0.28, spd * dt + slk)

    def _max_bridge(dt_total: float) -> float:
        return min(0.32, spd * dt_total + slk)

    def _interior_dts(idx: int) -> tuple[float, float] | None:
        if idx <= 0 or idx >= n - 1:
            return None
        if not (valid[idx] and valid[idx - 1] and valid[idx + 1]):
            return None
        dt_prev = ts[idx] - ts[idx - 1]
        dt_next = ts[idx + 1] - ts[idx]
        if not (0.0 < dt_prev <= mdt and 0.0 < dt_next <= mdt):
            return None
        return dt_prev, dt_next

    # Split snapshot by color
    orig_by_color: dict[str, list[list[dict[str, Any]]]] = {
        "warm": [[d for d in orig_dots[i] if d["color"] == "warm"] for i in range(n)],
        "cool": [[d for d in orig_dots[i] if d["color"] == "cool"] for i in range(n)],
    }

    # Pass 1: Conservative lone 1-frame low-score FP suppression
    after_p1: dict[str, list[list[dict[str, Any]]]] = {
        "warm": [[dict(d) for d in orig_by_color["warm"][i]] for i in range(n)],
        "cool": [[dict(d) for d in orig_by_color["cool"][i]] for i in range(n)],
    }

    if supp_thr is not None:
        for i in range(1, n - 1):
            dts = _interior_dts(i)
            if dts is None:
                continue
            dt_prev, dt_next = dts
            s_prev = _max_step(dt_prev)
            s_next = _max_step(dt_next)
            for c in ("warm", "cool"):
                prev_c = orig_by_color[c][i - 1]
                next_c = orig_by_color[c][i + 1]
                kept: list[dict[str, Any]] = []
                for d in orig_by_color[c][i]:
                    sc = _finite_float(d.get("score")) if "score" in d else None
                    if sc is not None and sc < supp_thr:
                        pt = (d["x"], d["y"])
                        near_prev = any(_dist(pt, (p["x"], p["y"]), asp) <= s_prev for p in prev_c)
                        near_next = any(_dist(pt, (q["x"], q["y"]), asp) <= s_next for q in next_c)
                        if not near_prev and not near_next:
                            continue
                    kept.append(dict(d))
                after_p1[c][i] = kept

    # Pass 2: Hungarian ±2s overlap & dropout bridging + low-thresh snap
    result: list[list[dict[str, Any]]] = [
        [dict(d) for d in after_p1["warm"][i]] + [dict(d) for d in after_p1["cool"][i]]
        for i in range(n)
    ]

    for i in range(1, n - 1):
        dts = _interior_dts(i)
        if dts is None:
            continue
        dt_prev, dt_next = dts
        step_prev = _max_step(dt_prev)
        step_next = _max_step(dt_next)
        step_bridge = _max_bridge(dt_prev + dt_next)
        w_t = dt_prev / (dt_prev + dt_next)

        all_accepted_at_i = list(result[i])

        for c in ("warm", "cool"):
            prev_c = after_p1[c][i - 1]
            cur_c = [dict(d) for d in after_p1[c][i]]
            next_c = after_p1[c][i + 1]

            cap = cap_default
            if alive_counts is not None and i < len(alive_counts):
                ac_i = alive_counts[i]
                if isinstance(ac_i, dict) and c in ac_i and ac_i[c] is not None:
                    try:
                        cap = min(cap, max(0, int(ac_i[c])))
                    except (TypeError, ValueError):
                        pass

            slots = max(0, cap - len(cur_c))
            if slots <= 0 or not prev_c or not next_c:
                continue

            pts_prev = [(d["x"], d["y"]) for d in prev_c]
            pts_cur = [(d["x"], d["y"]) for d in cur_c]
            pts_next = [(d["x"], d["y"]) for d in next_c]

            _, unm_prev, _ = min_cost_match(pts_prev, pts_cur, step_prev, aspect=asp)
            _, _, unm_next = min_cost_match(pts_cur, pts_next, step_next, aspect=asp)
            if not unm_prev or not unm_next:
                continue

            pts_up = [pts_prev[idx] for idx in unm_prev]
            pts_un = [pts_next[idx] for idx in unm_next]
            bridges, _, _ = min_cost_match(pts_up, pts_un, step_bridge, aspect=asp)
            if not bridges:
                continue

            bridges.sort(key=lambda b: b[2])
            low_c = [d for d in low_dots[i] if d["color"] == c]
            used_low: set[int] = set()
            added = 0

            for bi, bj, _bdist in bridges:
                if added >= slots:
                    break
                p = prev_c[unm_prev[bi]]
                q = next_c[unm_next[bj]]
                mx = p["x"] + w_t * (q["x"] - p["x"])
                my = p["y"] + w_t * (q["y"] - p["y"])

                best_li = -1
                best_lsc = -1.0
                for li, lc in enumerate(low_c):
                    if li in used_low:
                        continue
                    lpt = (lc["x"], lc["y"])
                    if _dist(lpt, (mx, my), asp) > snap_r:
                        continue
                    if any(_dist(lpt, (e["x"], e["y"]), asp) <= 0.015 for e in cur_c):
                        continue
                    lsc = _finite_float(lc.get("score"), 0.0) or 0.0
                    if lsc > best_lsc:
                        best_lsc = lsc
                        best_li = li

                if best_li >= 0:
                    cand = low_c[best_li]
                    rx, ry = cand["x"], cand["y"]
                    snapped = True
                    used_low.add(best_li)
                else:
                    cand = None
                    rx, ry = mx, my
                    snapped = False

                if mask_bool is not None:
                    clamped = _clamp_to_mask(rx, ry, mask_bool, max_snap=0.03, aspect=asp)
                    if clamped is None:
                        continue
                    rx, ry = clamped

                is_overlap = any(_dist((rx, ry), (e["x"], e["y"]), asp) <= 0.06 for e in all_accepted_at_i)
                if is_overlap:
                    tag = "overlap_snap" if snapped else "overlap_interp"
                else:
                    tag = "snap_low" if snapped else "interp"

                p_sc = _finite_float(p.get("score"), 0.8) or 0.8
                q_sc = _finite_float(q.get("score"), 0.8) or 0.8
                base_sc = 0.85 * min(p_sc, q_sc)
                if snapped and cand is not None:
                    cand_sc = _finite_float(cand.get("score"), 0.0) or 0.0
                    sc = round(max(cand_sc, base_sc), 4)
                else:
                    sc = round(base_sc, 4)

                area = int(round(0.5 * ((_finite_float(p.get("area"), 100.0) or 100.0) +
                                        (_finite_float(q.get("area"), 100.0) or 100.0))))
                w_px = int(round(0.5 * ((_finite_float(p.get("w"), 12.0) or 12.0) +
                                        (_finite_float(q.get("w"), 12.0) or 12.0))))
                h_px = int(round(0.5 * ((_finite_float(p.get("h"), 12.0) or 12.0) +
                                        (_finite_float(q.get("h"), 12.0) or 12.0))))

                new_dot: dict[str, Any] = {
                    "color": c,
                    "x": float(rx),
                    "y": float(ry),
                    "area": max(1, area),
                    "k": 1,
                    "w": max(1, w_px),
                    "h": max(1, h_px),
                    "score": sc,
                    "temporal": tag,
                }
                cur_c.append(new_dot)
                result[i].append(new_dot)
                all_accepted_at_i.append(new_dot)
                added += 1

    # Pass 3: Enforce alive_counts upper bounds (per-color and/or total)
    if alive_counts is not None:
        def _dot_priority(d: dict[str, Any]) -> tuple[int, float]:
            is_obs = 0 if str(d.get("temporal") or "") in ("interp", "overlap_interp") else 1
            sc = _finite_float(d.get("score"), 0.5)
            return (is_obs, sc if sc is not None else 0.5)

        for i in range(n):
            if i >= len(alive_counts) or not isinstance(alive_counts[i], dict):
                continue
            ac = alive_counts[i]
            cur = result[i]
            for c_key in ("warm", "cool"):
                if c_key in ac and ac[c_key] is not None:
                    try:
                        lim = max(0, int(ac[c_key]))
                    except (TypeError, ValueError):
                        continue
                    c_dots = [d for d in cur if d.get("color") == c_key]
                    if len(c_dots) > lim:
                        c_dots.sort(key=_dot_priority, reverse=True)
                        keep_ids = {id(d) for d in c_dots[:lim]}
                        cur = [d for d in cur if d.get("color") != c_key or id(d) in keep_ids]
            if "total" in ac and ac["total"] is not None:
                try:
                    t_lim = max(0, int(ac["total"]))
                except (TypeError, ValueError):
                    t_lim = None
                if t_lim is not None and len(cur) > t_lim:
                    cur.sort(key=_dot_priority, reverse=True)
                    cur = cur[:t_lim]
            result[i] = cur

    return result


# 同時展開しうるドームスモーク数の上限（コントローラー＋クラウドバースト）
DOME_SMOKE_CAPS: dict[str, int] = {
    "omen": 2,
    "astra": 2,
    "brimstone": 3,
    "clove": 2,
    "viper": 1,
    "harbor": 1,
    "miks": 2,
    "jett": 2,
}


def max_dome_smokes_for_comp(agents: list[str] | None) -> int | None:
    """両チームの出場エージェント構成から同時展開しうるドームスモーク上限を返す。

    構成情報が空/未指定の場合は制約なし（``None``）を返す。
    """
    if not agents:
        return None
    cleaned = [str(a).strip().lower() for a in agents if a and str(a).strip()]
    if not cleaned:
        return None
    return sum(DOME_SMOKE_CAPS.get(a, 0) for a in cleaned)


def correct_smokes_sequence(
    frames: list[dict[str, Any]],
    *,
    match_dist: float = 0.05,
    max_dt: float = 3.5,
    suppress_lone_conf: float = 0.90,
    bridge_min_conf: float = 0.55,
    aspect: float = 1.0,
    min_duration_sec: float = 0.0,
    max_duration_sec: float = 0.0,
    controller_agents: list[str] | None = None,
    max_active_smokes: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Apply ±2s temporal consistency, stationary duration filtering, and comp caps to smokes.

    Never mutates ``frames`` or any input smoke dict.
    """
    if not isinstance(frames, list) or not frames:
        return []

    n = len(frames)
    asp = _finite_float(aspect, 1.0)
    if asp is None or asp <= 0.0:
        asp = 1.0
    mdist = max(0.0, _finite_float(match_dist, 0.05) or 0.0)
    mdt = max(0.0, _finite_float(max_dt, 3.5) or 0.0)
    supp_c = _finite_float(suppress_lone_conf, 0.90)
    if supp_c is None:
        supp_c = 0.90
    bridge_c = _finite_float(bridge_min_conf, 0.55)
    if bridge_c is None:
        bridge_c = 0.55
    min_dur = max(0.0, _finite_float(min_duration_sec, 0.0) or 0.0)
    max_dur = max(0.0, _finite_float(max_duration_sec, 0.0) or 0.0)

    ts: list[float] = []
    valid: list[bool] = []
    orig_smokes: list[list[dict[str, Any]]] = []
    for fr in frames:
        if not isinstance(fr, dict):
            ts.append(0.0)
            valid.append(False)
            orig_smokes.append([])
            continue
        t_val = _finite_float(fr.get("t"))
        v_flag = bool(fr.get("valid", True)) and (t_val is not None)
        ts.append(t_val if t_val is not None else 0.0)
        valid.append(v_flag)
        orig_smokes.append(_copy_valid_smokes(fr.get("smokes")))

    def _interior_dts(idx: int) -> tuple[float, float] | None:
        if idx <= 0 or idx >= n - 1:
            return None
        if not (valid[idx] and valid[idx - 1] and valid[idx + 1]):
            return None
        dt_prev = ts[idx] - ts[idx - 1]
        dt_next = ts[idx + 1] - ts[idx]
        if not (0.0 < dt_prev <= mdt and 0.0 < dt_next <= mdt):
            return None
        return dt_prev, dt_next

    # Pass 1: Isolated 1-frame FP suppression
    after_p1: list[list[dict[str, Any]]] = [[dict(s) for s in orig_smokes[i]] for i in range(n)]
    for i in range(1, n - 1):
        if _interior_dts(i) is None:
            continue
        prev_s = orig_smokes[i - 1]
        next_s = orig_smokes[i + 1]
        kept: list[dict[str, Any]] = []
        for s in orig_smokes[i]:
            conf = _finite_float(s.get("conf"), 0.0) or 0.0
            if conf < supp_c:
                pt = (s["x"], s["y"])
                near_prev = any(_dist(pt, (p["x"], p["y"]), asp) <= mdist for p in prev_s)
                near_next = any(_dist(pt, (q["x"], q["y"]), asp) <= mdist for q in next_s)
                if not near_prev and not near_next:
                    continue
            kept.append(dict(s))
        after_p1[i] = kept

    # Pass 2: 1-frame dropout bridging
    result: list[list[dict[str, Any]]] = [[dict(s) for s in after_p1[i]] for i in range(n)]
    for i in range(1, n - 1):
        dts = _interior_dts(i)
        if dts is None:
            continue
        dt_prev, dt_next = dts
        w_t = dt_prev / (dt_prev + dt_next)
        prev_s = after_p1[i - 1]
        next_s = after_p1[i + 1]
        if not prev_s or not next_s:
            continue
        pts_prev = [(s["x"], s["y"]) for s in prev_s]
        pts_next = [(s["x"], s["y"]) for s in next_s]
        matches, _, _ = min_cost_match(pts_prev, pts_next, mdist, aspect=asp)
        for pi, qi, _ in matches:
            p = prev_s[pi]
            q = next_s[qi]
            p_conf = _finite_float(p.get("conf"), 0.0) or 0.0
            q_conf = _finite_float(q.get("conf"), 0.0) or 0.0
            if 0.5 * (p_conf + q_conf) < bridge_c:
                continue
            mx = p["x"] + w_t * (q["x"] - p["x"])
            my = p["y"] + w_t * (q["y"] - p["y"])
            if any(_dist((mx, my), (e["x"], e["y"]), asp) <= mdist for e in result[i]):
                continue
            pr = _finite_float(p.get("r"), 0.03) or 0.03
            qr = _finite_float(q.get("r"), 0.03) or 0.03
            pa = _finite_float(p.get("area"), 0.0) or 0.0
            qa = _finite_float(q.get("area"), 0.0) or 0.0
            bridged = dict(p)
            bridged.update({
                "kind": str(p.get("kind", "smoke")),
                "subkind": str(p.get("subkind", "dome")),
                "x": float(mx),
                "y": float(my),
                "r": float(0.5 * (pr + qr)),
                "zone": p.get("zone") or q.get("zone"),
                "area": int(round(0.5 * (pa + qa))),
                "team": p.get("team") if p.get("team") == q.get("team") else None,
                "conf": round(0.85 * min(p_conf, q_conf), 3),
                "temporal": "interp",
            })
            result[i].append(bridged)
        result[i].sort(key=lambda s: float(_finite_float(s.get("conf"), 0.0) or 0.0), reverse=True)

    # Pass 3: Stationary persistence duration filter (dome smokes persist ~13.5-19.25s, never > max_dur)
    if (min_dur > 0.0 or max_dur > 0.0) and n >= 2:
        tracks: list[list[tuple[int, int]]] = []
        active_tid: dict[tuple[int, int], int] = {}
        for i in range(n):
            if not valid[i] or not result[i]:
                continue
            prev_i = i - 1
            can_link = (
                prev_i >= 0
                and valid[prev_i]
                and bool(result[prev_i])
                and 0.0 < (ts[i] - ts[prev_i]) <= mdt
            )
            matched_cur: set[int] = set()
            if can_link:
                pts_p = [(s["x"], s["y"]) for s in result[prev_i]]
                pts_c = [(s["x"], s["y"]) for s in result[i]]
                m_pairs, _, _ = min_cost_match(pts_p, pts_c, mdist, aspect=asp)
                for pi, ci, _ in m_pairs:
                    tid = active_tid.get((prev_i, pi))
                    if tid is None:
                        tid = len(tracks)
                        tracks.append([(prev_i, pi)])
                        active_tid[(prev_i, pi)] = tid
                    tracks[tid].append((i, ci))
                    active_tid[(i, ci)] = tid
                    matched_cur.add(ci)
            for ci in range(len(result[i])):
                if ci not in matched_cur:
                    tid = len(tracks)
                    tracks.append([(i, ci)])
                    active_tid[(i, ci)] = tid

        drop_nodes: set[tuple[int, int]] = set()
        for tr in tracks:
            t_span = ts[tr[-1][0]] - ts[tr[0][0]]
            peak_c = max(float(_finite_float(result[fi][si].get("conf"), 0.0) or 0.0) for fi, si in tr)
            if min_dur > 0.0 and t_span + 1e-6 < min_dur and peak_c < supp_c:
                drop_nodes.update(tr)
            if max_dur > 0.0 and t_span > max_dur + 1e-6:
                drop_nodes.update(tr)
        if drop_nodes:
            for i in range(n):
                result[i] = [s for si, s in enumerate(result[i]) if (i, si) not in drop_nodes]

    # Pass 4: Controller composition / simultaneous smoke upper bound
    smoke_cap = max_active_smokes if max_active_smokes is not None else max_dome_smokes_for_comp(controller_agents)
    if smoke_cap is not None:
        smoke_cap = max(0, int(smoke_cap))
        for i in range(n):
            if not result[i]:
                continue
            if smoke_cap == 0:
                result[i] = [
                    s for s in result[i]
                    if (float(_finite_float(s.get("conf"), 0.0) or 0.0)) >= 0.96
                ]
            elif len(result[i]) > smoke_cap:
                result[i].sort(
                    key=lambda s: (
                        0 if str(s.get("temporal") or "") == "interp" else 1,
                        float(_finite_float(s.get("conf"), 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                result[i] = result[i][:smoke_cap]

    return result


def analyze_round_frames(
    frames_bgr: list[np.ndarray],
    timestamps: list[float],
    *,
    map_name: str | None = None,
    bg_bgr: np.ndarray | None = None,
    per_color: int = 5,
    include_smokes: bool = True,
    low_thresh: float = 0.25,
) -> dict[str, Any]:
    """Run exact-mask registration, single-pass detection, and ±2s temporal correction on a 1-round frame list."""
    empty_summary = {
        "n_frames": 0,
        "valid_frames": 0,
        "raw_agents": 0,
        "corrected_agents": 0,
        "recovered_agents": 0,
        "suppressed_agents": 0,
        "raw_smokes": 0,
        "corrected_smokes": 0,
        "bridged_smokes": 0,
        "suppressed_smokes": 0,
    }
    if not frames_bgr or not timestamps:
        return {"map_name": map_name, "art_ok": False, "frames": [], "summary": empty_summary}

    n = min(len(frames_bgr), len(timestamps))
    frames_bgr = frames_bgr[:n]
    timestamps = [float(t) for t in timestamps[:n]]

    valid_imgs = [
        f for f in frames_bgr
        if isinstance(f, np.ndarray) and f.size > 0 and f.ndim == 3 and f.shape[2] >= 3
    ]
    if not valid_imgs:
        empty_summary["n_frames"] = n
        return {"map_name": map_name, "art_ok": False, "frames": [], "summary": empty_summary}

    H, W = valid_imgs[0].shape[:2]
    aspect = float(W) / float(max(1, H))

    from . import agent_dots as AD
    from . import minimap as MM
    from . import smokes as SM
    from . import tactics as TC

    same_shape = [f[:, :, :3] for f in valid_imgs if f.shape[:2] == (H, W)]
    if bg_bgr is not None and isinstance(bg_bgr, np.ndarray) and bg_bgr.ndim == 3 and bg_bgr.shape[:2] == (H, W):
        bg = bg_bgr[:, :, :3]
    elif len(same_shape) >= 3:
        bg = np.median(np.stack(same_shape[:40]), axis=0).astype(np.uint8)
    else:
        bg = same_shape[0]

    raw_m = MM.map_mask(bg)
    det_map_name, art_alpha, T_art = MM.register_art_auto(bg, map_name=map_name, mask_hint=raw_m)
    art_ok = bool(isinstance(T_art, dict) and T_art.get("ok") and art_alpha is not None)
    if art_ok:
        ref_mask = MM.warp_art(art_alpha, T_art, bg.shape) > 128
    else:
        ok_bg, det_m = MM.detect_minimap_region(bg, mask=raw_m)
        ref_mask = det_m if ok_bg and det_m is not None else None

    dot_mask = None
    if ref_mask is not None or art_ok:
        _, dot_mask = MM.playable_region(
            bg, mask=ref_mask, art_alpha=art_alpha if art_ok else None, T_art=T_art if art_ok else None
        )

    use_learned = AD.load_model() is not None
    agent_seq: list[dict[str, Any]] = []
    smoke_seq: list[dict[str, Any]] = []

    for i in range(n):
        fr = frames_bgr[i]
        t_i = timestamps[i]
        is_img = isinstance(fr, np.ndarray) and fr.size > 0 and fr.ndim == 3 and fr.shape[2] >= 3 and fr.shape[:2] == (H, W)
        if not is_img:
            is_valid = False
        elif ref_mask is not None:
            is_valid = bool(MM._frame_matches_mask(ref_mask, MM._outline(fr), min_f=0.30, max_line_frac=0.048))
        else:
            is_valid = bool(int(fr[:, :, :3].max()) > 0)

        if not is_valid:
            agent_seq.append({"t": t_i, "dots": [], "low_dots": [], "valid": False})
            smoke_seq.append({"t": t_i, "smokes": [], "valid": False})
            continue

        if use_learned:
            primary, low = AD.detect_agent_dots_with_low(
                fr,
                per_color=per_color,
                mask=ref_mask,
                art_alpha=art_alpha if art_ok else None,
                T_art=T_art if art_ok else None,
                low_thresh=low_thresh,
            )
        else:
            primary = TC.detect_dots(
                fr,
                per_color=per_color,
                mask=ref_mask,
                art_alpha=art_alpha if art_ok else None,
                T_art=T_art if art_ok else None,
            )
            low = []

        if include_smokes:
            raw_sm = SM.detect_smokes(
                fr,
                bg,
                map_name=det_map_name or map_name,
                mask=ref_mask,
                found_agents=primary,
                art_alpha=art_alpha if art_ok else None,
                T_art=T_art if art_ok else None,
            )
        else:
            raw_sm = []

        agent_seq.append({"t": t_i, "dots": primary, "low_dots": low, "valid": True})
        smoke_seq.append({"t": t_i, "smokes": raw_sm, "valid": True})

    corr_agents = correct_agent_dots_sequence(
        agent_seq,
        per_color=per_color,
        aspect=aspect,
        dot_mask=dot_mask,
    )
    corr_smokes = (
        correct_smokes_sequence(smoke_seq, aspect=aspect)
        if include_smokes
        else [[] for _ in range(n)]
    )

    out_frames: list[dict[str, Any]] = []
    raw_agents = 0
    corrected_agents = 0
    recovered_agents = 0
    raw_smokes = 0
    corrected_smokes = 0
    bridged_smokes = 0
    valid_frames = 0

    for i in range(n):
        v = bool(agent_seq[i]["valid"])
        if v:
            valid_frames += 1
        rd = agent_seq[i]["dots"]
        cd = corr_agents[i]
        rs = smoke_seq[i]["smokes"]
        cs = corr_smokes[i]
        raw_agents += len(rd)
        corrected_agents += len(cd)
        recovered_agents += sum(1 for d in cd if d.get("temporal"))
        raw_smokes += len(rs)
        corrected_smokes += len(cs)
        bridged_smokes += sum(1 for s in cs if s.get("temporal"))
        out_frames.append({
            "t": timestamps[i],
            "valid": v,
            "raw_dots": rd,
            "low_dots": agent_seq[i]["low_dots"],
            "dots": cd,
            "raw_smokes": rs,
            "smokes": cs,
        })

    suppressed_agents = max(0, raw_agents + recovered_agents - corrected_agents)
    suppressed_smokes = max(0, raw_smokes + bridged_smokes - corrected_smokes)

    return {
        "map_name": det_map_name or map_name,
        "art_ok": art_ok,
        "frames": out_frames,
        "summary": {
            "n_frames": n,
            "valid_frames": valid_frames,
            "raw_agents": raw_agents,
            "corrected_agents": corrected_agents,
            "recovered_agents": recovered_agents,
            "suppressed_agents": suppressed_agents,
            "raw_smokes": raw_smokes,
            "corrected_smokes": corrected_smokes,
            "bridged_smokes": bridged_smokes,
            "suppressed_smokes": suppressed_smokes,
        },
    }
