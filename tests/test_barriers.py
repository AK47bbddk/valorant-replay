import json
from pathlib import Path

import cv2
import numpy as np

from vreplay import agents as AG
from vreplay import barriers as BR
from vreplay import minimap as MM
from vreplay import tactival as TV

FX = Path(__file__).parent / "fixtures"


def test_load_barriers_all_maps():
    for mid in TV.MAP_ROT:
        bars = BR.load_barriers(mid)
        assert bars.get("ATK") and bars.get("DEF"), mid
        for side in ("ATK", "DEF"):
            for poly in bars[side]:
                assert len(poly) >= 2
                for u, v in poly:
                    assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0, (mid, side, u, v)
    assert BR.load_barriers("UnknownMap") == {}


def test_opening_masks_sunset_fixture():
    bg = cv2.imread(str(FX / "minimap_sunset_riot.jpg"))
    mask = MM.map_mask(bg)
    m = json.loads((FX / "sunset_callouts.json").read_text("utf-8"))
    calls = [
        {
            "name": n,
            "super": s,
            "x": m["xMultiplier"] * y + m["xScalarToAdd"],
            "y": m["yMultiplier"] * x + m["yScalarToAdd"],
        }
        for n, s, x, y in m["callouts"]
    ]
    spawns = {
        "攻撃": next((c["x"], c["y"]) for c in calls if c["name"].lower() == "spawn" and "attacker" in c["super"].lower()),
        "防衛": next((c["x"], c["y"]) for c in calls if c["name"].lower() == "spawn" and "defender" in c["super"].lower()),
    }
    T = MM.fit_mask(mask, {"callouts": calls, "spawns": spawns}, bg=bg)
    assert T["ok"]

    dmaps = BR.opening_dist_maps(mask, T, {"spawns": spawns}, "Sunset")
    assert "ATK" in dmaps and "DEF" in dmaps

    H, W = mask.shape
    atk_m = BR.opening_mask(dmaps, "攻撃", dt=1.0)
    def_m = BR.opening_mask(dmaps, "防衛", dt=1.0)
    assert atk_m is not None and def_m is not None

    ax, ay = MM.apply(T, *spawns["攻撃"])
    dx, dy = MM.apply(T, *spawns["防衛"])
    apx, apy = int(round(ax * H)), int(round(ay * H))
    dpx, dpy = int(round(dx * H)), int(round(dy * H))

    assert atk_m[apy, apx] and not atk_m[dpy, dpx]
    assert def_m[dpy, dpx] and not def_m[apy, apx]
    assert BR.opening_mask(dmaps, "攻撃", dt=2.5).sum() > atk_m.sum()


def test_agents_match_respects_team_masks():
    # 200x200 のミニマップに、左（自陣側 x=40）と右（敵陣側 x=160）へ同じアイコン模様を置く。
    # 右側の方がくっきり（高スコア）でも、team_masks で右側を禁止すれば左側が選ばれる。
    img = np.full((200, 200, 3), 40, np.uint8)
    icon = np.zeros((24, 24, 3), np.uint8)
    cv2.circle(icon, (12, 12), 8, (220, 220, 220), -1)
    cv2.circle(icon, (9, 10), 2, (40, 40, 40), -1)

    def draw_unit(cx, cy, contrast=1.0):
        r = 6
        face = cv2.resize(icon, (2 * r, 2 * r))
        face = np.clip(face.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
        img[cy - r : cy + r, cx - r : cx + r] = face
        cv2.circle(img, (cx, cy), r + 2, (0, 0, 255), 2)  # 赤（暖色）の縁

    draw_unit(40, 100, contrast=0.85)   # 自陣側（本物）
    draw_unit(160, 100, contrast=1.0)   # 敵陣側（誤検出候補）

    icons = {"Jett": icon}
    teams = {1: ["Jett"], 2: []}
    color_team = {"warm": 1, "cool": 2}

    no_mask = AG.match(img, icons, teams, color_team, icon_frac=0.07)
    assert len(no_mask) == 1 and no_mask[0]["x"] > 0.5  # 制限なしだとスコアの高い右側を拾う

    tm1 = np.zeros((200, 200), bool)
    tm1[:, :100] = True
    with_mask = AG.match(img, icons, teams, color_team, icon_frac=0.07, team_masks={1: tm1})
    assert len(with_mask) == 1 and with_mask[0]["x"] < 0.3  # team_masks で自陣側（左側）が選ばれる
