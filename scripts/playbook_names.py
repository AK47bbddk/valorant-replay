"""プレイブックの ATK／DEF の行の名前を、合成データ（tests/deckfx.py）から出す。

型の見分け方を変えたときに、名前がどう変わったかを前後で比べるために使う。
ネットワークは使わない（コールアウトは tests/fixtures の実データ）。

    python scripts/playbook_names.py            # 名前の一覧
    python scripts/playbook_names.py --lines    # 本文の箇条書きも出す
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vreplay import deck as D                                      # noqa: E402
from vreplay import tactics as T                                   # noqa: E402

T.load_map = lambda *a, **k: None                                  # valorant-api を叩かない

from tests.deckfx import report                                    # noqa: E402


def main(show_lines: bool = False) -> None:
    reps = [report(str(i), "G2", "SEN" if i % 2 else "FNC", i, strong_retake="G2") for i in range(6)]
    rows = [r for rep in reps for r in D.records(rep)]
    g2 = [r for r in rows if r["team"] == "G2"]
    for side, label in ((D.ATTACK, "ATK"), (D.DEFENSE, "DEF")):
        print(f"== {label}（{side}）")
        for g in D.playbook_rows([r for r in g2 if r["side"] == side], side):
            print(f"  {g['name']}｜{g['kind']}｜{g['n']} 回・勝ち {g['w']}")
            if show_lines:
                for ln in g["lines"]:
                    print(f"      - {ln}")


if __name__ == "__main__":
    main("--lines" in sys.argv)
