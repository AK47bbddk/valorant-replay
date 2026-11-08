"""冊子（HTML）のスライドのうち、見出しに言葉を含むものを PNG に撮る（確認用）。

    python scripts/shot_slides.py <冊子.html> <出力フォルダ> 言葉1 言葉2 ...
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(deck, out, words):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1400, "height": 900})
        pg.goto(Path(deck).resolve().as_uri())
        pg.wait_for_timeout(1500)
        for i, s in enumerate(pg.query_selector_all(".slide")):
            t = s.inner_text()
            head = " ".join(t.split())[:90]
            if not words or any(w in head for w in words):
                f = out / f"slide_{i:03d}.png"
                s.screenshot(path=str(f))
                print(f, head)
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
