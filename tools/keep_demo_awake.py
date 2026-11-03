"""Keep the public Streamlit Community Cloud deployments awake.

Community Cloud puts an app to sleep after a stretch without visitors and then
shows a "Zzzz" page with a wake-up button. A stranger who follows a link from a
forum post lands on that page, waits a minute, and usually leaves. This script
opens each deployment in a real browser (Playwright, Chromium), presses the
wake-up button if the page is asleep, waits until the app itself has rendered,
and keeps the session open for a moment so the platform registers a visit.

It is run every few hours by ``.github/workflows/keep-demo-awake.yml`` and can
be run by hand::

    python tools/keep_demo_awake.py https://reformulation-assurance-demo.streamlit.app/

Exit status is non-zero when a deployment could not be brought up, so the
workflow run shows red instead of silently passing.
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

DEFAULT_URLS = (
    "https://reformulation-assurance-demo.streamlit.app/",
    "https://reformulation-assurance.streamlit.app/",
)
APP_TITLE = "Reformulation Assurance"
WAKE_BUTTON = "Yes, get this app back up!"
WAKE_TIMEOUT_S = 240
SESSION_HOLD_S = 20


def _app_is_up(page) -> bool:
    try:
        return APP_TITLE in (page.title() or "")
    except Exception:  # noqa: BLE001 - the page may be mid-navigation
        return False


def visit(page, url: str) -> str:
    """Return 'up', 'woken' or 'down' for one deployment."""
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    deadline = time.time() + WAKE_TIMEOUT_S
    clicked = False
    while time.time() < deadline:
        if _app_is_up(page):
            time.sleep(SESSION_HOLD_S)
            return "woken" if clicked else "up"
        button = page.get_by_role("button", name=WAKE_BUTTON)
        try:
            if not clicked and button.count() > 0:
                button.first.click(timeout=5_000)
                clicked = True
        except PlaywrightTimeout:
            pass
        time.sleep(5)
    return "down"


def main(argv: list[str]) -> int:
    urls = argv[1:] or list(DEFAULT_URLS)
    failures = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for url in urls:
            page = browser.new_page()
            try:
                status = visit(page, url)
            except Exception as exc:  # noqa: BLE001 - report and continue with the next URL
                status = f"down ({type(exc).__name__}: {exc})"
            finally:
                page.close()
            print(f"{url}  {status}", flush=True)
            if status.startswith("down"):
                failures += 1
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
