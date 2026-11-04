"""Keep the public Streamlit Community Cloud deployments awake.

Community Cloud puts an app to sleep after a stretch without visitors and then
shows a "Zzzz" page with a wake-up button. A stranger who follows a link from a
forum post lands on that page, waits a minute, and usually leaves. This script
opens each deployment in a real browser (Playwright, Chromium), presses the
wake-up button if the page is asleep, waits until the app itself has rendered,
and keeps the session open for a while so the platform registers a visit.

It is run every couple of hours by ``.github/workflows/keep-demo-awake.yml``
and can be run by hand::

    python tools/keep_demo_awake.py https://reformulation-assurance-demo.streamlit.app/

Exit status is non-zero when a deployment could not be brought up, so the
workflow run shows red instead of silently passing. On every attempt the script
writes a screenshot and the page's text to ``keepalive-artifacts/`` so a failed
run can be read instead of guessed at; the workflow uploads that folder.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

DEFAULT_URLS = (
    "https://reformulation-assurance-demo.streamlit.app/",
    "https://reformulation-assurance.streamlit.app/",
)
APP_TITLE = "Reformulation Assurance"
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
WAKE_TIMEOUT_S = int(os.environ.get("KEEPALIVE_WAKE_TIMEOUT_S", "480"))
SESSION_HOLD_S = int(os.environ.get("KEEPALIVE_SESSION_HOLD_S", "45"))
ARTIFACT_DIR = Path(os.environ.get("KEEPALIVE_ARTIFACT_DIR", "keepalive-artifacts"))
# A plain desktop Chrome user agent: headless Chromium announces itself as
# "HeadlessChrome", which some front doors treat differently from a visitor.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def _slug(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url.lower()).strip("-")


def _app_is_up(page) -> bool:
    try:
        return APP_TITLE in (page.title() or "")
    except Exception:  # noqa: BLE001 - the page may be mid-navigation
        return False


def _press_wake_button(page) -> bool:
    """Press the sleeping page's button wherever it lives; True if something was pressed."""
    frames = [page.main_frame, *page.frames]
    for frame in frames:
        try:
            button = frame.get_by_role("button", name=WAKE_BUTTON)
            if button.count() == 0:
                button = frame.get_by_text(WAKE_BUTTON)
            if button.count() == 0:
                continue
            button.first.scroll_into_view_if_needed(timeout=5_000)
            button.first.click(timeout=5_000)
            return True
        except PlaywrightTimeout:
            try:  # a plain DOM click as the fallback
                frame.evaluate(
                    """() => { const b = [...document.querySelectorAll('button')]
                        .find(x => /get this app back up/i.test(x.textContent || ''));
                        if (b) { b.click(); return true; } return false; }"""
                )
                return True
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001 - detached frame and the like
            continue
    return False


def _snapshot(page, url: str, label: str) -> None:
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        stem = ARTIFACT_DIR / f"{_slug(url)}-{label}"
        page.screenshot(path=f"{stem}.png", full_page=True)
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
        Path(f"{stem}.txt").write_text(f"title: {page.title()}\nurl: {page.url}\n\n{text}", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the run
        print(f"  (snapshot {label} failed: {type(exc).__name__}: {exc})", flush=True)


def visit(page, url: str) -> str:
    """Return 'up', 'woken' or 'down' for one deployment."""
    page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    time.sleep(5)
    _snapshot(page, url, "arrival")
    deadline = time.time() + WAKE_TIMEOUT_S
    clicked_at: float | None = None
    presses = 0
    while time.time() < deadline:
        if _app_is_up(page):
            time.sleep(SESSION_HOLD_S)
            _snapshot(page, url, "final")
            return "woken" if presses else "up"
        # Press the button on arrival, and again every two minutes if nothing happened.
        if clicked_at is None or time.time() - clicked_at > 120:
            if _press_wake_button(page):
                presses += 1
                clicked_at = time.time()
                print(f"  pressed the wake button (press {presses})", flush=True)
            elif clicked_at is None and time.time() - (deadline - WAKE_TIMEOUT_S) > 60:
                # No button and no app for a minute: reload in case the page is stale.
                try:
                    page.reload(wait_until="domcontentloaded", timeout=90_000)
                except Exception:  # noqa: BLE001
                    pass
                clicked_at = time.time()
        time.sleep(5)
    _snapshot(page, url, "timeout")
    return "down"


def main(argv: list[str]) -> int:
    urls = argv[1:] or list(DEFAULT_URLS)
    failures = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 900})
        for url in urls:
            page = context.new_page()
            started = time.time()
            try:
                status = visit(page, url)
            except Exception as exc:  # noqa: BLE001 - report and continue with the next URL
                status = f"down ({type(exc).__name__}: {exc})"
                _snapshot(page, url, "error")
            finally:
                page.close()
            print(f"{url}  {status}  ({time.time() - started:.0f}s)", flush=True)
            if status.startswith("down"):
                failures += 1
        context.close()
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
