"""Headless smoke tests for the Streamlit app (sign-in screen and workspace creation).

These run the real ``app.py`` through ``streamlit.testing.v1.AppTest`` with an isolated
database, so a broken import or a mis-wired setting shows up in the suite instead of in
the browser. Skipped when Streamlit is not installed (the core library has no Streamlit
dependency).
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - import guard
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    HAS_STREAMLIT = True
except Exception:  # noqa: BLE001
    HAS_STREAMLIT = False

from pilot_store import PilotStore

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app.py"
SETTINGS = (
    "REFORMULATION_DB_PATH",
    "REFORMULATION_ARTIFACT_ROOT",
    "REFORMULATION_OPEN_SIGNUP",
    "REFORMULATION_DEMO_MODE",
    "REFORMULATION_DATABASE_URL",
    "REFORMULATION_PUBLIC_URL",
)


@unittest.skipUnless(HAS_STREAMLIT, "streamlit is not installed")
class AppSmokeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "smoke.db"
        self._saved = {name: os.environ.get(name) for name in SETTINGS}
        for name in SETTINGS:
            os.environ.pop(name, None)
        os.environ["REFORMULATION_DB_PATH"] = str(self.db_path)
        os.environ["REFORMULATION_ARTIFACT_ROOT"] = str(self.root / "artifacts")
        os.environ["REFORMULATION_PUBLIC_URL"] = "https://assurance.example.test"
        # The store is cached per process; every test gets a fresh database.
        st.cache_resource.clear()

    def tearDown(self):
        st.cache_resource.clear()
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()

    def _app(self) -> "AppTest":
        return AppTest.from_file(str(APP_FILE), default_timeout=240)

    @staticmethod
    def _fill(container, values: list[str]) -> None:
        for widget, value in zip(container.text_input, values):
            widget.input(value)

    def test_fresh_install_creates_first_owner(self):
        at = self._app().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.title[0].value, "Reformulation Assurance")
        self.assertTrue(any("first workspace owner" in s.value for s in at.subheader))

        self._fill(at, ["Smoke Lab", "Smoke Owner", "owner@example.test", "SecurePass123", "SecurePass123"])
        at.button[0].click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertFalse(at.error, [e.value for e in at.error])

        user = at.session_state["current_user"]
        self.assertEqual(user["email"], "owner@example.test")
        store = PilotStore(self.db_path)
        self.assertTrue(store.has_users())
        self.assertEqual(len(store.user_organizations(user["id"])), 1)

    def test_private_install_has_no_signup_tab(self):
        PilotStore(self.db_path).register_owner(
            email="owner@example.test", display_name="Owner", password="SecurePass123",
            organization_name="Private Lab",
        )
        at = self._app().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual([tab.label for tab in at.tabs], ["Sign in", "Accept invitation", "Reset password"])

        sign_in = at.tabs[0]
        self._fill(sign_in, ["owner@example.test", "wrong-password"])
        sign_in.button[0].click().run()
        self.assertTrue(any("incorrect" in e.value for e in at.error))

        sign_in = at.tabs[0]
        self._fill(sign_in, ["owner@example.test", "SecurePass123"])
        sign_in.button[0].click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state["current_user"]["email"], "owner@example.test")

    def test_open_signup_creates_a_second_workspace(self):
        PilotStore(self.db_path).register_owner(
            email="owner@example.test", display_name="Owner", password="SecurePass123",
            organization_name="First Lab",
        )
        os.environ["REFORMULATION_OPEN_SIGNUP"] = "true"
        at = self._app().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(
            [tab.label for tab in at.tabs],
            ["Sign in", "Create a workspace", "Accept invitation", "Reset password"],
        )

        signup = at.tabs[1]
        self._fill(signup, ["CHEM 3410 Fall 2026", "Course Instructor", "instructor@example.test", "SecurePass123", "SecurePass123"])
        signup.button[0].click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertFalse(at.error, [e.value for e in at.error])
        self.assertEqual(at.session_state["current_user"]["email"], "instructor@example.test")

        store = PilotStore(self.db_path)
        orgs = store.user_organizations(at.session_state["current_user"]["id"])
        self.assertEqual(orgs["name"].tolist(), ["CHEM 3410 Fall 2026"])
        self.assertEqual(orgs["role"].tolist(), ["owner"])
        # The first workspace is untouched by the new one.
        first = store.authenticate("owner@example.test", "SecurePass123")
        self.assertEqual(store.user_organizations(first["id"])["name"].tolist(), ["First Lab"])

    def test_signup_rejects_mismatched_passwords(self):
        PilotStore(self.db_path).register_owner(
            email="owner@example.test", display_name="Owner", password="SecurePass123",
            organization_name="First Lab",
        )
        os.environ["REFORMULATION_OPEN_SIGNUP"] = "true"
        at = self._app().run()
        signup = at.tabs[1]
        self._fill(signup, ["Lab", "Person", "person@example.test", "SecurePass123", "SecurePass124"])
        signup.button[0].click().run()
        self.assertTrue(any("do not match" in e.value for e in at.error))
        self.assertNotIn("current_user", at.session_state)


if __name__ == "__main__":
    unittest.main()
