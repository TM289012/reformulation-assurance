"""Second entry point for a hosted-workspaces deployment of this repository.

Streamlit Community Cloud allows one app per repository + branch + file, so the
public demo (deployed from ``app.py``) and a hosted workspaces instance cannot
both point at ``app.py``. Deploy the hosted instance from this file instead: it
runs ``app.py`` unchanged, and the deployment's secrets decide the behaviour
(``REFORMULATION_DATABASE_URL``, ``REFORMULATION_OPEN_SIGNUP``,
``REFORMULATION_ARTIFACT_KEY``, ``REFORMULATION_PUBLIC_URL``; see DEPLOY.md).
"""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
