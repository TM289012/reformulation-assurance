"""Demo-mode seeding for public sandbox deployments (e.g. Streamlit Community Cloud).

When ``REFORMULATION_DEMO_MODE`` is enabled, an empty database is seeded with a
shared demo workspace and the sandbox projects preloaded (the cosmetics emulsifier
swap and the food color replacement, see ``demo_catalog.SANDBOX_KEYS``), so a
visitor can explore recommendations, qualification stages, approvals, and the
dossier export without installing anything or entering real data.

The sandbox is shared and periodically wiped. Never enable demo mode on a
deployment that holds real formulation data.
"""
from __future__ import annotations

from demo_catalog import ROOT, SANDBOX_KEYS, cosmetics_config, create_demo_project, demo_dataset
from pilot_store import PilotStore

DEMO_OWNER_EMAIL = "demo@example.com"
DEMO_OWNER_PASSWORD = "try-the-demo-2026"
DEMO_ORGANIZATION = "Public Demo Lab"
DEMO_CSV = ROOT / demo_dataset("cosmetics").filename

__all__ = [
    "DEMO_CSV",
    "DEMO_ORGANIZATION",
    "DEMO_OWNER_EMAIL",
    "DEMO_OWNER_PASSWORD",
    "cosmetics_config",
    "seed_demo",
]


def _demo_workspace(store: PilotStore) -> tuple[str, str] | None:
    """Return (owner user id, organization id) of an existing demo workspace, or None."""
    owner = store.find_user_by_email(DEMO_OWNER_EMAIL)
    if owner is None:
        return None
    organizations = store.user_organizations(owner["id"])
    if organizations.empty:
        return None
    named = organizations[organizations["name"] == DEMO_ORGANIZATION]
    chosen = named if not named.empty else organizations
    return str(owner["id"]), str(chosen.iloc[0]["id"])


def seed_demo(store: PilotStore) -> str | None:
    """Create or complete the shared demo workspace.

    * Empty database: register the demo owner and organization, then create every
      sandbox project.
    * Database that already holds the demo workspace: create only the sandbox
      projects that are missing, so a redeploy that ships a new dataset picks it up
      without waiting for the next wipe. Nothing already there is touched.
    * Any other populated database: do nothing. Demo mode must never write into a
      workspace that is not its own.

    Returns the id of the first project created by this call, else None.
    """
    if not store.has_users():
        user_id, organization_id = store.register_owner(
            email=DEMO_OWNER_EMAIL,
            display_name="Demo Explorer",
            password=DEMO_OWNER_PASSWORD,
            organization_name=DEMO_ORGANIZATION,
        )
    else:
        workspace = _demo_workspace(store)
        if workspace is None:
            return None
        user_id, organization_id = workspace

    existing = store.list_projects(organization_id)
    present = set(existing["name"].astype(str)) if not existing.empty else set()
    created: list[str] = []
    for key in SANDBOX_KEYS:
        if demo_dataset(key).project_name in present:
            continue
        created.append(
            create_demo_project(store, key, organization_id=organization_id, created_by_user_id=user_id)
        )
    return created[0] if created else None
