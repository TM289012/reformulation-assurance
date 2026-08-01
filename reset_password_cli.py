"""Generate a password-reset link from the server side.

Local deployments often run without SMTP, so reset emails cannot be
delivered. This helper is the supported alternative: an operator with access
to the machine that hosts the database (someone who could already modify it
directly) generates a link and hands it to the account owner out of band.

Reset links are deliberately never shown in the in-app admin outbox: an
administrator who could read another user's reset link could take over that
account and sign approvals as them.

Usage:
    python reset_password_cli.py user@example.com
    python reset_password_cli.py user@example.com --base-url https://assurance.example.com
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from pilot_store import PilotStore

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB = Path(os.environ.get("REFORMULATION_DB_PATH", APP_DIR / "data" / "reformulation_assurance_v06.db"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a single-use password-reset link.")
    parser.add_argument("email", help="Email address of the account to reset")
    parser.add_argument("--base-url", default=os.environ.get("REFORMULATION_PUBLIC_URL", "http://localhost:8501"))
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to the application database")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    store = PilotStore(db_path)
    result = store.request_password_reset(args.email, base_url=args.base_url)
    if result is None:
        # Deliberately the same outcome as the in-app flow: do not reveal
        # whether the account exists beyond this operator context.
        print("No active account with that email address.")
        return 1

    print("Single-use reset link (expires at "
          f"{result['expires_at']}):\n\n  {result['reset_url']}\n")
    print("Deliver it to the account owner directly. It is not shown in the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
