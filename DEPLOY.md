# Deploying a hosted workspace

Reformulation Assurance is local-first: the default install keeps everything in a SQLite file on one machine, and that is still the recommended way to handle confidential formulations. This guide covers the other mode, introduced in v0.12: one app instance serving several workspaces (labs, courses, teams) from a PostgreSQL database, with self-serve sign-up. It costs nothing to run at small scale on Streamlit Community Cloud plus a free Neon PostgreSQL project, which is the setup described here. Any other host that runs a Python process and any other PostgreSQL work the same way; only the secrets mechanism differs.

## How the hosted mode differs

| | Local (default) | Hosted |
|---|---|---|
| Storage | SQLite file at `data/reformulation_assurance_v06.db` | PostgreSQL, `REFORMULATION_DATABASE_URL` |
| Workspaces | One, created by the first user | Many; anyone can create one when `REFORMULATION_OPEN_SIGNUP` is on |
| Encrypted exports | Files under `data/artifacts/`, key file next to them | Ciphertext rows in the database, key from `REFORMULATION_ARTIFACT_KEY` |
| Backups | Verified backup files and the PostgreSQL migration bundle from the app | Your database provider's backups (Neon: point-in-time restore); the in-app SQLite tools are hidden |
| Password reset | Link sent by SMTP, or generated on the server with `reset_password_cli.py` | SMTP only: there is no shell on a managed host, so configure mail before people depend on the instance |

The application code is identical. The store picks the backend from the value it is given: a filesystem path opens SQLite, a `postgresql://` URL opens PostgreSQL through psycopg 3 (`db_backend.py`). Workspace isolation is the same in both modes: every query is scoped by organization and role, and members only see the workspaces they belong to.

## Settings

All settings are read from environment variables first, then from Streamlit secrets (`.streamlit/secrets.toml` locally, the Secrets panel on Community Cloud). Top-level string secrets are also exported as environment variables by Streamlit, so the SMTP settings below work from either place.

| Setting | Purpose |
|---|---|
| `REFORMULATION_DATABASE_URL` | PostgreSQL connection URL. Unset = local SQLite. |
| `REFORMULATION_OPEN_SIGNUP` | `true` adds a **Create a workspace** tab to the sign-in screen. Off by default. |
| `REFORMULATION_ARTIFACT_KEY` | Fernet key that encrypts stored exports. **Required for hosted mode**; without it a key is generated on the server's disposable disk and everything encrypted with it becomes unreadable at the next redeploy. The app shows a red banner to owners while this is missing. |
| `REFORMULATION_PUBLIC_URL` | The app's public address, used to build invitation and password-reset links. |
| `REFORMULATION_SMTP_HOST`, `REFORMULATION_SMTP_PORT`, `REFORMULATION_SMTP_USERNAME`, `REFORMULATION_SMTP_PASSWORD`, `REFORMULATION_SMTP_TLS`, `REFORMULATION_EMAIL_FROM` | Outgoing mail for invitations and password resets. Without SMTP, invitation links stay in the workspace outbox (Team page) and an admin can pass them on by hand; password-reset links are deliberately never shown in the app, so resets need SMTP. |
| `REFORMULATION_DEMO_MODE` | Public sandbox with the demo project preloaded and a one-click demo login. Never combine with a real database. |

Generate an artifact key once and keep it somewhere safe (a password manager); losing it means losing every stored encrypted export:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Step by step: Streamlit Community Cloud + Neon

1. **Create the database.** Sign in at neon.tech, create a project (pick the region closest to your users; Community Cloud apps run in the United States), and copy the connection string from the project dashboard. Use the **pooled** connection string if it is offered; the app does not use server-side prepared statements, so it works behind Neon's PgBouncer endpoint. The Free plan (as of September 2026: 0.5 GB per project, 100 compute-hours a month, compute suspends after five idle minutes and wakes on the next query) is enough for a handful of small labs or one course.

2. **Deploy the app.** At share.streamlit.io, choose *Create app*, point it at your fork of this repository, branch `main`, main file `hosted.py` (Community Cloud allows one app per repository, branch and file, and `app.py` is usually taken by a public demo; `hosted.py` runs `app.py` unchanged). Community Cloud installs `requirements.txt`, which already includes `psycopg[binary]`. Under *Advanced settings → Secrets*, paste:

   ```toml
   REFORMULATION_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
   REFORMULATION_OPEN_SIGNUP = "true"
   REFORMULATION_ARTIFACT_KEY = "the key you generated above"
   REFORMULATION_PUBLIC_URL = "https://your-app-name.streamlit.app"
   ```

   Add the SMTP settings as soon as you have a sending account (any transactional mail provider's SMTP credentials work; a personal mailbox with an app password is fine for a pilot). Until then, nobody can reset a forgotten password on this instance.

3. **Create the first workspace.** Open the app. Because the database is empty you get the *Create the first workspace owner* form; this account is simply the first owner, it has no special powers over other workspaces. Then confirm the sign-in screen shows four tabs: *Sign in*, *Create a workspace*, *Accept invitation*, *Reset password*.

4. **Check the red banner is gone.** If an owner sees the "Hosted deployment without REFORMULATION_ARTIFACT_KEY" error after signing in, the key secret is missing or misspelled. Fix it before anyone exports a dossier.

5. **Know the platform's limits.** Community Cloud apps get roughly 0.7 to 2.7 GB of memory and go to sleep after twelve hours without traffic; the next visitor sees a wake-up prompt and waits about half a minute. Model fitting for a recommendation batch takes 30 to 60 seconds of CPU either way. If a workspace outgrows that, the same repository deploys unchanged to any container host; only the secrets move.

## Running the hosted mode locally

Useful for trying it before deploying, or for a lab that wants PostgreSQL on its own server:

```bash
export REFORMULATION_DATABASE_URL="postgresql://user:password@localhost:5432/assurance"
export REFORMULATION_OPEN_SIGNUP=true
export REFORMULATION_ARTIFACT_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
python -m streamlit run app.py
```

## Moving an existing local workspace to PostgreSQL

The local app's **Pilot operations → PostgreSQL migration bundle** exports every table as checksummed CSV plus a manifest; load it into the new database with your usual tooling (`\copy` in psql, or a short pandas script), then point the app at the database. Encrypted export files stored on disk under `data/artifacts/` stay readable by the local app; re-export what you need after the move.

## Testing against PostgreSQL

The whole test suite runs unchanged against PostgreSQL. Point it at any scratch database; each test's store gets its own schema, named from the temporary path it would have used, so runs do not collide:

```bash
REFORMULATION_FORCE_DATABASE_URL="postgresql://user:password@localhost:5432/scratch" \
    python -m unittest discover -s tests
```

The three tests that read the SQLite file directly (verified backups, the migration bundle, the pre-v0.8 migration) skip in this mode. `tests/test_app_smoke.py` runs the real `app.py` headlessly through Streamlit's `AppTest` and is skipped where Streamlit is not installed.

For a course section on a hosted instance: the instructor creates the workspace from the *Create a workspace* tab, pastes the roster on the Team page, and follows the section from the Workspace overview page. Details and a lab outline are on the for-courses page (`docs/for-courses.html`).

## What hosted mode is not

Hosted mode does not change the security posture stated in the README: passwords are hashed with PBKDF2 and sign-ins are throttled (eight failures lock an email for fifteen minutes), but there is no MFA, no SSO and no independent security review. It is meant for a small team or a course that has decided a shared instance is acceptable for its data, not for formulations a company would consider trade secrets. For those, run it locally.
