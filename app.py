"""Streamlit application for Reformulation Assurance v0.10.0."""
from __future__ import annotations

import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from assurance_v4 import (
    calibration_report,
    default_variation_config,
    result_for_storage,
    simulate_manufacturing_variation,
)
from closed_loop import (
    QUALIFICATION_STAGES,
    create_recommendation_batch,
    ensure_v04_config,
    qualification_progress,
    refresh_after_result,
)
from dossier import evidence_snapshot_and_hash, generate_dossier, generate_workbook
from artifact_vault import ArtifactVault
from backup_service import create_backup
from notifications import deliver_queued_notifications
from postgres_migration import create_postgres_migration_bundle
from ingestion import import_readiness_report, load_table, workbook_preview
from pilot_store import ADMIN_ROLES, APPROVAL_ROLES, EDIT_ROLES, PilotStore, ROLES, PRIORITIES, TASK_STATUSES
from reformulation_engine import infer_numeric_bounds
from process_window import DESIGN_MODES, design_process_window


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB = Path(os.environ.get("REFORMULATION_DB_PATH", APP_DIR / "data" / "reformulation_assurance_v06.db"))
ARTIFACT_ROOT = Path(os.environ.get("REFORMULATION_ARTIFACT_ROOT", APP_DIR / "data" / "artifacts"))
DEMO_FILE = APP_DIR / "demo_coatings_reformulation.csv"


def _demo_mode_enabled() -> bool:
    """Public-sandbox mode, set via env var or Streamlit secrets. Off by default."""
    flag = os.environ.get("REFORMULATION_DEMO_MODE", "")
    if not flag:
        try:
            flag = str(st.secrets.get("REFORMULATION_DEMO_MODE", ""))
        except Exception:
            flag = ""
    return flag.strip().lower() in {"1", "true", "yes", "on"}


DEMO_MODE = _demo_mode_enabled()

st.set_page_config(page_title="Reformulation Assurance v0.10.0", page_icon="🧪", layout="wide")
print(f"[boot] page config set, demo_mode={DEMO_MODE}", flush=True)
st.title("Reformulation Assurance")
st.caption("v0.10.0 · design → run → verify → qualify → approve → export")
print("[boot] title rendered", flush=True)


@st.cache_resource
def get_store() -> PilotStore:
    return PilotStore(DEFAULT_DB)


@st.cache_resource
def get_vault() -> ArtifactVault:
    return ArtifactVault(ARTIFACT_ROOT)


store = get_store()
vault = get_vault()
print("[boot] store and vault ready", flush=True)

if DEMO_MODE:
    from demo_seed import DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo

    seed_demo(store)
    print("[boot] demo seed complete", flush=True)


def set_flash(message: str, level: str = "success") -> None:
    st.session_state["flash_message"] = {"message": message, "level": level}


def show_flash() -> None:
    payload = st.session_state.pop("flash_message", None)
    if not payload:
        return
    renderer = getattr(st, payload.get("level", "success"), st.success)
    renderer(payload.get("message", "Saved."))


def first_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s]+", str(text or ""))
    return match.group(0).rstrip(".,)") if match else None


def authentication_screen() -> dict[str, Any]:
    print("[boot] auth screen reached", flush=True)
    if DEMO_MODE and not st.session_state.get("current_user"):
        st.info(
            "**Public demo.** Shared sandbox preloaded with a cosmetics reformulation "
            "project (88 lots). Anything you enter is visible to other visitors and is "
            "wiped periodically. Never enter real formulas here — for real work, run the "
            "app locally so data stays on your machine."
        )
        if st.button("Enter the demo", type="primary", use_container_width=True):
            demo_user = store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
            if demo_user:
                st.session_state["current_user"] = demo_user
                st.rerun()
            else:
                st.error(
                    "The demo account is unavailable (someone may have changed the "
                    "sandbox password). It comes back at the next reset — or reboot "
                    "the app if you are the operator."
                )
        st.caption(f"Or sign in manually: {DEMO_OWNER_EMAIL} / {DEMO_OWNER_PASSWORD}")
    if not store.has_users():
        st.subheader("Create the first workspace owner")
        st.info("This account controls the first organization. Use a private deployment for real company data.")
        with st.form("bootstrap_owner"):
            organization_name = st.text_input("Organization", value="Reformulation Lab")
            display_name = st.text_input("Your full name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password", help="At least 10 characters, including a letter and number.")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create owner account", type="primary")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    user_id, organization_id = store.register_owner(
                        email=email, display_name=display_name, password=password,
                        organization_name=organization_name,
                    )
                    st.session_state["current_user"] = store.get_user(user_id)
                    st.session_state["active_organization_id"] = organization_id
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.stop()

    user = st.session_state.get("current_user")
    if user:
        return user

    invite_default = str(st.query_params.get("invite", ""))
    reset_default = str(st.query_params.get("reset", ""))
    sign_in_tab, invite_tab, reset_tab = st.tabs(["Sign in", "Accept invitation", "Reset password"])
    with sign_in_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            authenticated = store.authenticate(email, password)
            if authenticated:
                st.session_state["current_user"] = authenticated
                st.rerun()
            st.error("Email or password is incorrect.")

    with invite_tab:
        with st.form("accept_invitation"):
            token = st.text_input("Invitation token", value=invite_default)
            display_name = st.text_input("Full name")
            password = st.text_input("Choose password", type="password")
            accepted = st.form_submit_button("Join workspace", type="primary")
        if accepted:
            try:
                user_id, organization_id = store.accept_invitation(
                    token, display_name=display_name, password=password
                )
                st.session_state["current_user"] = store.get_user(user_id)
                st.session_state["active_organization_id"] = organization_id
                st.query_params.clear()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with reset_tab:
        st.markdown("#### Request a reset")
        with st.form("request_reset"):
            reset_email = st.text_input("Account email")
            requested = st.form_submit_button("Create reset link")
        if requested:
            store.request_password_reset(reset_email, base_url=os.environ.get("REFORMULATION_PUBLIC_URL", "http://localhost:8501"))
            st.success("If the account exists, a reset message was added to the email outbox.")
        st.markdown("#### Use a reset token")
        with st.form("complete_reset"):
            reset_token = st.text_input("Reset token", value=reset_default)
            new_password = st.text_input("New password", type="password")
            reset_confirm = st.text_input("Confirm new password", type="password")
            completed = st.form_submit_button("Reset password", type="primary")
        if completed:
            if new_password != reset_confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    store.reset_password(reset_token, new_password)
                    st.query_params.clear()
                    st.success("Password reset. Sign in with the new password.")
                except Exception as exc:
                    st.error(str(exc))
    st.stop()


current_user = authentication_screen()
organizations = store.user_organizations(current_user["id"])
if organizations.empty:
    st.error("Your account is not assigned to a workspace.")
    st.stop()
organization_ids = organizations["id"].tolist()
active_org = st.session_state.get("active_organization_id")
if active_org not in organization_ids:
    active_org = organization_ids[0]
st.session_state["active_organization_id"] = active_org
organization_row = organizations[organizations["id"] == active_org].iloc[0]
organization_id = str(organization_row["id"])
current_role = str(organization_row["role"])
can_edit = current_role in EDIT_ROLES
can_approve = current_role in APPROVAL_ROLES
can_admin = current_role in ADMIN_ROLES
if DEMO_MODE:
    st.warning(
        "Public demo sandbox — shared with other visitors, resets periodically. "
        "Explore freely; don't enter anything real."
    )
show_flash()


def demo_config(data: pd.DataFrame) -> dict[str, Any]:
    mixture_columns = [column for column in data.columns if column.startswith("ingredient_")]
    completed = data[data["status"] == "completed"]
    baseline = completed.iloc[0][
        [*mixture_columns, "mix_temperature_c", "mix_time_min", "supplier_family"]
    ].to_dict()
    base = {
        "mixture_columns": mixture_columns,
        "process_columns": ["mix_temperature_c", "mix_time_min"],
        "categorical_columns": ["supplier_family"],
        "response_specs": [
            {"response": "adhesion", "minimum": 8.0, "maximum": None, "weight": 1.0},
            {"response": "viscosity_cp", "minimum": 2000.0, "maximum": 2600.0, "weight": 1.0},
            {"response": "dry_time_min", "minimum": None, "maximum": 40.0, "weight": 1.0},
            {"response": "gloss", "minimum": 85.0, "maximum": None, "weight": 1.0},
        ],
        "mixture_bounds": {
            "ingredient_resin": [34.0, 62.0],
            "ingredient_crosslinker": [8.0, 20.0],
            "ingredient_solvent": [12.0, 35.0],
            "ingredient_legacy_plasticizer": [0.0, 0.0],
            "ingredient_substitute_a": [0.0, 18.0],
            "ingredient_substitute_b": [0.0, 15.0],
        },
        "process_bounds": {
            "mix_temperature_c": [48.0, 72.0],
            "mix_time_min": [18.0, 42.0],
        },
        "category_values": {"supplier_family": ["A", "B", "C"]},
        "mixture_total": 100.0,
        "ingredient_to_remove": "ingredient_legacy_plasticizer",
        "baseline": baseline,
        "ingredient_costs": {
            "ingredient_resin": 2.50,
            "ingredient_crosslinker": 4.80,
            "ingredient_solvent": 0.90,
            "ingredient_legacy_plasticizer": 1.30,
            "ingredient_substitute_a": 1.85,
            "ingredient_substitute_b": 2.20,
        },
        "status_column": "status",
        "n_recommendations": 5,
        "candidate_pool_size": 2500,
        "min_distance": 0.06,
    }
    config = ensure_v04_config(base)
    config["manufacturing_variation"] = default_variation_config(config)
    return config


def format_project(row: pd.Series) -> str:
    return f"{row['name']} · {int(row['experiment_count'])} experiments · {int(row['batch_count'])} batches"


def create_demo_project() -> str:
    data = pd.read_csv(DEMO_FILE)
    project_id = store.create_project(
        "Coating Plasticizer Replacement v0.6",
        demo_config(data),
        description="Replace the legacy plasticizer and prove repeatability, robustness, approval, and qualification evidence.",
        source_filename=DEMO_FILE.name,
        organization_id=organization_id,
        created_by_user_id=current_user["id"],
    )
    store.import_history(project_id, data)
    return project_id


def project_selector() -> str | None:
    projects = store.list_projects(organization_id)
    if projects.empty:
        return None
    ids = projects["id"].tolist()
    current = st.session_state.get("active_project_id")
    index = ids.index(current) if current in ids else 0
    selected = st.sidebar.selectbox(
        "Active project",
        ids,
        index=index,
        format_func=lambda pid: format_project(projects[projects["id"] == pid].iloc[0]),
    )
    st.session_state["active_project_id"] = selected
    return selected


with st.sidebar:
    st.header("Workspace")
    selected_org = st.selectbox(
        "Organization",
        organization_ids,
        index=organization_ids.index(organization_id),
        format_func=lambda oid: str(organizations[organizations["id"] == oid].iloc[0]["name"]),
    )
    if selected_org != organization_id:
        st.session_state["active_organization_id"] = selected_org
        st.session_state.pop("active_project_id", None)
        st.rerun()
    st.caption(f"{current_user['display_name']} · {current_role}")
    pages = [
        "Project overview",
        "Recommendations",
        "Experiment loop",
        "Robustness",
        "Calibration",
        "Process window",
        "Qualification",
        "Collaboration",
        "Approvals & dossier",
        "Audit trail",
    ]
    if can_edit:
        pages.append("New project")
    if can_admin:
        pages.extend(["Team", "Pilot operations"])
    page = st.radio("Go to", pages)
    if st.button("Sign out", use_container_width=True):
        for key in ["current_user", "active_organization_id", "active_project_id", "dossier_download"]:
            st.session_state.pop(key, None)
        st.rerun()

project_id = project_selector()


if page == "Team":
    st.header("Workspace team")
    members = store.list_members(organization_id, current_user["id"])
    st.dataframe(members[["display_name", "email", "role", "is_active", "last_login_at"]], use_container_width=True, hide_index=True)
    st.markdown("### Invite a member")
    with st.form("invite_member"):
        email = st.text_input("Email")
        role = st.selectbox("Role", list(ROLES), index=list(ROLES).index("scientist"))
        expires_hours = st.number_input("Invitation validity (hours)", min_value=1, max_value=720, value=72)
        submitted = st.form_submit_button("Create invitation", type="primary")
    if submitted:
        try:
            invitation = store.create_invitation(
                organization_id,
                email=email,
                role=role,
                actor_user_id=current_user["id"],
                base_url=os.environ.get("REFORMULATION_PUBLIC_URL", "http://localhost:8501"),
                expires_hours=int(expires_hours),
            )
            st.success("Invitation created and queued for delivery.")
            st.code(invitation["invite_url"])
        except Exception as exc:
            st.error(str(exc))
    invitations = store.list_invitations(organization_id, current_user["id"])
    if not invitations.empty:
        st.markdown("### Invitation history")
        st.dataframe(invitations[["email", "role", "status", "expires_at", "invited_by", "created_at"]], use_container_width=True, hide_index=True)
    st.markdown("### Email outbox")
    delivery = deliver_queued_notifications(store)
    if delivery["queued"]:
        st.info(f"{delivery['queued']} message(s) are queued. Configure SMTP environment variables to deliver them.")
    notifications = store.list_outbox(organization_id, current_user["id"], limit=50)
    if not notifications.empty:
        notifications = notifications.copy()
        notifications["action_link"] = notifications["body"].map(first_url)
        st.dataframe(
            notifications[["recipient_email", "kind", "subject", "status", "action_link", "error", "created_at"]],
            use_container_width=True,
            hide_index=True,
            column_config={"action_link": st.column_config.LinkColumn("Invitation link")},
        )
        with st.expander("Copy a queued message or link"):
            selected_notification = st.selectbox(
                "Message",
                notifications["id"].tolist(),
                format_func=lambda nid: f"{notifications[notifications['id'] == nid].iloc[0]['kind']} · {notifications[notifications['id'] == nid].iloc[0]['recipient_email']}",
            )
            selected_row = notifications[notifications["id"] == selected_notification].iloc[0]
            if selected_row.get("action_link"):
                st.code(str(selected_row["action_link"]))
            st.text_area("Message body", value=str(selected_row["body"]), height=180, disabled=True)
    st.caption(
        "The outbox shows this organization's invitation links only. Password-reset links are never "
        "shown here — they are delivered by email, or generated server-side with reset_password_cli.py. "
        "SSO and MFA remain future enterprise controls."
    )
    st.stop()


if page == "New project":
    if not can_edit:
        st.error("Your role does not permit creating projects.")
        st.stop()
    st.header("Create a project")
    demo_tab, upload_tab = st.tabs(["Start with demo", "Import your CSV"])
    with demo_tab:
        st.write("Create a coatings project with completed and failed historical trials.")
        if st.button("Create v0.6 demo project", type="primary"):
            new_id = create_demo_project()
            st.session_state["active_project_id"] = new_id
            st.success("Demo created. Generate the first recommendation batch next.")
    with upload_tab:
        uploaded = st.file_uploader("Historical experiments", type=["csv", "xlsx", "xlsm"])
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            try:
                preview = workbook_preview(file_bytes, uploaded.name)
                sheet_name = preview.sheets[0]
                if len(preview.sheets) > 1:
                    sheet_name = st.selectbox("Worksheet", preview.sheets)
                data = load_table(file_bytes, uploaded.name, sheet_name=sheet_name)
            except Exception as exc:
                st.error(f"Could not read the file: {exc}")
                st.stop()
            st.dataframe(data.head(20), use_container_width=True, hide_index=True)
            with st.expander("Import readiness"):
                st.dataframe(import_readiness_report(data), use_container_width=True, hide_index=True)
            numeric_columns = data.select_dtypes(include="number").columns.tolist()
            text_columns = [column for column in data.columns if column not in numeric_columns]
            name = st.text_input("Project name", value=Path(uploaded.name).stem.replace("_", " ").title())
            description = st.text_area("Project goal")
            mixture_columns = st.multiselect(
                "Mixture columns",
                numeric_columns,
                default=[column for column in numeric_columns if column.startswith("ingredient_")],
            )
            ingredient_to_remove = st.selectbox(
                "Ingredient to remove",
                mixture_columns if mixture_columns else ["Select mixture columns first"],
            )
            process_options = [column for column in numeric_columns if column not in mixture_columns]
            process_columns = st.multiselect("Process variables", process_options)
            categorical_columns = st.multiselect("Categorical controls", text_columns)
            response_options = [column for column in numeric_columns if column not in mixture_columns + process_columns]
            response_columns = st.multiselect("Measured responses", response_options)
            status_options = ["None", *[column for column in text_columns if column not in categorical_columns]]
            status_column = st.selectbox("Status column", status_options)
            mixture_total = st.number_input("Required mixture total", min_value=0.001, value=100.0)

            mixture_bounds: dict[str, list[float]] = {}
            if mixture_columns:
                st.markdown("#### Mixture bounds")
                inferred = infer_numeric_bounds(data, mixture_columns, expansion_fraction=0.05, floor=0.0)
                cols = st.columns(min(3, len(mixture_columns)))
                for idx, column in enumerate(mixture_columns):
                    with cols[idx % len(cols)]:
                        if column == ingredient_to_remove:
                            mixture_bounds[column] = [0.0, 0.0]
                            st.caption(f"{column}: fixed at 0")
                        else:
                            lo = st.number_input(f"Min {column}", value=float(max(0, inferred[column][0])), key=f"mix_lo_{column}")
                            hi = st.number_input(f"Max {column}", value=float(max(inferred[column][1], lo + 0.1)), key=f"mix_hi_{column}")
                            mixture_bounds[column] = [lo, hi]

            process_bounds: dict[str, list[float]] = {}
            if process_columns:
                st.markdown("#### Process bounds")
                inferred_process = infer_numeric_bounds(data, process_columns)
                for column in process_columns:
                    c1, c2 = st.columns(2)
                    lo = c1.number_input(f"Min {column}", value=float(inferred_process[column][0]), key=f"proc_lo_{column}")
                    hi = c2.number_input(f"Max {column}", value=float(inferred_process[column][1]), key=f"proc_hi_{column}")
                    process_bounds[column] = [lo, hi]

            response_specs: list[dict[str, Any]] = []
            if response_columns:
                st.markdown("#### Product specifications")
                for response in response_columns:
                    values = pd.to_numeric(data[response], errors="coerce").dropna()
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**{response}**")
                    use_min = c2.checkbox("Use minimum", key=f"use_min_{response}")
                    use_max = c3.checkbox("Use maximum", key=f"use_max_{response}")
                    minimum = c2.number_input("Minimum", value=float(values.quantile(0.25)), key=f"min_{response}") if use_min else None
                    maximum = c3.number_input("Maximum", value=float(values.quantile(0.75)), key=f"max_{response}") if use_max else None
                    response_specs.append({"response": response, "minimum": minimum, "maximum": maximum, "weight": 1.0})

            can_create = bool(name and mixture_columns and response_specs and all(s["minimum"] is not None or s["maximum"] is not None for s in response_specs))
            if st.button("Create imported project", type="primary", disabled=not can_create):
                completed = data.copy()
                if status_column != "None":
                    completed = data[data[status_column].astype(str).str.lower().isin(["completed", "complete", "success", "successful", "ok", "passed"])]
                if completed.empty:
                    st.error("No completed experiments were found for the baseline.")
                    st.stop()
                feature_columns = [*mixture_columns, *process_columns, *categorical_columns]
                base_config = {
                    "mixture_columns": mixture_columns,
                    "process_columns": process_columns,
                    "categorical_columns": categorical_columns,
                    "response_specs": response_specs,
                    "mixture_bounds": mixture_bounds,
                    "process_bounds": process_bounds,
                    "category_values": {column: sorted(data[column].dropna().astype(str).unique().tolist()) for column in categorical_columns},
                    "mixture_total": mixture_total,
                    "ingredient_to_remove": ingredient_to_remove,
                    "baseline": completed.iloc[0][feature_columns].to_dict(),
                    "ingredient_costs": {column: 1.0 for column in mixture_columns},
                    "status_column": None if status_column == "None" else status_column,
                    "n_recommendations": 5,
                    "candidate_pool_size": 2500,
                    "min_distance": 0.06,
                }
                config = ensure_v04_config(base_config)
                config["manufacturing_variation"] = default_variation_config(config)
                new_id = store.create_project(
                    name,
                    config,
                    description=description,
                    source_filename=f"{uploaded.name}::{sheet_name}" if len(preview.sheets) > 1 else uploaded.name,
                    organization_id=organization_id,
                    created_by_user_id=current_user["id"],
                )
                store.import_history(new_id, data)
                st.session_state["active_project_id"] = new_id
                st.success("Project created.")
    st.stop()

if project_id is None:
    st.info("Create a demo or imported project to begin.")
    if can_edit and st.button("Create demo project", type="primary"):
        st.session_state["active_project_id"] = create_demo_project()
        st.rerun()
    elif not can_edit:
        st.caption("Ask a workspace scientist or administrator to create the first project.")
    st.stop()

store.require_project_access(current_user["id"], project_id)
project = store.get_project(project_id)
config = ensure_v04_config(project["config"])
if config != project["config"] and can_edit:
    store.update_project_config(project_id, config)
response_columns = [item["response"] for item in config["response_specs"]]


if page == "Collaboration":
    st.header("Project collaboration")
    comments_tab, tasks_tab = st.tabs(["Comments", "Assignments"])
    with comments_tab:
        comments = store.list_comments(project_id, current_user["id"])
        if comments.empty:
            st.info("No project comments yet.")
        else:
            view = comments[["id", "author_name", "body", "entity_type", "entity_id", "created_at", "resolved_at"]]
            st.dataframe(view, use_container_width=True, hide_index=True)
        with st.form("add_project_comment"):
            comment_body = st.text_area("Add a comment")
            submitted = st.form_submit_button("Post comment", type="primary")
        if submitted:
            try:
                store.add_comment(project_id, author_user_id=current_user["id"], body=comment_body)
                st.success("Comment added.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        unresolved = comments[comments["resolved_at"].isna()] if not comments.empty else comments
        if not unresolved.empty:
            selected_comment = st.selectbox(
                "Resolve comment",
                unresolved["id"].tolist(),
                format_func=lambda cid: str(unresolved[unresolved["id"] == cid].iloc[0]["body"])[:90],
            )
            if st.button("Mark selected comment resolved"):
                try:
                    store.resolve_comment(selected_comment, current_user["id"])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with tasks_tab:
        assignments = store.list_assignments(project_id, current_user["id"])
        if assignments.empty:
            st.info("No assignments yet.")
        else:
            st.dataframe(
                assignments[["id", "title", "assignee_name", "status", "priority", "due_at", "created_by_name", "created_at"]],
                use_container_width=True,
                hide_index=True,
            )
        members = store.list_members(organization_id, current_user["id"])
        if current_role in {"owner", "admin", "scientist", "approver"}:
            with st.form("create_assignment"):
                title = st.text_input("Assignment title")
                description = st.text_area("Description")
                assignee = st.selectbox(
                    "Assignee",
                    members["id"].tolist(),
                    format_func=lambda uid: str(members[members["id"] == uid].iloc[0]["display_name"]),
                )
                priority = st.selectbox("Priority", list(PRIORITIES), index=list(PRIORITIES).index("normal"))
                due_at = st.text_input("Due date or timestamp (optional)")
                created = st.form_submit_button("Create assignment", type="primary")
            if created:
                try:
                    store.create_assignment(
                        project_id,
                        title=title,
                        description=description,
                        assignee_user_id=assignee,
                        created_by_user_id=current_user["id"],
                        priority=priority,
                        due_at=due_at.strip() or None,
                    )
                    st.success("Assignment created.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        open_tasks = assignments[~assignments["status"].isin(["done", "cancelled"])] if not assignments.empty else assignments
        if not open_tasks.empty:
            selected_task = st.selectbox(
                "Update assignment",
                open_tasks["id"].tolist(),
                format_func=lambda aid: str(open_tasks[open_tasks["id"] == aid].iloc[0]["title"]),
            )
            new_status = st.selectbox("New status", list(TASK_STATUSES), index=list(TASK_STATUSES).index("in_progress"))
            if st.button("Update assignment status"):
                try:
                    store.update_assignment(selected_task, current_user["id"], status=new_status)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

elif page == "Pilot operations":
    st.header("Pilot operations")
    st.markdown("### Encrypted qualification artifacts")
    if st.button("Generate and encrypt current dossier", type="primary"):
        dossier_bytes, manifest = generate_dossier(store, project_id, generated_by_user_id=current_user["id"])
        filename = f"{project['name'].lower().replace(' ', '_')}_qualification_dossier_v{manifest['version']}.zip"
        artifact_id = vault.store_project_artifact(
            store,
            project_id,
            created_by_user_id=current_user["id"],
            payload=dossier_bytes,
            filename=filename,
            artifact_type="qualification_dossier",
            content_type="application/zip",
            metadata=manifest,
        )
        st.success(f"Encrypted artifact stored: {artifact_id}")
    artifacts = store.list_artifacts(project_id, current_user["id"])
    if not artifacts.empty:
        st.dataframe(
            artifacts[["id", "artifact_type", "filename", "created_by", "size_bytes", "plaintext_sha256", "created_at"]],
            use_container_width=True,
            hide_index=True,
        )
        selected_artifact = st.selectbox("Artifact to decrypt", artifacts["id"].tolist(), format_func=lambda aid: str(artifacts[artifacts["id"] == aid].iloc[0]["filename"]))
        if selected_artifact:
            payload, record = vault.retrieve_project_artifact(store, selected_artifact, current_user["id"])
            st.download_button("Download verified decrypted artifact", payload, file_name=record["filename"], mime=record["content_type"], use_container_width=True)

    st.markdown("### Verified encrypted backups")
    if st.button("Create verified backup now"):
        backup_id, _ = create_backup(store, vault, organization_id=organization_id, created_by_user_id=current_user["id"])
        st.success(f"Backup created and verified: {backup_id}")
    backups = store.list_backups(organization_id=organization_id)
    if not backups.empty:
        st.dataframe(backups[["id", "filename", "status", "size_bytes", "created_at", "verified_at"]], use_container_width=True, hide_index=True)

    st.markdown("### PostgreSQL migration bundle")
    st.caption("The included application still runs on SQLite. This bundle supports a controlled migration into an already-provisioned PostgreSQL schema.")
    if st.button("Build migration bundle"):
        migration_bytes, manifest = create_postgres_migration_bundle(store.database_path)
        st.session_state["postgres_bundle"] = migration_bytes
        st.success(f"Exported {sum(manifest['table_counts'].values())} records across {len(manifest['table_counts'])} tables.")
    if st.session_state.get("postgres_bundle"):
        st.download_button("Download PostgreSQL migration bundle", st.session_state["postgres_bundle"], file_name="reformulation_postgres_migration.zip", mime="application/zip", use_container_width=True)

elif page == "Project overview":
    st.header(project["name"])
    st.write(project["description"] or "No project description provided.")
    history = store.project_dataframe(project_id)
    batches = store.list_batches(project_id)
    snapshots = store.list_snapshots(project_id)
    progress = qualification_progress(store, project_id)
    calibration = calibration_report(store, project_id)

    metrics = st.columns(6)
    metrics[0].metric("Experiments", len(history))
    metrics[1].metric("Batches", len(batches))
    metrics[2].metric("Completed live", progress["completed_platform_experiments"])
    metrics[3].metric("Compliant live", progress["compliant_platform_experiments"])
    metrics[4].metric("Qualification", f"{progress['score']:.0f}%")
    overview_brier = calibration.get("formulation_brier_score")
    if overview_brier is None:
        overview_brier = calibration.get("brier_score")
    metrics[5].metric("Formulation Brier", "—" if overview_brier is None else f"{overview_brier:.3f}")

    stage_view = progress["stage_progress"].copy()
    stage_view["completion"] = stage_view["completion"].map(lambda x: f"{x:.0%}")
    stage_view["success_rate"] = stage_view["success_rate"].map(lambda x: f"{x:.0%}")
    st.markdown("### Qualification gates")
    st.dataframe(stage_view, use_container_width=True, hide_index=True)

    if not snapshots.empty:
        st.markdown("### Learning history")
        chart = snapshots[["created_at", "best_success_probability", "qualification_score"]].copy()
        chart["created_at"] = pd.to_datetime(chart["created_at"])
        st.line_chart(chart.set_index("created_at"), use_container_width=True)
    with st.expander("Project configuration"):
        st.json(config)


elif page == "Recommendations":
    st.header("Recommendation batches")
    batches = store.list_batches(project_id)
    open_proposed = batches[batches["status"] == "proposed"] if not batches.empty else pd.DataFrame()
    if open_proposed.empty:
        if st.button("Generate next batch", type="primary", use_container_width=True, disabled=not can_edit):
            with st.spinner("Training models and searching feasible reformulations..."):
                _, batch_id = create_recommendation_batch(store, project_id)
            set_flash(f"Batch {store.get_batch(batch_id)['batch_number']} generated. Review and approve it before running experiments.")
            st.rerun()
    else:
        batch = open_proposed.iloc[0]
        batch_id = str(batch["id"])
        st.subheader(f"Proposed batch {int(batch['batch_number'])}")
        st.info(f"**{batch['decision']}** — {batch['decision_reason']}")
        experiments = store.list_experiments(project_id, batch_id=batch_id)
        evidence = []
        display_columns = ["experiment_code", "purpose", *config["mixture_columns"], *config.get("process_columns", []), *config.get("categorical_columns", [])]
        for _, row in experiments.iterrows():
            record = {column: row.get(column) for column in display_columns}
            metadata = row["recommendation"]
            record.update({
                "probability_all_specs": metadata.get("probability_all_specs"),
                "probability_feasible": metadata.get("probability_feasible"),
                "projected_ingredient_cost": metadata.get("projected_ingredient_cost"),
                "extrapolation_warning": metadata.get("extrapolation_warning"),
            })
            evidence.append(record)
        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
        st.caption(
            "probability_all_specs is a modeled estimate that assumes responses are independent; "
            "probability_feasible is a separate, uncalibrated estimate. Both are hypotheses to "
            "test in the lab, not guarantees — the calibration page tracks how honest they turn out to be."
        )
        if st.button("Approve and freeze batch", type="primary", use_container_width=True, disabled=not can_edit):
            store.approve_batch(batch_id)
            set_flash(f"Batch {int(batch['batch_number'])} approved and its predictions frozen.")
            st.rerun()

    batches = store.list_batches(project_id)
    if not batches.empty:
        st.markdown("### Batch history")
        st.dataframe(
            batches[["batch_number", "status", "decision", "experiment_count", "resolved_count", "created_at", "completed_at"]],
            use_container_width=True,
            hide_index=True,
        )
        if can_edit:
            with st.expander("Close a batch or cancel unused experiments"):
                management_batch_id = st.selectbox(
                    "Batch to manage",
                    batches["id"].tolist(),
                    format_func=lambda value: f"Batch {int(batches[batches['id'] == value].iloc[0]['batch_number'])} · {batches[batches['id'] == value].iloc[0]['status']}",
                    key="batch_management_id",
                )
                batch_experiments = store.list_experiments(project_id, batch_id=management_batch_id)
                unresolved = batch_experiments[~batch_experiments["status"].isin(["completed", "failed", "cancelled"])]
                if unresolved.empty:
                    st.success("This batch has no unresolved experiments.")
                else:
                    selected_unused = st.multiselect(
                        "Unused experiments to cancel",
                        unresolved["id"].tolist(),
                        default=unresolved["id"].tolist(),
                        format_func=lambda eid: f"{unresolved[unresolved['id'] == eid].iloc[0]['experiment_code']} · {unresolved[unresolved['id'] == eid].iloc[0]['status']}",
                    )
                    if st.button("Cancel selected unused experiments", disabled=not selected_unused):
                        cancelled = store.cancel_experiments(selected_unused)
                        set_flash(f"Cancelled {cancelled} unused experiment(s).")
                        st.rerun()
                cancel_on_close = st.checkbox("Cancel every unresolved experiment when closing", value=not unresolved.empty)
                if st.button("Formally close batch", type="primary"):
                    try:
                        result = store.close_batch(management_batch_id, cancel_unresolved=cancel_on_close)
                        set_flash(f"Batch closed. {result['cancelled']} unresolved experiment(s) were cancelled.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))


elif page == "Experiment loop":
    st.header("Run experiments and enter results")
    batches = store.list_batches(project_id)
    active = batches[batches["status"].isin(["approved", "running", "completed"])] if not batches.empty else pd.DataFrame()
    if active.empty:
        st.info("Approve a recommendation or process-window batch first.")
        st.stop()
    batch_id = st.selectbox(
        "Batch",
        active["id"].tolist(),
        format_func=lambda value: f"Batch {int(active[active['id'] == value].iloc[0]['batch_number'])} · {active[active['id'] == value].iloc[0]['status']}",
    )
    experiments = store.list_experiments(project_id, batch_id=batch_id)
    status_table = experiments[["experiment_code", "purpose", "replicate_group", "replicate_index", "status", "qualification_stage"]].copy()
    st.dataframe(status_table, use_container_width=True, hide_index=True)
    selectable = experiments[experiments["status"] != "cancelled"]
    if selectable.empty:
        st.info("Every experiment in this batch is cancelled or resolved. Close the batch from Recommendations.")
        st.stop()
    experiment_id = st.selectbox(
        "Experiment",
        selectable["id"].tolist(),
        format_func=lambda value: f"{selectable[selectable['id'] == value].iloc[0]['experiment_code']} · {selectable[selectable['id'] == value].iloc[0]['purpose']}",
    )
    experiment = selectable[selectable["id"] == experiment_id].iloc[0]
    condition_columns = [*config["mixture_columns"], *config.get("process_columns", []), *config.get("categorical_columns", [])]
    st.dataframe(pd.DataFrame([{column: experiment.get(column) for column in condition_columns}]), use_container_width=True, hide_index=True)

    group = store.replicate_group_status(experiment_id)
    completed_count = int((group["status"] == "completed").sum())
    unresolved_count = int((~group["status"].isin(["completed", "failed", "cancelled"])).sum())
    cancelled_count = int((group["status"] == "cancelled").sum())
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Replicate group", str(experiment.get("replicate_group") or experiment["experiment_code"]))
    g2.metric("Total runs", len(group))
    g3.metric("Completed", completed_count)
    g4.metric("Remaining", unresolved_count)
    with st.expander("Replicate group details", expanded=unresolved_count > 0):
        st.dataframe(
            group[["experiment_code", "replicate_index", "status", "qualification_stage"]],
            use_container_width=True,
            hide_index=True,
        )
        if cancelled_count:
            st.caption(f"{cancelled_count} cancelled run(s) are retained in the audit history but excluded from evidence.")

    if can_edit:
        with st.expander("Create missing linked replicates"):
            target_default = max(3, len(group))
            target_total = st.number_input(
                "Target total runs in this replicate group",
                min_value=1,
                max_value=12,
                value=int(target_default),
                help="The app creates only the missing runs. Re-entering the same target cannot create duplicates.",
            )
            missing_target = max(int(target_total) - len(group), 0)
            st.caption(f"Existing: {len(group)} · Missing to target: {missing_target}")
            if st.button("Create missing replicates", use_container_width=True, disabled=missing_target == 0):
                try:
                    created = store.ensure_replicate_count(experiment_id, int(target_total))
                    set_flash(f"Created {len(created)} linked replicate(s). The group now contains {int(target_total)} runs.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with st.form("record_result"):
        status_options = ["planned", "running", "completed", "failed", "cancelled"]
        status = st.selectbox("Status", status_options, index=status_options.index(experiment["status"]) if experiment["status"] in status_options else 0)
        stage = st.selectbox("Qualification stage", QUALIFICATION_STAGES, index=QUALIFICATION_STAGES.index(experiment["qualification_stage"]) if experiment["qualification_stage"] in QUALIFICATION_STAGES else 0)
        st.caption("All measurements are required and must be non-zero before a run can be marked completed.")
        result_values: dict[str, float | None] = {}
        response_cols = st.columns(min(4, len(response_columns)))
        for index, response in enumerate(response_columns):
            with response_cols[index % len(response_cols)]:
                existing = experiment.get(response)
                default = float(existing) if existing is not None and pd.notna(existing) else None
                result_values[response] = st.number_input(
                    response,
                    value=default,
                    placeholder="Required for completed runs",
                    key=f"result_{experiment_id}_{response}",
                )
        notes = st.text_area("Notes", value=str(experiment.get("notes", "")))
        auto_retrain = st.checkbox("Retrain after final result", value=True)
        submitted = st.form_submit_button("Save experiment update", type="primary", use_container_width=True, disabled=not can_edit)
    if submitted:
        try:
            store.update_experiment(
                experiment_id,
                status=status,
                responses=result_values if status == "completed" else None,
                notes=notes,
                qualification_stage=stage,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            message = f"{experiment['experiment_code']} saved as {status}."
            if auto_retrain and status in {"completed", "failed"}:
                with st.spinner("Retraining models and updating evidence..."):
                    try:
                        result, new_batch_id = refresh_after_result(store, project_id)
                        message += f" Current decision: {result.decision}."
                        if new_batch_id:
                            message += " A new recommendation batch was proposed."
                    except Exception as exc:
                        set_flash(message + f" Retraining did not finish: {exc}", "warning")
                        st.rerun()
            set_flash(message)
            st.rerun()


elif page == "Robustness":
    st.header("Manufacturing-variation simulation")
    experiments = store.list_experiments(project_id, source_type="recommended")
    if experiments.empty:
        st.info("Generate a recommendation batch first.")
        st.stop()
    experiment_id = st.selectbox(
        "Candidate",
        experiments["id"].tolist(),
        format_func=lambda value: experiments[experiments["id"] == value].iloc[0]["experiment_code"],
    )
    candidate_row = experiments[experiments["id"] == experiment_id].iloc[0]
    defaults = {**default_variation_config(config), **config.get("manufacturing_variation", {})}
    st.write("Enter expected one-standard-deviation manufacturing tolerances.")
    variation: dict[str, float] = {}
    numeric_columns = [*config["mixture_columns"], *config.get("process_columns", [])]
    cols = st.columns(min(3, len(numeric_columns)))
    for idx, column in enumerate(numeric_columns):
        with cols[idx % len(cols)]:
            variation[column] = st.number_input(
                f"σ {column}",
                min_value=0.0,
                value=float(defaults.get(column, 0.0)),
                key=f"variation_{column}",
            )
    simulations = st.select_slider("Monte Carlo runs", options=[250, 500, 1000, 2500, 5000], value=1000)
    if st.button("Run robustness simulation", type="primary", use_container_width=True, disabled=not can_edit):
        candidate = candidate_row.to_dict()
        candidate.update(candidate_row.get("recommendation") or {})
        with st.spinner("Perturbing process inputs and sampling response uncertainty..."):
            result = simulate_manufacturing_variation(
                store.project_dataframe(project_id),
                config=config,
                candidate=candidate,
                variation_std=variation,
                n_simulations=int(simulations),
            )
            store.save_robustness_run(
                project_id,
                experiment_id,
                simulation_count=int(simulations),
                variation=variation,
                result=result_for_storage(result),
            )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Optimizer nominal",
            "—" if result["optimizer_nominal_success_probability"] is None else f"{result['optimizer_nominal_success_probability']:.0%}",
            help="The recommendation engine's point estimate at the exact proposed settings.",
        )
        m2.metric(
            "Monte Carlo nominal",
            f"{result['monte_carlo_nominal_success_probability']:.0%}",
            help="The exact nominal settings evaluated using the same Monte Carlo method as the robustness calculation.",
        )
        m3.metric("Robust success", f"{result['robust_success_probability']:.0%}")
        m4.metric("Disposition", result["recommended_disposition"])
        st.info(
            "Why can the optimizer nominal and robust numbers differ? The optimizer combines model probabilities "
            "using its recommendation-scoring method, while the robustness study jointly samples response "
            "uncertainty and manufacturing variation. For an apples-to-apples comparison, compare **Monte Carlo "
            "nominal** with **Robust success**. Robust success can occasionally be slightly higher because clipping "
            "and mixture rebalancing may move simulated runs toward safer regions, or because of Monte Carlo noise."
        )
        st.dataframe(result["response_summary"], use_container_width=True, hide_index=True)
        st.markdown("### Strongest sensitivities")
        st.dataframe(result["sensitivity"].groupby("response").head(3), use_container_width=True, hide_index=True)
    runs = store.list_robustness_runs(project_id)
    if not runs.empty:
        summary = []
        for _, row in runs.iterrows():
            summary.append({
                "created_at": row["created_at"],
                "experiment_code": row["experiment_code"],
                "simulation_count": row["simulation_count"],
                "optimizer_nominal_probability": row["result"].get("optimizer_nominal_success_probability", row["result"].get("nominal_success_probability")),
                "monte_carlo_nominal_probability": row["result"].get("monte_carlo_nominal_success_probability"),
                "robust_success_probability": row["result"].get("robust_success_probability"),
                "disposition": row["result"].get("recommended_disposition"),
            })
        st.markdown("### Saved robustness evidence")
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


elif page == "Calibration":
    st.header("Predicted-versus-actual calibration")
    report = calibration_report(store, project_id)
    run_report = report["run_level"]
    formulation_report = report["formulation_level"]
    if run_report["observations"].empty:
        st.info("Complete platform-recommended experiments to build prospective calibration evidence.")
    else:
        st.info(
            "Run-level calibration treats every physical run separately. Formulation-level calibration averages "
            "linked replicates and counts each formulation once, which is the safer view for judging how broadly "
            "the model has been tested."
        )
        run_tab, formulation_tab = st.tabs(["Run-level calibration", "Formulation-level calibration"])

        def render_calibration_view(view: dict[str, Any], *, label: str) -> None:
            brier = view["brier_score"]
            st.metric(
                f"{label} all-specification Brier score",
                "—" if brier is None else f"{brier:.3f}",
                help="Lower is better; 0 is perfect probability calibration.",
            )
            st.markdown("#### Response accuracy")
            summary = view["response_summary"].copy()
            if not summary.empty:
                summary["coverage_90"] = summary["coverage_90"].map(lambda x: f"{x:.0%}")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.markdown("#### Prediction observations")
            st.dataframe(view["observations"], use_container_width=True, hide_index=True)
            if not view["probability_bins"].empty:
                st.markdown("#### Probability calibration")
                st.dataframe(view["probability_bins"], use_container_width=True, hide_index=True)

        with run_tab:
            st.caption("Best for monitoring repeatability and individual-run prediction error.")
            render_calibration_view(run_report, label="Run-level")
        with formulation_tab:
            st.caption("Best for assessing model generalization across distinct candidate formulations.")
            render_calibration_view(formulation_report, label="Formulation-level")


elif page == "Process window":
    st.header("Process Window Designer")
    st.write(
        "Preserve a confirmed formulation while deliberately varying selected process controls. "
        "The resulting experiments are created as a dedicated **process_window** qualification batch."
    )
    recommended = store.list_experiments(project_id, source_type="recommended")
    confirmed = recommended[
        (recommended["qualification_stage"] == "confirmation")
        & (recommended["status"] == "completed")
    ] if not recommended.empty else pd.DataFrame()
    process_columns = list(config.get("process_columns", []))
    if not process_columns:
        st.warning("This project has no configured process variables.")
        st.stop()
    if confirmed.empty:
        st.info("Complete at least one confirmation-stage experiment before designing a process-window study.")
        st.stop()

    group_options = confirmed["replicate_group"].dropna().drop_duplicates().tolist()
    source_group = st.selectbox(
        "Confirmed formulation",
        group_options,
        format_func=lambda group: f"{group} · {int((confirmed['replicate_group'] == group).sum())} completed confirmation run(s)",
    )
    group_rows = confirmed[confirmed["replicate_group"] == source_group].sort_values("replicate_index")
    source_row = group_rows.iloc[0]
    feature_columns = [*config["mixture_columns"], *process_columns, *config.get("categorical_columns", [])]
    nominal_inputs = {column: source_row.get(column) for column in feature_columns}
    st.markdown("### Confirmed nominal conditions")
    st.dataframe(pd.DataFrame([nominal_inputs]), use_container_width=True, hide_index=True)

    selected_process = st.multiselect(
        "Process variables to challenge",
        process_columns,
        default=process_columns[: min(2, len(process_columns))],
        help="Select up to three variables for a full grid. The formulation itself remains unchanged.",
    )
    if len(selected_process) > 3:
        st.warning("Select at most three process variables for a practical qualification matrix.")
    mode_label = st.selectbox("Design", list(DESIGN_MODES), index=1 if len(DESIGN_MODES) > 1 else 0)
    mode = DESIGN_MODES[mode_label]
    deltas: dict[str, float] = {}
    variation_defaults = {**default_variation_config(config), **config.get("manufacturing_variation", {})}
    if selected_process:
        st.markdown("### Low/high excursions from nominal")
        delta_cols = st.columns(min(3, len(selected_process)))
        for index, column in enumerate(selected_process):
            nominal = float(nominal_inputs[column])
            lo, hi = map(float, config["process_bounds"][column])
            default_delta = min(max(float(variation_defaults.get(column, 0.1)) * 3.0, 0.1), max(nominal - lo, hi - nominal))
            with delta_cols[index % len(delta_cols)]:
                deltas[column] = st.number_input(
                    f"± {column}",
                    min_value=0.0001,
                    max_value=float(max(hi - lo, 0.0001)),
                    value=float(default_delta),
                    help=f"Allowed project range: {lo:g} to {hi:g}; nominal: {nominal:g}",
                    key=f"process_window_delta_{column}",
                )

    preview = pd.DataFrame()
    design_error = None
    if selected_process and len(selected_process) <= 3:
        try:
            preview = design_process_window(
                config=config,
                nominal_inputs=nominal_inputs,
                source_replicate_group=str(source_group),
                process_columns=selected_process,
                deltas=deltas,
                mode=mode,
            )
        except Exception as exc:
            design_error = str(exc)
    if design_error:
        st.error(design_error)
    elif not preview.empty:
        st.markdown(f"### Proposed process-window matrix · {len(preview)} runs")
        preview_columns = ["purpose", *config["mixture_columns"], *process_columns, *config.get("categorical_columns", [])]
        st.dataframe(preview[preview_columns], use_container_width=True, hide_index=True)
        fingerprint = str(preview.iloc[0]["design_fingerprint"])
        duplicate = store.has_design_fingerprint(project_id, fingerprint)
        if duplicate:
            st.warning("This exact process-window design already exists. Change the variables, excursions, or design type instead of creating a duplicate.")
        if st.button(
            "Create and approve process-window batch",
            type="primary",
            use_container_width=True,
            disabled=(not can_edit) or duplicate,
        ):
            batch_id = store.create_batch(
                project_id,
                preview,
                decision="RUN PROCESS WINDOW",
                decision_reason=(
                    f"Challenge confirmed formulation {source_group} across {len(preview)} bounded process settings "
                    f"using the {mode_label.lower()} design."
                ),
                qualification_stage="process_window",
            )
            store.approve_batch(batch_id)
            batch_number = store.get_batch(batch_id)["batch_number"]
            set_flash(f"Process-window batch {batch_number} created and approved with {len(preview)} planned runs.")
            st.rerun()

    existing_window = recommended[recommended["qualification_stage"] == "process_window"] if not recommended.empty else pd.DataFrame()
    if not existing_window.empty:
        st.markdown("### Existing process-window evidence")
        st.dataframe(
            existing_window[["experiment_code", "purpose", "status", "replicate_group", *process_columns]],
            use_container_width=True,
            hide_index=True,
        )


elif page == "Qualification":
    st.header("Path to qualification")
    progress = qualification_progress(store, project_id)
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall progress", f"{progress['score']:.0f}%")
    m2.metric("All gates passed", "Yes" if progress["all_gates_passed"] else "No")
    m3.metric("Best robust probability", "—" if progress["best_robust_probability"] is None else f"{progress['best_robust_probability']:.0%}")
    stage_view = progress["stage_progress"].copy()
    stage_view["completion"] = stage_view["completion"].map(lambda x: f"{x:.0%}")
    stage_view["success_rate"] = stage_view["success_rate"].map(lambda x: f"{x:.0%}")
    st.dataframe(stage_view, use_container_width=True, hide_index=True)

    if not progress["replicate_summary"].empty:
        st.markdown("### Replicate repeatability")
        st.caption(
            "Two-stage check, after Donald Wheeler: first each replicate is judged against "
            "limits built from its siblings (the consistency screen), and only groups whose "
            "replicates agree are scored on CV. With 3-6 replicates the screen is indicative, "
            "not definitive."
        )
        st.dataframe(progress["replicate_summary"], use_container_width=True, hide_index=True)
        with st.expander("Running records — look before computing"):
            replicate_experiments = store.list_experiments(project_id, source_type="recommended")
            if not replicate_experiments.empty:
                completed_runs = replicate_experiments[replicate_experiments["status"] == "completed"]
                group_labels = sorted(
                    str(g) for g in completed_runs.get("replicate_group", pd.Series(dtype=str)).dropna().unique()
                )
                if group_labels:
                    chosen_group = st.selectbox("Replicate group", group_labels, key="running_record_group")
                    group_frame = completed_runs[completed_runs["replicate_group"].astype(str) == chosen_group]
                    response_names = [item["response"] for item in config["response_specs"]]
                    for response_name in response_names:
                        series = pd.to_numeric(group_frame.get(response_name), errors="coerce").dropna()
                        if len(series) >= 2:
                            st.caption(f"{response_name} — {len(series)} replicates in run order")
                            st.line_chart(series.reset_index(drop=True))

    st.markdown("### Configure qualification gates")
    selected_stage = st.selectbox("Stage to configure", QUALIFICATION_STAGES)
    gate = config["qualification_gates"][selected_stage]
    with st.form("gate_config"):
        c1, c2, c3 = st.columns(3)
        required_completed = c1.number_input("Required completed", min_value=0, value=int(gate["required_completed"]))
        required_compliant = c2.number_input("Required compliant", min_value=0, value=int(gate["required_compliant"]))
        minimum_success_rate = c3.number_input("Minimum success rate", min_value=0.0, max_value=1.0, value=float(gate["minimum_success_rate"]), step=0.05)
        c4, c5 = st.columns(2)
        required_groups = c4.number_input("Required replicate groups", min_value=0, value=int(gate["required_replicate_groups"]))
        min_replicates = c5.number_input("Minimum replicates per group", min_value=1, value=int(gate["minimum_replicates_per_group"]))
        use_robust = st.checkbox("Require robustness probability", value=gate.get("minimum_robust_probability") is not None)
        robust_threshold = st.number_input("Minimum robust probability", min_value=0.0, max_value=1.0, value=float(gate.get("minimum_robust_probability") or 0.80), step=0.05, disabled=not use_robust)
        cv_values: dict[str, float] = {}
        if required_groups > 0:
            st.caption("Maximum coefficient of variation for qualifying replicate groups")
            cols = st.columns(min(4, len(response_columns)))
            for idx, response in enumerate(response_columns):
                with cols[idx % len(cols)]:
                    cv_values[response] = st.number_input(
                        response,
                        min_value=0.0,
                        max_value=1.0,
                        value=float(gate.get("max_cv_by_response", {}).get(response, 0.08)),
                        step=0.01,
                        key=f"cv_gate_{selected_stage}_{response}",
                    )
        if st.form_submit_button("Save gate configuration", type="primary", disabled=not can_edit):
            updated = ensure_v04_config(config)
            updated["qualification_gates"][selected_stage] = {
                "required_completed": int(required_completed),
                "required_compliant": int(required_compliant),
                "minimum_success_rate": float(minimum_success_rate),
                "required_replicate_groups": int(required_groups),
                "minimum_replicates_per_group": int(min_replicates),
                "max_cv_by_response": cv_values if required_groups > 0 else {},
                "minimum_robust_probability": float(robust_threshold) if use_robust else None,
            }
            store.update_project_config(project_id, updated)
            st.success("Qualification gate updated.")
            st.rerun()


elif page == "Approvals & dossier":
    st.header("Approvals and qualification dossier")
    evidence_snapshot, evidence_hash = evidence_snapshot_and_hash(store, project_id)
    progress = qualification_progress(store, project_id)
    approvals = store.list_approvals(project_id)

    m1, m2, m3 = st.columns(3)
    m1.metric("Qualification progress", f"{progress['score']:.0f}%")
    m2.metric("All gates passed", "Yes" if progress["all_gates_passed"] else "No")
    current_signatures = 0
    if not approvals.empty:
        current_signatures = int(((approvals["status"] == "signed") & (approvals["evidence_hash"] == evidence_hash)).sum())
    m3.metric("Current signatures", current_signatures)
    st.caption(f"Current scientific evidence SHA-256: `{evidence_hash}`")

    if not approvals.empty:
        approval_view = approvals.copy()
        approval_view["matches_current_evidence"] = approval_view["evidence_hash"] == evidence_hash
        st.markdown("### Approval history")
        st.dataframe(
            approval_view[["stage", "status", "signer_name", "signer_role", "signature_meaning", "signed_at", "matches_current_evidence", "comment"]],
            use_container_width=True,
            hide_index=True,
        )
        if (~approval_view["matches_current_evidence"]).any():
            st.warning("Some signatures refer to an earlier evidence state. New results or configuration changes require a new signature.")
        with st.expander("View the exact evidence a signature covers"):
            viewer_rows = approvals.reset_index(drop=True)
            viewer_labels = [
                f"{row['stage']} · {row['signer_name']} · {row['signed_at']} · {row['status']}"
                for _, row in viewer_rows.iterrows()
            ]
            chosen = st.selectbox(
                "Signature",
                range(len(viewer_labels)),
                format_func=lambda idx: viewer_labels[idx],
                key="signed_snapshot_choice",
            )
            chosen_row = viewer_rows.iloc[int(chosen)]
            stored_snapshot = chosen_row.get("evidence_snapshot") if "evidence_snapshot" in viewer_rows.columns else None
            if isinstance(stored_snapshot, str) and stored_snapshot:
                recomputed_hash = sha256(stored_snapshot.encode("utf-8")).hexdigest()
                if recomputed_hash == str(chosen_row["evidence_hash"]):
                    st.success("Stored snapshot verifies: re-hashing it reproduces the hash recorded at signing.")
                else:
                    st.error("Integrity problem: the stored snapshot no longer matches the hash recorded at signing.")
                st.download_button(
                    "Download signed evidence snapshot (JSON)",
                    data=stored_snapshot.encode("utf-8"),
                    file_name=f"signed_evidence_{chosen_row['stage']}_{str(chosen_row['evidence_hash'])[:12]}.json",
                    mime="application/json",
                    key="signed_snapshot_download",
                )
            else:
                st.caption("Signed before v0.8.0, when only the evidence hash was stored. The exact signed snapshot is not available for this signature.")

    st.markdown("### Multi-signer approval policies")
    policies = store.list_approval_policies(project_id)
    policy_status = store.approval_policy_status(project_id, evidence_hash)
    if not policy_status.empty:
        policy_view = policy_status.copy()
        policy_view["requirements"] = policy_view["requirements"].map(
            lambda items: "; ".join(f"{item['role']} {item['signed']}/{item['required']}" for item in items)
        )
        st.dataframe(policy_view[["policy_id", "stage", "name", "requirements", "complete"]], use_container_width=True, hide_index=True)
    else:
        st.info("No multi-signer policies have been configured.")

    if can_admin:
        with st.expander("Create approval policy"):
            with st.form("create_approval_policy"):
                policy_name = st.text_input("Policy name", value="Technical and quality approval")
                policy_stage = st.selectbox("Policy stage", [*QUALIFICATION_STAGES, "qualification_dossier"], key="policy_stage")
                c1, c2, c3 = st.columns(3)
                owner_count = c1.number_input("Owners required", min_value=0, max_value=5, value=1)
                admin_count = c2.number_input("Admins required", min_value=0, max_value=5, value=0)
                approver_count = c3.number_input("Approvers required", min_value=0, max_value=5, value=1)
                create_policy = st.form_submit_button("Create policy", type="primary")
            if create_policy:
                requirements = [
                    {"role": role, "count": int(count)}
                    for role, count in [("owner", owner_count), ("admin", admin_count), ("approver", approver_count)]
                    if int(count) > 0
                ]
                try:
                    store.create_approval_policy(
                        project_id,
                        stage=policy_stage,
                        name=policy_name,
                        requirements=requirements,
                        actor_user_id=current_user["id"],
                    )
                    st.success("Approval policy created.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("### Sign current evidence")
    if not can_approve:
        st.info("Your role can review and export evidence but cannot sign approvals.")
    else:
        active_policies = store.list_approval_policies(project_id, active_only=True)
        policy_options = ["Unscoped signature", *active_policies["id"].tolist()] if not active_policies.empty else ["Unscoped signature"]
        with st.form("approval_signature"):
            policy_choice = st.selectbox(
                "Approval policy",
                policy_options,
                format_func=lambda value: value if value == "Unscoped signature" else str(active_policies[active_policies["id"] == value].iloc[0]["name"]),
            )
            if policy_choice == "Unscoped signature":
                stage = st.selectbox("Approval stage", [*QUALIFICATION_STAGES, "qualification_dossier"])
                selected_policy_id = None
            else:
                selected_policy = active_policies[active_policies["id"] == policy_choice].iloc[0]
                stage = str(selected_policy["stage"])
                selected_policy_id = str(selected_policy["id"])
                st.caption(f"Stage: {stage} · Requirements: {selected_policy['requirements']}")
            meaning = st.selectbox(
                "Signature meaning",
                [
                    "I reviewed the evidence for this stage and approve progression.",
                    "I reviewed the evidence and approve the qualification dossier for internal use.",
                    "I reviewed the evidence and approve with the limitations noted below.",
                ],
            )
            typed_name = st.text_input("Type your full account name", value=current_user["display_name"])
            comment = st.text_area("Approval comment or limitations")
            password = st.text_input("Re-enter your password", type="password")
            acknowledge = st.checkbox("I understand this signature is attached to the current evidence snapshot and its hash.")
            signed = st.form_submit_button("Sign approval", type="primary")
        if signed:
            if not acknowledge:
                st.error("Check the evidence-hash acknowledgement before signing.")
            else:
                try:
                    store.sign_approval(
                        project_id,
                        stage=stage,
                        signer_user_id=current_user["id"],
                        typed_name=typed_name,
                        password=password,
                        signature_meaning=meaning,
                        evidence_hash=evidence_hash,
                        evidence_snapshot=evidence_snapshot,
                        comment=comment,
                        policy_id=selected_policy_id,
                    )
                    set_flash("Approval signed; the frozen evidence snapshot was stored with it.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("### Export qualification dossier")
    st.write("The ZIP package includes a printable HTML dossier, scientific evidence JSON, signed evidence snapshots, CSV evidence tables, signatures, audit history, and SHA-256 checksums.")
    if st.button("Generate dossier package", type="primary", use_container_width=True):
        with st.spinner("Assembling evidence and checksums..."):
            dossier_bytes, manifest = generate_dossier(
                store,
                project_id,
                generated_by_user_id=current_user["id"],
            )
        st.session_state["dossier_download"] = {
            "project_id": project_id,
            "bytes": dossier_bytes,
            "filename": f"{project['name'].lower().replace(' ', '_')}_qualification_dossier_v{manifest['version']}.zip",
            "manifest": manifest,
        }
    download = st.session_state.get("dossier_download")
    if download and download.get("project_id") == project_id:
        st.success(f"Dossier v{download['manifest']['version']} is ready. Evidence hash: {download['manifest']['scientific_evidence_sha256'][:16]}…")
        st.download_button(
            "Download dossier ZIP",
            data=download["bytes"],
            file_name=download["filename"],
            mime="application/zip",
            use_container_width=True,
        )
    st.markdown("### Export Excel workbook")
    st.write("One click, one .xlsx: experiments, recommendations, gates, calibration, robustness, approvals, and the audit trail as tabs. Your spreadsheet stays the system of record; this hands the analysis back.")
    if st.button("Generate Excel workbook", use_container_width=True):
        with st.spinner("Assembling workbook..."):
            workbook_bytes, workbook_manifest = generate_workbook(
                store,
                project_id,
                generated_by_user_id=current_user["id"],
            )
        st.session_state["workbook_download"] = {
            "project_id": project_id,
            "bytes": workbook_bytes,
            "filename": f"{project['name'].lower().replace(' ', '_')}_workbench_export.xlsx",
            "sheet_count": len(workbook_manifest["sheets"]),
        }
    workbook_download = st.session_state.get("workbook_download")
    if workbook_download and workbook_download.get("project_id") == project_id:
        st.success(f"Workbook ready: {workbook_download['sheet_count']} tabs.")
        st.download_button(
            "Download Excel workbook (.xlsx)",
            data=workbook_download["bytes"],
            file_name=workbook_download["filename"],
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="workbook_download_button",
        )

    dossiers = store.list_dossiers(project_id)
    if not dossiers.empty:
        st.markdown("### Generated dossier history")
        st.dataframe(dossiers[["version", "evidence_hash", "generated_by", "created_at"]], use_container_width=True, hide_index=True)
    st.caption("These prototype signatures are not represented as FDA 21 CFR Part 11, EU Annex 11, or other regulated electronic-signature compliance.")


elif page == "Audit trail":
    st.header("Audit trail")
    audit = store.audit_log(project_id)
    if audit.empty:
        st.info("No recorded events.")
    else:
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.download_button("Download audit trail", audit.to_csv(index=False).encode("utf-8"), file_name="reformulation_audit_trail.csv", mime="text/csv")

st.divider()
st.caption("Prototype decision support only. Qualified professionals remain responsible for chemical safety, regulatory review, physical execution, and final product approval.")
