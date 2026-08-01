"""Pilot-ready collaboration persistence for Reformulation Assurance v0.6."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import pandas as pd

from product_store import (
    ADMIN_ROLES,
    APPROVAL_ROLES,
    EDIT_ROLES,
    ROLES,
    ProductStore,
)
from project_store import _now, dumps, loads
from security import (
    expires_at,
    generate_single_use_token,
    hash_password,
    is_expired,
    token_digest,
    validate_email,
    verify_password,
    verify_token,
)

COMMENT_ROLES = set(ROLES)
TASK_EDIT_ROLES = {"owner", "admin", "scientist", "approver"}
TASK_STATUSES = ("open", "in_progress", "blocked", "done", "cancelled")
PRIORITIES = ("low", "normal", "high", "urgent")


class PilotStore(ProductStore):
    def _initialize(self) -> None:
        super()._initialize()
        with self.connection() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS invitations (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_by_user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accepted_at TEXT,
                    accepted_by_user_id TEXT REFERENCES users(id),
                    revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS password_resets (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    token_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    used_at TEXT
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT REFERENCES organizations(id) ON DELETE CASCADE,
                    recipient_email TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                );

                CREATE TABLE IF NOT EXISTS project_comments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    author_user_id TEXT NOT NULL REFERENCES users(id),
                    entity_type TEXT NOT NULL DEFAULT 'project',
                    entity_id TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    edited_at TEXT,
                    resolved_at TEXT,
                    resolved_by_user_id TEXT REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS assignments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    entity_type TEXT NOT NULL DEFAULT 'project',
                    entity_id TEXT,
                    assignee_user_id TEXT NOT NULL REFERENCES users(id),
                    created_by_user_id TEXT NOT NULL REFERENCES users(id),
                    status TEXT NOT NULL DEFAULT 'open',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    due_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approval_policies (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    name TEXT NOT NULL,
                    requirements_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_by_user_id TEXT NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS encrypted_artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    created_by_user_id TEXT NOT NULL REFERENCES users(id),
                    artifact_type TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    plaintext_sha256 TEXT NOT NULL,
                    ciphertext_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    encryption_method TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backup_records (
                    id TEXT PRIMARY KEY,
                    organization_id TEXT REFERENCES organizations(id) ON DELETE SET NULL,
                    created_by_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
                    filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    plaintext_sha256 TEXT NOT NULL,
                    ciphertext_sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    encryption_method TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    verified_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_invites_org ON invitations(organization_id, status);
                CREATE INDEX IF NOT EXISTS idx_resets_user ON password_resets(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_comments_project ON project_comments(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_assignments_project ON assignments(project_id, status);
                CREATE INDEX IF NOT EXISTS idx_policies_project ON approval_policies(project_id, stage);
                CREATE INDEX IF NOT EXISTS idx_artifacts_project ON encrypted_artifacts(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_backups_created ON backup_records(created_at);
                """
            )
            approval_columns = self._column_names(con, "approvals")
            if "policy_id" not in approval_columns:
                con.execute("ALTER TABLE approvals ADD COLUMN policy_id TEXT")

    # ------------------------------------------------------------------
    # Outbox, invitations, and recovery
    # ------------------------------------------------------------------
    def queue_notification(
        self,
        *,
        recipient_email: str,
        kind: str,
        subject: str,
        body: str,
        organization_id: str | None = None,
    ) -> str:
        notification_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO notifications
                (id, organization_id, recipient_email, kind, subject, body, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)""",
                (notification_id, organization_id, validate_email(recipient_email), kind, subject, body, _now()),
            )
        return notification_id

    def list_notifications(self, *, status: str | None = None, limit: int = 100) -> pd.DataFrame:
        """Server-side delivery queue listing.

        Intentionally unscoped so the SMTP delivery worker can process every
        queued message, including password resets. This must never back a
        user-facing view — use :meth:`list_outbox` for anything rendered in
        the interface.
        """
        query = "SELECT * FROM notifications"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_outbox(
        self,
        organization_id: str,
        actor_user_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Organization-scoped outbox for the admin interface.

        Only messages addressed to the actor's own organization are returned,
        and password-reset messages are excluded entirely: their single-use
        links must reach only the account owner. An administrator who could
        read another user's reset link could take over that account and sign
        approvals as them, which would void the e-signature record.
        """
        self.require_role(actor_user_id, organization_id, ADMIN_ROLES)
        query = (
            "SELECT * FROM notifications "
            "WHERE organization_id = ? AND kind != 'password_reset'"
        )
        params: list[Any] = [organization_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def mark_notification(self, notification_id: str, status: str, *, error: str = "") -> None:
        if status not in {"queued", "sent", "failed"}:
            raise ValueError("unsupported notification status")
        with self.connection() as con:
            con.execute(
                "UPDATE notifications SET status = ?, error = ?, sent_at = ? WHERE id = ?",
                (status, error[:1000], _now() if status == "sent" else None, notification_id),
            )

    def create_invitation(
        self,
        organization_id: str,
        *,
        email: str,
        role: str,
        actor_user_id: str,
        base_url: str = "http://localhost:8501",
        expires_hours: int = 72,
    ) -> dict[str, str]:
        self.require_role(actor_user_id, organization_id, ADMIN_ROLES)
        if role not in ROLES:
            raise ValueError(f"unsupported role: {role}")
        email = validate_email(email)
        token = generate_single_use_token()
        invitation_id = str(uuid.uuid4())
        expiration = expires_at(hours=expires_hours)
        with self.connection() as con:
            con.execute(
                """UPDATE invitations SET status = 'revoked', revoked_at = ?
                WHERE organization_id = ? AND email = ? AND status = 'pending'""",
                (_now(), organization_id, email),
            )
            con.execute(
                """INSERT INTO invitations
                (id, organization_id, email, role, token_hash, status, created_by_user_id,
                 expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (invitation_id, organization_id, email, role, token_digest(token), actor_user_id, expiration, _now()),
            )
            org = con.execute("SELECT name FROM organizations WHERE id = ?", (organization_id,)).fetchone()
        invite_url = f"{base_url.rstrip('/')}?invite={token}"
        self.queue_notification(
            recipient_email=email,
            organization_id=organization_id,
            kind="workspace_invitation",
            subject=f"Invitation to {org['name'] if org else 'Reformulation Assurance'}",
            body=(
                f"You were invited as {role}.\n\nAccept the invitation:\n{invite_url}\n\n"
                f"This invitation expires at {expiration}."
            ),
        )
        return {"id": invitation_id, "token": token, "invite_url": invite_url, "expires_at": expiration}

    def accept_invitation(self, token: str, *, display_name: str, password: str) -> tuple[str, str]:
        digest = token_digest(token)
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM invitations WHERE token_hash = ?", (digest,)
            ).fetchone()
            if row is None or not verify_token(token, row["token_hash"]):
                raise ValueError("invitation token is invalid")
            if row["status"] != "pending":
                raise ValueError("invitation is no longer active")
            if is_expired(row["expires_at"]):
                con.execute("UPDATE invitations SET status = 'expired' WHERE id = ?", (row["id"],))
                raise ValueError("invitation has expired")
            existing = con.execute("SELECT * FROM users WHERE email = ?", (row["email"],)).fetchone()
            if existing:
                if not verify_password(password, existing["password_hash"]):
                    raise PermissionError("existing account password is incorrect")
                user_id = str(existing["id"])
            else:
                name = display_name.strip()
                if not name:
                    raise ValueError("display name is required")
                user_id = str(uuid.uuid4())
                con.execute(
                    """INSERT INTO users
                    (id, email, display_name, password_hash, is_active, created_at)
                    VALUES (?, ?, ?, ?, 1, ?)""",
                    (user_id, row["email"], name, hash_password(password), _now()),
                )
            con.execute(
                """INSERT INTO memberships (organization_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(organization_id, user_id) DO UPDATE SET role = excluded.role""",
                (row["organization_id"], user_id, row["role"], _now()),
            )
            con.execute(
                """UPDATE invitations SET status = 'accepted', accepted_at = ?, accepted_by_user_id = ?
                WHERE id = ?""",
                (_now(), user_id, row["id"]),
            )
        return user_id, str(row["organization_id"])

    def list_invitations(self, organization_id: str, actor_user_id: str) -> pd.DataFrame:
        self.require_role(actor_user_id, organization_id, ADMIN_ROLES)
        with self.connection() as con:
            rows = con.execute(
                """SELECT i.*, u.display_name AS invited_by
                FROM invitations i JOIN users u ON u.id = i.created_by_user_id
                WHERE i.organization_id = ? ORDER BY i.created_at DESC""",
                (organization_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def request_password_reset(
        self,
        email: str,
        *,
        base_url: str = "http://localhost:8501",
        expires_minutes: int = 30,
    ) -> dict[str, str] | None:
        try:
            email = validate_email(email)
        except ValueError:
            return None
        with self.connection() as con:
            user = con.execute("SELECT id FROM users WHERE email = ? AND is_active = 1", (email,)).fetchone()
            if user is None:
                return None
            token = generate_single_use_token()
            reset_id = str(uuid.uuid4())
            expiration = expires_at(minutes=expires_minutes)
            con.execute(
                "UPDATE password_resets SET status = 'revoked' WHERE user_id = ? AND status = 'pending'",
                (user["id"],),
            )
            con.execute(
                """INSERT INTO password_resets
                (id, user_id, token_hash, status, expires_at, created_at)
                VALUES (?, ?, ?, 'pending', ?, ?)""",
                (reset_id, user["id"], token_digest(token), expiration, _now()),
            )
        reset_url = f"{base_url.rstrip('/')}?reset={token}"
        self.queue_notification(
            recipient_email=email,
            kind="password_reset",
            subject="Reset your Reformulation Assurance password",
            body=f"Reset your password:\n{reset_url}\n\nThis link expires at {expiration}.",
        )
        return {"id": reset_id, "token": token, "reset_url": reset_url, "expires_at": expiration}

    def reset_password(self, token: str, new_password: str) -> str:
        digest = token_digest(token)
        with self.connection() as con:
            row = con.execute("SELECT * FROM password_resets WHERE token_hash = ?", (digest,)).fetchone()
            if row is None or not verify_token(token, row["token_hash"]):
                raise ValueError("reset token is invalid")
            if row["status"] != "pending":
                raise ValueError("reset token is no longer active")
            if is_expired(row["expires_at"]):
                con.execute("UPDATE password_resets SET status = 'expired' WHERE id = ?", (row["id"],))
                raise ValueError("reset token has expired")
            con.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(new_password), row["user_id"]),
            )
            con.execute(
                "UPDATE password_resets SET status = 'used', used_at = ? WHERE id = ?",
                (_now(), row["id"]),
            )
        return str(row["user_id"])

    # ------------------------------------------------------------------
    # Comments and assignments
    # ------------------------------------------------------------------
    def add_comment(
        self,
        project_id: str,
        *,
        author_user_id: str,
        body: str,
        entity_type: str = "project",
        entity_id: str | None = None,
    ) -> str:
        self.require_project_access(author_user_id, project_id, COMMENT_ROLES)
        body = body.strip()
        if not body:
            raise ValueError("comment cannot be empty")
        comment_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO project_comments
                (id, project_id, author_user_id, entity_type, entity_id, body, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (comment_id, project_id, author_user_id, entity_type, entity_id, body, _now()),
            )
        self.audit(project_id, "comment_added", entity_type="comment", entity_id=comment_id,
                   detail={"target_type": entity_type, "target_id": entity_id})
        return comment_id

    def list_comments(self, project_id: str, actor_user_id: str) -> pd.DataFrame:
        self.require_project_access(actor_user_id, project_id)
        with self.connection() as con:
            rows = con.execute(
                """SELECT c.*, u.display_name AS author_name,
                          r.display_name AS resolved_by_name
                FROM project_comments c
                JOIN users u ON u.id = c.author_user_id
                LEFT JOIN users r ON r.id = c.resolved_by_user_id
                WHERE c.project_id = ? ORDER BY c.created_at""",
                (project_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def resolve_comment(self, comment_id: str, actor_user_id: str) -> None:
        with self.connection() as con:
            row = con.execute("SELECT * FROM project_comments WHERE id = ?", (comment_id,)).fetchone()
            if row is None:
                raise KeyError("unknown comment")
            role = self.require_project_access(actor_user_id, row["project_id"])
            if actor_user_id != row["author_user_id"] and role not in ADMIN_ROLES:
                raise PermissionError("only the author or an administrator can resolve this comment")
            con.execute(
                "UPDATE project_comments SET resolved_at = ?, resolved_by_user_id = ? WHERE id = ?",
                (_now(), actor_user_id, comment_id),
            )
        self.audit(row["project_id"], "comment_resolved", entity_type="comment", entity_id=comment_id)

    def create_assignment(
        self,
        project_id: str,
        *,
        title: str,
        assignee_user_id: str,
        created_by_user_id: str,
        description: str = "",
        entity_type: str = "project",
        entity_id: str | None = None,
        due_at: str | None = None,
        priority: str = "normal",
    ) -> str:
        self.require_project_access(created_by_user_id, project_id, TASK_EDIT_ROLES)
        organization_id = self.project_organization(project_id)
        if organization_id and not self.role_for(assignee_user_id, organization_id):
            raise ValueError("assignee is not a member of this organization")
        if priority not in PRIORITIES:
            raise ValueError("unsupported priority")
        title = title.strip()
        if not title:
            raise ValueError("assignment title is required")
        assignment_id = str(uuid.uuid4())
        now = _now()
        with self.connection() as con:
            con.execute(
                """INSERT INTO assignments
                (id, project_id, title, description, entity_type, entity_id,
                 assignee_user_id, created_by_user_id, status, priority, due_at,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)""",
                (assignment_id, project_id, title, description.strip(), entity_type, entity_id,
                 assignee_user_id, created_by_user_id, priority, due_at, now, now),
            )
        self.audit(project_id, "assignment_created", entity_type="assignment", entity_id=assignment_id,
                   detail={"assignee_user_id": assignee_user_id, "priority": priority})
        return assignment_id

    def list_assignments(self, project_id: str, actor_user_id: str) -> pd.DataFrame:
        self.require_project_access(actor_user_id, project_id)
        with self.connection() as con:
            rows = con.execute(
                """SELECT a.*, u.display_name AS assignee_name, c.display_name AS created_by_name
                FROM assignments a
                JOIN users u ON u.id = a.assignee_user_id
                JOIN users c ON c.id = a.created_by_user_id
                WHERE a.project_id = ? ORDER BY
                CASE a.priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 WHEN 'normal' THEN 3 ELSE 4 END,
                a.created_at""",
                (project_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def update_assignment(self, assignment_id: str, actor_user_id: str, *, status: str) -> None:
        if status not in TASK_STATUSES:
            raise ValueError("unsupported assignment status")
        with self.connection() as con:
            row = con.execute("SELECT * FROM assignments WHERE id = ?", (assignment_id,)).fetchone()
            if row is None:
                raise KeyError("unknown assignment")
            role = self.require_project_access(actor_user_id, row["project_id"])
            if actor_user_id not in {row["assignee_user_id"], row["created_by_user_id"]} and role not in ADMIN_ROLES:
                raise PermissionError("you cannot update this assignment")
            con.execute("UPDATE assignments SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), assignment_id))
        self.audit(row["project_id"], "assignment_updated", entity_type="assignment", entity_id=assignment_id,
                   detail={"status": status})

    # ------------------------------------------------------------------
    # Multi-signer approval policies
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_requirements(requirements: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not requirements:
            raise ValueError("at least one signer requirement is required")
        cleaned: list[dict[str, Any]] = []
        for item in requirements:
            role = str(item.get("role", ""))
            count = int(item.get("count", 0))
            if role not in ROLES:
                raise ValueError(f"unsupported signer role: {role}")
            if role not in APPROVAL_ROLES:
                raise ValueError(f"role {role} cannot sign approvals")
            if count < 1:
                raise ValueError("signer count must be at least 1")
            cleaned.append({"role": role, "count": count})
        return cleaned

    def create_approval_policy(
        self,
        project_id: str,
        *,
        stage: str,
        name: str,
        requirements: Sequence[Mapping[str, Any]],
        actor_user_id: str,
    ) -> str:
        self.require_project_access(actor_user_id, project_id, ADMIN_ROLES)
        policy_id = str(uuid.uuid4())
        now = _now()
        cleaned = self._validate_requirements(requirements)
        with self.connection() as con:
            con.execute(
                """INSERT INTO approval_policies
                (id, project_id, stage, name, requirements_json, is_active,
                 created_by_user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (policy_id, project_id, stage, name.strip() or f"{stage} approval", dumps(cleaned), actor_user_id, now, now),
            )
        self.audit(project_id, "approval_policy_created", entity_type="approval_policy", entity_id=policy_id,
                   detail={"stage": stage, "requirements": cleaned})
        return policy_id

    def list_approval_policies(self, project_id: str, *, active_only: bool = False) -> pd.DataFrame:
        query = "SELECT * FROM approval_policies WHERE project_id = ?"
        params: list[Any] = [project_id]
        if active_only:
            query += " AND is_active = 1"
        query += " ORDER BY stage, created_at"
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["requirements"] = loads(item.pop("requirements_json"), [])
            item["is_active"] = bool(item["is_active"])
            records.append(item)
        return pd.DataFrame(records)

    def sign_approval(
        self,
        project_id: str,
        *,
        stage: str,
        signer_user_id: str,
        typed_name: str,
        password: str,
        signature_meaning: str,
        evidence_hash: str,
        comment: str = "",
        policy_id: str | None = None,
    ) -> str:
        organization_id = self.project_organization(project_id)
        role = self.require_role(signer_user_id, organization_id, APPROVAL_ROLES) if organization_id else "owner"
        user = self.get_user(signer_user_id)
        if typed_name.strip().casefold() != user["display_name"].strip().casefold():
            raise ValueError("typed name must exactly match the account display name")
        if not self.reauthenticate(signer_user_id, password):
            raise PermissionError("password verification failed")
        if not signature_meaning.strip():
            raise ValueError("signature meaning is required")
        with self.connection() as con:
            if policy_id:
                policy = con.execute("SELECT * FROM approval_policies WHERE id = ? AND project_id = ? AND is_active = 1", (policy_id, project_id)).fetchone()
                if policy is None:
                    raise ValueError("approval policy is not active")
                if policy["stage"] != stage:
                    raise ValueError("approval stage does not match the policy")
                requirements = loads(policy["requirements_json"], [])
                if role not in {item["role"] for item in requirements}:
                    raise PermissionError("your role is not required by this approval policy")
                duplicate = con.execute(
                    """SELECT id FROM approvals WHERE policy_id = ? AND signer_user_id = ?
                    AND evidence_hash = ? AND status = 'signed'""",
                    (policy_id, signer_user_id, evidence_hash),
                ).fetchone()
                if duplicate:
                    raise ValueError("you already signed this policy for the current evidence")
            else:
                duplicate = con.execute(
                    """SELECT id FROM approvals WHERE project_id = ? AND stage = ?
                    AND signer_user_id = ? AND evidence_hash = ? AND status = 'signed'
                    AND policy_id IS NULL""",
                    (project_id, stage, signer_user_id, evidence_hash),
                ).fetchone()
                if duplicate:
                    raise ValueError("you already signed this stage for the current evidence")
        approval_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO approvals
                (id, project_id, stage, status, signer_user_id, signer_name,
                 signer_role, signature_meaning, comment, evidence_hash, signed_at, policy_id)
                VALUES (?, ?, ?, 'signed', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (approval_id, project_id, stage, signer_user_id, user["display_name"], role,
                 signature_meaning.strip(), comment.strip(), evidence_hash, _now(), policy_id),
            )
        self.audit(project_id, "approval_signed", entity_type="approval", entity_id=approval_id,
                   detail={"stage": stage, "evidence_hash": evidence_hash, "signer_role": role, "policy_id": policy_id})
        return approval_id

    def approval_policy_status(self, project_id: str, evidence_hash: str) -> pd.DataFrame:
        policies = self.list_approval_policies(project_id, active_only=True)
        approvals = self.list_approvals(project_id)
        records: list[dict[str, Any]] = []
        for _, policy in policies.iterrows():
            current = approvals[
                (approvals.get("policy_id") == policy["id"])
                & (approvals["status"] == "signed")
                & (approvals["evidence_hash"] == evidence_hash)
            ] if not approvals.empty else pd.DataFrame()
            requirement_results = []
            all_met = True
            for requirement in policy["requirements"]:
                role = requirement["role"]
                required = int(requirement["count"])
                signed = 0 if current.empty else int(current[current["signer_role"] == role]["signer_user_id"].nunique())
                met = signed >= required
                all_met &= met
                requirement_results.append({"role": role, "required": required, "signed": signed, "met": met})
            records.append({
                "policy_id": policy["id"],
                "stage": policy["stage"],
                "name": policy["name"],
                "requirements": requirement_results,
                "complete": bool(all_met),
                "current_signature_count": int(len(current)),
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Artifact and backup metadata
    # ------------------------------------------------------------------
    def save_artifact_record(self, project_id: str, *, created_by_user_id: str, artifact_type: str,
                             filename: str, content_type: str, storage_path: str,
                             plaintext_sha256: str, ciphertext_sha256: str, size_bytes: int,
                             encryption_method: str, metadata: Mapping[str, Any] | None = None) -> str:
        self.require_project_access(created_by_user_id, project_id)
        artifact_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO encrypted_artifacts
                (id, project_id, created_by_user_id, artifact_type, filename, content_type,
                 storage_path, plaintext_sha256, ciphertext_sha256, size_bytes,
                 encryption_method, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (artifact_id, project_id, created_by_user_id, artifact_type, filename, content_type,
                 storage_path, plaintext_sha256, ciphertext_sha256, int(size_bytes), encryption_method,
                 dumps(metadata or {}), _now()),
            )
        self.audit(project_id, "encrypted_artifact_saved", entity_type="artifact", entity_id=artifact_id,
                   detail={"artifact_type": artifact_type, "filename": filename, "plaintext_sha256": plaintext_sha256})
        return artifact_id

    def list_artifacts(self, project_id: str, actor_user_id: str) -> pd.DataFrame:
        self.require_project_access(actor_user_id, project_id)
        with self.connection() as con:
            rows = con.execute(
                """SELECT a.*, u.display_name AS created_by
                FROM encrypted_artifacts a JOIN users u ON u.id = a.created_by_user_id
                WHERE a.project_id = ? ORDER BY a.created_at DESC""",
                (project_id,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["metadata"] = loads(item.pop("metadata_json"), {})
            records.append(item)
        return pd.DataFrame(records)

    def get_artifact(self, artifact_id: str, actor_user_id: str) -> dict[str, Any]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM encrypted_artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError("unknown artifact")
        self.require_project_access(actor_user_id, row["project_id"])
        item = dict(row)
        item["metadata"] = loads(item.pop("metadata_json"), {})
        return item

    def save_backup_record(self, *, filename: str, storage_path: str, plaintext_sha256: str,
                           ciphertext_sha256: str, size_bytes: int, encryption_method: str,
                           organization_id: str | None = None, created_by_user_id: str | None = None,
                           metadata: Mapping[str, Any] | None = None) -> str:
        if organization_id and created_by_user_id:
            self.require_role(created_by_user_id, organization_id, ADMIN_ROLES)
        backup_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO backup_records
                (id, organization_id, created_by_user_id, filename, storage_path,
                 plaintext_sha256, ciphertext_sha256, size_bytes, encryption_method,
                 status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'created', ?, ?)""",
                (backup_id, organization_id, created_by_user_id, filename, storage_path,
                 plaintext_sha256, ciphertext_sha256, int(size_bytes), encryption_method,
                 dumps(metadata or {}), _now()),
            )
        return backup_id

    def mark_backup_verified(self, backup_id: str) -> None:
        with self.connection() as con:
            con.execute("UPDATE backup_records SET status = 'verified', verified_at = ? WHERE id = ?", (_now(), backup_id))

    def list_backups(self, *, organization_id: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM backup_records"
        params: list[Any] = []
        if organization_id:
            query += " WHERE organization_id = ? OR organization_id IS NULL"
            params.append(organization_id)
        query += " ORDER BY created_at DESC"
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["metadata"] = loads(item.pop("metadata_json"), {})
            records.append(item)
        return pd.DataFrame(records)
