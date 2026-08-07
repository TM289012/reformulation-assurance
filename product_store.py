"""Product persistence for Reformulation Assurance v0.5.

Extends the v0.4 scientific store with accounts, organizations, role-based
project isolation, approval signatures, and generated dossier records.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import sqlite3
import uuid

import pandas as pd

from project_store import ProjectStore, _now, dumps, loads
from security import hash_password, validate_email, verify_password

ROLES = ("owner", "admin", "scientist", "approver", "viewer")
EDIT_ROLES = {"owner", "admin", "scientist"}
APPROVAL_ROLES = {"owner", "admin", "approver"}
ADMIN_ROLES = {"owner", "admin"}


class ProductStore(ProjectStore):
    def _initialize(self) -> None:
        super()._initialize()
        with self.connection() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memberships (
                    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'signed',
                    signer_user_id TEXT NOT NULL REFERENCES users(id),
                    signer_name TEXT NOT NULL,
                    signer_role TEXT NOT NULL,
                    signature_meaning TEXT NOT NULL,
                    comment TEXT NOT NULL DEFAULT '',
                    evidence_hash TEXT NOT NULL,
                    evidence_snapshot TEXT,
                    signed_at TEXT NOT NULL,
                    withdrawn_at TEXT
                );

                CREATE TABLE IF NOT EXISTS dossiers (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    generated_by_user_id TEXT NOT NULL REFERENCES users(id),
                    version INTEGER NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, version)
                );

                CREATE INDEX IF NOT EXISTS idx_memberships_user ON memberships(user_id);
                CREATE INDEX IF NOT EXISTS idx_approvals_project ON approvals(project_id, signed_at);
                CREATE INDEX IF NOT EXISTS idx_dossiers_project ON dossiers(project_id, version);
                """
            )
            project_columns = self._column_names(con, "projects")
            if "organization_id" not in project_columns:
                con.execute("ALTER TABLE projects ADD COLUMN organization_id TEXT")
            if "created_by_user_id" not in project_columns:
                con.execute("ALTER TABLE projects ADD COLUMN created_by_user_id TEXT")
            approval_columns = self._column_names(con, "approvals")
            if "evidence_snapshot" not in approval_columns:
                con.execute("ALTER TABLE approvals ADD COLUMN evidence_snapshot TEXT")

    def has_users(self) -> bool:
        with self.connection() as con:
            return bool(con.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def register_owner(
        self,
        *,
        email: str,
        display_name: str,
        password: str,
        organization_name: str,
    ) -> tuple[str, str]:
        email = validate_email(email)
        display_name = display_name.strip()
        organization_name = organization_name.strip()
        if not display_name:
            raise ValueError("display name is required")
        if not organization_name:
            raise ValueError("organization name is required")
        user_id = str(uuid.uuid4())
        organization_id = str(uuid.uuid4())
        now = _now()
        password_hash = hash_password(password)
        with self.connection() as con:
            con.execute(
                "INSERT INTO organizations (id, name, created_at) VALUES (?, ?, ?)",
                (organization_id, organization_name, now),
            )
            con.execute(
                """INSERT INTO users
                (id, email, display_name, password_hash, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)""",
                (user_id, email, display_name, password_hash, now),
            )
            con.execute(
                """INSERT INTO memberships
                (organization_id, user_id, role, created_at) VALUES (?, ?, 'owner', ?)""",
                (organization_id, user_id, now),
            )
        return user_id, organization_id

    def create_member(
        self,
        organization_id: str,
        *,
        email: str,
        display_name: str,
        password: str,
        role: str,
        actor_user_id: str,
    ) -> str:
        self.require_role(actor_user_id, organization_id, ADMIN_ROLES)
        if role not in ROLES:
            raise ValueError(f"unsupported role: {role}")
        email = validate_email(email)
        display_name = display_name.strip()
        if not display_name:
            raise ValueError("display name is required")
        user_id = str(uuid.uuid4())
        now = _now()
        with self.connection() as con:
            existing = con.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                user_id = existing["id"]
            else:
                con.execute(
                    """INSERT INTO users
                    (id, email, display_name, password_hash, is_active, created_at)
                    VALUES (?, ?, ?, ?, 1, ?)""",
                    (user_id, email, display_name, hash_password(password), now),
                )
            con.execute(
                """INSERT INTO memberships (organization_id, user_id, role, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(organization_id, user_id) DO UPDATE SET role = excluded.role""",
                (organization_id, user_id, role, now),
            )
        return user_id

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        try:
            email = validate_email(email)
        except ValueError:
            return None
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM users WHERE email = ? AND is_active = 1", (email,)
            ).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                return None
            con.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), row["id"]))
        result = dict(row)
        result.pop("password_hash", None)
        result["is_active"] = bool(result["is_active"])
        return result

    def reauthenticate(self, user_id: str, password: str) -> bool:
        with self.connection() as con:
            row = con.execute(
                "SELECT password_hash, is_active FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return bool(row and row["is_active"] and verify_password(password, row["password_hash"]))

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self.connection() as con:
            row = con.execute(
                "SELECT id, email, display_name, is_active, created_at, last_login_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown user: {user_id}")
        result = dict(row)
        result["is_active"] = bool(result["is_active"])
        return result

    def user_organizations(self, user_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT o.id, o.name, m.role, o.created_at
                FROM organizations o
                JOIN memberships m ON m.organization_id = o.id
                WHERE m.user_id = ?
                ORDER BY o.name""",
                (user_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def list_members(self, organization_id: str, actor_user_id: str) -> pd.DataFrame:
        self.require_role(actor_user_id, organization_id, set(ROLES))
        with self.connection() as con:
            rows = con.execute(
                """SELECT u.id, u.email, u.display_name, u.is_active, u.last_login_at,
                          m.role, m.created_at
                FROM users u JOIN memberships m ON m.user_id = u.id
                WHERE m.organization_id = ? ORDER BY u.display_name""",
                (organization_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def role_for(self, user_id: str, organization_id: str) -> str | None:
        with self.connection() as con:
            row = con.execute(
                "SELECT role FROM memberships WHERE user_id = ? AND organization_id = ?",
                (user_id, organization_id),
            ).fetchone()
        return None if row is None else str(row["role"])

    def require_role(self, user_id: str, organization_id: str, allowed: set[str]) -> str:
        role = self.role_for(user_id, organization_id)
        if role not in allowed:
            raise PermissionError("your role does not permit this action")
        return role

    def create_project(
        self,
        name: str,
        config: Mapping[str, Any],
        *,
        description: str = "",
        source_filename: str | None = None,
        organization_id: str | None = None,
        created_by_user_id: str | None = None,
    ) -> str:
        if organization_id and created_by_user_id:
            self.require_role(created_by_user_id, organization_id, EDIT_ROLES)
        project_id = super().create_project(
            name, config, description=description, source_filename=source_filename
        )
        with self.connection() as con:
            con.execute(
                "UPDATE projects SET organization_id = ?, created_by_user_id = ? WHERE id = ?",
                (organization_id, created_by_user_id, project_id),
            )
        return project_id

    def list_projects(self, organization_id: str | None = None) -> pd.DataFrame:
        query = """SELECT p.*,
                    (SELECT COUNT(*) FROM experiments e WHERE e.project_id = p.id) AS experiment_count,
                    (SELECT COUNT(*) FROM batches b WHERE b.project_id = p.id) AS batch_count,
                    (SELECT COUNT(*) FROM snapshots s WHERE s.project_id = p.id) AS snapshot_count
                FROM projects p"""
        params: list[Any] = []
        if organization_id:
            query += " WHERE p.organization_id = ?"
            params.append(organization_id)
        query += " ORDER BY p.updated_at DESC"
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def project_organization(self, project_id: str) -> str | None:
        with self.connection() as con:
            row = con.execute("SELECT organization_id FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown project: {project_id}")
        return row["organization_id"]

    def require_project_access(self, user_id: str, project_id: str, allowed: set[str] | None = None) -> str:
        organization_id = self.project_organization(project_id)
        if not organization_id:
            return "owner"  # backward-compatible local v0.4 project
        return self.require_role(user_id, organization_id, allowed or set(ROLES))

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
        evidence_snapshot: str | None = None,
        comment: str = "",
    ) -> str:
        organization_id = self.project_organization(project_id)
        if organization_id:
            role = self.require_role(signer_user_id, organization_id, APPROVAL_ROLES)
        else:
            role = "owner"
        user = self.get_user(signer_user_id)
        if typed_name.strip().casefold() != user["display_name"].strip().casefold():
            raise ValueError("typed name must exactly match the account display name")
        if not self.reauthenticate(signer_user_id, password):
            raise PermissionError("password verification failed")
        if not signature_meaning.strip():
            raise ValueError("signature meaning is required")
        with self.connection() as con:
            duplicate = con.execute(
                """SELECT id FROM approvals WHERE project_id = ? AND stage = ?
                AND signer_user_id = ? AND evidence_hash = ? AND status = 'signed'""",
                (project_id, stage, signer_user_id, evidence_hash),
            ).fetchone()
            if duplicate:
                raise ValueError("you already signed this stage for the current evidence")
        approval_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO approvals
                (id, project_id, stage, status, signer_user_id, signer_name,
                 signer_role, signature_meaning, comment, evidence_hash, evidence_snapshot, signed_at)
                VALUES (?, ?, ?, 'signed', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval_id,
                    project_id,
                    stage,
                    signer_user_id,
                    user["display_name"],
                    role,
                    signature_meaning.strip(),
                    comment.strip(),
                    evidence_hash,
                    evidence_snapshot,
                    _now(),
                ),
            )
        self.audit(
            project_id,
            "approval_signed",
            entity_type="approval",
            entity_id=approval_id,
            detail={"stage": stage, "evidence_hash": evidence_hash, "signer_role": role},
        )
        return approval_id

    def withdraw_approval(self, approval_id: str, actor_user_id: str) -> None:
        with self.connection() as con:
            row = con.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown approval: {approval_id}")
            organization_id = self.project_organization(row["project_id"])
            if organization_id:
                actor_role = self.require_role(actor_user_id, organization_id, APPROVAL_ROLES)
                if actor_user_id != row["signer_user_id"] and actor_role not in ADMIN_ROLES:
                    raise PermissionError("only the signer or an administrator can withdraw this approval")
            con.execute(
                "UPDATE approvals SET status = 'withdrawn', withdrawn_at = ? WHERE id = ?",
                (_now(), approval_id),
            )
        self.audit(row["project_id"], "approval_withdrawn", entity_type="approval", entity_id=approval_id)

    def list_approvals(self, project_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT a.*, u.email AS signer_email
                FROM approvals a JOIN users u ON u.id = a.signer_user_id
                WHERE a.project_id = ? ORDER BY a.signed_at""",
                (project_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def save_dossier_record(
        self,
        project_id: str,
        *,
        generated_by_user_id: str,
        evidence_hash: str,
        manifest: Mapping[str, Any],
    ) -> tuple[str, int]:
        self.require_project_access(generated_by_user_id, project_id)
        dossier_id = str(uuid.uuid4())
        with self.connection() as con:
            version = int(
                con.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM dossiers WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0]
            )
            con.execute(
                """INSERT INTO dossiers
                (id, project_id, generated_by_user_id, version, evidence_hash, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (dossier_id, project_id, generated_by_user_id, version, evidence_hash, dumps(manifest), _now()),
            )
        self.audit(
            project_id,
            "dossier_generated",
            entity_type="dossier",
            entity_id=dossier_id,
            detail={"version": version, "evidence_hash": evidence_hash},
        )
        return dossier_id, version

    def list_dossiers(self, project_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT d.*, u.display_name AS generated_by
                FROM dossiers d JOIN users u ON u.id = d.generated_by_user_id
                WHERE d.project_id = ? ORDER BY d.version DESC""",
                (project_id,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["manifest"] = loads(item.pop("manifest_json"), {})
            records.append(item)
        return pd.DataFrame(records)
