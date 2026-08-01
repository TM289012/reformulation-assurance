"""Verified encrypted backups for SQLite pilot deployments."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from artifact_vault import ArtifactVault
from pilot_store import PilotStore


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_backup(
    store: PilotStore,
    vault: ArtifactVault,
    *,
    organization_id: str | None = None,
    created_by_user_id: str | None = None,
) -> tuple[str, dict]:
    """Create a transactionally consistent SQLite backup, package it, encrypt it, and verify it."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_db = Path(temp_dir) / "reformulation_assurance.db"
        source = sqlite3.connect(store.database_path)
        target = sqlite3.connect(temp_db)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        raw_db = temp_db.read_bytes()
        manifest = {
            "format": "reformulation-assurance-sqlite-backup",
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "database_sha256": hashlib.sha256(raw_db).hexdigest(),
            "database_size_bytes": len(raw_db),
        }
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("reformulation_assurance.db", raw_db)
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        payload = output.getvalue()
    filename = f"reformulation_assurance_backup_{_timestamp()}.zip"
    encrypted = vault.store_backup_payload(payload, filename)
    backup_id = store.save_backup_record(
        **encrypted,
        organization_id=organization_id,
        created_by_user_id=created_by_user_id,
        metadata=manifest,
    )
    verify_backup(store, vault, backup_id)
    return backup_id, {**encrypted, "metadata": manifest}


def verify_backup(store: PilotStore, vault: ArtifactVault, backup_id: str) -> bool:
    backups = store.list_backups()
    match = backups[backups["id"] == backup_id]
    if match.empty:
        raise KeyError("unknown backup")
    record = match.iloc[0].to_dict()
    payload = vault.retrieve_backup_payload(record)
    with ZipFile(BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        database = archive.read("reformulation_assurance.db")
        if hashlib.sha256(database).hexdigest() != manifest["database_sha256"]:
            raise ValueError("database checksum inside backup does not match")
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "verify.db"
            db_path.write_bytes(database)
            con = sqlite3.connect(db_path)
            try:
                result = con.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                con.close()
            if result != "ok":
                raise ValueError(f"SQLite integrity check failed: {result}")
    store.mark_backup_verified(backup_id)
    return True


def restore_backup_payload(vault: ArtifactVault, record: dict, target_path: str | Path) -> Path:
    payload = vault.retrieve_backup_payload(record)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(payload)) as archive:
        target_path.write_bytes(archive.read("reformulation_assurance.db"))
    con = sqlite3.connect(target_path)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    if result != "ok":
        target_path.unlink(missing_ok=True)
        raise ValueError("restored database failed integrity verification")
    return target_path
