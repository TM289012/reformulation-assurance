"""Portable migration bundle and optional PostgreSQL loader for v0.6.

The application runtime remains SQLite for the included single-instance pilot.
This module creates a checksummed, table-by-table migration bundle and can load
it into PostgreSQL when SQLAlchemy and a PostgreSQL driver are installed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib
import json
import sqlite3
from typing import Any


TABLE_ORDER = [
    "organizations", "users", "memberships", "projects", "batches",
    "experiments", "snapshots", "robustness_runs", "audit_events",
    "approvals", "dossiers", "invitations", "password_resets",
    "notifications", "project_comments", "assignments", "approval_policies",
    "encrypted_artifacts", "backup_records",
]


def _clean_row(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def create_postgres_migration_bundle(database_path: str | Path) -> tuple[bytes, dict[str, Any]]:
    database_path = Path(database_path)
    con = sqlite3.connect(database_path)
    con.row_factory = sqlite3.Row
    try:
        existing = {
            row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        table_counts: dict[str, int] = {}
        files: dict[str, bytes] = {}
        for table in TABLE_ORDER:
            if table not in existing:
                continue
            rows = con.execute(f'SELECT * FROM "{table}"').fetchall()
            table_counts[table] = len(rows)
            content = "".join(json.dumps(_clean_row(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
            files[f"tables/{table}.jsonl"] = content.encode("utf-8")
        schema_rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type IN ('table','index') AND sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
        sqlite_schema = "\n\n".join(str(row["sql"]) + ";" for row in schema_rows)
        files["sqlite_schema.sql"] = sqlite_schema.encode("utf-8")
        files["POSTGRES_IMPORT.md"] = (
            "Install SQLAlchemy and psycopg, provision an empty PostgreSQL database, "
            "create an equivalent schema using your migration tooling, then run:\n\n"
            "python postgres_migration.py load bundle.zip postgresql+psycopg://...\n\n"
            "The loader is intended for controlled migration testing; the included app runtime remains SQLite.\n"
        ).encode("utf-8")
        manifest = {
            "format": "reformulation-assurance-postgres-migration",
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(database_path),
            "table_counts": table_counts,
        }
        files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        checksums = [f"{hashlib.sha256(content).hexdigest()}  {name}" for name, content in sorted(files.items())]
        files["SHA256SUMS.txt"] = ("\n".join(checksums) + "\n").encode("utf-8")
        output = BytesIO()
        with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return output.getvalue(), manifest
    finally:
        con.close()


def load_bundle_to_postgres(bundle: bytes | str | Path, database_url: str) -> dict[str, int]:
    """Load rows into an already-created PostgreSQL schema.

    Schema creation is deliberately left to Alembic/DBA-controlled migrations.
    This function verifies checksums and inserts the exported records in a stable
    foreign-key order.
    """
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SQLAlchemy is required for PostgreSQL migration") from exc
    payload = Path(bundle).read_bytes() if isinstance(bundle, (str, Path)) else bundle
    with ZipFile(BytesIO(payload)) as archive:
        checksum_lines = archive.read("SHA256SUMS.txt").decode("utf-8").strip().splitlines()
        for line in checksum_lines:
            digest, filename = line.split("  ", 1)
            if hashlib.sha256(archive.read(filename)).hexdigest() != digest:
                raise ValueError(f"migration bundle checksum failed for {filename}")
        engine = create_engine(database_url, future=True)
        inserted: dict[str, int] = {}
        with engine.begin() as connection:
            for table in TABLE_ORDER:
                name = f"tables/{table}.jsonl"
                if name not in archive.namelist():
                    continue
                rows = [json.loads(line) for line in archive.read(name).decode("utf-8").splitlines() if line.strip()]
                for row in rows:
                    columns = list(row)
                    quoted = ", ".join(f'"{column}"' for column in columns)
                    values = ", ".join(f":{column}" for column in columns)
                    connection.execute(text(f'INSERT INTO "{table}" ({quoted}) VALUES ({values})'), row)
                inserted[table] = len(rows)
        return inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("database")
    export.add_argument("output")
    load = sub.add_parser("load")
    load.add_argument("bundle")
    load.add_argument("database_url")
    args = parser.parse_args()
    if args.command == "export":
        payload, manifest = create_postgres_migration_bundle(args.database)
        Path(args.output).write_bytes(payload)
        print(json.dumps(manifest, indent=2))
    else:
        print(json.dumps(load_bundle_to_postgres(args.bundle, args.database_url), indent=2))
