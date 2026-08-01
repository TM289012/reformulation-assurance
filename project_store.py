"""SQLite persistence for Reformulation Assurance v0.4.

The store separates immutable recommendations, physical outcomes, robustness
runs, model snapshots, and audit events. Replicates are linked through a
replicate group so repeatability evidence can be evaluated explicitly rather
than counted as unrelated experiments.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import sqlite3
import uuid

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=True)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    return json.loads(value)


class ProjectStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.database_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _column_names(con: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}

    def _initialize(self) -> None:
        with self.connection() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    source_filename TEXT,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    batch_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    decision_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    completed_at TEXT,
                    UNIQUE(project_id, batch_number)
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    batch_id TEXT REFERENCES batches(id) ON DELETE SET NULL,
                    experiment_code TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    purpose TEXT,
                    qualification_stage TEXT NOT NULL DEFAULT 'discovery',
                    status TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    responses_json TEXT NOT NULL DEFAULT '{}',
                    recommendation_json TEXT NOT NULL DEFAULT '{}',
                    notes TEXT NOT NULL DEFAULT '',
                    replicate_group TEXT,
                    replicate_index INTEGER NOT NULL DEFAULT 1,
                    parent_experiment_id TEXT REFERENCES experiments(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(project_id, experiment_code)
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    batch_id TEXT REFERENCES batches(id) ON DELETE SET NULL,
                    trigger TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    best_success_probability REAL,
                    best_feasibility_probability REAL,
                    qualification_score REAL NOT NULL,
                    completed_platform_experiments INTEGER NOT NULL,
                    compliant_platform_experiments INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS robustness_runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                    simulation_count INTEGER NOT NULL,
                    variation_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_batch ON experiments(batch_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_replicate ON experiments(project_id, replicate_group);
                CREATE INDEX IF NOT EXISTS idx_snapshots_project ON snapshots(project_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_robustness_project ON robustness_runs(project_id, created_at);
                """
            )
            # Migrate databases created by v0.3 without discarding evidence.
            columns = self._column_names(con, "experiments")
            if "replicate_group" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN replicate_group TEXT")
            if "replicate_index" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN replicate_index INTEGER NOT NULL DEFAULT 1")
            if "parent_experiment_id" not in columns:
                con.execute("ALTER TABLE experiments ADD COLUMN parent_experiment_id TEXT")
            con.execute(
                "UPDATE experiments SET replicate_group = COALESCE(replicate_group, experiment_code)"
            )

    def audit(
        self,
        project_id: str,
        event_type: str,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                """INSERT INTO audit_events
                (project_id, event_type, entity_type, entity_id, detail_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (project_id, event_type, entity_type, entity_id, dumps(detail or {}), _now()),
            )

    def create_project(
        self,
        name: str,
        config: Mapping[str, Any],
        *,
        description: str = "",
        source_filename: str | None = None,
    ) -> str:
        name = name.strip()
        if not name:
            raise ValueError("project name is required")
        project_id = str(uuid.uuid4())
        now = _now()
        with self.connection() as con:
            con.execute(
                """INSERT INTO projects
                (id, name, description, source_filename, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (project_id, name, description, source_filename, dumps(config), now, now),
            )
        self.audit(project_id, "project_created", entity_type="project", entity_id=project_id)
        return project_id

    def list_projects(self) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT p.*,
                    (SELECT COUNT(*) FROM experiments e WHERE e.project_id = p.id) AS experiment_count,
                    (SELECT COUNT(*) FROM batches b WHERE b.project_id = p.id) AS batch_count,
                    (SELECT COUNT(*) FROM snapshots s WHERE s.project_id = p.id) AS snapshot_count
                FROM projects p ORDER BY p.updated_at DESC"""
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown project: {project_id}")
        result = dict(row)
        result["config"] = loads(result.pop("config_json"), {})
        return result

    def update_project_config(self, project_id: str, config: Mapping[str, Any]) -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE projects SET config_json = ?, updated_at = ? WHERE id = ?",
                (dumps(config), _now(), project_id),
            )
        self.audit(project_id, "project_config_updated", entity_type="project", entity_id=project_id)

    def import_history(self, project_id: str, data: pd.DataFrame) -> int:
        project = self.get_project(project_id)
        config = project["config"]
        feature_columns = [
            *config["mixture_columns"],
            *config.get("process_columns", []),
            *config.get("categorical_columns", []),
        ]
        response_columns = [item["response"] for item in config["response_specs"]]
        status_column = config.get("status_column")
        replicate_column = config.get("replicate_group_column")
        imported = 0
        now = _now()
        with self.connection() as con:
            for position, (_, row) in enumerate(data.iterrows(), start=1):
                code_value = row.get("experiment_id", f"HIST-{position:04d}")
                code = str(code_value) if pd.notna(code_value) else f"HIST-{position:04d}"
                inputs = {column: row.get(column) for column in feature_columns}
                responses = {
                    column: row.get(column)
                    for column in response_columns
                    if column in row.index and pd.notna(row.get(column))
                }
                status = "completed"
                if status_column and status_column in row.index:
                    status = str(row.get(status_column, "completed")).strip().lower() or "completed"
                if status not in {"completed", "failed", "planned", "running", "cancelled"}:
                    status = "completed" if responses else "failed"
                notes = str(row.get("failure_reason", "")) if pd.notna(row.get("failure_reason", "")) else ""
                replicate_group = code
                if replicate_column and replicate_column in row.index and pd.notna(row.get(replicate_column)):
                    replicate_group = str(row.get(replicate_column))
                experiment_id = str(uuid.uuid4())
                con.execute(
                    """INSERT OR REPLACE INTO experiments
                    (id, project_id, batch_id, experiment_code, source_type, purpose,
                     qualification_stage, status, inputs_json, responses_json,
                     recommendation_json, notes, replicate_group, replicate_index,
                     parent_experiment_id, created_at, updated_at)
                    VALUES (?, ?, NULL, ?, 'historical', NULL, 'historical', ?, ?, ?, '{}', ?, ?, 1, NULL, ?, ?)""",
                    (
                        experiment_id,
                        project_id,
                        code,
                        status,
                        dumps(inputs),
                        dumps(responses),
                        notes,
                        replicate_group,
                        now,
                        now,
                    ),
                )
                imported += 1
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        self.audit(project_id, "history_imported", detail={"rows": imported})
        return imported

    def create_batch(
        self,
        project_id: str,
        recommendations: pd.DataFrame,
        *,
        decision: str,
        decision_reason: str,
        qualification_stage: str = "discovery",
    ) -> str:
        with self.connection() as con:
            next_number = con.execute(
                "SELECT COALESCE(MAX(batch_number), 0) + 1 FROM batches WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
            batch_id = str(uuid.uuid4())
            now = _now()
            con.execute(
                """INSERT INTO batches
                (id, project_id, batch_number, status, decision, decision_reason, created_at)
                VALUES (?, ?, ?, 'proposed', ?, ?, ?)""",
                (batch_id, project_id, int(next_number), decision, decision_reason, now),
            )
            project = self.get_project(project_id)
            config = project["config"]
            feature_columns = [
                *config["mixture_columns"],
                *config.get("process_columns", []),
                *config.get("categorical_columns", []),
            ]
            for index, row in recommendations.reset_index(drop=True).iterrows():
                code = f"B{int(next_number):02d}-E{index + 1:02d}"
                experiment_id = str(uuid.uuid4())
                inputs = {column: row[column] for column in feature_columns}
                recommendation = row.to_dict()
                con.execute(
                    """INSERT INTO experiments
                    (id, project_id, batch_id, experiment_code, source_type, purpose,
                     qualification_stage, status, inputs_json, responses_json,
                     recommendation_json, notes, replicate_group, replicate_index,
                     parent_experiment_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'recommended', ?, ?, 'proposed', ?, '{}', ?, '', ?, 1, NULL, ?, ?)""",
                    (
                        experiment_id,
                        project_id,
                        batch_id,
                        code,
                        str(row.get("purpose", "Recommended candidate")),
                        qualification_stage,
                        dumps(inputs),
                        dumps(recommendation),
                        code,
                        now,
                        now,
                    ),
                )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        self.audit(
            project_id,
            "batch_created",
            entity_type="batch",
            entity_id=batch_id,
            detail={"batch_number": int(next_number), "recommendations": len(recommendations), "qualification_stage": qualification_stage},
        )
        return batch_id

    def list_batches(self, project_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT b.*,
                    COUNT(e.id) AS experiment_count,
                    SUM(CASE WHEN e.status IN ('completed', 'failed', 'cancelled') THEN 1 ELSE 0 END) AS resolved_count
                FROM batches b
                LEFT JOIN experiments e ON e.batch_id = b.id
                WHERE b.project_id = ?
                GROUP BY b.id
                ORDER BY b.batch_number DESC""",
                (project_id,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def close_batch(self, batch_id: str, *, cancel_unresolved: bool = False) -> dict[str, int]:
        """Formally close a batch, optionally cancelling every unresolved experiment."""
        batch = self.get_batch(batch_id)
        now = _now()
        with self.connection() as con:
            unresolved_rows = con.execute(
                """SELECT id FROM experiments WHERE batch_id = ?
                AND status NOT IN ('completed', 'failed', 'cancelled')""",
                (batch_id,),
            ).fetchall()
            unresolved_ids = [row[0] for row in unresolved_rows]
            if unresolved_ids and not cancel_unresolved:
                raise ValueError(
                    f"batch has {len(unresolved_ids)} unresolved experiment(s); cancel them or resolve them before closing"
                )
            cancelled = 0
            if unresolved_ids:
                placeholders = ",".join("?" for _ in unresolved_ids)
                con.execute(
                    f"UPDATE experiments SET status = 'cancelled', updated_at = ? WHERE id IN ({placeholders})",
                    [now, *unresolved_ids],
                )
                cancelled = len(unresolved_ids)
            con.execute(
                "UPDATE batches SET status = 'closed', completed_at = ? WHERE id = ?",
                (now, batch_id),
            )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, batch["project_id"]))
        self.audit(
            batch["project_id"],
            "batch_closed",
            entity_type="batch",
            entity_id=batch_id,
            detail={"cancelled_unresolved": cancelled},
        )
        return {"cancelled": cancelled, "closed": 1}

    def cancel_experiments(self, experiment_ids: list[str]) -> int:
        """Cancel a selected set of unresolved experiments from one project."""
        unique_ids = list(dict.fromkeys(str(value) for value in experiment_ids if value))
        if not unique_ids:
            return 0
        with self.connection() as con:
            placeholders = ",".join("?" for _ in unique_ids)
            rows = con.execute(
                f"SELECT id, project_id, batch_id, status FROM experiments WHERE id IN ({placeholders})",
                unique_ids,
            ).fetchall()
            if len(rows) != len(unique_ids):
                raise KeyError("one or more experiments were not found")
            project_ids = {row["project_id"] for row in rows}
            if len(project_ids) != 1:
                raise ValueError("bulk cancellation must be limited to one project")
            cancellable = [row["id"] for row in rows if row["status"] not in {"completed", "failed", "cancelled"}]
            if not cancellable:
                return 0
            placeholders = ",".join("?" for _ in cancellable)
            now = _now()
            con.execute(
                f"UPDATE experiments SET status = 'cancelled', updated_at = ? WHERE id IN ({placeholders})",
                [now, *cancellable],
            )
            project_id = next(iter(project_ids))
            batch_ids = {row["batch_id"] for row in rows if row["batch_id"]}
            for batch_id in batch_ids:
                unresolved = con.execute(
                    """SELECT COUNT(*) FROM experiments WHERE batch_id = ?
                    AND status NOT IN ('completed', 'failed', 'cancelled')""",
                    (batch_id,),
                ).fetchone()[0]
                if unresolved == 0:
                    con.execute(
                        "UPDATE batches SET status = 'completed', completed_at = COALESCE(completed_at, ?) WHERE id = ? AND status != 'closed'",
                        (now, batch_id),
                    )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        self.audit(
            project_id,
            "experiments_bulk_cancelled",
            entity_type="experiment",
            detail={"experiment_ids": cancellable, "count": len(cancellable)},
        )
        return len(cancellable)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown batch: {batch_id}")
        return dict(row)

    def approve_batch(self, batch_id: str) -> None:
        batch = self.get_batch(batch_id)
        now = _now()
        with self.connection() as con:
            con.execute(
                "UPDATE batches SET status = 'approved', approved_at = ? WHERE id = ?",
                (now, batch_id),
            )
            con.execute(
                "UPDATE experiments SET status = 'planned', updated_at = ? WHERE batch_id = ? AND status = 'proposed'",
                (now, batch_id),
            )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, batch["project_id"]))
        self.audit(batch["project_id"], "batch_approved", entity_type="batch", entity_id=batch_id)

    def list_experiments(
        self,
        project_id: str,
        *,
        batch_id: str | None = None,
        source_type: str | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM experiments WHERE project_id = ?"
        params: list[Any] = [project_id]
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        query += " ORDER BY created_at, experiment_code"
        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            inputs = loads(item.pop("inputs_json"), {})
            responses = loads(item.pop("responses_json"), {})
            recommendation = loads(item.pop("recommendation_json"), {})
            item.update(inputs)
            item.update(responses)
            item["recommendation"] = recommendation
            records.append(item)
        return pd.DataFrame(records)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown experiment: {experiment_id}")
        item = dict(row)
        item["inputs"] = loads(item.pop("inputs_json"), {})
        item["responses"] = loads(item.pop("responses_json"), {})
        item["recommendation"] = loads(item.pop("recommendation_json"), {})
        return item

    def replicate_group_status(self, experiment_id: str) -> pd.DataFrame:
        source = self.get_experiment(experiment_id)
        group = source.get("replicate_group") or source["experiment_code"]
        experiments = self.list_experiments(source["project_id"])
        return experiments[experiments["replicate_group"] == group].sort_values("replicate_index").reset_index(drop=True)

    def ensure_replicate_count(self, experiment_id: str, target_total: int) -> list[str]:
        """Create only the missing replicates required to reach a target group size."""
        if target_total < 1:
            raise ValueError("target_total must be at least 1")
        group = self.replicate_group_status(experiment_id)
        existing = int(len(group))
        if target_total < existing:
            raise ValueError(
                f"replicate group already contains {existing} run(s); cancel extras instead of creating a smaller target"
            )
        if target_total == existing:
            raise ValueError(f"replicate group already contains the requested {target_total} run(s)")
        return self.create_replicates(experiment_id, target_total - existing)

    def create_replicates(self, experiment_id: str, additional_replicates: int) -> list[str]:
        if additional_replicates < 1:
            raise ValueError("additional_replicates must be at least 1")
        source = self.get_experiment(experiment_id)
        if source["source_type"] != "recommended":
            raise ValueError("replicates can only be created from platform-recommended experiments")
        project_id = source["project_id"]
        group = source.get("replicate_group") or source["experiment_code"]
        created: list[str] = []
        now = _now()
        with self.connection() as con:
            max_index = con.execute(
                "SELECT COALESCE(MAX(replicate_index), 1) FROM experiments WHERE project_id = ? AND replicate_group = ?",
                (project_id, group),
            ).fetchone()[0]
            for offset in range(1, additional_replicates + 1):
                replicate_index = int(max_index) + offset
                code = f"{group}-R{replicate_index}"
                new_id = str(uuid.uuid4())
                con.execute(
                    """INSERT INTO experiments
                    (id, project_id, batch_id, experiment_code, source_type, purpose,
                     qualification_stage, status, inputs_json, responses_json,
                     recommendation_json, notes, replicate_group, replicate_index,
                     parent_experiment_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'recommended', ?, ?, 'planned', ?, '{}', ?, '', ?, ?, ?, ?, ?)""",
                    (
                        new_id,
                        project_id,
                        source["batch_id"],
                        code,
                        f"Replicate {replicate_index} of {group}",
                        source["qualification_stage"],
                        dumps(source["inputs"]),
                        dumps(source["recommendation"]),
                        group,
                        replicate_index,
                        experiment_id,
                        now,
                        now,
                    ),
                )
                created.append(new_id)
            if source["batch_id"]:
                con.execute(
                    "UPDATE batches SET status = CASE WHEN status = 'completed' THEN 'running' ELSE status END, completed_at = NULL WHERE id = ?",
                    (source["batch_id"],),
                )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        self.audit(
            project_id,
            "replicates_created",
            entity_type="experiment",
            entity_id=experiment_id,
            detail={"replicate_group": group, "additional_replicates": additional_replicates},
        )
        return created

    def has_design_fingerprint(self, project_id: str, fingerprint: str) -> bool:
        experiments = self.list_experiments(project_id, source_type="recommended")
        if experiments.empty:
            return False
        for recommendation in experiments.get("recommendation", []):
            if isinstance(recommendation, Mapping) and recommendation.get("design_fingerprint") == fingerprint:
                return True
        return False

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: str,
        responses: Mapping[str, Any] | None = None,
        notes: str = "",
        qualification_stage: str | None = None,
    ) -> str:
        allowed = {"proposed", "planned", "running", "completed", "failed", "cancelled"}
        if status not in allowed:
            raise ValueError(f"unsupported status: {status}")
        with self.connection() as con:
            row = con.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown experiment: {experiment_id}")
            project_id = row["project_id"]
            stage = qualification_stage or row["qualification_stage"]
            existing_responses = loads(row["responses_json"], {})
            provided_responses = existing_responses if responses is None else dict(responses)
            if status == "completed":
                project_config = loads(
                    con.execute("SELECT config_json FROM projects WHERE id = ?", (project_id,)).fetchone()[0],
                    {},
                )
                required_responses = [item["response"] for item in project_config.get("response_specs", [])]
                missing = [
                    response for response in required_responses
                    if response not in provided_responses
                    or provided_responses[response] is None
                    or (isinstance(provided_responses[response], str) and not provided_responses[response].strip())
                    or pd.isna(provided_responses[response])
                ]
                if missing:
                    raise ValueError(f"completed experiments require every measurement: {', '.join(missing)}")
                invalid = []
                zero_values = []
                zero_allowed = set(project_config.get("zero_allowed_responses", []))
                for response in required_responses:
                    try:
                        value = float(provided_responses[response])
                    except (TypeError, ValueError):
                        invalid.append(response)
                        continue
                    if not np.isfinite(value):
                        invalid.append(response)
                    if abs(value) <= 1e-15 and response not in zero_allowed:
                        zero_values.append(response)
                    provided_responses[response] = value
                if invalid:
                    raise ValueError(f"completed experiment measurements must be finite numbers: {', '.join(invalid)}")
                if zero_values:
                    raise ValueError(
                        "zero is not a valid completed measurement for: " + ", ".join(zero_values)
                        + ". Enter the measured value or mark the run failed/cancelled."
                    )
            response_json = dumps(provided_responses)
            now = _now()
            con.execute(
                """UPDATE experiments
                SET status = ?, responses_json = ?, notes = ?, qualification_stage = ?, updated_at = ?
                WHERE id = ?""",
                (status, response_json, notes, stage, now, experiment_id),
            )
            if row["batch_id"]:
                unresolved = con.execute(
                    """SELECT COUNT(*) FROM experiments
                    WHERE batch_id = ? AND status NOT IN ('completed', 'failed', 'cancelled')""",
                    (row["batch_id"],),
                ).fetchone()[0]
                if unresolved == 0:
                    con.execute(
                        "UPDATE batches SET status = 'completed', completed_at = ? WHERE id = ?",
                        (now, row["batch_id"]),
                    )
                else:
                    con.execute(
                        "UPDATE batches SET status = 'running' WHERE id = ? AND status IN ('approved', 'completed')",
                        (row["batch_id"],),
                    )
            con.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        self.audit(
            project_id,
            "experiment_updated",
            entity_type="experiment",
            entity_id=experiment_id,
            detail={"status": status, "qualification_stage": stage},
        )
        return project_id

    def project_dataframe(self, project_id: str) -> pd.DataFrame:
        project = self.get_project(project_id)
        config = project["config"]
        feature_columns = [
            *config["mixture_columns"],
            *config.get("process_columns", []),
            *config.get("categorical_columns", []),
        ]
        response_columns = [item["response"] for item in config["response_specs"]]
        status_column = config.get("status_column") or "status"
        experiments = self.list_experiments(project_id)
        rows: list[dict[str, Any]] = []
        for _, experiment in experiments.iterrows():
            record = {column: experiment.get(column) for column in feature_columns + response_columns}
            record[status_column] = experiment["status"]
            record["experiment_id"] = experiment["experiment_code"]
            record["source_type"] = experiment["source_type"]
            record["qualification_stage"] = experiment["qualification_stage"]
            record["replicate_group"] = experiment.get("replicate_group")
            record["replicate_index"] = experiment.get("replicate_index")
            record["notes"] = experiment["notes"]
            rows.append(record)
        return pd.DataFrame(rows)

    def save_snapshot(
        self,
        project_id: str,
        *,
        batch_id: str | None,
        trigger: str,
        decision: str,
        best_success_probability: float | None,
        best_feasibility_probability: float | None,
        qualification_score: float,
        completed_platform_experiments: int,
        compliant_platform_experiments: int,
        result_payload: Mapping[str, Any],
    ) -> str:
        snapshot_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO snapshots
                (id, project_id, batch_id, trigger, decision, best_success_probability,
                 best_feasibility_probability, qualification_score,
                 completed_platform_experiments, compliant_platform_experiments,
                 result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    project_id,
                    batch_id,
                    trigger,
                    decision,
                    best_success_probability,
                    best_feasibility_probability,
                    qualification_score,
                    completed_platform_experiments,
                    compliant_platform_experiments,
                    dumps(result_payload),
                    _now(),
                ),
            )
        self.audit(project_id, "snapshot_saved", entity_type="snapshot", entity_id=snapshot_id, detail={"trigger": trigger})
        return snapshot_id

    def list_snapshots(self, project_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                "SELECT * FROM snapshots WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["result"] = loads(item.pop("result_json"), {})
            records.append(item)
        return pd.DataFrame(records)

    def save_robustness_run(
        self,
        project_id: str,
        experiment_id: str,
        *,
        simulation_count: int,
        variation: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> str:
        run_id = str(uuid.uuid4())
        with self.connection() as con:
            con.execute(
                """INSERT INTO robustness_runs
                (id, project_id, experiment_id, simulation_count, variation_json, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, project_id, experiment_id, int(simulation_count), dumps(variation), dumps(result), _now()),
            )
        self.audit(
            project_id,
            "robustness_run_saved",
            entity_type="robustness_run",
            entity_id=run_id,
            detail={"experiment_id": experiment_id, "simulation_count": simulation_count},
        )
        return run_id

    def list_robustness_runs(self, project_id: str) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT r.*, e.experiment_code
                FROM robustness_runs r
                JOIN experiments e ON e.id = r.experiment_id
                WHERE r.project_id = ? ORDER BY r.created_at""",
                (project_id,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["variation"] = loads(item.pop("variation_json"), {})
            item["result"] = loads(item.pop("result_json"), {})
            records.append(item)
        return pd.DataFrame(records)

    def audit_log(self, project_id: str, limit: int = 500) -> pd.DataFrame:
        with self.connection() as con:
            rows = con.execute(
                """SELECT * FROM audit_events WHERE project_id = ?
                ORDER BY id DESC LIMIT ?""",
                (project_id, int(limit)),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["detail"] = loads(item.pop("detail_json"), {})
            records.append(item)
        return pd.DataFrame(records)
