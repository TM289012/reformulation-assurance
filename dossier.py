"""Qualification dossier and workbook exports for Reformulation Assurance v0.11.0."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import BytesIO
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile
import json

import numpy as np
import pandas as pd

from assurance_v4 import calibration_report, replicate_summary
from closed_loop import qualification_progress
from product_store import ProductStore


def _clean(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_clean(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _clean(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, pd.Interval):
        return {
            "left": _clean(value.left),
            "right": _clean(value.right),
            "closed": str(value.closed),
            "label": str(value),
        }
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def scientific_evidence_bundle(store: ProductStore, project_id: str) -> dict[str, Any]:
    project = store.get_project(project_id)
    experiments = store.list_experiments(project_id)
    batches = store.list_batches(project_id)
    snapshots = store.list_snapshots(project_id)
    robustness = store.list_robustness_runs(project_id)
    progress = qualification_progress(store, project_id)
    calibration = calibration_report(store, project_id)
    replicates = replicate_summary(store, project_id)

    project_fields = {
        "id": project["id"],
        "name": project["name"],
        "description": project["description"],
        "source_filename": project.get("source_filename"),
        "config": project["config"],
    }
    experiment_columns = [
        column for column in experiments.columns
        if column not in {"created_at", "updated_at"}
    ]
    batch_columns = [column for column in batches.columns if column not in {"created_at", "approved_at", "completed_at"}]
    snapshot_columns = [column for column in snapshots.columns if column != "created_at"]
    robustness_columns = [column for column in robustness.columns if column != "created_at"]

    return _clean({
        "project": project_fields,
        "experiments": experiments[experiment_columns] if not experiments.empty else experiments,
        "batches": batches[batch_columns] if not batches.empty else batches,
        "snapshots": snapshots[snapshot_columns] if not snapshots.empty else snapshots,
        "robustness_runs": robustness[robustness_columns] if not robustness.empty else robustness,
        "qualification": {
            "score": progress["score"],
            "all_gates_passed": progress["all_gates_passed"],
            "best_robust_probability": progress["best_robust_probability"],
            "stage_progress": progress["stage_progress"],
        },
        "replicate_summary": replicates,
        "calibration": {
            "run_level": calibration["run_level"],
            "formulation_level": calibration["formulation_level"],
        },
    })


def evidence_snapshot_and_hash(store: ProductStore, project_id: str) -> tuple[str, str]:
    """Freeze the current evidence as canonical JSON and hash those exact bytes.

    The snapshot string is what gets stored beside a signature: hashing its
    UTF-8 encoding always reproduces the returned hash, so any stored snapshot
    can be re-verified against the hash recorded at signing time.
    """
    payload = _canonical_json(scientific_evidence_bundle(store, project_id))
    return payload.decode("utf-8"), sha256(payload).hexdigest()


def compute_evidence_hash(store: ProductStore, project_id: str) -> str:
    return evidence_snapshot_and_hash(store, project_id)[1]


def _df_csv(data: pd.DataFrame) -> bytes:
    if data.empty:
        return b""
    frame = data.copy()
    for column in frame.columns:
        frame[column] = frame[column].map(
            lambda value: json.dumps(_clean(value), ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple)) else value
        )
    return frame.to_csv(index=False).encode("utf-8")


def _table_html(data: pd.DataFrame, max_rows: int = 50) -> str:
    if data.empty:
        return "<p><em>No evidence recorded.</em></p>"
    return data.head(max_rows).to_html(index=False, border=0, classes="evidence-table", escape=True)


def _dossier_html(
    *,
    project: Mapping[str, Any],
    version: int,
    evidence_hash: str,
    progress: Mapping[str, Any],
    approvals: pd.DataFrame,
    batches: pd.DataFrame,
    experiments: pd.DataFrame,
    calibration: Mapping[str, Any],
    robustness: pd.DataFrame,
    policy_status: pd.DataFrame,
    comments: pd.DataFrame,
    assignments: pd.DataFrame,
    generated_by: str,
    generated_at: str,
) -> str:
    current_approvals = approvals.copy()
    if not current_approvals.empty:
        current_approvals["matches_current_evidence"] = current_approvals["evidence_hash"] == evidence_hash
        if "evidence_snapshot" in current_approvals.columns:
            current_approvals["snapshot_stored"] = current_approvals["evidence_snapshot"].map(
                lambda value: bool(isinstance(value, str) and value)
            )
        else:
            current_approvals["snapshot_stored"] = False
        approval_view = current_approvals[[
            "stage", "status", "signer_name", "signer_role", "signature_meaning",
            "signed_at", "matches_current_evidence", "snapshot_stored", "comment"
        ]]
    else:
        approval_view = pd.DataFrame()
    stage_progress = progress["stage_progress"].copy()
    if not stage_progress.empty:
        stage_progress["completion"] = stage_progress["completion"].map(lambda value: f"{float(value):.0%}")
        stage_progress["success_rate"] = stage_progress["success_rate"].map(lambda value: f"{float(value):.0%}")
    response_summary = calibration["run_level"]["response_summary"].copy()
    formulation_summary = calibration["formulation_level"]["response_summary"].copy()
    if not response_summary.empty and "coverage_90" in response_summary:
        response_summary["coverage_90"] = response_summary["coverage_90"].map(lambda value: f"{float(value):.0%}")
    if not formulation_summary.empty and "coverage_90" in formulation_summary:
        formulation_summary["coverage_90"] = formulation_summary["coverage_90"].map(lambda value: f"{float(value):.0%}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(str(project['name']))} — Qualification Dossier v{version}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#17202a;line-height:1.45}}
h1,h2,h3{{color:#12344d}} .meta{{background:#f3f6f8;padding:16px;border-radius:8px}}
.metric-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}}
.metric{{border:1px solid #d5dde3;border-radius:8px;padding:14px}} .metric strong{{display:block;font-size:1.35rem}}
.evidence-table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:10px 0 24px}}
.evidence-table th,.evidence-table td{{border:1px solid #d5dde3;padding:7px;vertical-align:top}}
.evidence-table th{{background:#edf2f5;text-align:left}} code{{word-break:break-all}}
.warning{{background:#fff4d6;border-left:5px solid #d79b00;padding:12px}} .footer{{margin-top:50px;font-size:.82rem;color:#566}}
@media print{{body{{max-width:none;margin:0}}}}
</style></head><body>
<h1>Qualification Dossier</h1>
<div class="meta"><strong>{escape(str(project['name']))}</strong><br>{escape(str(project.get('description') or ''))}<br>
Version {version} · Generated {escape(generated_at)} by {escape(generated_by)}<br>
Evidence SHA-256: <code>{evidence_hash}</code></div>
<div class="warning"><strong>Decision-support record:</strong> This dossier documents software evidence and approvals. It does not replace chemical-safety review, regulatory review, a validated quality system, or final release authority.</div>
<div class="metric-grid">
<div class="metric">Qualification progress<strong>{float(progress['score']):.0f}%</strong></div>
<div class="metric">All gates passed<strong>{'Yes' if progress['all_gates_passed'] else 'No'}</strong></div>
<div class="metric">Best robust probability<strong>{'—' if progress['best_robust_probability'] is None else f"{float(progress['best_robust_probability']):.0%}"}</strong></div>
</div>
<h2>Qualification gates</h2>{_table_html(stage_progress)}
<h2>Approval policies</h2>{_table_html(policy_status)}
<h2>Approvals and signatures</h2>{_table_html(approval_view)}
<h2>Recommendation batches</h2>{_table_html(batches)}
<h2>Experiment evidence</h2>{_table_html(experiments, max_rows=200)}
<h2>Prospective calibration</h2>
<h3>Run level</h3>
<p>Brier score: {'—' if calibration['run_level']['brier_score'] is None else f"{float(calibration['run_level']['brier_score']):.4f}"}</p>
{_table_html(response_summary)}
<h3>Formulation level</h3>
<p>Brier score: {'—' if calibration['formulation_level']['brier_score'] is None else f"{float(calibration['formulation_level']['brier_score']):.4f}"}</p>
{_table_html(formulation_summary)}
<h2>Robustness evidence</h2>{_table_html(robustness)}
<h2>Project comments</h2>{_table_html(comments)}
<h2>Assignments</h2>{_table_html(assignments)}
<div class="footer">Generated by Reformulation Assurance v0.11.0. Verify the evidence hash and the SHA256SUMS file before relying on an exported copy.</div>
</body></html>"""


def generate_dossier(
    store: ProductStore,
    project_id: str,
    *,
    generated_by_user_id: str,
) -> tuple[bytes, dict[str, Any]]:
    store.require_project_access(generated_by_user_id, project_id)
    project = store.get_project(project_id)
    user = store.get_user(generated_by_user_id)
    evidence = scientific_evidence_bundle(store, project_id)
    evidence_hash = sha256(_canonical_json(evidence)).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    approvals = store.list_approvals(project_id)
    progress = qualification_progress(store, project_id)
    calibration = calibration_report(store, project_id)
    batches = store.list_batches(project_id)
    experiments = store.list_experiments(project_id)
    robustness = store.list_robustness_runs(project_id)
    policy_status = store.approval_policy_status(project_id, evidence_hash) if hasattr(store, "approval_policy_status") else pd.DataFrame()
    comments = store.list_comments(project_id, generated_by_user_id) if hasattr(store, "list_comments") else pd.DataFrame()
    assignments = store.list_assignments(project_id, generated_by_user_id) if hasattr(store, "list_assignments") else pd.DataFrame()
    audit_before = store.audit_log(project_id)

    base_manifest = {
        "format": "reformulation-assurance-qualification-dossier",
        "format_version": 2,
        "project_id": project_id,
        "project_name": project["name"],
        "generated_at": generated_at,
        "generated_by": {"user_id": user["id"], "display_name": user["display_name"], "email": user["email"]},
        "scientific_evidence_sha256": evidence_hash,
        "approval_count": int(len(approvals)),
        "approval_policy_count": int(len(policy_status)),
        "completed_approval_policy_count": int(policy_status["complete"].sum()) if not policy_status.empty else 0,
        "all_qualification_gates_passed": bool(progress["all_gates_passed"]),
        "disclaimer": "Prototype decision-support evidence; not a validated regulated quality system.",
    }
    dossier_id, version = store.save_dossier_record(
        project_id,
        generated_by_user_id=generated_by_user_id,
        evidence_hash=evidence_hash,
        manifest=base_manifest,
    )
    manifest = {**base_manifest, "dossier_id": dossier_id, "version": version}
    audit = store.audit_log(project_id)

    files: dict[str, bytes] = {
        "qualification_dossier.html": _dossier_html(
            project=project,
            version=version,
            evidence_hash=evidence_hash,
            progress=progress,
            approvals=approvals,
            batches=batches,
            experiments=experiments,
            calibration=calibration,
            robustness=robustness,
            policy_status=policy_status,
            comments=comments,
            assignments=assignments,
            generated_by=user["display_name"],
            generated_at=generated_at,
        ).encode("utf-8"),
        "manifest.json": json.dumps(_clean(manifest), indent=2, sort_keys=True).encode("utf-8"),
        "scientific_evidence.json": json.dumps(_clean(evidence), indent=2, sort_keys=True).encode("utf-8"),
        "experiments.csv": _df_csv(experiments),
        "recommendation_batches.csv": _df_csv(batches),
        "qualification_gates.csv": _df_csv(progress["stage_progress"]),
        "replicate_summary.csv": _df_csv(progress["replicate_summary"]),
        "calibration_run_response_summary.csv": _df_csv(calibration["run_level"]["response_summary"]),
        "calibration_run_observations.csv": _df_csv(calibration["run_level"]["observations"]),
        "calibration_formulation_response_summary.csv": _df_csv(calibration["formulation_level"]["response_summary"]),
        "calibration_formulation_observations.csv": _df_csv(calibration["formulation_level"]["observations"]),
        "robustness_runs.csv": _df_csv(robustness),
        "approval_policies.csv": _df_csv(policy_status),
        "approvals.csv": _df_csv(approvals.drop(columns=["evidence_snapshot"], errors="ignore")),
        "comments.csv": _df_csv(comments),
        "assignments.csv": _df_csv(assignments),
        "audit_trail.csv": _df_csv(audit),
    }
    if not approvals.empty and "evidence_snapshot" in approvals.columns:
        for _, approval_row in approvals.iterrows():
            snapshot = approval_row.get("evidence_snapshot")
            if isinstance(snapshot, str) and snapshot:
                snapshot_name = (
                    f"signed_evidence_snapshots/{approval_row['stage']}_{str(approval_row['evidence_hash'])[:12]}.json"
                )
                files[snapshot_name] = snapshot.encode("utf-8")
    checksums = []
    for name, content in sorted(files.items()):
        checksums.append(f"{sha256(content).hexdigest()}  {name}")
    files["SHA256SUMS.txt"] = ("\n".join(checksums) + "\n").encode("utf-8")

    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue(), manifest


def generate_workbook(
    store: ProductStore,
    project_id: str,
    *,
    generated_by_user_id: str,
) -> tuple[bytes, dict[str, Any]]:
    """One-click Excel export: the whole project as tabs in a single .xlsx.

    The workbench treats the user's spreadsheet as the system of record, so the
    analysis has to travel back into spreadsheet-land: every evidence table
    becomes a tab, with a Read Me cover sheet carrying the evidence hash.
    """
    store.require_project_access(generated_by_user_id, project_id)
    project = store.get_project(project_id)
    user = store.get_user(generated_by_user_id)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evidence_hash = compute_evidence_hash(store, project_id)
    progress = qualification_progress(store, project_id)
    calibration = calibration_report(store, project_id)
    approvals = store.list_approvals(project_id)
    if not approvals.empty:
        approvals = approvals.drop(columns=["evidence_snapshot"], errors="ignore")

    def _cell(value: Any) -> Any:
        cleaned = _clean(value)
        if isinstance(cleaned, (dict, list)):
            return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        return cleaned

    def _sheet_frame(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return pd.DataFrame({"note": ["No records yet."]})
        frame = data.copy()
        for column in frame.columns:
            frame[column] = frame[column].map(_cell)
        return frame

    cover = pd.DataFrame(
        [
            ["Project", str(project["name"])],
            ["Description", str(project.get("description") or "")],
            ["Generated at (UTC)", generated_at],
            ["Generated by", f"{user['display_name']} ({user['email']})"],
            ["Scientific evidence SHA-256", evidence_hash],
            ["Qualification progress", f"{float(progress['score']):.0f}%"],
            ["All gates passed", "Yes" if progress["all_gates_passed"] else "No"],
            ["", ""],
            ["About this file", "One-click export of the whole project, one evidence table per tab. Your spreadsheet stays the system of record; this hands the analysis back."],
            ["Disclaimer", "Prototype decision-support evidence; not a validated regulated quality system."],
        ],
        columns=["field", "value"],
    )

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("Read Me", cover),
        ("Experiments", _sheet_frame(store.list_experiments(project_id))),
        ("Recommendation Batches", _sheet_frame(store.list_batches(project_id))),
        ("Qualification Gates", _sheet_frame(progress["stage_progress"])),
        ("Replicate Summary", _sheet_frame(progress["replicate_summary"])),
        ("Calibration Run Summary", _sheet_frame(calibration["run_level"]["response_summary"])),
        ("Calibration Run Obs", _sheet_frame(calibration["run_level"]["observations"])),
        ("Calibration Form Summary", _sheet_frame(calibration["formulation_level"]["response_summary"])),
        ("Calibration Form Obs", _sheet_frame(calibration["formulation_level"]["observations"])),
        ("Robustness Runs", _sheet_frame(store.list_robustness_runs(project_id))),
        ("Approvals", _sheet_frame(approvals)),
        ("Audit Trail", _sheet_frame(store.audit_log(project_id))),
    ]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            has_header = sheet_name != "Read Me"
            frame.to_excel(writer, sheet_name=sheet_name, index=False, header=has_header)
            worksheet = writer.sheets[sheet_name]
            for column_cells in worksheet.columns:
                sample = [str(cell.value) for cell in column_cells[:20] if cell.value is not None]
                width = min(52, max(12, *(len(text) for text in sample)) + 2) if sample else 12
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
            if has_header:
                worksheet.freeze_panes = "A2"

    manifest = {
        "format": "reformulation-assurance-workbook-export",
        "project_id": project_id,
        "project_name": project["name"],
        "generated_at": generated_at,
        "scientific_evidence_sha256": evidence_hash,
        "sheets": [name for name, _ in sheets],
    }
    store.audit(
        project_id,
        "workbook_exported",
        entity_type="project",
        entity_id=project_id,
        detail={"sheets": len(sheets), "evidence_hash": evidence_hash},
    )
    return output.getvalue(), manifest
