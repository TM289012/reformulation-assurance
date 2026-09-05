"""ELN archive export (.eln, RO-Crate) for Reformulation Assurance v0.11.0.

An .eln file is a zipped RO-Crate: a single root folder holding a
``ro-crate-metadata.json`` that describes everything beside it. It is the
interchange format of the ELN Consortium and is imported by eLabFTW, RSpace
and other electronic lab notebooks, so a project's evidence can be filed in
the notebook a lab already keeps instead of a folder of loose ZIPs.

The export packages one project's qualification dossier as ONE experiment
entry: a compact HTML summary as the entry body, and every evidence table,
the scientific-evidence JSON, the signed evidence snapshots, the checksum
list and (optionally) the Excel workbook as attached files, each described
with its size, media type and SHA-256 in the crate metadata.

Verified against the ELN file format specification (TheELNConsortium) and
eLabFTW's importer (src/Import/Eln.php): the importer walks the root
dataset's ``hasPart``, files a dataset with ``genre: "experiment"`` as an
experiment, uses ``name`` as the title, ``text`` as the body, ``temporal``
as the entry date, ``keywords`` as tags, resolves each ``File`` by its
``@id`` relative to the root folder, verifies ``sha256`` when asked to, and
reads extra fields from a ``PropertyValue`` whose ``propertyID`` is
``elabftw_metadata``. Everything the importer touches is written inline so
no lookup can fail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from io import BytesIO
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile
import json
import re

import pandas as pd

from closed_loop import qualification_progress
from dossier import _canonical_json, _table_html, generate_dossier, generate_workbook
from product_store import ProductStore

SOFTWARE_NAME = "Reformulation Assurance"
SOFTWARE_VERSION = "0.11.0"
SOFTWARE_URL = "https://github.com/TM289012/reformulation-assurance"
ELN_MEDIA_TYPE = "application/vnd.eln+zip"
RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
RO_CRATE_CONFORMS_TO = "https://w3id.org/ro/crate/1.1"

_MEDIA_TYPES = {
    ".html": "text/html",
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_FILE_DESCRIPTIONS = {
    "qualification_dossier.html": "Printable qualification dossier (the full report).",
    "manifest.json": "Dossier manifest: version, evidence hash, generating user, disclaimer.",
    "scientific_evidence.json": "Complete scientific evidence as readable JSON.",
    "scientific_evidence.canonical.json": "The same evidence as canonical bytes: sha256sum of this file IS the evidence hash.",
    "experiments.csv": "Every experiment with recipe, responses and pass/fail status.",
    "recommendation_batches.csv": "Recommendation batches and their approval state.",
    "qualification_gates.csv": "Stage-by-stage qualification gate progress.",
    "replicate_summary.csv": "Replicate groups: consistency screen, CV and gate status.",
    "calibration_run_response_summary.csv": "Run-level prospective calibration summary.",
    "calibration_run_observations.csv": "Run-level calibration observations.",
    "calibration_formulation_response_summary.csv": "Formulation-level prospective calibration summary.",
    "calibration_formulation_observations.csv": "Formulation-level calibration observations.",
    "robustness_runs.csv": "Robustness (process-window) runs.",
    "approval_policies.csv": "Approval policies and their completion.",
    "approvals.csv": "Signed approvals (evidence hash per signature).",
    "comments.csv": "Project comments.",
    "assignments.csv": "Project assignments.",
    "audit_trail.csv": "Audit trail of every recorded event.",
    "SHA256SUMS.txt": "SHA-256 of every file in the dossier package.",
}


def _slug(value: str, fallback: str = "project") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or fallback


def _media_type(name: str) -> str:
    for suffix, media_type in _MEDIA_TYPES.items():
        if name.lower().endswith(suffix):
            return media_type
    return "application/octet-stream"


def _split_name(display_name: str) -> tuple[str, str]:
    parts = [part for part in str(display_name or "").strip().split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]


def _describe(name: str) -> str:
    if name in _FILE_DESCRIPTIONS:
        return _FILE_DESCRIPTIONS[name]
    if name.startswith("signed_evidence_snapshots/"):
        return "Evidence snapshot frozen at signing; re-hashing it reproduces the hash recorded with the signature."
    if name.endswith(".xlsx"):
        return "Excel workbook: every evidence table as a tab, evidence hash on the cover sheet."
    return "Exported evidence file."


def _fmt_probability(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def _eln_body_html(
    *,
    project: Mapping[str, Any],
    version: int,
    evidence_hash: str,
    progress: Mapping[str, Any],
    approvals: pd.DataFrame,
    files: Mapping[str, bytes],
    generated_by: str,
    generated_at: str,
) -> str:
    """Compact HTML fragment for the notebook entry body (no <html>/<style> wrapper)."""
    stage_progress = progress["stage_progress"].copy()
    if not stage_progress.empty:
        for column in ("completion", "success_rate"):
            if column in stage_progress:
                stage_progress[column] = stage_progress[column].map(lambda value: f"{float(value):.0%}")
    if not approvals.empty:
        approval_view = approvals.copy()
        approval_view["matches_this_evidence"] = approval_view["evidence_hash"] == evidence_hash
        keep = [
            column for column in
            ["stage", "status", "signer_name", "signer_role", "signed_at", "matches_this_evidence"]
            if column in approval_view.columns
        ]
        approval_view = approval_view[keep]
    else:
        approval_view = pd.DataFrame()
    attachments = "".join(
        f"<li><code>{escape(name)}</code> ({len(content)} bytes, sha256 {sha256(content).hexdigest()[:16]}…)</li>"
        for name, content in sorted(files.items())
    )
    best = _fmt_probability(progress.get("best_robust_probability"))
    return (
        f"<h1>Qualification dossier v{version}: {escape(str(project['name']))}</h1>"
        f"<p>{escape(str(project.get('description') or ''))}</p>"
        f"<p>Generated {escape(generated_at)} by {escape(generated_by)} with {SOFTWARE_NAME} v{SOFTWARE_VERSION}.</p>"
        f"<p><strong>Qualification progress:</strong> {float(progress['score']):.0f}% &nbsp; "
        f"<strong>All gates passed:</strong> {'Yes' if progress['all_gates_passed'] else 'No'} &nbsp; "
        f"<strong>Best robust probability:</strong> {best}</p>"
        f"<p><strong>Scientific evidence SHA-256:</strong> <code>{evidence_hash}</code></p>"
        "<p><em>Decision-support record: this entry documents software evidence and approvals. "
        "It does not replace chemical-safety review, regulatory review, a validated quality system, "
        "or final release authority.</em></p>"
        f"<h2>Qualification gates</h2>{_table_html(stage_progress)}"
        f"<h2>Approvals and signatures</h2>{_table_html(approval_view)}"
        f"<h2>Attached files</h2><ul>{attachments}</ul>"
        "<p>The full printable dossier is attached as <code>qualification_dossier.html</code>. "
        "<code>scientific_evidence.canonical.json</code> holds the complete evidence as canonical bytes: "
        "its sha256sum is the evidence hash above.</p>"
    )


def _elabftw_metadata(
    *,
    project: Mapping[str, Any],
    version: int,
    dossier_id: str,
    evidence_hash: str,
    progress: Mapping[str, Any],
    generated_by: str,
    generated_at: str,
) -> str:
    """eLabFTW extra-fields JSON (shown as structured fields beside the entry body)."""
    fields = [
        ("Project", "text", str(project["name"])),
        ("Project id", "text", str(project["id"])),
        ("Dossier id", "text", str(dossier_id)),
        ("Dossier version", "number", str(int(version))),
        ("Scientific evidence SHA-256", "text", evidence_hash),
        ("Qualification progress (%)", "number", f"{float(progress['score']):.0f}"),
        ("All gates passed", "checkbox", "on" if progress["all_gates_passed"] else ""),
        ("Best robust probability", "text", _fmt_probability(progress.get("best_robust_probability"))),
        ("Generated at (UTC)", "text", generated_at),
        ("Generated by", "text", generated_by),
        ("Software", "text", f"{SOFTWARE_NAME} v{SOFTWARE_VERSION}"),
        ("Software URL", "url", SOFTWARE_URL),
        ("Disclaimer", "text", "Prototype decision-support evidence; not a validated regulated quality system."),
    ]
    extra_fields = {
        name: {"type": kind, "value": value, "group_id": 1, "position": position, "readonly": True}
        for position, (name, kind, value) in enumerate(fields)
    }
    return json.dumps(
        {
            "elabftw": {
                "display_main_text": True,
                "extra_fields_groups": [{"id": 1, "name": SOFTWARE_NAME}],
            },
            "extra_fields": extra_fields,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def generate_eln(
    store: ProductStore,
    project_id: str,
    *,
    generated_by_user_id: str,
    include_workbook: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Export the project's qualification dossier as an .eln archive (zipped RO-Crate).

    Returns the archive bytes and a manifest (archive name, root folder, file
    list with hashes, dossier version, evidence hash). Generating the dossier
    records a dossier version exactly as the ZIP export does, so the notebook
    entry and the dossier history agree on the version number.
    """
    store.require_project_access(generated_by_user_id, project_id)
    project = store.get_project(project_id)
    user = store.get_user(generated_by_user_id)

    dossier_bytes, dossier_manifest = generate_dossier(store, project_id, generated_by_user_id=generated_by_user_id)
    files: dict[str, bytes] = {}
    with ZipFile(BytesIO(dossier_bytes)) as archive:
        for member in archive.namelist():
            if not member.endswith("/"):
                files[member] = archive.read(member)
    if include_workbook:
        workbook_bytes, _ = generate_workbook(store, project_id, generated_by_user_id=generated_by_user_id)
        files[f"{_slug(project['name'])}_workbench_export.xlsx"] = workbook_bytes

    version = int(dossier_manifest["version"])
    dossier_id = str(dossier_manifest["dossier_id"])
    evidence_hash = str(dossier_manifest["scientific_evidence_sha256"])
    # Ship the evidence in its canonical byte form too, so anyone holding the
    # archive can reproduce the evidence hash with a plain sha256sum.
    canonical = _canonical_json(json.loads(files["scientific_evidence.json"].decode("utf-8")))
    if sha256(canonical).hexdigest() != evidence_hash:
        raise RuntimeError("canonical evidence bytes do not reproduce the dossier's evidence hash")
    files["scientific_evidence.canonical.json"] = canonical
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress = qualification_progress(store, project_id)
    approvals = store.list_approvals(project_id)
    generated_by = str(user["display_name"])
    given_name, family_name = _split_name(generated_by)

    root_folder = f"{_slug(project['name'])}_qualification_dossier_v{version}"
    archive_name = f"{root_folder}.eln"
    experiment_folder = f"dossier-v{version}"
    experiment_id = f"./{experiment_folder}/"
    author_id = f"./author/{user['id']}"
    metadata_property_id = "#elabftw-metadata"
    evidence_property_id = "#scientific-evidence-sha256"
    version_property_id = "#dossier-version"

    body_html = _eln_body_html(
        project=project,
        version=version,
        evidence_hash=evidence_hash,
        progress=progress,
        approvals=approvals,
        files=files,
        generated_by=generated_by,
        generated_at=generated_at,
    )
    elabftw_metadata = _elabftw_metadata(
        project=project,
        version=version,
        dossier_id=dossier_id,
        evidence_hash=evidence_hash,
        progress=progress,
        generated_by=generated_by,
        generated_at=generated_at,
    )

    # PropertyValue nodes are written inline under variableMeasured (what
    # eLabFTW reads) and repeated as graph nodes (what RO-Crate tooling reads).
    property_values = [
        {
            "@id": metadata_property_id,
            "@type": "PropertyValue",
            "propertyID": "elabftw_metadata",
            "name": "eLabFTW extra fields",
            "value": elabftw_metadata,
        },
        {
            "@id": evidence_property_id,
            "@type": "PropertyValue",
            "propertyID": "scientific_evidence_sha256",
            "name": "Scientific evidence SHA-256",
            "value": evidence_hash,
        },
        {
            "@id": version_property_id,
            "@type": "PropertyValue",
            "propertyID": "dossier_version",
            "name": "Dossier version",
            "value": str(version),
        },
    ]

    file_nodes: list[dict[str, Any]] = []
    file_manifest: list[dict[str, Any]] = []
    for name, content in sorted(files.items()):
        file_id = f"{experiment_id}{name}"
        digest = sha256(content).hexdigest()
        file_nodes.append(
            {
                "@id": file_id,
                "@type": "File",
                "name": name.rsplit("/", 1)[-1],
                "description": _describe(name),
                "encodingFormat": _media_type(name),
                "contentSize": str(len(content)),
                "sha256": digest,
                "dateCreated": generated_at,
            }
        )
        file_manifest.append({"path": f"{experiment_folder}/{name}", "sha256": digest, "bytes": len(content)})

    experiment_node = {
        "@id": experiment_id,
        "@type": "Dataset",
        "name": f"{project['name']}: qualification dossier v{version}",
        "identifier": dossier_id,
        "genre": "experiment",
        "author": {"@id": author_id},
        "dateCreated": generated_at,
        "dateModified": generated_at,
        "temporal": generated_at,
        "text": body_html,
        "keywords": ["reformulation assurance", "qualification dossier", "formulation"],
        "url": SOFTWARE_URL,
        "hasPart": [{"@id": node["@id"]} for node in file_nodes],
        "variableMeasured": property_values,
    }
    organization_node = {
        "@id": SOFTWARE_URL,
        "@type": "Organization",
        "name": SOFTWARE_NAME,
        "url": SOFTWARE_URL,
    }
    person_node = {
        "@id": author_id,
        "@type": "Person",
        "givenName": given_name,
        "familyName": family_name,
        "name": generated_by,
    }
    if user.get("email"):
        person_node["email"] = str(user["email"])
    graph = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": RO_CRATE_CONFORMS_TO},
            "dateCreated": generated_at,
            "sdPublisher": {"@id": SOFTWARE_URL},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": f"{project['name']}: qualification dossier v{version} ({SOFTWARE_NAME})",
            "description": (
                f"Qualification evidence for '{project['name']}' exported from {SOFTWARE_NAME} "
                f"v{SOFTWARE_VERSION} as one notebook entry with attached evidence files."
            ),
            "datePublished": generated_at,
            "hasPart": [{"@id": experiment_id}],
        },
        organization_node,
        person_node,
        experiment_node,
        *file_nodes,
        *property_values,
    ]
    metadata_bytes = json.dumps(
        {"@context": RO_CRATE_CONTEXT, "@graph": graph}, indent=2, ensure_ascii=False
    ).encode("utf-8")

    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root_folder}/", b"")
        archive.writestr(f"{root_folder}/ro-crate-metadata.json", metadata_bytes)
        for name, content in sorted(files.items()):
            archive.writestr(f"{root_folder}/{experiment_folder}/{name}", content)
    eln_bytes = output.getvalue()

    manifest = {
        "format": "eln-ro-crate",
        "media_type": ELN_MEDIA_TYPE,
        "archive_name": archive_name,
        "root_folder": root_folder,
        "experiment_id": experiment_id,
        "project_id": project_id,
        "project_name": project["name"],
        "dossier_id": dossier_id,
        "dossier_version": version,
        "scientific_evidence_sha256": evidence_hash,
        "generated_at": generated_at,
        "generated_by": {"user_id": user["id"], "display_name": generated_by, "email": user.get("email")},
        "metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "archive_sha256": sha256(eln_bytes).hexdigest(),
        "file_count": len(files),
        "files": file_manifest,
        "workbook_included": bool(include_workbook),
    }
    store.audit(
        project_id,
        "eln_exported",
        entity_type="dossier",
        entity_id=dossier_id,
        detail={
            "archive_name": archive_name,
            "dossier_version": version,
            "evidence_hash": evidence_hash,
            "file_count": len(files),
            "archive_sha256": manifest["archive_sha256"],
        },
    )
    return eln_bytes, manifest


def _main(argv: list[str] | None = None) -> int:
    """CLI: python eln_export.py --db data/reformulation_assurance_v06.db --project <id> --email you@lab.org --out project.eln"""
    import argparse
    from pathlib import Path

    from pilot_store import PilotStore

    parser = argparse.ArgumentParser(description="Export a project's qualification dossier as an .eln archive.")
    parser.add_argument("--db", required=True, help="path to the workspace database")
    parser.add_argument("--project", required=True, help="project id")
    parser.add_argument("--email", required=True, help="email of the exporting user (must have project access)")
    parser.add_argument("--out", help="output path (defaults to the archive name in the current directory)")
    parser.add_argument("--no-workbook", action="store_true", help="skip the Excel workbook attachment")
    args = parser.parse_args(argv)

    store = PilotStore(Path(args.db))
    with store.connection() as con:
        row = con.execute("SELECT id FROM users WHERE lower(email) = lower(?)", (args.email,)).fetchone()
    if row is None:
        parser.error(f"no user with email {args.email}")
    eln_bytes, manifest = generate_eln(
        store, args.project, generated_by_user_id=row["id"], include_workbook=not args.no_workbook
    )
    out = Path(args.out) if args.out else Path(manifest["archive_name"])
    out.write_bytes(eln_bytes)
    print(f"wrote {out} ({len(eln_bytes)} bytes, {manifest['file_count']} files, dossier v{manifest['dossier_version']})")
    print(f"evidence sha256 {manifest['scientific_evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
