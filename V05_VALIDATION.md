# v0.5 Validation

Run:

```bash
python -m unittest discover -s tests -v
python demo_v05.py
```

## Automated result

Twelve tests pass:

- seven retained v0.4 scientific-assurance tests;
- authentication and wrong-password rejection;
- organization-level project isolation;
- role and password re-authentication at approval;
- stale evidence-hash detection;
- multi-sheet Excel ingestion;
- qualification dossier content and checksum verification.

## End-to-end demonstration

The demonstration creates a workspace owner, imports a two-sheet Excel workbook, creates an isolated project, records an evidence-bound approval, generates a dossier, and opens the dossier ZIP to verify its structure.

Representative result:

```text
Excel sheets: ['Experiments', 'Metadata']
Imported rows: 72
Tenant projects: 1
Evidence hash: <SHA-256 prefix>
Dossier version: 1
Dossier files: 13
```

## Not validated in this environment

Streamlit was not installed in the execution environment, so the browser UI was syntax-compiled but not interactively launched. The product lifecycle, persistence, security primitives, ingestion, approvals, and dossier generation were exercised directly through Python.
