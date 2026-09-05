"""Tests for the .eln (RO-Crate) notebook export (v0.11.0).

The checks mirror what an importing notebook actually does: find the single
root folder, read ro-crate-metadata.json, walk the root dataset's hasPart,
resolve every file by its @id, and verify sizes and SHA-256 hashes.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_seed import DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD, seed_demo  # noqa: E402
from dossier import evidence_snapshot_and_hash  # noqa: E402
from eln_export import ELN_MEDIA_TYPE, RO_CRATE_CONFORMS_TO, RO_CRATE_CONTEXT, generate_eln  # noqa: E402
from pilot_store import PilotStore  # noqa: E402


class ElnExportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = PilotStore(Path(self.tempdir.name) / "workspace.db")
        self.project_id = seed_demo(self.store)
        self.user = self.store.authenticate(DEMO_OWNER_EMAIL, DEMO_OWNER_PASSWORD)
        snapshot, evidence_hash = evidence_snapshot_and_hash(self.store, self.project_id)
        self.store.sign_approval(
            self.project_id,
            stage="screening",
            signer_user_id=self.user["id"],
            typed_name=self.user["display_name"],
            password=DEMO_OWNER_PASSWORD,
            signature_meaning="I reviewed the evidence for this stage and approve progression.",
            evidence_hash=evidence_hash,
            evidence_snapshot=snapshot,
        )
        self.evidence_hash = evidence_hash

    def tearDown(self):
        self.tempdir.cleanup()

    def _export(self, **kwargs):
        eln_bytes, manifest = generate_eln(
            self.store, self.project_id, generated_by_user_id=self.user["id"], **kwargs
        )
        archive = ZipFile(BytesIO(eln_bytes))
        metadata = json.loads(archive.read(f"{manifest['root_folder']}/ro-crate-metadata.json"))
        return eln_bytes, manifest, archive, metadata

    def test_archive_has_single_root_folder_with_metadata(self):
        eln_bytes, manifest, archive, metadata = self._export()
        self.assertTrue(manifest["archive_name"].endswith(".eln"))
        self.assertEqual(manifest["media_type"], ELN_MEDIA_TYPE)
        root = manifest["root_folder"]
        self.assertEqual(manifest["archive_name"], f"{root}.eln")
        for name in archive.namelist():
            self.assertTrue(name.startswith(f"{root}/"), name)
        self.assertIn(f"{root}/ro-crate-metadata.json", archive.namelist())
        self.assertEqual(metadata["@context"], RO_CRATE_CONTEXT)
        self.assertEqual(manifest["archive_sha256"], sha256(eln_bytes).hexdigest())

    def test_metadata_descriptor_and_root_dataset(self):
        _, manifest, _, metadata = self._export()
        nodes = {node["@id"]: node for node in metadata["@graph"]}
        descriptor = nodes["ro-crate-metadata.json"]
        self.assertEqual(descriptor["@type"], "CreativeWork")
        self.assertEqual(descriptor["about"], {"@id": "./"})
        self.assertEqual(descriptor["conformsTo"], {"@id": RO_CRATE_CONFORMS_TO})
        publisher = nodes[descriptor["sdPublisher"]["@id"]]
        self.assertEqual(publisher["@type"], "Organization")
        root = nodes["./"]
        self.assertEqual(root["@type"], "Dataset")
        self.assertEqual(root["hasPart"], [{"@id": manifest["experiment_id"]}])
        # eLabFTW reads a "version" on the root dataset as ITS OWN internal format marker; never set one.
        self.assertNotIn("version", root)

    def test_experiment_entry_carries_what_an_eln_importer_reads(self):
        _, manifest, _, metadata = self._export()
        nodes = {node["@id"]: node for node in metadata["@graph"]}
        experiment = nodes[manifest["experiment_id"]]
        self.assertEqual(experiment["@type"], "Dataset")
        self.assertEqual(experiment["genre"], "experiment")
        self.assertIn("qualification dossier v", experiment["name"])
        for key in ("dateCreated", "dateModified", "temporal", "text", "keywords", "hasPart", "variableMeasured"):
            self.assertIn(key, experiment)
        self.assertIn(self.evidence_hash, experiment["text"])
        self.assertIn("<h1>", experiment["text"])
        self.assertNotIn("<html", experiment["text"])  # a body fragment, not a document
        self.assertIsInstance(experiment["keywords"], list)
        author = nodes[experiment["author"]["@id"]]
        self.assertEqual(author["@type"], "Person")
        self.assertTrue(author["givenName"] or author["familyName"] or author["name"])
        # Every PropertyValue is inline (eLabFTW reads propertyID directly) and also a graph node.
        inline = {value["propertyID"]: value for value in experiment["variableMeasured"]}
        self.assertIn("elabftw_metadata", inline)
        self.assertIn("scientific_evidence_sha256", inline)
        self.assertEqual(inline["scientific_evidence_sha256"]["value"], self.evidence_hash)
        for value in experiment["variableMeasured"]:
            self.assertEqual(nodes[value["@id"]], value)
        extra = json.loads(inline["elabftw_metadata"]["value"])
        self.assertIn("extra_fields", extra)
        self.assertEqual(extra["extra_fields"]["Scientific evidence SHA-256"]["value"], self.evidence_hash)
        self.assertEqual(extra["extra_fields"]["Dossier version"]["value"], str(manifest["dossier_version"]))

    def test_every_file_resolves_with_matching_size_and_sha256(self):
        _, manifest, archive, metadata = self._export()
        nodes = {node["@id"]: node for node in metadata["@graph"]}
        experiment = nodes[manifest["experiment_id"]]
        root = manifest["root_folder"]
        described = set()
        for part in experiment["hasPart"]:
            node = nodes[part["@id"]]
            self.assertEqual(node["@type"], "File")
            member = f"{root}/{part['@id'][len('./'):]}"
            content = archive.read(member)
            described.add(member)
            self.assertEqual(node["contentSize"], str(len(content)), member)
            self.assertEqual(node["sha256"], sha256(content).hexdigest(), member)
            self.assertTrue(node["encodingFormat"], member)
            self.assertTrue(node["description"], member)
        # No undocumented files: everything in the archive is either the metadata file or a described File.
        members = {name for name in archive.namelist() if not name.endswith("/")}
        self.assertEqual(members - described, {f"{root}/ro-crate-metadata.json"})
        self.assertEqual(len(described), manifest["file_count"])

    def test_json_evidence_workbook_and_signed_snapshot_are_attached(self):
        _, manifest, _, metadata = self._export()
        names = {node["name"] for node in metadata["@graph"] if node.get("@type") == "File"}
        self.assertIn("scientific_evidence.json", names)
        self.assertIn("qualification_dossier.html", names)
        self.assertIn("SHA256SUMS.txt", names)
        self.assertTrue(any(name.endswith(".xlsx") for name in names))
        snapshot_paths = [entry["path"] for entry in manifest["files"] if "signed_evidence_snapshots/" in entry["path"]]
        self.assertEqual(len(snapshot_paths), 1)
        # The canonical evidence file hashes to the evidence hash carried in the metadata,
        # so a plain sha256sum on the extracted archive reproduces it.
        self.assertIn("scientific_evidence.canonical.json", names)
        canonical_node = next(
            node for node in metadata["@graph"] if node.get("name") == "scientific_evidence.canonical.json"
        )
        self.assertEqual(canonical_node["sha256"], manifest["scientific_evidence_sha256"])
        self.assertEqual(canonical_node["sha256"], self.evidence_hash)

    def test_workbook_can_be_left_out(self):
        _, manifest, _, metadata = self._export(include_workbook=False)
        names = {node["name"] for node in metadata["@graph"] if node.get("@type") == "File"}
        self.assertFalse(any(name.endswith(".xlsx") for name in names))
        self.assertFalse(manifest["workbook_included"])

    def test_export_records_a_dossier_version_and_is_audited(self):
        _, manifest, _, _ = self._export()
        dossiers = self.store.list_dossiers(self.project_id)
        self.assertIn(manifest["dossier_version"], set(int(value) for value in dossiers["version"]))
        audit = self.store.audit_log(self.project_id)
        self.assertIn("eln_exported", set(audit["event_type"]))


if __name__ == "__main__":
    unittest.main()
