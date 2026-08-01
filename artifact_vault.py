"""Encrypted artifact storage for Reformulation Assurance v0.6."""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

from cryptography.fernet import Fernet, InvalidToken

from pilot_store import PilotStore


class ArtifactVault:
    def __init__(self, root: str | Path, *, key: str | bytes | None = None, key_file: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key_file = Path(key_file) if key_file else self.root / ".artifact_key"
        self._fernet = Fernet(self._resolve_key(key))

    def _resolve_key(self, provided: str | bytes | None) -> bytes:
        if provided is None:
            provided = os.environ.get("REFORMULATION_ARTIFACT_KEY")
        if provided:
            raw = provided.encode("ascii") if isinstance(provided, str) else provided
            Fernet(raw)  # validate
            return raw
        if self.key_file.exists():
            raw = self.key_file.read_bytes().strip()
            Fernet(raw)
            return raw
        raw = Fernet.generate_key()
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.key_file.write_bytes(raw)
        try:
            os.chmod(self.key_file, 0o600)
        except OSError:
            pass
        return raw

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, encrypted: bytes) -> bytes:
        try:
            return self._fernet.decrypt(encrypted)
        except InvalidToken as exc:
            raise ValueError("artifact could not be decrypted with the configured key") from exc

    def store_project_artifact(
        self,
        store: PilotStore,
        project_id: str,
        *,
        created_by_user_id: str,
        payload: bytes,
        filename: str,
        artifact_type: str,
        content_type: str = "application/octet-stream",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        encrypted = self.encrypt(payload)
        artifact_dir = self.root / "projects" / project_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        storage_path = artifact_dir / f"{uuid.uuid4()}.fernet"
        storage_path.write_bytes(encrypted)
        return store.save_artifact_record(
            project_id,
            created_by_user_id=created_by_user_id,
            artifact_type=artifact_type,
            filename=filename,
            content_type=content_type,
            storage_path=str(storage_path),
            plaintext_sha256=self.sha256(payload),
            ciphertext_sha256=self.sha256(encrypted),
            size_bytes=len(payload),
            encryption_method="fernet-aes128-cbc-hmac-sha256",
            metadata=metadata,
        )

    def retrieve_project_artifact(self, store: PilotStore, artifact_id: str, actor_user_id: str) -> tuple[bytes, dict[str, Any]]:
        record = store.get_artifact(artifact_id, actor_user_id)
        encrypted = Path(record["storage_path"]).read_bytes()
        if self.sha256(encrypted) != record["ciphertext_sha256"]:
            raise ValueError("encrypted artifact checksum does not match its metadata")
        payload = self.decrypt(encrypted)
        if self.sha256(payload) != record["plaintext_sha256"]:
            raise ValueError("decrypted artifact checksum does not match its metadata")
        return payload, record

    def store_backup_payload(self, payload: bytes, filename: str) -> dict[str, Any]:
        encrypted = self.encrypt(payload)
        backup_dir = self.root / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        storage_path = backup_dir / f"{uuid.uuid4()}.fernet"
        storage_path.write_bytes(encrypted)
        return {
            "filename": filename,
            "storage_path": str(storage_path),
            "plaintext_sha256": self.sha256(payload),
            "ciphertext_sha256": self.sha256(encrypted),
            "size_bytes": len(payload),
            "encryption_method": "fernet-aes128-cbc-hmac-sha256",
        }

    def retrieve_backup_payload(self, record: Mapping[str, Any]) -> bytes:
        encrypted = Path(str(record["storage_path"])).read_bytes()
        if self.sha256(encrypted) != record["ciphertext_sha256"]:
            raise ValueError("backup ciphertext checksum mismatch")
        payload = self.decrypt(encrypted)
        if self.sha256(payload) != record["plaintext_sha256"]:
            raise ValueError("backup plaintext checksum mismatch")
        return payload
