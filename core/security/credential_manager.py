from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Protocol

from cryptography.fernet import Fernet

from monitoring.telemetry import audit_log


class SecretsBackend(Protocol):
    """Simple protocol for external secret backends."""

    def get_secret(self, name: str) -> Optional[str]:
        """Return secret value or ``None`` if unavailable."""
        ...


class CredentialManager:
    """Load secrets from environment, backend, or an encrypted file."""

    def __init__(
        self,
        secrets_file: Optional[str] = None,
        *,
        encryption_key: Optional[str] = None,
        backend: Optional[SecretsBackend] = None,
    ) -> None:
        secrets_file = secrets_file or os.getenv("SECRETS_FILE")
        encryption_key = encryption_key or os.getenv("SECRETS_KEY")

        self.backend = backend
        self.secrets_file = Path(secrets_file) if secrets_file else None
        self.encryption_key = encryption_key
        self._cache: Dict[str, str] = {}

        if self.secrets_file and self.secrets_file.exists():
            self._load_file()

    def _load_file(self) -> None:
        data = self.secrets_file.read_bytes()
        if self.encryption_key:
            cipher = Fernet(self.encryption_key.encode())
            data = cipher.decrypt(data)
        secrets = json.loads(data.decode())
        self._cache.update(secrets)
        audit_log(
            "secrets_file_loaded",
            path=str(self.secrets_file),
            encrypted=bool(self.encryption_key),
        )

    def get(self, name: str) -> Optional[str]:
        """Retrieve a credential by name."""
        if name in os.environ:
            audit_log("credential_retrieved", name=name, source="env")
            return os.environ.get(name)
        if name in self._cache:
            audit_log("credential_retrieved", name=name, source="cache")
            return self._cache[name]
        if self.backend:
            try:
                value = self.backend.get_secret(name)
                if value:
                    self._cache[name] = value
                    audit_log("credential_retrieved", name=name, source="backend")
                return value
            except Exception:
                return None
        return None
