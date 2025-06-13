import json

import pytest

from cryptography.fernet import Fernet

from core.security.credential_manager import CredentialManager
from monitoring.telemetry import configure_logging


@pytest.fixture
def temp_secrets_file(tmp_path):
    key = Fernet.generate_key()
    data = {"TEST_SECRET": "value"}
    cipher = Fernet(key)
    encrypted = cipher.encrypt(json.dumps(data).encode())
    file_path = tmp_path / "secrets.json"
    file_path.write_bytes(encrypted)
    return file_path, key.decode()


def test_load_encrypted_secrets(temp_secrets_file, monkeypatch):
    file_path, key = temp_secrets_file
    monkeypatch.setenv("SECRETS_FILE", str(file_path))
    monkeypatch.setenv("SECRETS_KEY", key)
    manager = CredentialManager()
    assert manager.get("TEST_SECRET") == "value"


def test_audit_log_capture(temp_secrets_file, monkeypatch, caplog):
    file_path, key = temp_secrets_file
    monkeypatch.setenv("SECRETS_FILE", str(file_path))
    monkeypatch.setenv("SECRETS_KEY", key)
    configure_logging(fmt="json")
    caplog.set_level("INFO")
    manager = CredentialManager()
    manager.get("TEST_SECRET")
    events = [record.message for record in caplog.records]
    assert any("secrets_file_loaded" in e for e in events)
    assert any("credential_retrieved" in e for e in events)
