import pytest

from poultry_disease import weights
from poultry_disease.constants import WEIGHTS_SHA256
from tests.conftest import ROOT


def test_checksum_constant_matches_committed_file():
    lines = (ROOT / "models" / "checksums.sha256").read_text("utf-8").splitlines()
    assert f"{WEIGHTS_SHA256}  poultry_resnet50.keras" in lines


def test_verify_rejects_wrong_file(tmp_path):
    f = tmp_path / "poultry_resnet50.keras"
    f.write_bytes(b"not the model")
    with pytest.raises(weights.ChecksumError):
        weights.verify(f)
    weights.verify(f, weights.sha256_file(f))


def test_env_override_is_verified(tmp_path, monkeypatch):
    f = tmp_path / "fake.keras"
    f.write_bytes(b"x")
    monkeypatch.setenv("POULTRY_DISEASE_WEIGHTS", str(f))
    with pytest.raises(weights.ChecksumError):
        weights.get_weights()


def test_no_download_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("POULTRY_DISEASE_WEIGHTS", raising=False)
    monkeypatch.setenv("POULTRY_DISEASE_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        weights.get_weights(allow_download=False)
