import pytest
from pydantic import ValidationError

from quorex.settings import Settings


def test_refuse_sans_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUOREX_DATABASE_URL", raising=False)
    monkeypatch.setenv("QUOREX_LLM_PROVIDER", "fake")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_refuse_seuils_incoherents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUOREX_DATABASE_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("QUOREX_LLM_PROVIDER", "fake")
    monkeypatch.setenv("QUOREX_SIM_MATCH", "0.7")
    monkeypatch.setenv("QUOREX_SIM_AMBIGUOUS", "0.8")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_gemini_exige_cle_et_modele(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUOREX_DATABASE_URL", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("QUOREX_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("QUOREX_GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]