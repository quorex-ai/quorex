"""Configuration : toutes les variables QUOREX_*, validées au démarrage.

Le service refuse de démarrer si une variable obligatoire manque
(pydantic-settings lève une ValidationError explicite).
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUOREX_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    llm_provider: Literal["gemini", "fake"]
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    embedding_model: str = "intfloat/multilingual-e5-base"
    sim_match: float = Field(0.85, ge=0, le=1)
    sim_ambiguous: float = Field(0.70, ge=0, le=1)
    min_confidence: float = Field(0.5, ge=0, le=1)
    log_level: str = "INFO"

    @field_validator("database_url")
    @classmethod
    def _must_be_postgres(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError("QUOREX_DATABASE_URL doit être une URL postgresql (ADR-001)")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.sim_ambiguous >= self.sim_match:
            raise ValueError("QUOREX_SIM_AMBIGUOUS doit être strictement inférieur à QUOREX_SIM_MATCH")
        if self.llm_provider == "gemini" and not (self.gemini_api_key and self.gemini_model):
            raise ValueError(
                "QUOREX_GEMINI_API_KEY et QUOREX_GEMINI_MODEL sont obligatoires avec llm_provider=gemini"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]