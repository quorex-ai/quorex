"""Schémas d'entrée/sortie de l'API v1."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quorex.domain import Durability, Fact


class RememberStructured(BaseModel):
    user: str = Field(min_length=1, max_length=256)
    subject: str = Field(default="user", min_length=1, max_length=128)
    attribute: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1, max_length=4096)
    durability: Durability = Durability.DURABLE
    valid_from: datetime | None = None
    expires_at: datetime | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def _transient_needs_expiry(self) -> "RememberStructured":
        if self.durability is Durability.TRANSIENT and self.expires_at is None:
            raise ValueError("un fait transient exige expires_at")
        return self


class FactOut(BaseModel):
    id: UUID
    subject: str
    attribute: str
    attribute_raw: str
    value: str
    confidence: float
    durability: Durability
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    expires_at: datetime | None
    source_message_id: UUID | None
    replaced_by: UUID | None
    end_reason: str | None

    @classmethod
    def from_fact(cls, f: Fact) -> "FactOut":
        return cls(
            id=f.id, subject=f.subject, attribute=f.attribute, attribute_raw=f.attribute_raw,
            value=f.value, confidence=f.confidence, durability=f.durability,
            valid_from=f.valid_from, valid_to=f.valid_to, recorded_at=f.recorded_at,
            expires_at=f.expires_at, source_message_id=f.source_message_id,
            replaced_by=f.replaced_by, end_reason=f.end_reason.value if f.end_reason else None,
        )


class RememberResult(BaseModel):
    action: Literal["created", "replaced", "ignored"]
    fact: FactOut | None = None
    replaced_fact_id: UUID | None = None
    level: str | None = None


class RememberResponse(BaseModel):
    facts: list[RememberResult]
    warnings: list[str] = Field(default_factory=list)


class RecallRequest(BaseModel):
    user: str = Field(min_length=1, max_length=256)
    attributes: list[str] | None = None
    as_of: datetime | None = None
    limit: int = Field(default=50, ge=1, le=500)


class RecallResponse(BaseModel):
    facts: list[FactOut]
    as_of: datetime | None = None


class DiffRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user: str = Field(min_length=1, max_length=256)
    from_: datetime = Field(alias="from")
    to: datetime


class ReplacedPair(BaseModel):
    old: FactOut
    new: FactOut


class DiffResponse(BaseModel):
    added: list[FactOut]
    replaced: list[ReplacedPair]
    invalidated: list[FactOut]


class ForgetRequest(BaseModel):
    user: str = Field(min_length=1, max_length=256)
    fact_id: UUID | None = None
    attribute: str | None = None

    @model_validator(mode="after")
    def _one_target(self) -> "ForgetRequest":
        if self.fact_id is None and self.attribute is None:
            raise ValueError("fact_id ou attribute requis")
        return self


class ForgetResponse(BaseModel):
    closed: list[FactOut]