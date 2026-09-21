from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class Durability(StrEnum):
    PERMANENT = "permanent"
    DURABLE = "durable"
    TRANSIENT = "transient"


class EndReason(StrEnum):
    REPLACED = "replaced"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"
    FORGOTTEN = "forgotten"


@dataclass(frozen=True, slots=True)
class NewFact:
    """Ce que le moteur veut écrire. Les identifiants et recorded_at sont posés par le store."""

    tenant_id: UUID
    user_id: UUID
    attribute: str          # normalisé
    attribute_raw: str
    value: str
    confidence: float
    durability: Durability
    valid_from: datetime
    subject: str = "user"
    expires_at: datetime | None = None
    source_message_id: UUID | None = None
    attribute_embedding: list[float] | None = None
    value_embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence doit être entre 0 et 1")
        if self.durability is Durability.TRANSIENT and self.expires_at is None:
            raise ValueError("un fait transient doit avoir expires_at")


@dataclass(frozen=True, slots=True)
class Fact:
    """Un fait tel qu'il est en base."""

    id: UUID
    tenant_id: UUID
    user_id: UUID
    subject: str
    attribute: str
    attribute_raw: str
    value: str
    confidence: float
    durability: Durability
    expires_at: datetime | None
    source_message_id: UUID | None
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    replaced_by: UUID | None
    end_reason: EndReason | None

    @property
    def is_active(self) -> bool:
        return self.valid_to is None


@dataclass(frozen=True, slots=True)
class Diff:
    added: list[Fact] = field(default_factory=list)
    replaced: list[tuple[Fact, Fact]] = field(default_factory=list)   # (ancien, nouveau)
    invalidated: list[Fact] = field(default_factory=list)