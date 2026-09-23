"""Modèle de fait bitemporel (ADR-002). Aucune dépendance à SQL ici."""
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
    """Ce que le moteur veut écrire.

    valid_from et recorded_at à None signifient « maintenant, horloge de la base » :
    le store les pose dans la même instruction SQL, donc identiques (jalon 3).
    recorded_at n'est jamais exposé par l'API ; il sert aux tests et aux imports.
    """

    tenant_id: UUID
    user_id: UUID
    attribute: str          # normalisé
    attribute_raw: str
    value: str
    confidence: float
    durability: Durability
    subject: str = "user"
    valid_from: datetime | None = None
    recorded_at: datetime | None = None
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