"""Interface FactStore (ADR-001, architecture.md 6.2).

Seule storage/ l'implémente. Aucun appelant ne connaît SQL.
Toute méthode reçoit implicitement le tenant via les identifiants qu'on lui passe :
une implémentation DOIT filtrer sur tenant_id dans chaque requête.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from quorex.domain.models import Diff, EndReason, Fact, NewFact


class ConcurrentWrite(Exception):
    """Levée quand un replace/create perd deux fois de suite face à une écriture concurrente."""


class FactStore(Protocol):
    def get_active(self, tenant_id: UUID, user_id: UUID, subject: str, attribute: str) -> Fact | None: ...

    def list_active(
        self, tenant_id: UUID, user_id: UUID, attributes: list[str] | None = None, now: datetime | None = None
    ) -> list[Fact]: ...

    def list_as_of(
        self, tenant_id: UUID, user_id: UUID, as_of: datetime, attributes: list[str] | None = None
    ) -> list[Fact]: ...

    def find_similar_active(
        self, tenant_id: UUID, user_id: UUID, attribute_embedding: list[float], limit: int = 5
    ) -> list[tuple[Fact, float]]: ...

    def search(
        self, tenant_id: UUID, user_id: UUID, query_embedding: list[float],
        as_of: datetime | None = None, limit: int = 20,
    ) -> list[tuple[Fact, float]]: ...

    def diff(self, tenant_id: UUID, user_id: UUID, from_: datetime, to: datetime) -> Diff: ...

    def create(self, fact: NewFact) -> Fact: ...

    def replace(self, old_id: UUID, new: NewFact, reason: EndReason = EndReason.REPLACED) -> tuple[Fact, Fact]:
        """Atomique : clôt old_id, crée new, lie replaced_by. Retourne (ancien clos, nouveau)."""
        ...

    def close(self, tenant_id: UUID, fact_id: UUID, reason: EndReason, at: datetime | None = None) -> Fact: ...