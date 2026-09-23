from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from quorex.domain.models import Diff, EndReason, Fact, NewFact


class ConcurrentWrite(Exception):
    """Un replace/create a perdu deux fois de suite face à une écriture concurrente."""


class ActiveFactExists(Exception):
    """create() a rencontré un fait actif sur (user, subject, attribute). Passer par replace()."""

    def __init__(self, existing: Fact) -> None:
        super().__init__(f"fait actif existant : {existing.subject}.{existing.attribute}")
        self.existing = existing


class StaleFact(Exception):
    """replace() : le fait à clore n'est plus actif (quelqu'un l'a clos entre-temps)."""

    def __init__(self, fact_id: UUID) -> None:
        super().__init__(f"fait {fact_id} déjà clos")
        self.fact_id = fact_id


class FactStore(Protocol):
    def get_active(self, tenant_id: UUID, user_id: UUID, subject: str, attribute: str) -> Fact | None: ...

    def get_by_id(self, tenant_id: UUID, fact_id: UUID) -> Fact | None: ...

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

    def create(self, fact: NewFact) -> Fact:
        """Crée un fait actif. Lève ActiveFactExists si l'index unique refuse."""
        ...

    def replace(self, old_id: UUID, new: NewFact) -> tuple[Fact, Fact]:
        """Atomique : clôt old_id (replaced), crée new, lie replaced_by.
        Lève StaleFact si old_id n'est plus actif, ActiveFactExists si un autre fait
        actif occupe déjà (user, subject, attribute) du nouveau."""
        ...

    def close(self, tenant_id: UUID, fact_id: UUID, reason: EndReason, at: datetime | None = None) -> Fact: ...