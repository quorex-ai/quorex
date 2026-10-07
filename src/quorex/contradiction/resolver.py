from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import structlog

from quorex.domain import (
    ActiveFactExists,
    ConcurrentWrite,
    Durability,
    Fact,
    FactStore,
    NewFact,
    StaleFact
)

log = structlog.get_logger()

Action = Literal["created", "replaced", "ignored"]
_MAX_ATTEMPTS = 2

@dataclass(frozen=True, slots=True)
class Outcome:
    action: Action
    fact: Fact | None # Non si ingored
    replaced: Fact | None = None # l'ancien fait clos, si replaced
    level: str | None = None # "a" | "b" | "c" | None
    warning: str | None = None

def _transient_over_durable(new: NewFact, existing: Fact) -> bool:
    return new.durability is Durability.TRANSIENT and existing.durability is not Durability.TRANSIENT

class Resolver:
    def __init__(self, store: FactStore):
        self._store = store

    def remember(self, new: NewFact) -> Outcome:
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            existing = self._store.get_active(new.tenant_id, new.user_id, new.subject, new.attribute)
            try:
                if existing is None:
                    created = self._store.create(new)
                    log.info("contradiction.resolver", level=None, action="created", attempt=attempt)
                    return Outcome(action="created", fact=created)

                # Niveau (a) : même attrivut normalisé
                if _transient_over_durable(new, existing):
                    log.info("contradiction.resolve", level="a", action="ignored", attempt=attempt)
                    return Outcome(
                        action="ignored", fact=None, level="a",
                        warning="transient_ignored_durable_exists"
                    )
                old, created = self._store.replace(existing.id, new)
                log.info("contradiction.resolve", level="a", action="replaced", attempt=attempt)
                return Outcome(action="replaced", fact=created, replaced=old, level="a")
            
            except (ActiveFactExists, StaleFact) as e:
                # Quelqu'un a écrit entre notre lecture et notre écriture : on relit et on rejoue.
                last_error = e
                log.info("contradiction.retry", reason=type(e).__name__, attempt=attempt)
                continue
        raise ConcurrentWrite(str(last_error))