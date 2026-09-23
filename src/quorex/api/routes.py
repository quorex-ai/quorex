"""Routes v1. Jalon 3 : recall avec as_of, diff, forget."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from quorex.api import schemas
from quorex.api.deps import (
    AuthContext, get_auth, get_fact_store, get_normalizer, get_resolver, get_tenant_store,
)
from quorex.api.errors import ConcurrentWriteError, FactNotFound, InvalidTimeRange, UserNotFound
from quorex.contradiction import Resolver
from quorex.domain import ConcurrentWrite, EndReason, NewFact, StaleFact
from quorex.normalization import Normalizer
from quorex.storage import PostgresFactStore, PostgresTenantStore

router = APIRouter()


def _ensure_tz(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/remember", response_model=schemas.RememberResponse)
def remember(
    body: schemas.RememberStructured,
    auth: AuthContext = Depends(get_auth),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
    normalizer: Normalizer = Depends(get_normalizer),
    resolver: Resolver = Depends(get_resolver),
) -> schemas.RememberResponse:
    user = tenants.get_or_create_user(auth.tenant_id, body.user)
    new = NewFact(
        tenant_id=auth.tenant_id,
        user_id=user.id,
        subject=normalizer.normalize(body.subject),
        attribute=normalizer.normalize(body.attribute),
        attribute_raw=body.attribute,
        value=body.value,
        confidence=body.confidence,
        durability=body.durability,
        valid_from=_ensure_tz(body.valid_from) if body.valid_from else None,
        expires_at=_ensure_tz(body.expires_at) if body.expires_at else None,
    )
    try:
        outcome = resolver.remember(new)
    except ConcurrentWrite as e:
        raise ConcurrentWriteError("écriture concurrente non résolue, rejouer la requête") from e

    result = schemas.RememberResult(
        action=outcome.action,
        fact=schemas.FactOut.from_fact(outcome.fact) if outcome.fact else None,
        replaced_fact_id=outcome.replaced.id if outcome.replaced else None,
        level=outcome.level,
    )
    warnings = [outcome.warning] if outcome.warning else []
    return schemas.RememberResponse(facts=[result], warnings=warnings)


@router.post("/v1/recall", response_model=schemas.RecallResponse)
def recall(
    body: schemas.RecallRequest,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
    normalizer: Normalizer = Depends(get_normalizer),
) -> schemas.RecallResponse:
    user = tenants.get_user(auth.tenant_id, body.user)
    if user is None:
        raise UserNotFound(f"utilisateur inconnu : {body.user}")
    attributes = [normalizer.normalize(a) for a in body.attributes] if body.attributes else None
    if body.as_of is not None:
        as_of = _ensure_tz(body.as_of)
        if as_of > datetime.now(UTC):
            raise InvalidTimeRange("as_of ne peut pas être dans le futur")
        found = facts.list_as_of(auth.tenant_id, user.id, as_of, attributes=attributes)
    else:
        as_of = None
        found = facts.list_active(auth.tenant_id, user.id, attributes=attributes)
    return schemas.RecallResponse(
        facts=[schemas.FactOut.from_fact(f) for f in found[: body.limit]], as_of=as_of
    )


@router.post("/v1/diff", response_model=schemas.DiffResponse)
def diff(
    body: schemas.DiffRequest,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
) -> schemas.DiffResponse:
    user = tenants.get_user(auth.tenant_id, body.user)
    if user is None:
        raise UserNotFound(f"utilisateur inconnu : {body.user}")
    from_, to = _ensure_tz(body.from_), _ensure_tz(body.to)
    if from_ >= to:
        raise InvalidTimeRange("from doit être strictement antérieur à to")
    d = facts.diff(auth.tenant_id, user.id, from_, to)
    return schemas.DiffResponse(
        added=[schemas.FactOut.from_fact(f) for f in d.added],
        replaced=[
            schemas.ReplacedPair(old=schemas.FactOut.from_fact(o), new=schemas.FactOut.from_fact(n))
            for o, n in d.replaced
        ],
        invalidated=[schemas.FactOut.from_fact(f) for f in d.invalidated],
    )


@router.post("/v1/forget", response_model=schemas.ForgetResponse)
def forget(
    body: schemas.ForgetRequest,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
    normalizer: Normalizer = Depends(get_normalizer),
) -> schemas.ForgetResponse:
    user = tenants.get_user(auth.tenant_id, body.user)
    if user is None:
        raise UserNotFound(f"utilisateur inconnu : {body.user}")
    closed = []
    if body.fact_id is not None:
        fact = facts.get_by_id(auth.tenant_id, body.fact_id)
        if fact is None or fact.user_id != user.id or not fact.is_active:
            raise FactNotFound("fait introuvable ou déjà clos")
        try:
            closed.append(facts.close(auth.tenant_id, fact.id, EndReason.FORGOTTEN))
        except StaleFact as e:
            raise FactNotFound("fait déjà clos") from e
    else:
        attribute = normalizer.normalize(body.attribute or "")
        for fact in facts.list_active(auth.tenant_id, user.id, attributes=[attribute]):
            try:
                closed.append(facts.close(auth.tenant_id, fact.id, EndReason.FORGOTTEN))
            except StaleFact:
                continue
    return schemas.ForgetResponse(closed=[schemas.FactOut.from_fact(f) for f in closed])