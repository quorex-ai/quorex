from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from quorex.api import schemas
from quorex.api.deps import AuthContext, get_auth, get_fact_store, get_tenant_store
from quorex.api.errors import FactNotFound, FactRejected, UserNotFound
from quorex.domain import ActiveFactExists, EndReason, NewFact
from quorex.storage import PostgresFactStore, PostgresTenantStore

router = APIRouter()

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@router.post("/v1/remember", response_model=schemas.RememberResponse)
def remember(
    body: schemas.RememberStructured,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
) -> schemas.RememberResponse:
    user = tenants.get_or_create_user(auth.tenant_id, body.user)
    # TODO jalon 2 : attribte = normalize(body.attribute)
    new = NewFact(
        tenant_id=auth.tenant_id,
        user_id=user.id,  
        subject=body.subject,
        attribute=body.attribute,
        attribute_raw=body.attribute,
        value=body.value,
        confidence=body.confidence,
        durability=body.durability,
        valid_from=body.valid_from or datetime.now(UTC),
        expires_at=body.expires_at,  
    )
    try:
        created = facts.create(new)
    except ActiveFactExists as e:
        raise FactRejected(
            "un fait actif existe déjà pour cet attribut (remplacement disponible au jalon 2)",
            details={"reason": "active_fact_exists", "existing_fact_id": str(e.existing.id)}
        ) from e
    return schemas.RememberResponse(
        facts=[schemas.RememberResult(fact=schemas.FactOut.from_fact(created), action="created")]
    )

@router.post("/v1/recall", response_model=schemas.RecallResponse)
def recall(
    body: schemas.RecallRequest,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
) -> schemas.RecallResponse:
    user = tenants.get_user(auth.tenant_id, body.user)
    if user is None:
        raise UserNotFound(f"utilisateur inconnu : {body.user}")
    found = facts.list_active(auth.tenant_id, user.id, attributes=body.attributes)[: body.limit]
    return schemas.RecallResponse(facts=[schemas.FactOut.from_fact(f) for f in found])
 
 
@router.post("/v1/forget", response_model=schemas.ForgetResponse)
def forget(
    body: schemas.ForgetRequest,
    auth: AuthContext = Depends(get_auth),
    facts: PostgresFactStore = Depends(get_fact_store),
    tenants: PostgresTenantStore = Depends(get_tenant_store),
) -> schemas.ForgetResponse:
    user = tenants.get_user(auth.tenant_id, body.user)
    if user is None:
        raise UserNotFound(f"utilisateur inconnu : {body.user}")
    closed = []
    if body.fact_id is not None:
        fact = facts.get_by_id(auth.tenant_id, body.fact_id)
        if fact is None or fact.user_id != user.id or not fact.is_active:
            raise FactNotFound("fait introuvable ou déjà clos")
        closed.append(facts.close(auth.tenant_id, fact.id, EndReason.FORGOTTEN))
    else:
        for fact in facts.list_active(auth.tenant_id, user.id, attributes=[body.attribute or ""]):
            closed.append(facts.close(auth.tenant_id, fact.id, EndReason.FORGOTTEN))
    return schemas.ForgetResponse(closed=[schemas.FactOut.from_fact(f) for f in closed])
 
