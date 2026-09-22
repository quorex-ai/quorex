from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Depends, Header, Request

from quorex.api.errors import InvalidApiKey
from quorex.auth import extract_prefix, verify_api_key
from quorex.storage import PostgresFactStore, PostgresTenantStore

def get_fact_store(request: Request) -> PostgresFactStore:
    return request.app.state.facts

def get_tenant_store(request: Request) -> PostgresTenantStore:
    return request.app.state.tenants

@dataclass(frozen=True, slots=True)
class AuthContext:
    tenant_id: UUID
    api_key_id: UUID

def get_auth(authorization: str | None = Header(default=None), tenants: PostgresTenantStore = Depends(get_tenant_store)) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise InvalidApiKey("clé API absente")

    plain = authorization.removeprefix("Bearer ").strip()
    prefix = extract_prefix(plain)
    if prefix is None:
        raise InvalidApiKey("clé API mal formée")
    for key in tenants.find_active_keys_by_prefix(prefix):
        if verify_api_key(plain, key.key_hash):
            return AuthContext(tenant_id=key.tenant_id, api_key_id=key.id)
    raise InvalidApiKey("clé API inconnue ou révoquée")
    
