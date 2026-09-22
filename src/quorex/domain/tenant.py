from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

@dataclass(frozen=True, slots=True)
class Tenant:
    id: UUID
    name: str
    embedding_model: str
    embedding_dim: int
    created_at: datetime

@dataclass(frozen=True, slots=True)
class ApiKey:
    id: UUID
    tenant_id: UUID
    key_prefix: str
    key_hash: str
    label: str | None
    created_at: datetime
    revoked_at: datetime | None

    @property 
    def is_active(self) -> bool:
        return self.revoked_at is None

@dataclass(frozen=True, slots=True)
class User:
    id: UUID
    tenant_id: UUID
    external_id: str
    created_at: datetime

class TenantStore(Protocol):
    def create_tenant(self, name: str, embedding_model: str, embedding_dim: int) -> Tenant: ...
    def get_tenant(self, tenant_id: UUID) -> Tenant | None: ...
    def create_api_key(self, tenant_id: UUID, key_prefix: str, key_hash: str, label: str | None) -> ApiKey: ...
    def find_active_keys_by_prefix(self, key_prefix: str) -> list[ApiKey]: ...
    def revoke_api_key(self, key_id: UUID) -> None: ...
    def get_or_create_user(self, tenant_id: UUID, external_id: str) -> User: ...
    def get_user(self, tenant_id: UUID, external_id: str) -> User | None: ...
