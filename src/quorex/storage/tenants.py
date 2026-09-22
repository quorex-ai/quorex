"""Tenants, clés API, utilisateurs (ADR-007)."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import text

from quorex.domain.tenant import ApiKey, Tenant, User
from quorex.ids import new_id
from quorex.storage.db import Database
from quorex.storage.rows import to_api_key, to_tenant, to_user


class PostgresTenantStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_tenant(self, name: str, embedding_model: str, embedding_dim: int) -> Tenant:
        sql = text("""
            INSERT INTO tenants (id, name, embedding_model, embedding_dim)
            VALUES (:id, :name, :model, :dim)
            RETURNING id, name, embedding_model, embedding_dim, created_at
        """)
        with self._db.transaction() as conn:
            row = conn.execute(
                sql, {"id": new_id(), "name": name, "model": embedding_model, "dim": embedding_dim}
            ).one()
        return to_tenant(row)

    def get_tenant(self, tenant_id: UUID) -> Tenant | None:
        sql = text(
            "SELECT id, name, embedding_model, embedding_dim, created_at FROM tenants WHERE id = :id"
        )
        with self._db.transaction() as conn:
            row = conn.execute(sql, {"id": tenant_id}).first()
        return to_tenant(row) if row else None

    def create_api_key(
        self, tenant_id: UUID, key_prefix: str, key_hash: str, label: str | None
    ) -> ApiKey:
        sql = text("""
            INSERT INTO api_keys (id, tenant_id, key_prefix, key_hash, label)
            VALUES (:id, :tenant_id, :prefix, :hash, :label)
            RETURNING id, tenant_id, key_prefix, key_hash, label, created_at, revoked_at
        """)
        with self._db.transaction() as conn:
            row = conn.execute(
                sql,
                {"id": new_id(), "tenant_id": tenant_id, "prefix": key_prefix,
                 "hash": key_hash, "label": label},
            ).one()
        return to_api_key(row)

    def find_active_keys_by_prefix(self, key_prefix: str) -> list[ApiKey]:
        sql = text("""
            SELECT id, tenant_id, key_prefix, key_hash, label, created_at, revoked_at
            FROM api_keys WHERE key_prefix = :prefix AND revoked_at IS NULL
        """)
        with self._db.transaction() as conn:
            rows = conn.execute(sql, {"prefix": key_prefix}).all()
        return [to_api_key(r) for r in rows]

    def revoke_api_key(self, key_id: UUID) -> None:
        sql = text("UPDATE api_keys SET revoked_at = now() WHERE id = :id AND revoked_at IS NULL")
        with self._db.transaction() as conn:
            conn.execute(sql, {"id": key_id})

    def get_user(self, tenant_id: UUID, external_id: str) -> User | None:
        sql = text("""
            SELECT id, tenant_id, external_id, created_at FROM users
            WHERE tenant_id = :tenant_id AND external_id = :external_id
        """)
        with self._db.transaction() as conn:
            row = conn.execute(sql, {"tenant_id": tenant_id, "external_id": external_id}).first()
        return to_user(row) if row else None

    def get_or_create_user(self, tenant_id: UUID, external_id: str) -> User:
        sql = text("""
            INSERT INTO users (id, tenant_id, external_id)
            VALUES (:id, :tenant_id, :external_id)
            ON CONFLICT (tenant_id, external_id) DO UPDATE SET external_id = EXCLUDED.external_id
            RETURNING id, tenant_id, external_id, created_at
        """)
        with self._db.transaction() as conn:
            row = conn.execute(
                sql, {"id": new_id(), "tenant_id": tenant_id, "external_id": external_id}
            ).one()
        return to_user(row)