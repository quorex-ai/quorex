"""Conversion lignes SQL -> dataclasses du domaine, et l'inverse pour les vecteurs."""
from __future__ import annotations

from typing import Any

from quorex.domain.models import Durability, EndReason, Fact
from quorex.domain.tenant import ApiKey, Tenant, User

FACT_COLUMNS = """
    id, tenant_id, user_id, subject, attribute, attribute_raw, value, confidence,
    durability, expires_at, source_message_id, valid_from, valid_to, recorded_at,
    replaced_by, end_reason
"""


def to_fact(row: Any) -> Fact:
    m = row._mapping
    return Fact(
        id=m["id"],
        tenant_id=m["tenant_id"],
        user_id=m["user_id"],
        subject=m["subject"],
        attribute=m["attribute"],
        attribute_raw=m["attribute_raw"],
        value=m["value"],
        confidence=float(m["confidence"]),
        durability=Durability(m["durability"]),
        expires_at=m["expires_at"],
        source_message_id=m["source_message_id"],
        valid_from=m["valid_from"],
        valid_to=m["valid_to"],
        recorded_at=m["recorded_at"],
        replaced_by=m["replaced_by"],
        end_reason=EndReason(m["end_reason"]) if m["end_reason"] else None,
    )


def to_tenant(row: Any) -> Tenant:
    m = row._mapping
    return Tenant(m["id"], m["name"], m["embedding_model"], m["embedding_dim"], m["created_at"])


def to_api_key(row: Any) -> ApiKey:
    m = row._mapping
    return ApiKey(
        m["id"], m["tenant_id"], m["key_prefix"], m["key_hash"], m["label"],
        m["created_at"], m["revoked_at"],
    )


def to_user(row: Any) -> User:
    m = row._mapping
    return User(m["id"], m["tenant_id"], m["external_id"], m["created_at"])


def vector_literal(v: list[float] | None) -> str | None:
    """pgvector accepte le format texte '[0.1,0.2,...]' casté en ::vector."""
    if v is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in v) + "]"