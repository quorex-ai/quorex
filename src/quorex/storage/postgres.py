from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from quorex.domain.models import Diff, EndReason, Fact, NewFact
from quorex.domain.store import ActiveFactExists
from quorex.ids import new_id
from quorex.storage.db import Database
from quorex.storage.rows import FACT_COLUMNS, to_fact, vector_literal


class PostgresFactStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---------- lecture ----------

    def get_active(self, tenant_id: UUID, user_id: UUID, subject: str, attribute: str) -> Fact | None:
        sql = text(f"""
            SELECT {FACT_COLUMNS} FROM facts
            WHERE tenant_id = :tenant_id AND user_id = :user_id
              AND subject = :subject AND attribute = :attribute
              AND valid_to IS NULL
        """)
        with self._db.transaction() as conn:
            row = conn.execute(
                sql, {"tenant_id": tenant_id, "user_id": user_id, "subject": subject, "attribute": attribute}
            ).first()
        return to_fact(row) if row else None

    def get_by_id(self, tenant_id: UUID, fact_id: UUID) -> Fact | None:
        sql = text(f"SELECT {FACT_COLUMNS} FROM facts WHERE tenant_id = :tenant_id AND id = :id")
        with self._db.transaction() as conn:
            row = conn.execute(sql, {"tenant_id": tenant_id, "id": fact_id}).first()
        return to_fact(row) if row else None

    def list_active(
        self, tenant_id: UUID, user_id: UUID, attributes: list[str] | None = None, now: datetime | None = None
    ) -> list[Fact]:
        clauses = ["tenant_id = :tenant_id", "user_id = :user_id", "valid_to IS NULL",
                   "(expires_at IS NULL OR expires_at > :now)"]
        params: dict = {"tenant_id": tenant_id, "user_id": user_id, "now": now or datetime.now(UTC)}
        if attributes:
            clauses.append("attribute = ANY(:attributes)")
            params["attributes"] = attributes
        sql = text(f"SELECT {FACT_COLUMNS} FROM facts WHERE {' AND '.join(clauses)} ORDER BY recorded_at DESC")
        with self._db.transaction() as conn:
            rows = conn.execute(sql, params).all()
        return [to_fact(r) for r in rows]

    def list_as_of(self, tenant_id: UUID, user_id: UUID, as_of: datetime, attributes: list[str] | None = None) -> list[Fact]:
        raise NotImplementedError("jalon 3")

    def find_similar_active(self, tenant_id: UUID, user_id: UUID, attribute_embedding: list[float], limit: int = 5) -> list[tuple[Fact, float]]:
        raise NotImplementedError("jalon 5")

    def search(self, tenant_id: UUID, user_id: UUID, query_embedding: list[float], as_of: datetime | None = None, limit: int = 20) -> list[tuple[Fact, float]]:
        raise NotImplementedError("jalon 5")

    def diff(self, tenant_id: UUID, user_id: UUID, from_: datetime, to: datetime) -> Diff:
        raise NotImplementedError("jalon 3")

    # ---------- écriture ----------

    def create(self, fact: NewFact) -> Fact:
        sql = text(f"""
            INSERT INTO facts (
                id, tenant_id, user_id, subject, attribute, attribute_raw, value, confidence,
                durability, expires_at, source_message_id, valid_from,
                attribute_embedding, value_embedding
            ) VALUES (
                :id, :tenant_id, :user_id, :subject, :attribute, :attribute_raw, :value, :confidence,
                :durability, :expires_at, :source_message_id, :valid_from,
                CAST(:attribute_embedding AS vector), CAST(:value_embedding AS vector)
            )
            RETURNING {FACT_COLUMNS}
        """)
        params = {
            "id": new_id(),
            "tenant_id": fact.tenant_id,
            "user_id": fact.user_id,
            "subject": fact.subject,
            "attribute": fact.attribute,
            "attribute_raw": fact.attribute_raw,
            "value": fact.value,
            "confidence": fact.confidence,
            "durability": fact.durability.value,
            "expires_at": fact.expires_at,
            "source_message_id": fact.source_message_id,
            "valid_from": fact.valid_from,
            "attribute_embedding": vector_literal(fact.attribute_embedding),
            "value_embedding": vector_literal(fact.value_embedding),
        }
        try:
            with self._db.transaction() as conn:
                row = conn.execute(sql, params).one()
                return to_fact(row)
        except IntegrityError as e:
            if "facts_active_unique" in str(e.orig):
                existing = self.get_active(fact.tenant_id, fact.user_id, fact.subject, fact.attribute)
                if existing is not None:
                    raise ActiveFactExists(existing) from e
            raise

    def replace(self, old_id: UUID, new: NewFact, reason: EndReason = EndReason.REPLACED) -> tuple[Fact, Fact]:
        raise NotImplementedError("jalon 2")

    def close(self, tenant_id: UUID, fact_id: UUID, reason: EndReason, at: datetime | None = None) -> Fact:
        """Clôt un fait actif. Jamais de DELETE (ADR-002)."""
        sql = text(f"""
            UPDATE facts
               SET valid_to = :at, end_reason = :reason
             WHERE tenant_id = :tenant_id AND id = :id AND valid_to IS NULL
            RETURNING {FACT_COLUMNS}
        """)
        with self._db.transaction() as conn:
            row = conn.execute(
                sql, {"at": at or datetime.now(UTC), "reason": reason.value, "tenant_id": tenant_id, "id": fact_id}
            ).first()
        if row is None:
            raise LookupError(f"fait {fact_id} introuvable ou déjà clos")
        return to_fact(row)