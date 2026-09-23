from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from quorex.domain.models import Diff, EndReason, Fact, NewFact
from quorex.domain.store import ActiveFactExists, StaleFact
from quorex.ids import new_id
from quorex.storage.db import Database
from quorex.storage.rows import FACT_COLUMNS, to_fact, vector_literal

_INSERT = text(f"""
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


def _insert_params(fact: NewFact, fact_id: UUID) -> dict:
    return {
        "id": fact_id,
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
        try:
            with self._db.transaction() as conn:
                row = conn.execute(_INSERT, _insert_params(fact, new_id())).one()
                return to_fact(row)
        except IntegrityError as e:
            self._raise_if_active_conflict(e, fact)
            raise

    def replace(self, old_id: UUID, new: NewFact) -> tuple[Fact, Fact]:
        """Une transaction : clore l'ancien, insérer le nouveau, lier replaced_by.

        L'UPDATE de clôture porte 'AND valid_to IS NULL' : si une autre écriture a déjà
        clos l'ancien, zéro ligne -> StaleFact, et rien n'est écrit (rollback).
        L'INSERT peut violer facts_active_unique si un autre fait actif est apparu
        entre-temps -> ActiveFactExists, rollback aussi. L'appelant (contradiction/)
        relit et rejoue une fois.
        """
        new_id_ = new_id()
        try:
            with self._db.transaction() as conn:
                old = self._close_in(conn, new.tenant_id, old_id, EndReason.REPLACED, at=new.valid_from)
                if old is None:
                    raise StaleFact(old_id)
                created = to_fact(conn.execute(_INSERT, _insert_params(new, new_id_)).one())
                conn.execute(
                    text("UPDATE facts SET replaced_by = :new_id WHERE id = :old_id"),
                    {"new_id": new_id_, "old_id": old_id},
                )
                old_linked = to_fact(conn.execute(
                    text(f"SELECT {FACT_COLUMNS} FROM facts WHERE id = :id"), {"id": old_id}
                ).one())
                return old_linked, created
        except IntegrityError as e:
            self._raise_if_active_conflict(e, new)
            raise

    def close(self, tenant_id: UUID, fact_id: UUID, reason: EndReason, at: datetime | None = None) -> Fact:
        """Clôt un fait actif. Jamais de DELETE (ADR-002)."""
        with self._db.transaction() as conn:
            fact = self._close_in(conn, tenant_id, fact_id, reason, at)
        if fact is None:
            raise StaleFact(fact_id)
        return fact

    # ---------- internes ----------

    @staticmethod
    def _close_in(conn: Connection, tenant_id: UUID, fact_id: UUID, reason: EndReason,
                  at: datetime | None) -> Fact | None:
        row = conn.execute(
            text(f"""
                UPDATE facts
                   SET valid_to = GREATEST(:at, valid_from), end_reason = :reason
                 WHERE tenant_id = :tenant_id AND id = :id AND valid_to IS NULL
                RETURNING {FACT_COLUMNS}
            """),
            {"at": at or datetime.now(UTC), "reason": reason.value, "tenant_id": tenant_id, "id": fact_id},
        ).first()
        return to_fact(row) if row else None

    def _raise_if_active_conflict(self, e: IntegrityError, fact: NewFact) -> None:
        if "facts_active_unique" in str(e.orig):
            existing = self.get_active(fact.tenant_id, fact.user_id, fact.subject, fact.attribute)
            if existing is not None:
                raise ActiveFactExists(existing) from e