"""PostgresFactStore : la seule implémentation de FactStore en v1 (ADR-001).

Jalon 1 : create, get_active, get_by_id, list_active, close.
Jalon 2 : replace (atomique).
Jalon 3 : list_as_of, diff, horloge unique (recorded_at et valid_from posés par la base).
Jalon 5 : find_similar_active, search.
"""
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

# COALESCE(:x, now()) : quand l'appelant ne fixe pas la date, la base la pose,
# et recorded_at / valid_from sortent de la même horloge, dans la même instruction.
_INSERT = text(f"""
    INSERT INTO facts (
        id, tenant_id, user_id, subject, attribute, attribute_raw, value, confidence,
        durability, expires_at, source_message_id, valid_from, recorded_at,
        attribute_embedding, value_embedding
    ) VALUES (
        :id, :tenant_id, :user_id, :subject, :attribute, :attribute_raw, :value, :confidence,
        :durability, :expires_at, :source_message_id,
        COALESCE(:valid_from, now()), COALESCE(:recorded_at, now()),
        CAST(:attribute_embedding AS vector), CAST(:value_embedding AS vector)
    )
    RETURNING {FACT_COLUMNS}
""")

# Condition bitemporelle (architecture.md 5.1) : ce que le système savait à as_of,
# et qui était vrai à as_of.
_AS_OF = """
    recorded_at <= :as_of
    AND valid_from <= :as_of
    AND (valid_to IS NULL OR valid_to > :as_of)
    AND (expires_at IS NULL OR expires_at > :as_of)
"""


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
        "recorded_at": fact.recorded_at,
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

    def list_as_of(
        self, tenant_id: UUID, user_id: UUID, as_of: datetime, attributes: list[str] | None = None
    ) -> list[Fact]:
        clauses = ["tenant_id = :tenant_id", "user_id = :user_id", _AS_OF]
        params: dict = {"tenant_id": tenant_id, "user_id": user_id, "as_of": as_of}
        if attributes:
            clauses.append("attribute = ANY(:attributes)")
            params["attributes"] = attributes
        sql = text(f"SELECT {FACT_COLUMNS} FROM facts WHERE {' AND '.join(clauses)} ORDER BY recorded_at DESC")
        with self._db.transaction() as conn:
            rows = conn.execute(sql, params).all()
        return [to_fact(r) for r in rows]

    def diff(self, tenant_id: UUID, user_id: UUID, from_: datetime, to: datetime) -> Diff:
        """Ce qui a changé dans (from_, to] (architecture.md 5.2)."""
        base = "tenant_id = :tenant_id AND user_id = :user_id"
        params = {"tenant_id": tenant_id, "user_id": user_id, "from_": from_, "to": to}
        with self._db.transaction() as conn:
            added = conn.execute(text(f"""
                SELECT {FACT_COLUMNS} FROM facts
                WHERE {base} AND recorded_at > :from_ AND recorded_at <= :to
                  AND (valid_to IS NULL OR valid_to > :to)
                ORDER BY recorded_at
            """), params).all()
            replaced = conn.execute(text(f"""
                SELECT {FACT_COLUMNS} FROM facts
                WHERE {base} AND valid_to > :from_ AND valid_to <= :to AND end_reason = 'replaced'
                ORDER BY valid_to
            """), params).all()
            invalidated = conn.execute(text(f"""
                SELECT {FACT_COLUMNS} FROM facts
                WHERE {base} AND valid_to > :from_ AND valid_to <= :to
                  AND end_reason IN ('invalidated', 'expired', 'forgotten')
                ORDER BY valid_to
            """), params).all()
            pairs: list[tuple[Fact, Fact]] = []
            for r in replaced:
                old = to_fact(r)
                new_row = conn.execute(
                    text(f"SELECT {FACT_COLUMNS} FROM facts WHERE id = :id"), {"id": old.replaced_by}
                ).one()
                pairs.append((old, to_fact(new_row)))
        return Diff(
            added=[to_fact(r) for r in added],
            replaced=pairs,
            invalidated=[to_fact(r) for r in invalidated],
        )

    def find_similar_active(self, tenant_id: UUID, user_id: UUID, attribute_embedding: list[float], limit: int = 5) -> list[tuple[Fact, float]]:
        raise NotImplementedError("jalon 5")

    def search(self, tenant_id: UUID, user_id: UUID, query_embedding: list[float], as_of: datetime | None = None, limit: int = 20) -> list[tuple[Fact, float]]:
        raise NotImplementedError("jalon 5")

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

        L'ancien est clos à valid_from du nouveau (ou now() si non fourni), jamais avant
        son propre valid_from. Si l'ancien n'est plus actif -> StaleFact, rollback.
        Si un autre fait actif occupe déjà l'attribut -> ActiveFactExists, rollback.
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
                   SET valid_to = GREATEST(COALESCE(:at, now()), valid_from), end_reason = :reason
                 WHERE tenant_id = :tenant_id AND id = :id AND valid_to IS NULL
                RETURNING {FACT_COLUMNS}
            """),
            {"at": at, "reason": reason.value, "tenant_id": tenant_id, "id": fact_id},
        ).first()
        return to_fact(row) if row else None

    def _raise_if_active_conflict(self, e: IntegrityError, fact: NewFact) -> None:
        if "facts_active_unique" in str(e.orig):
            existing = self.get_active(fact.tenant_id, fact.user_id, fact.subject, fact.attribute)
            if existing is not None:
                raise ActiveFactExists(existing) from e