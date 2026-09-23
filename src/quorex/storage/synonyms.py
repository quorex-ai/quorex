"""Lecture de la table synonyms (chargée par la migration 0001)."""
from __future__ import annotations

from sqlalchemy import text

from quorex.storage.db import Database


def load_synonyms(db: Database) -> dict[str, str]:
    with db.transaction() as conn:
        rows = conn.execute(text("SELECT variant, canonical FROM synonyms")).all()
    return {r._mapping["variant"]: r._mapping["canonical"] for r in rows}