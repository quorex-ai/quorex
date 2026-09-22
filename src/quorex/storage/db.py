"""Connexion et transactions. Seul endroit qui crée un engine."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, create_engine
from sqlalchemy.engine import Engine


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.engine: Engine = create_engine(url, echo=echo, pool_pre_ping=True, future=True)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Une transaction : commit à la sortie, rollback sur exception."""
        with self.engine.begin() as conn:
            yield conn

    def dispose(self) -> None:
        self.engine.dispose()