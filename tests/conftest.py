"""Fixtures partagées : base, stores, application et client HTTP.

Les tests d'intégration et d'acceptation exigent la base du docker-compose
avec la migration appliquée. Les tables sont tronquées avant chaque test.
"""
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from quorex.api import create_app
from quorex.auth import generate_api_key
from quorex.settings import get_settings
from quorex.storage import Database, PostgresFactStore, PostgresTenantStore

TABLES = ["facts", "messages", "users", "api_keys", "tenants"]


@pytest.fixture(scope="session")
def db() -> Iterator[Database]:
    database = Database(get_settings().database_url)
    try:
        with database.transaction() as conn:
            conn.execute(text("SELECT 1 FROM facts LIMIT 1"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"base indisponible ou migration non appliquée : {e}")
    yield database
    database.dispose()


@pytest.fixture(autouse=True)
def clean_db(request: pytest.FixtureRequest) -> None:
    # Ne tronque que si le test utilise la base (évite de la réclamer pour les unitaires).
    if "db" in request.fixturenames or "client" in request.fixturenames:
        database = request.getfixturevalue("db")
        with database.transaction() as conn:
            conn.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


@pytest.fixture
def facts(db: Database) -> PostgresFactStore:
    return PostgresFactStore(db)


@pytest.fixture
def tenants(db: Database) -> PostgresTenantStore:
    return PostgresTenantStore(db)


@pytest.fixture
def client(db: Database) -> Iterator[TestClient]:
    app = create_app(db=db)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_tenant(tenants: PostgresTenantStore):
    """Crée un tenant + une clé, retourne (tenant, clé en clair)."""
    def _make(name: str = "t1") -> tuple:
        t = tenants.create_tenant(name, "intfloat/multilingual-e5-base", 768)
        k = generate_api_key()
        tenants.create_api_key(t.id, k.prefix, k.hash, "test")
        return t, k.plain
    return _make