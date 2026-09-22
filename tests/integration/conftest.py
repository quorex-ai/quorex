from collections.abc import Iterator

import pytest
from sqlalchemy import text

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
def clean_db(db: Database) -> None:
    with db.transaction() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))


@pytest.fixture
def facts(db: Database) -> PostgresFactStore:
    return PostgresFactStore(db)


@pytest.fixture
def tenants(db: Database) -> PostgresTenantStore:
    return PostgresTenantStore(db)