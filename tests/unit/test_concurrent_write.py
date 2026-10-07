"""Écriture concurrente non résolue : après _MAX_ATTEMPTS échecs, ConcurrentWrite,
puis 409 via l'API.

Sans base : le store est un faux, l'auth et les tenants sont surchargés.
"""
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from quorex.api import create_app
from quorex.api.deps import AuthContext, get_auth, get_normalizer, get_tenant_store
from quorex.contradiction import Resolver
from quorex.domain import ActiveFactExists, ConcurrentWrite, Durability, Fact, NewFact
from quorex.domain.tenant import User
from quorex.normalization import Normalizer

T0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _fact(tenant_id: UUID, user_id: UUID) -> Fact:
    return Fact(
        id=uuid4(), tenant_id=tenant_id, user_id=user_id, subject="user",
        attribute="couleur_preferee", attribute_raw="couleur_preferee", value="bleu",
        confidence=1.0, durability=Durability.DURABLE, expires_at=None, source_message_id=None,
        valid_from=T0, valid_to=None, recorded_at=T0, replaced_by=None, end_reason=None,
    )


class FakeFactStore:
    """create() et replace() perdent toujours la course face à une écriture concurrente."""

    def __init__(self, existing: Fact | None) -> None:
        self.existing = existing
        self.writes = 0

    def get_active(
        self, tenant_id: UUID, user_id: UUID, subject: str, attribute: str
    ) -> Fact | None:
        return self.existing

    def create(self, fact: NewFact) -> Fact:
        self.writes += 1
        raise ActiveFactExists(_fact(fact.tenant_id, fact.user_id))

    def replace(self, old_id: UUID, new: NewFact) -> tuple[Fact, Fact]:
        self.writes += 1
        raise ActiveFactExists(_fact(new.tenant_id, new.user_id))


class FakeTenantStore:
    def get_or_create_user(self, tenant_id: UUID, external_id: str) -> User:
        return User(id=uuid4(), tenant_id=tenant_id, external_id=external_id, created_at=T0)


def _new(tenant_id: UUID, user_id: UUID) -> NewFact:
    return NewFact(
        tenant_id=tenant_id, user_id=user_id, attribute="couleur_preferee",
        attribute_raw="couleur_preferee", value="rouge", confidence=1.0,
        durability=Durability.DURABLE,
    )


@pytest.fixture(params=["create", "replace"])
def store(request: pytest.FixtureRequest) -> FakeFactStore:
    existing = _fact(uuid4(), uuid4()) if request.param == "replace" else None
    return FakeFactStore(existing)


def test_resolver_leve_concurrent_write_apres_deux_echecs(store: FakeFactStore) -> None:
    with pytest.raises(ConcurrentWrite):
        Resolver(store).remember(_new(uuid4(), uuid4()))
    assert store.writes == 2


def test_api_repond_409_concurrent_write(store: FakeFactStore) -> None:
    app = create_app()
    # Hors `with` : le lifespan (qui ouvrirait la base) ne tourne pas.
    app.state.resolver = Resolver(store)
    app.dependency_overrides[get_auth] = lambda: AuthContext(tenant_id=uuid4(), api_key_id=uuid4())
    app.dependency_overrides[get_tenant_store] = FakeTenantStore
    app.dependency_overrides[get_normalizer] = lambda: Normalizer({})

    r = TestClient(app).post(
        "/v1/remember", json={"user": "u1", "attribute": "couleur_preferee", "value": "rouge"}
    )

    assert r.status_code == 409
    assert r.json()["error"]["code"] == "concurrent_write"
    assert store.writes == 2
