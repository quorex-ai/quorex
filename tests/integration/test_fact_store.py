from datetime import UTC, datetime, timedelta

import pytest

from quorex.domain import Durability, NewFact
from quorex.domain.store import ActiveFactExists
from quorex.storage import PostgresFactStore, PostgresTenantStore


def _fact(tenant_id, user_id, attribute="couleur_preferee", value="bleu", **kw) -> NewFact:
    base = dict(
        tenant_id=tenant_id, user_id=user_id, attribute=attribute, attribute_raw=attribute,
        value=value, confidence=0.9, durability=Durability.DURABLE, valid_from=datetime.now(UTC),
    )
    base.update(kw)
    return NewFact(**base)


@pytest.fixture
def ctx(tenants: PostgresTenantStore):
    t = tenants.create_tenant("t1", "intfloat/multilingual-e5-base", 768)
    u = tenants.get_or_create_user(t.id, "alice")
    return t, u


def test_create_puis_get_active(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    created = facts.create(_fact(t.id, u.id))
    assert created.is_active
    assert created.subject == "user"
    assert created.recorded_at is not None

    found = facts.get_active(t.id, u.id, "user", "couleur_preferee")
    assert found == created


def test_index_unique_refuse_un_second_fait_actif(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    first = facts.create(_fact(t.id, u.id, value="bleu"))
    with pytest.raises(ActiveFactExists) as exc:
        facts.create(_fact(t.id, u.id, value="rouge"))
    assert exc.value.existing.id == first.id
    assert facts.get_active(t.id, u.id, "user", "couleur_preferee").value == "bleu"


def test_list_active_filtre_expires(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    facts.create(_fact(t.id, u.id, attribute="ville_de_residence", value="Paris"))
    facts.create(_fact(t.id, u.id, attribute="localisation_actuelle", value="Lyon",
                       durability=Durability.TRANSIENT, expires_at=datetime.now(UTC) - timedelta(hours=1)))
    active = facts.list_active(t.id, u.id)
    assert [f.attribute for f in active] == ["ville_de_residence"]


def test_list_active_filtre_par_attributs(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    facts.create(_fact(t.id, u.id, attribute="a", value="1"))
    facts.create(_fact(t.id, u.id, attribute="b", value="2"))
    facts.create(_fact(t.id, u.id, attribute="c", value="3"))
    got = facts.list_active(t.id, u.id, attributes=["a", "c"])
    assert sorted(f.attribute for f in got) == ["a", "c"]


def test_transient_sans_expiration_refuse() -> None:
    from uuid import uuid4
    with pytest.raises(ValueError):
        _fact(uuid4(), uuid4(), durability=Durability.TRANSIENT)


def test_get_or_create_user_idempotent(tenants: PostgresTenantStore) -> None:
    t = tenants.create_tenant("t1", "m", 768)
    u1 = tenants.get_or_create_user(t.id, "alice")
    u2 = tenants.get_or_create_user(t.id, "alice")
    assert u1.id == u2.id


def test_isolation_tenants_meme_external_id(facts: PostgresFactStore, tenants: PostgresTenantStore) -> None:
    """Prémisse de S9 côté store : même external_id, deux tenants, deux users, deux mémoires."""
    t1 = tenants.create_tenant("t1", "m", 768)
    t2 = tenants.create_tenant("t2", "m", 768)
    a1 = tenants.get_or_create_user(t1.id, "alice")
    a2 = tenants.get_or_create_user(t2.id, "alice")
    assert a1.id != a2.id
    facts.create(_fact(t1.id, a1.id, value="bleu"))
    assert facts.list_active(t2.id, a2.id) == []
    assert facts.get_active(t2.id, a1.id, "user", "couleur_preferee") is None  # mauvais tenant, bon user