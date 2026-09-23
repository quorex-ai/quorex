"""S6, S6b, S7, S11 au niveau du store, avec des dates contrôlées via NewFact.recorded_at."""
from datetime import UTC, datetime, timedelta

import pytest

from quorex.domain import Durability, EndReason, NewFact
from quorex.storage import PostgresFactStore, PostgresTenantStore

J0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def J(n: int) -> datetime:
    return J0 + timedelta(days=n)


def _fact(t, u, attribute, value, day, valid_from=None) -> NewFact:
    return NewFact(tenant_id=t, user_id=u, attribute=attribute, attribute_raw=attribute, value=value,
                   confidence=0.9, durability=Durability.DURABLE,
                   valid_from=valid_from or J(day), recorded_at=J(day))


@pytest.fixture
def ctx(tenants: PostgresTenantStore):
    t = tenants.create_tenant("t1", "m", 768)
    u = tenants.get_or_create_user(t.id, "alice")
    return t.id, u.id


def test_horloge_unique_quand_dates_non_fournies(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    f = facts.create(NewFact(tenant_id=t, user_id=u, attribute="x", attribute_raw="x", value="1",
                             confidence=1, durability=Durability.DURABLE))
    assert f.recorded_at == f.valid_from


def test_s6_time_travel(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    bleu = facts.create(_fact(t, u, "couleur_preferee", "bleu", 0))
    facts.replace(bleu.id, _fact(t, u, "couleur_preferee", "rouge", 12))

    assert [f.value for f in facts.list_as_of(t, u, J(5))] == ["bleu"]
    assert [f.value for f in facts.list_as_of(t, u, J(12))] == ["rouge"]   # borne : rouge inclus, bleu exclu
    assert [f.value for f in facts.list_as_of(t, u, J(20))] == ["rouge"]
    assert facts.list_as_of(t, u, J(-1)) == []


def test_s6b_connaissance_retroactive(facts: PostgresFactStore, ctx) -> None:
    """Un fait vrai depuis toujours mais appris à J30 n'apparaît pas dans as_of(J10)."""
    t, u = ctx
    facts.create(_fact(t, u, "ville_de_naissance", "Lyon", 30, valid_from=J(-3650)))
    assert facts.list_as_of(t, u, J(10)) == []
    assert [f.value for f in facts.list_as_of(t, u, J(31))] == ["Lyon"]


def test_s7_diff(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    bleu = facts.create(_fact(t, u, "couleur_preferee", "bleu", 0))
    julie = facts.create(_fact(t, u, "manager", "Julie", 0))
    _, rouge = facts.replace(bleu.id, _fact(t, u, "couleur_preferee", "rouge", 12))
    facts.create(_fact(t, u, "langue", "fr", 15))
    facts.close(t, julie.id, EndReason.INVALIDATED, at=J(20))

    d = facts.diff(t, u, J(10), J(25))
    assert sorted(f.value for f in d.added) == ["fr", "rouge"]
    assert [(o.value, n.value) for o, n in d.replaced] == [("bleu", "rouge")]
    assert d.replaced[0][1].id == rouge.id
    assert [f.value for f in d.invalidated] == ["Julie"]

    d0 = facts.diff(t, u, J(-1), J(5))
    assert sorted(f.value for f in d0.added) == ["Julie", "bleu"]
    assert d0.replaced == [] and d0.invalidated == []


def test_s11_forget_clot_sans_supprimer(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    rouge = facts.create(_fact(t, u, "couleur_preferee", "rouge", 0))
    facts.close(t, rouge.id, EndReason.FORGOTTEN, at=J(3))
    assert facts.list_active(t, u) == []
    assert [f.value for f in facts.list_as_of(t, u, J(1))] == ["rouge"]
    gone = facts.get_by_id(t, rouge.id)
    assert gone is not None and gone.end_reason is EndReason.FORGOTTEN and gone.replaced_by is None