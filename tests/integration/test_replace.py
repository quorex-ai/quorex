from datetime import UTC, datetime

import pytest

from quorex.domain import Durability, EndReason, NewFact, StaleFact
from quorex.storage import PostgresFactStore, PostgresTenantStore


def _fact(tenant_id, user_id, value, attribute="couleur_preferee") -> NewFact:
    return NewFact(tenant_id=tenant_id, user_id=user_id, attribute=attribute, attribute_raw=attribute,
                   value=value, confidence=0.9, durability=Durability.DURABLE, valid_from=datetime.now(UTC))


@pytest.fixture
def ctx(tenants: PostgresTenantStore):
    t = tenants.create_tenant("t1", "m", 768)
    u = tenants.get_or_create_user(t.id, "alice")
    return t, u


def test_replace_clot_et_lie(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    bleu = facts.create(_fact(t.id, u.id, "bleu"))
    old, rouge = facts.replace(bleu.id, _fact(t.id, u.id, "rouge"))
    assert old.id == bleu.id
    assert old.valid_to is not None and old.end_reason is EndReason.REPLACED
    assert old.replaced_by == rouge.id
    assert rouge.is_active and rouge.value == "rouge"
    assert facts.get_active(t.id, u.id, "user", "couleur_preferee").id == rouge.id


def test_replace_sur_fait_deja_clos_leve_stale(facts: PostgresFactStore, ctx) -> None:
    t, u = ctx
    bleu = facts.create(_fact(t.id, u.id, "bleu"))
    facts.close(t.id, bleu.id, EndReason.FORGOTTEN)
    with pytest.raises(StaleFact):
        facts.replace(bleu.id, _fact(t.id, u.id, "rouge"))
    # rien n'a été écrit
    assert facts.get_active(t.id, u.id, "user", "couleur_preferee") is None


def test_valid_to_jamais_avant_valid_from(facts: PostgresFactStore, ctx) -> None:
    """Un remplacement avec un valid_from dans le passé ne doit pas violer facts_valid_range."""
    t, u = ctx
    bleu = facts.create(_fact(t.id, u.id, "bleu"))
    passe = NewFact(tenant_id=t.id, user_id=u.id, attribute="couleur_preferee", attribute_raw="x",
                    value="rouge", confidence=0.9, durability=Durability.DURABLE,
                    valid_from=datetime(2000, 1, 1, tzinfo=UTC))
    old, _ = facts.replace(bleu.id, passe)
    assert old.valid_to >= old.valid_from