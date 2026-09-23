"""S8. Vingt remember concurrents sur le même attribut (tests/acceptance/SCENARIOS.md)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from quorex.contradiction import Resolver
from quorex.domain import ConcurrentWrite, Durability, NewFact
from quorex.storage import PostgresFactStore, PostgresTenantStore

pytestmark = pytest.mark.acceptance

N = 20


def test_s8_un_seul_actif_et_une_chaine_complete(facts: PostgresFactStore, tenants: PostgresTenantStore) -> None:
    t = tenants.create_tenant("t1", "m", 768)
    u = tenants.get_or_create_user(t.id, "alice")
    resolver = Resolver(facts)

    def write(i: int):
        new = NewFact(tenant_id=t.id, user_id=u.id, attribute="couleur_preferee", attribute_raw="x",
                      value=f"valeur_{i}", confidence=0.9, durability=Durability.DURABLE,
                      valid_from=datetime.now(UTC))
        try:
            return resolver.remember(new)
        except ConcurrentWrite:
            return None

    with ThreadPoolExecutor(max_workers=N) as pool:
        outcomes = list(pool.map(write, range(N)))

    ok = [o for o in outcomes if o is not None]
    conflicts = N - len(ok)

    # Exactement un fait actif.
    active = facts.list_active(t.id, u.id)
    assert len(active) == 1

    # Total en base = nombre de succès ; aucun fantôme.
    from sqlalchemy import text
    with facts._db.transaction() as conn:  # accès direct toléré dans un test
        total = conn.execute(text("SELECT count(*) FROM facts WHERE user_id = :u"), {"u": u.id}).scalar()
        rows = conn.execute(text("SELECT id, replaced_by, end_reason FROM facts WHERE user_id = :u"),
                            {"u": u.id}).all()
    assert total == len(ok)

    # Chaîne replaced_by : en partant de l'actif, remonter les prédécesseurs couvre tout, sans cycle.
    by_successor = {r._mapping["replaced_by"]: r._mapping["id"] for r in rows if r._mapping["replaced_by"]}
    seen = set()
    cur = active[0].id
    while cur is not None and cur not in seen:
        seen.add(cur)
        cur = by_successor.get(cur)
    assert len(seen) == total, f"chaîne incomplète : {len(seen)} / {total} (conflits : {conflicts})"
    closed = [r for r in rows if r._mapping["end_reason"] is not None]
    assert all(r._mapping["end_reason"] == "replaced" for r in closed)