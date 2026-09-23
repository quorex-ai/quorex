"""S1. Remplacement simple (tests/acceptance/SCENARIOS.md). Le cas de la démo."""
import pytest

pytestmark = pytest.mark.acceptance


def _h(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_s1_remplacement_simple(client, make_tenant) -> None:
    _, key = make_tenant()

    r1 = client.post("/v1/remember", headers=_h(key),
                     json={"user": "u1", "attribute": "couleur préférée", "value": "bleu",
                           "valid_from": "2026-09-01T10:00:00Z"})
    assert r1.status_code == 200, r1.text
    b = r1.json()["facts"][0]
    assert b["action"] == "created"
    assert b["fact"]["attribute"] == "couleur_preferee"
    assert b["fact"]["attribute_raw"] == "couleur préférée"
    bleu_id = b["fact"]["id"]

    r2 = client.post("/v1/remember", headers=_h(key),
                     json={"user": "u1", "attribute": "Couleur favorite", "value": "rouge",
                           "valid_from": "2026-09-13T10:00:00Z"})
    assert r2.status_code == 200, r2.text
    rr = r2.json()["facts"][0]
    assert rr["action"] == "replaced"
    assert rr["level"] == "a"
    assert rr["replaced_fact_id"] == bleu_id
    assert rr["fact"]["value"] == "rouge"
    assert rr["fact"]["valid_from"].startswith("2026-09-13")

    r3 = client.post("/v1/recall", headers=_h(key), json={"user": "u1"})
    facts = r3.json()["facts"]
    assert [f["value"] for f in facts] == ["rouge"]
    assert facts[0]["valid_to"] is None


def test_s1_ancien_fait_est_clos_non_supprime(client, make_tenant, facts, tenants) -> None:
    t, key = make_tenant()
    client.post("/v1/remember", headers=_h(key),
                json={"user": "u1", "attribute": "couleur_preferee", "value": "bleu"})
    r = client.post("/v1/remember", headers=_h(key),
                    json={"user": "u1", "attribute": "couleur_preferee", "value": "rouge"})
    old_id = r.json()["facts"][0]["replaced_fact_id"]
    old = facts.get_by_id(t.id, __import__("uuid").UUID(old_id))
    assert old is not None
    assert old.end_reason.value == "replaced"
    assert str(old.replaced_by) == r.json()["facts"][0]["fact"]["id"]