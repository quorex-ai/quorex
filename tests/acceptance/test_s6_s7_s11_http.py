"""S6, S7, S11 via l'API. Les dates d'enregistrement sont posées côté store (recorded_at
n'est jamais exposé par l'API), les lectures passent par HTTP."""
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from quorex.domain import Durability, EndReason, NewFact

pytestmark = pytest.mark.acceptance

J0 = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def J(n: int) -> datetime:
    return J0 + timedelta(days=n)


def _h(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _seed(facts, t, u) -> None:
    def f(attr, val, day):
        return NewFact(tenant_id=t, user_id=u, attribute=attr, attribute_raw=attr, value=val,
                       confidence=0.9, durability=Durability.DURABLE, valid_from=J(day), recorded_at=J(day))
    bleu = facts.create(f("couleur_preferee", "bleu", 0))
    julie = facts.create(f("manager", "Julie", 0))
    facts.replace(bleu.id, f("couleur_preferee", "rouge", 12))
    facts.create(f("langue", "fr", 15))
    facts.close(t, julie.id, EndReason.INVALIDATED, at=J(20))


def _by_attr(resp) -> dict[str, str]:
    return {f["attribute"]: f["value"] for f in resp.json()["facts"]}


def test_s6_recall_as_of(client, make_tenant, tenants, facts) -> None:
    t, key = make_tenant()
    u = tenants.get_or_create_user(t.id, "u1")
    _seed(facts, t.id, u.id)

    r = client.post("/v1/recall", headers=_h(key), json={"user": "u1", "as_of": J(5).isoformat()})
    assert r.status_code == 200
    assert _by_attr(r) == {"couleur_preferee": "bleu", "manager": "Julie"}

    r = client.post("/v1/recall", headers=_h(key), json={"user": "u1", "as_of": J(25).isoformat()})
    assert _by_attr(r) == {"couleur_preferee": "rouge", "langue": "fr"}

    r = client.post("/v1/recall", headers=_h(key), json={"user": "u1"})
    assert _by_attr(r) == {"couleur_preferee": "rouge", "langue": "fr"}

    futur = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    r = client.post("/v1/recall", headers=_h(key), json={"user": "u1", "as_of": futur})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_time_range"


def test_s7_diff_http(client, make_tenant, tenants, facts) -> None:
    t, key = make_tenant()
    u = tenants.get_or_create_user(t.id, "u1")
    _seed(facts, t.id, u.id)

    r = client.post("/v1/diff", headers=_h(key),
                    json={"user": "u1", "from": J(10).isoformat(), "to": J(25).isoformat()})
    assert r.status_code == 200, r.text
    d = r.json()
    assert sorted(f["value"] for f in d["added"]) == ["fr", "rouge"]
    assert [(p["old"]["value"], p["new"]["value"]) for p in d["replaced"]] == [("bleu", "rouge")]
    assert [f["value"] for f in d["invalidated"]] == ["Julie"]

    r = client.post("/v1/diff", headers=_h(key),
                    json={"user": "u1", "from": J(25).isoformat(), "to": J(10).isoformat()})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_time_range"


def test_s11_forget_http(client, make_tenant, tenants, facts) -> None:
    t, key = make_tenant()
    client.post("/v1/remember", headers=_h(key),
                json={"user": "u1", "attribute": "couleur_preferee", "value": "rouge"})
    r = client.post("/v1/remember", headers=_h(key),
                    json={"user": "u1", "attribute": "langue", "value": "fr"})
    langue_id = r.json()["facts"][0]["fact"]["id"]

    r = client.post("/v1/forget", headers=_h(key), json={"user": "u1", "attribute": "couleur préférée"})
    assert r.status_code == 200
    assert [f["end_reason"] for f in r.json()["closed"]] == ["forgotten"]

    r = client.post("/v1/forget", headers=_h(key), json={"user": "u1", "fact_id": langue_id})
    assert r.status_code == 200 and r.json()["closed"][0]["replaced_by"] is None

    r = client.post("/v1/forget", headers=_h(key), json={"user": "u1", "fact_id": str(UUID(int=0))})
    assert r.status_code == 404 and r.json()["error"]["code"] == "fact_not_found"

    assert client.post("/v1/recall", headers=_h(key), json={"user": "u1"}).json()["facts"] == []
    assert facts.get_by_id(t.id, UUID(langue_id)) is not None   # jamais de DELETE