"""S9. Isolation entre tenants (tests/acceptance/SCENARIOS.md).

Deux tenants, chacun avec un utilisateur external_id = "alice".
"""
import pytest

pytestmark = pytest.mark.acceptance


def _h(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_s9_isolation_entre_tenants(client, make_tenant, tenants) -> None:
    _, key1 = make_tenant("T1")
    _, key2 = make_tenant("T2")

    # 1. T1 : remember couleur_preferee = bleu pour alice
    r = client.post("/v1/remember", headers=_h(key1),
                    json={"user": "alice", "attribute": "couleur_preferee", "value": "bleu"})
    assert r.status_code == 200, r.text
    fact_id = r.json()["facts"][0]["fact"]["id"]
    assert r.json()["facts"][0]["action"] == "created"

    # 2. T2 : recall alice -> alice n'existe pas encore chez T2
    r = client.post("/v1/recall", headers=_h(key2), json={"user": "alice"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "user_not_found"

    # 3. T2 : forget par id du fait de T1 -> introuvable (alice créée au passage chez T2)
    client.post("/v1/remember", headers=_h(key2),
                json={"user": "alice", "attribute": "langue", "value": "fr"})
    r = client.post("/v1/forget", headers=_h(key2), json={"user": "alice", "fact_id": fact_id})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "fact_not_found"

    # 4. T2 : remember couleur_preferee = vert pour alice -> création dans T2
    r = client.post("/v1/remember", headers=_h(key2),
                    json={"user": "alice", "attribute": "couleur_preferee", "value": "vert"})
    assert r.status_code == 200, r.text

    # 5. T1 : recall alice -> bleu uniquement
    r = client.post("/v1/recall", headers=_h(key1), json={"user": "alice"})
    assert r.status_code == 200
    values = {f["attribute"]: f["value"] for f in r.json()["facts"]}
    assert values == {"couleur_preferee": "bleu"}

    # Clé révoquée -> 401 sur toute route
    keys = tenants.find_active_keys_by_prefix(key2[3:11])
    tenants.revoke_api_key(keys[0].id)
    r = client.post("/v1/recall", headers=_h(key2), json={"user": "alice"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_cle_absente_ou_mal_formee(client) -> None:
    r = client.post("/v1/recall", json={"user": "alice"})
    assert r.status_code == 401
    r = client.post("/v1/recall", headers=_h("qx_pasunecle"), json={"user": "alice"})
    assert r.status_code == 401
    assert "request_id" in r.json()["error"]
    assert r.headers["X-Request-Id"]


def test_health_sans_auth(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}