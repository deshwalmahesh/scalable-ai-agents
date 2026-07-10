"""Smoke: the app boots (lifespan runs), liveness is green, OpenAPI + readiness respond."""


async def test_health_liveness(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_openapi_served(client):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    assert "/documents" in r.json()["paths"]


async def test_readiness_shape(client):
    r = await client.get("/health/ready")
    assert r.status_code in (200, 503)
    assert "checks" in r.json()


async def test_requires_auth(client):
    r = await client.get("/documents")
    assert r.status_code == 401
