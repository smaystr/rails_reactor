def test_get_health(client):
    response = client.get("/app/v1/health")

    assert response.json == {"status": "ok"}
