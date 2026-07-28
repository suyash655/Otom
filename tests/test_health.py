def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "device" in payload
    assert isinstance(payload["classes"], list)
    assert len(payload["classes"]) > 0
