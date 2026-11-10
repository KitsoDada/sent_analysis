import os

import pytest

from app.review_api import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app({
        "TESTING": True,
        "DATABASE": str(tmp_path / "reviews.sqlite3"),
        "REVIEW_API_KEY": "test-secret",
    })
    return app.test_client()


def headers():
    return {"X-API-Key": "test-secret"}


def test_analyze_and_create_review_requires_auth(client):
    assert client.post("/api/reviews", json={"text": "Great product"}).status_code == 401
    response = client.post("/api/reviews/analyze", headers=headers(), json={"text": "Package arrived late and damaged", "rating": 1})
    assert response.status_code == 200
    result = response.get_json()
    assert result["sentiment"] == "negative"
    assert "shipping" in result["themes"]
    assert result["requires_approval"] is True


def test_review_stays_pending_until_explicit_approval(client):
    created = client.post("/api/reviews", headers=headers(), json={"text": "The delivery was late", "product": "Shoes"})
    assert created.status_code == 201
    review_id = created.get_json()["id"]
    assert created.get_json()["status"] == "pending"

    listing = client.get("/api/reviews?status=pending", headers=headers()).get_json()["reviews"]
    assert len(listing) == 1
    assert listing[0]["id"] == review_id

    approved = client.patch(
        f"/api/reviews/{review_id}",
        headers=headers(),
        json={"status": "approved", "approved_reply": "Sorry about the delay. We will help."},
    )
    assert approved.status_code == 200
    assert approved.get_json()["status"] == "approved"


def test_batch_provider_ingestion(client):
    response = client.post(
        "/api/integrations/reviews",
        headers=headers(),
        json={"source": "shopify_review_app", "reviews": [{"external_id": "provider-123", "text": "Excellent quality", "rating": 5}]},
    )
    assert response.status_code == 201
    assert response.get_json()["imported"] == 1


def test_invalid_review_is_rejected(client):
    response = client.post("/api/reviews", headers=headers(), json={"text": "", "rating": 7})
    assert response.status_code == 400


def test_approval_requires_reply(client):
    created = client.post("/api/reviews", headers=headers(), json={"text": "Nice item"}).get_json()
    response = client.patch(f"/api/reviews/{created['id']}", headers=headers(), json={"status": "approved"})
    assert response.status_code == 400
