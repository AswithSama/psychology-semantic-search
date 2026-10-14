from fastapi.testclient import TestClient

from psych_search.api import create_app
from psych_search.retrieval import SearchService
from tests.test_store import FakeCollection, make_chunk
from psych_search.settings import AppSettings
from psych_search.store import ChromaStore


def make_client():
    store = ChromaStore(AppSettings(), collection=FakeCollection())
    store.upsert_chunks([make_chunk("1.1", 0, "Classical conditioning example")])
    return TestClient(create_app(SearchService(store)))


def test_health_is_liveness_only():
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_serves_ui():
    response = make_client().get("/")
    assert response.status_code == 200
    assert "Search your psychology textbook" in response.text


def test_stats():
    assert make_client().get("/api/stats").json() == {"indexed_chunks": 1}


def test_post_search():
    response = make_client().post("/api/search", json={"query": "conditioning", "k": 3})
    assert response.status_code == 200
    assert response.json()["hits"][0]["metadata"]["section_id"] == "1.1"


def test_get_search():
    response = make_client().get("/api/search?q=conditioning&chapter=1")
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_validation_blank_query():
    response = make_client().post("/api/search", json={"query": ""})
    assert response.status_code == 422


def test_validation_k_out_of_range():
    response = make_client().post("/api/search", json={"query": "example", "k": 0})
    assert response.status_code == 422


def test_validation_malformed_section():
    response = make_client().post("/api/search", json={"query": "example", "section": "abc"})
    assert response.status_code == 422


def test_conflicting_chapter_filter():
    response = make_client().post("/api/search", json={"query": "example", "chapter": 2, "section": "1.1"})
    assert response.status_code == 422
