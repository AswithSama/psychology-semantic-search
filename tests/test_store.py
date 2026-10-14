import pytest

from psych_search.chunking import Chunk
from psych_search.settings import AppSettings
from psych_search.store import ChromaStore


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.queries = []

    def count(self):
        return len(self.rows)

    def get(self, include):
        return {"ids": list(self.rows)}

    def upsert(self, ids, documents, metadatas):
        for identifier, text, metadata in zip(ids, documents, metadatas):
            self.rows[identifier] = (text, metadata)

    def delete(self, ids):
        for identifier in ids:
            self.rows.pop(identifier)

    def query(self, query_texts, n_results, where, include):
        self.queries.append((query_texts, n_results, where, include))
        rows = [(identifier, text, metadata) for identifier, (text, metadata) in self.rows.items()]
        def matches(metadata):
            if not where:
                return True
            if "$and" in where:
                return all(all(metadata.get(k) == v for k, v in clause.items()) for clause in where["$and"])
            return all(metadata.get(k) == v for k, v in where.items())
        selected = [(i, t, m) for i, t, m in rows if matches(m)][:n_results]
        return {"ids": [[x[0] for x in selected]],
                "documents": [[x[1] for x in selected]],
                "metadatas": [[x[2] for x in selected]],
                "distances": [[0.1 for _ in selected]]}


@pytest.fixture
def store():
    return ChromaStore(AppSettings(batch_size=2, max_results=10), collection=FakeCollection())


def make_chunk(section, number, text="Example text"):
    return Chunk(text, 0, number, {"chapter_id": 1, "section_id": section,
                                  "section_heading": "Heading", "source": "test.pdf",
                                  "start_page": 1, "end_page": 2})


def test_upsert_idempotent(store):
    chunks = [make_chunk("1.1", n) for n in range(3)]
    assert store.upsert_chunks(chunks) == 3
    assert store.upsert_chunks(chunks) == 3


def test_upsert_removes_old_chunks(store):
    store.upsert_chunks([make_chunk("1.1", 0), make_chunk("1.1", 1)])
    store.upsert_chunks([make_chunk("1.1", 0, "New text")])
    assert store.count() == 1
    assert store.collection.rows["sec1.1_chunk0"][0] == "New text"


def test_empty_index_refused(store):
    with pytest.raises(ValueError, match="zero chunks"):
        store.upsert_chunks([])


def test_duplicate_ids_refused(store):
    with pytest.raises(ValueError, match="Duplicate"):
        store.upsert_chunks([make_chunk("1.1", 0), make_chunk("1.1", 0)])


def test_empty_search_yields_no_results(store):
    assert store.search("test") == []


def test_search_with_both_filters(store):
    store.upsert_chunks([make_chunk("1.1", 0), make_chunk("1.2", 0)])
    result = store.search("Example", chapter=1, section="1.2")
    assert len(result) == 1
    assert result[0]["id"] == "sec1.2_chunk0"
    assert "$and" in store.collection.queries[-1][2]


def test_search_rejects_empty_query(store):
    with pytest.raises(ValueError, match="empty"):
        store.search(" ")


def test_search_rejects_large_k(store):
    with pytest.raises(ValueError, match="between"):
        store.search("hello", k=11)
