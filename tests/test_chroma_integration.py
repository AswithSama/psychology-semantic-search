"""Optional real persistent-Chroma test; skips when ChromaDB isn't installed.

No model download: test embeddings are deterministic vectors, independent
of the production all-MiniLM-L6-v2 provider.
"""
import pytest

from psych_search.chunking import Chunk
from psych_search.settings import AppSettings
from psych_search.store import ChromaStore


def test_persistent_chromadb_roundtrip(tmp_path):
    chromadb = pytest.importorskip("chromadb")

    class ToyEmbedding:
        def __call__(self, input):
            return [[float(len(text) % 7), float(sum(map(ord, text)) % 11), 1.0] for text in input]
        def name(self):
            return "toy-embedding"
        def get_config(self):
            return {}
        @staticmethod
        def build_from_config(config):
            return ToyEmbedding()

    path = tmp_path / "vectors"
    config = AppSettings(chroma_path=path, collection_name="toy_psych")
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name="toy_psych", embedding_function=ToyEmbedding(), metadata={"hnsw:space": "cosine"}
    )
    store = ChromaStore(config, collection=collection)
    chunks = [Chunk("Memory is a process of encoding information.", 0, 0,
                    {"section_id": "8.1", "chapter_id": 8, "section_heading": "Memory", "start_page": 2,
                     "end_page": 3, "source": "fixture.pdf"})]
    assert store.upsert_chunks(chunks) == 1
    assert store.search("memory", k=1)[0]["id"] == "sec8.1_chunk0"
    reopened = chromadb.PersistentClient(path=str(path)).get_collection(
        "toy_psych", embedding_function=ToyEmbedding()
    )
    assert reopened.count() == 1
