"""Persistent ChromaDB document collection with stable identifiers."""
from __future__ import annotations

from typing import Sequence

from .chunking import Chunk
from .settings import AppSettings


class ChromaStore:
    def __init__(self, settings: AppSettings, *, collection: object | None = None):
        self.settings = settings
        self._collection = collection

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.utils import embedding_functions

            embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.embedding_model
            )
            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self._collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                embedding_function=embedding,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks and remove leftover IDs from previous indexes of this book.

        Stable IDs make reruns idempotent. Changes to the source or chunk parameters
        don't leave stale tail chunks behind. Do not run multiple ingests concurrently.
        """
        if not chunks:
            raise ValueError("Refusing to replace the index with zero chunks")
        ids = [chunk.id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs")
        existing = set(self.collection.get(include=[])["ids"])
        batch = self.settings.batch_size
        for offset in range(0, len(chunks), batch):
            part = chunks[offset:offset + batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in part],
                documents=[chunk.text for chunk in part],
                metadatas=[{
                    "section_index": chunk.section_index,
                    "chunk_index_in_section": chunk.chunk_index_in_section,
                    **chunk.metadata,
                } for chunk in part],
            )
        stale = sorted(existing - set(ids))
        for offset in range(0, len(stale), batch):
            self.collection.delete(ids=stale[offset:offset + batch])
        return self.collection.count()

    def search(self, query: str, k: int = 5, *, chapter: int | None = None,
               section: str | None = None) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if k < 1 or k > self.settings.max_results:
            raise ValueError(f"k must be between 1 and {self.settings.max_results}")
        if self.count() == 0:
            return []
        where = None
        if chapter is not None and section is not None:
            where = {"$and": [{"chapter_id": chapter}, {"section_id": section}]}
        elif chapter is not None:
            where = {"chapter_id": chapter}
        elif section is not None:
            where = {"section_id": section}
        res = self.collection.query(
            query_texts=[query.strip()], n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": doc_id, "text": text, "metadata": metadata or {},
             "distance": distance}
            for doc_id, text, metadata, distance in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0],
            )
        ]
