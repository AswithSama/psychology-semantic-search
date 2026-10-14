"""Query service boundary shared by the command line and HTTP API."""
from __future__ import annotations

from dataclasses import dataclass

from .store import ChromaStore


@dataclass
class SearchService:
    store: ChromaStore

    def search(self, query: str, k: int = 5, *, chapter: int | None = None,
               section: str | None = None) -> list[dict]:
        if chapter is not None and chapter < 1:
            raise ValueError("chapter must be positive")
        if section is not None:
            try:
                matching_chapter = int(section.split(".")[0])
            except (ValueError, IndexError) as exc:
                raise ValueError("section must look like 6.2") from exc
            if "." not in section or (chapter is not None and matching_chapter != chapter):
                raise ValueError("section must look like 6.2 and match chapter")
        return self.store.search(query, k, chapter=chapter, section=section)
