"""Read-only FastAPI app exposing the indexed textbook and a small web UI."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .retrieval import SearchService
from .settings import AppSettings
from .store import ChromaStore


class Hit(BaseModel):
    id: str
    text: str
    metadata: dict[str, str | int | float | bool]
    distance: float


class SearchResponse(BaseModel):
    query: str
    hits: list[Hit]
    count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=100)
    chapter: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=32)


@lru_cache
def get_search_service() -> SearchService:
    config = AppSettings()
    return SearchService(ChromaStore(config))


def create_app(service: SearchService | None = None) -> FastAPI:
    app = FastAPI(
        title="Psychology Semantic Search",
        description="Grounded textbook passage retrieval; no generative answer synthesis.",
        version="0.1.0",
    )
    if service is not None:
        app.dependency_overrides[get_search_service] = lambda: service

    @app.get("/health")
    def health():
        # Liveness check does not download the embedding model or connect to Chroma.
        return {"status": "ok"}

    @app.get("/api/stats")
    def stats(search: Annotated[SearchService, Depends(get_search_service)]):
        try:
            return {"indexed_chunks": search.store.count()}
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Vector store unavailable") from exc

    @app.post("/api/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        search_service: Annotated[SearchService, Depends(get_search_service)],
    ):
        try:
            rows = search_service.search(
                request.query, request.k, chapter=request.chapter, section=request.section
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Search backend unavailable") from exc
        return SearchResponse(query=request.query, hits=rows, count=len(rows))

    @app.get("/api/search", response_model=SearchResponse)
    def search_get(
        search_service: Annotated[SearchService, Depends(get_search_service)],
        q: Annotated[str, Query(min_length=1, max_length=1000)],
        k: Annotated[int, Query(ge=1, le=100)] = 5,
        chapter: Annotated[int | None, Query(ge=1)] = None,
        section: str | None = None,
    ):
        return search(SearchRequest(query=q, k=k, chapter=chapter, section=section), search_service)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app


app = create_app()
