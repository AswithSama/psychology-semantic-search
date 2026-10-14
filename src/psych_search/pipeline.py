"""One-command deterministic PDF-to-Chroma indexing orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalog import DEFAULT_CATALOG, load_catalog
from .chunking import chunk_sections
from .pdf import read_pdf
from .sections import extract_sections
from .settings import AppSettings
from .store import ChromaStore


@dataclass(frozen=True)
class IndexReport:
    pdf_pages: int
    extracted_sections: int
    indexed_chunks: int
    collection_count: int


def index_pdf(
    pdf_path: str | Path,
    catalog_path: str | Path = DEFAULT_CATALOG,
    *,
    settings: AppSettings | None = None,
    strict: bool = True,
    store: ChromaStore | None = None,
) -> IndexReport:
    config = settings or AppSettings()
    pdf_path = Path(pdf_path)
    source, section_catalog = load_catalog(catalog_path)
    if pdf_path.name != source:
        # Keep the actual source filename for provenance; the original catalog
        # names the reference edition but can be used with a renamed PDF.
        source = pdf_path.name
    pages = read_pdf(pdf_path)
    documents = extract_sections(pages, section_catalog, source, strict=strict)
    chunks = chunk_sections(documents, config.chunk_size, config.overlap_sentences)
    if not chunks:
        raise ValueError("No indexed chunks created: check PDF page labels and catalog")
    vector_store = store or ChromaStore(config)
    count = vector_store.upsert_chunks(chunks)
    return IndexReport(len(pages), len(documents), len(chunks), count)
