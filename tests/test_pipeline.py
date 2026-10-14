import json

import pytest

from psych_search.pdf import Page
from psych_search.pipeline import index_pdf
from psych_search.settings import AppSettings
from psych_search.store import ChromaStore
from tests.test_store import FakeCollection


def test_pdf_to_mock_vector_store_end_to_end(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"source": "toy.pdf", "sections": [
        {"section_id": "1.1", "heading": "1.1 First", "pages": 1},
        {"section_id": "1.2", "heading": "1.2 Second", "pages": "2-3"},
    ]}))
    pages = [Page(1, "1.1 First\nThis paragraph introduces learning theories."),
             Page(2, "Previous ending.\n1.2 Second\nThis section describes memory."),
             Page(3, "The final page of the text.")]
    monkeypatch.setattr("psych_search.pipeline.read_pdf", lambda _: pages)
    store = ChromaStore(AppSettings(chunk_size=100, batch_size=2), collection=FakeCollection())
    first = index_pdf(tmp_path / "toy.pdf", catalog, settings=store.settings, store=store)
    second = index_pdf(tmp_path / "toy.pdf", catalog, settings=store.settings, store=store)
    assert first.pdf_pages == 3
    assert first.extracted_sections == 2
    assert first.indexed_chunks > 0
    assert second.collection_count == first.collection_count
    assert store.search("memory", section="1.2")


def test_pipeline_fails_without_text(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"sections": [{"section_id": "1.1", "heading": "One", "pages": "1-2"}]}))
    monkeypatch.setattr("psych_search.pipeline.read_pdf", lambda _: [Page(1, "No headings"), Page(2, "nothing")])
    with pytest.raises(ValueError, match="Could not extract headings"):
        index_pdf(tmp_path / "toy.pdf", catalog, store=ChromaStore(AppSettings(), collection=FakeCollection()))


def test_real_pdf_full_pipeline_with_fake_vector_collection(tmp_path):
    canvas_type = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf_path = tmp_path / "tiny_book.pdf"
    canvas = canvas_type.Canvas(str(pdf_path))
    canvas.drawString(50, 750, "1.1 First")
    canvas.drawString(50, 730, "Memory and attention are psychological processes.")
    canvas.showPage()
    canvas.drawString(50, 750, "1.2 Second")
    canvas.drawString(50, 730, "Conditioning is a form of learning.")
    canvas.showPage()
    canvas.drawString(50, 750, "Learning depends on experience.")
    canvas.save()
    catalog = tmp_path / "tiny_sections.json"
    catalog.write_text(json.dumps({"source": pdf_path.name, "sections": [
        {"section_id": "1.1", "heading": "1.1 First", "pages": 1},
        {"section_id": "1.2", "heading": "1.2 Second", "pages": "2-3"},
    ]}))
    settings = AppSettings(chunk_size=100)
    store = ChromaStore(settings, collection=FakeCollection())
    report = index_pdf(pdf_path, catalog, settings=settings, store=store)
    assert report == type(report)(3, 2, report.indexed_chunks, report.collection_count)
    assert report.indexed_chunks >= 2
    assert store.search("conditioning", section="1.2")
