import pytest

from psych_search.chunking import (
    chunk_sections, pack_sentences, sentences_from_blocks, split_oversize,
)


def test_section_boundaries_preserved(section_docs):
    chunks = chunk_sections(section_docs, max_size=80)
    assert {c.metadata["section_id"] for c in chunks} == {"1.1", "1.2"}
    assert all("different section" not in c.text for c in chunks if c.metadata["section_id"] == "1.1")


def test_chunk_size_never_exceeded():
    chunks = pack_sentences(["One." for _ in range(100)], max_size=17, overlap_sentences=2)
    assert len(chunks) > 1
    assert all(len(chunk) <= 17 for chunk in chunks)


def test_sentence_overlap_is_carried():
    output = pack_sentences(["First.", "Second.", "Third.", "Fourth."], max_size=22, overlap_sentences=1)
    assert output[0] == "First. Second. Third."
    assert output[1].startswith("Third.")
    assert "Fourth." in output[1]


def test_overlap_can_be_dropped_to_make_progress():
    output = pack_sentences(["123456789", "abcdefghi"], max_size=9, overlap_sentences=2)
    assert output == ["123456789", "abcdefghi"]


def test_split_punctuation_and_hard_fallback():
    pieces = split_oversize("Long sentence; second piece, third part.", 12)
    assert all(len(piece) <= 12 for piece in pieces)
    assert len(pieces) > 1


def test_arbitrary_long_unit_splits():
    assert split_oversize("X" * 41, 10) == ["X" * 10] * 4 + ["X"]


@pytest.mark.parametrize("max_size,overlap", [(0, 2), (10, -1)])
def test_invalid_chunk_settings(max_size, overlap):
    with pytest.raises(ValueError):
        pack_sentences(["Hello."], max_size, overlap)


def test_empty_chunk_input():
    assert pack_sentences([], 1000) == []


def test_stable_identifiers_and_metadata(section_docs):
    chunks = chunk_sections(section_docs, max_size=100)
    assert chunks[0].id == "sec1.1_chunk0"
    assert chunks[-1].metadata["chapter_id"] == 1
    assert chunks[-1].id.startswith("sec1.2_chunk")


def test_segment_newline_blocks():
    spacy = pytest.importorskip("spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    assert sentences_from_blocks("One. Two.\nThree.", nlp) == ["One.", "Two.", "Three."]
