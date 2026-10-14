import pytest

from psych_search.catalog import Section
from psych_search.pdf import Page
from psych_search.sections import extract_section, extract_sections, find_heading_index


def test_heading_match_ignores_line_wraps():
    text = "Running header 1.1\n1.1  What\n Is Psychology?\nReal body"
    assert find_heading_index(text, "1.1 What Is Psychology?") == text.index("1.1  What")


def test_heading_not_found():
    assert find_heading_index("No such heading", "1.1 What Is Psychology?") == -1


def test_extract_boundary_page(pages):
    sections = [Section("1.1", "1.1 What Is Psychology?", 1),
                Section("1.2", "1.2 History of Psychology", 2, 3)]
    first, used = extract_section({p.label: p for p in pages}, sections[0], sections[1])
    assert "Psychology is a scientific discipline" in first
    assert "Tail of previous section." in first
    assert "History starts here." not in first
    assert used == [1, 2]


def test_extract_explicit_hard_end(pages):
    sec = Section("1.2", "1.2 History of Psychology", 2, 3)
    result, used = extract_section({p.label: p for p in pages}, sec, None)
    assert "History continues on page three" in result
    assert used == [2, 3]


def test_extract_two_sections_with_metadata(pages):
    sections = [Section("1.1", "1.1 What Is Psychology?", 1),
                Section("1.2", "1.2 History of Psychology", 2, 3)]
    result = extract_sections(pages, sections, "mini.pdf")
    assert len(result) == 2
    assert result[0].metadata["section_id"] == "1.1"
    assert result[1].metadata["end_page"] == 3


def test_same_page_start_and_next_boundary():
    pages = [Page(1, "1.1 First\nFirst body.\n1.2 Second\nSecond body.")]
    parts = [Section("1.1", "1.1 First", 1), Section("1.2", "1.2 Second", 1, 1)]
    docs = extract_sections(pages, parts, "sample.pdf")
    assert "Second body" not in docs[0].text
    assert "Second body" in docs[1].text


def test_strict_missing_section_errors(pages):
    with pytest.raises(ValueError, match="1.1"):
        extract_sections(pages, [Section("1.1", "No heading", 1, 3)], "x.pdf")


def test_non_strict_skips_missing(pages):
    result = extract_sections(pages, [Section("1.1", "No heading", 1, 3)], "x.pdf", strict=False)
    assert result == []


def test_last_section_requires_end_page(pages):
    with pytest.raises(ValueError, match="requires an end page"):
        extract_sections(pages, [Section("1.1", "1.1 What Is Psychology?", 1)], "x.pdf")
