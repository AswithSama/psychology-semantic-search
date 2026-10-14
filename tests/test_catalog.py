import json

import pytest

from psych_search.catalog import load_catalog, parse_pages, sort_key


def test_original_catalog_has_88_sections():
    source, sections = load_catalog()
    assert source == "psychology_book.pdf"
    assert len(sections) == 88
    assert len({s.chapter_id for s in sections}) == 16
    assert sections[0].section_id == "1.1"
    assert sections[-1].section_id == "16.5"


@pytest.mark.parametrize("value,expected", [
    (20, (20, None)), ("38-41", (38, 41)), (" 10 - 13 ", (10, 13)), ("33", (33, None))
])
def test_parse_pages(value, expected):
    assert parse_pages(value) == expected


@pytest.mark.parametrize("value", ["5-3", "0-2", "garbage"])
def test_invalid_ranges(value):
    with pytest.raises(ValueError):
        parse_pages(value)


def test_numeric_section_order():
    assert sorted(["15.10", "15.2", "2.1"], key=sort_key) == ["2.1", "15.2", "15.10"]


def test_duplicate_section_rejected(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text(json.dumps({"sections": [
        {"section_id": "1.1", "heading": "One", "pages": 1},
        {"section_id": "1.1", "heading": "One again", "pages": 2},
    ]}))
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog(path)


def test_overlapping_ranges_rejected(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text(json.dumps({"sections": [
        {"section_id": "1.1", "heading": "One", "pages": "1-4"},
        {"section_id": "1.2", "heading": "Two", "pages": 3},
    ]}))
    with pytest.raises(ValueError, match="overlaps"):
        load_catalog(path)
