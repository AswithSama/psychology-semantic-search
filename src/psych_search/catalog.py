"""Explicit textbook section boundaries from the original preprocessing notebook."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).resolve().parent / "data" / "psychology_sections.json"


@dataclass(frozen=True)
class Section:
    section_id: str
    heading: str
    start_page: int
    hard_end_page: int | None = None

    @property
    def chapter_id(self) -> int:
        return int(self.section_id.split(".")[0])


def sort_key(section_id: str) -> tuple[int, int]:
    major, minor = section_id.split(".")
    return int(major), int(minor)


def parse_pages(value: int | str) -> tuple[int, int | None]:
    if isinstance(value, int):
        return value, None
    parts = str(value).split("-", 1)
    start = int(parts[0].strip())
    end = int(parts[1].strip()) if len(parts) == 2 else None
    if start < 1 or (end is not None and end < start):
        raise ValueError(f"Invalid page range: {value}")
    return start, end


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> tuple[str, list[Section]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload["sections"]
    sections = []
    seen: set[str] = set()
    for row in records:
        section_id = row["section_id"]
        if section_id in seen:
            raise ValueError(f"Duplicate section {section_id}")
        seen.add(section_id)
        start, end = parse_pages(row["pages"])
        sections.append(Section(section_id, row["heading"], start, end))
    sections.sort(key=lambda x: sort_key(x.section_id))
    if not sections:
        raise ValueError("Catalog must contain at least one section")
    for left, right in zip(sections, sections[1:]):
        if right.start_page < left.start_page:
            raise ValueError("Sections must have nondecreasing start pages")
        if left.hard_end_page and left.hard_end_page >= right.start_page:
            raise ValueError(f"Explicit range overlaps following section: {left.section_id}")
    return payload.get("source", "unknown.pdf"), sections
