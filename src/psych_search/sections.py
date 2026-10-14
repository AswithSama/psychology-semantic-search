"""Jump-based extraction: sections are defined by known textbook page starts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .catalog import Section
from .pdf import Page


@dataclass(frozen=True)
class SectionDocument:
    text: str
    metadata: dict[str, str | int]


def find_heading_index(text: str, heading: str) -> int:
    """Match the full heading regardless of whitespace, returning the real offset.

    Unlike the notebook's first-token approximation, this handles repeated section
    numbers in page running headers without truncating from the wrong location.
    """
    tokens = re.findall(r"\S+", heading)
    if not tokens:
        return -1
    pattern = r"\s*".join(re.escape(token) for token in tokens)
    match = re.search(pattern, text, re.IGNORECASE)
    return match.start() if match else -1


def extract_section(
    pages: dict[int, Page],
    section: Section,
    next_section: Section | None,
) -> tuple[str, list[int]]:
    """Take heading-to-next-heading spans with explicit last-section page ranges.

    Only the starting and boundary pages are searched for headings. Entire
    intermediate pages are included without section detection, as in the notebook.
    """
    first = pages.get(section.start_page)
    if first is None:
        return "", []
    start = find_heading_index(first.text, section.heading)
    if start < 0:
        return "", []
    remaining = first.text[start:]
    used = [section.start_page]
    if next_section is not None:
        same_page_boundary = find_heading_index(remaining, next_section.heading)
        if same_page_boundary >= 0:
            return remaining[:same_page_boundary].strip(), used
    parts = [remaining.rstrip()]
    boundary_page = next_section.start_page if next_section else None
    stop_page = section.hard_end_page if section.hard_end_page is not None else boundary_page
    if stop_page is None:
        raise ValueError(f"Last section {section.section_id} requires an end page")
    # Hard ranges include their end page; otherwise stop before the next heading's page.
    ending = (stop_page if section.hard_end_page is not None else stop_page - 1)
    for label in range(section.start_page + 1, ending + 1):
        if label in pages:
            parts.append(pages[label].text.rstrip())
            used.append(label)
    if section.hard_end_page is None and boundary_page is not None and next_section:
        boundary = pages.get(boundary_page)
        if boundary is not None:
            cut = find_heading_index(boundary.text, next_section.heading)
            # The original notebook conservatively omits an unrecognized boundary page.
            if cut >= 0:
                parts.append(boundary.text[:cut].rstrip())
                used.append(boundary_page)
    return "\n".join(p for p in parts if p).strip(), used


def extract_sections(
    pages: Sequence[Page],
    sections: Sequence[Section],
    source: str,
    strict: bool = True,
) -> list[SectionDocument]:
    lookup = {page.label: page for page in pages}
    output: list[SectionDocument] = []
    missing: list[str] = []
    for position, section in enumerate(sections):
        successor = sections[position + 1] if position + 1 < len(sections) else None
        text, used = extract_section(lookup, section, successor)
        if not text:
            missing.append(section.section_id)
            continue
        output.append(
            SectionDocument(
                text=text,
                metadata={
                    "chapter_id": section.chapter_id,
                    "section_id": section.section_id,
                    "section_heading": section.heading,
                    "start_page": section.start_page,
                    "end_page": max(used),
                    "source": source,
                },
            )
        )
    if missing and strict:
        raise ValueError(
            "Could not extract headings for sections: " + ", ".join(missing)
            + ". Verify the PDF edition and section page labels, or use --allow-missing."
        )
    return output
