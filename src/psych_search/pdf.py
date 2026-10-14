"""PDF page ingestion and notebook-compatible newline normalization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Page:
    label: int
    text: str


def clean_page_text(text: str) -> str:
    """Preserve headings/paragraph markers; merge the notebook's line wraps."""
    return re.sub(r"(?<![.!?])\n(?![A-Z])", " ", text)


def read_pdf(path: str | Path) -> list[Page]:
    from pypdf import PdfReader

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    try:
        labels = reader.page_labels
    except (AttributeError, ValueError):
        labels = []
    output: list[Page] = []
    seen: set[int] = set()
    for index, page in enumerate(reader.pages):
        raw_label = labels[index] if index < len(labels) else str(index + 1)
        try:
            label = int(str(raw_label).strip())
        except ValueError as exc:
            raise ValueError(
                f"Page {index + 1} has non-numeric PDF label {raw_label!r}; "
                "the notebook's section map requires integer labels"
            ) from exc
        if label in seen:
            raise ValueError(f"Duplicate PDF page label {label}")
        seen.add(label)
        output.append(Page(label, clean_page_text(page.extract_text() or "")))
    return output
