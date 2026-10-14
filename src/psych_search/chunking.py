"""Notebook-derived hierarchical chunking: section -> sentences -> bounded chunks.

Core invariants: never cross section boundaries, default 1000 characters,
2-sentence overlap, and punctuation/semicolon/comma fallback for long sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .sections import SectionDocument

FALLBACK_SEPARATORS = [r"(?<=[.!?]) (?=[A-Z])", "; ", ", "]


@dataclass(frozen=True)
class Chunk:
    text: str
    section_index: int
    chunk_index_in_section: int
    metadata: dict[str, str | int]

    @property
    def id(self) -> str:
        return f"sec{self.metadata['section_id']}_chunk{self.chunk_index_in_section}"


def split_oversize(text: str, max_size: int, separators: Sequence[str] = FALLBACK_SEPARATORS) -> list[str]:
    """Recursively split oversize units, hard-cutting only as a last resort."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    if len(text) <= max_size:
        return [text] if text.strip() else []
    if not separators:
        return [text[i:i + max_size].strip() for i in range(0, len(text), max_size)
                if text[i:i + max_size].strip()]
    separator = separators[0]
    if separator.startswith("(?"):
        pieces = re.split(separator, text)
    else:
        # Keep punctuation; do not drop semantic content while shortening chunks.
        parts = text.split(separator)
        pieces = [part + (separator.rstrip() if i < len(parts) - 1 else "")
                  for i, part in enumerate(parts)]
    if len(pieces) <= 1:
        return split_oversize(text, max_size, separators[1:])
    output: list[str] = []
    for piece in pieces:
        if piece.strip():
            output.extend(split_oversize(piece.strip(), max_size, separators[1:]))
    return output


def sentences_from_blocks(text: str, nlp: object) -> list[str]:
    """Preserve newline-separated textbook blocks, splitting within each block."""
    sentences = []
    for block in text.split("\n"):
        if not block.strip():
            continue
        for sentence in nlp(block.strip()).sents:
            if sentence.text.strip():
                sentences.append(sentence.text.strip())
    return sentences


def pack_sentences(
    sentences: Sequence[str],
    max_size: int = 1000,
    overlap_sentences: int = 2,
) -> list[str]:
    """Greedy sentence packing with carry-over overlap and size enforcement.

    When a carry-over would fill the entire budget, drop the oldest overlap
    unit(s) until the next unit fits. This prevents the source notebook's
    potential overflow and guarantees forward progress.
    """
    if max_size < 1 or overlap_sentences < 0:
        raise ValueError("max_size must be >=1; overlap_sentences must be >=0")
    units = [piece for sentence in sentences
             for piece in split_oversize(sentence.strip(), max_size) if piece]
    output: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join([*current, unit])
        if current and len(candidate) > max_size:
            output.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            while current and len(" ".join([*current, unit])) > max_size:
                current.pop(0)
        current.append(unit)
    if current:
        tail = " ".join(current)
        if not output or tail != output[-1]:
            output.append(tail)
    return output


def chunk_sections(
    documents: Sequence[SectionDocument],
    max_size: int = 1000,
    overlap_sentences: int = 2,
) -> list[Chunk]:
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    result: list[Chunk] = []
    for section_index, document in enumerate(documents):
        sentences = sentences_from_blocks(document.text, nlp)
        texts = pack_sentences(sentences, max_size, overlap_sentences)
        for chunk_index, text in enumerate(texts):
            result.append(
                Chunk(text, section_index, chunk_index, dict(document.metadata))
            )
    return result
