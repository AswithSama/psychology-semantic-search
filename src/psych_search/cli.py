"""CLI for indexing the original textbook and querying its Chroma collection."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .catalog import DEFAULT_CATALOG
from .pipeline import index_pdf
from .retrieval import SearchService
from .settings import AppSettings
from .store import ChromaStore


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Psychology textbook semantic search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Index a textbook PDF into persistent ChromaDB")
    ingest.add_argument("pdf", help="Path to the original psychology textbook PDF")
    ingest.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ingest.add_argument("--allow-missing", action="store_true", help="Skip sections with missing headings")

    query = subparsers.add_parser("search", help="Retrieve matching textbook passages")
    query.add_argument("query", help="Semantic query")
    query.add_argument("--k", type=int, default=5)
    query.add_argument("--chapter", type=int)
    query.add_argument("--section")

    subparsers.add_parser("stats", help="Get number of indexed passages")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    config = AppSettings()
    try:
        if args.command == "ingest":
            report = index_pdf(args.pdf, args.catalog, settings=config, strict=not args.allow_missing)
            print(json.dumps(asdict(report), indent=2))
        elif args.command == "search":
            hits = SearchService(ChromaStore(config)).search(
                args.query, args.k, chapter=args.chapter, section=args.section
            )
            print(json.dumps(hits, indent=2))
        else:
            print(json.dumps({"indexed_chunks": ChromaStore(config).count()}, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        create_parser().exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
