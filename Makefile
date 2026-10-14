.PHONY: install test run stats ingest
install:
	python -m pip install -e '.[test]'

test:
	pytest -q

run:
	uvicorn psych_search.api:app --reload --port 8000

stats:
	psych-search stats

ingest:
	@test -f data/psychology_book.pdf || (echo 'Place the textbook at data/psychology_book.pdf first' && exit 1)
	psych-search ingest data/psychology_book.pdf
