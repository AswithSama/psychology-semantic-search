import pytest

from psych_search.pdf import Page
from psych_search.sections import SectionDocument


@pytest.fixture
def pages():
    return [
        Page(1, "Cover\n1.1 What Is Psychology?\nPsychology is a scientific discipline.\nAdditional information."),
        Page(2, "Tail of previous section.\n1.2 History of Psychology\nHistory starts here."),
        Page(3, "History continues on page three."),
    ]


@pytest.fixture
def section_docs():
    return [
        SectionDocument(
            "One short sentence. Two more sentences follow. Third sentence appears.\nAnother new block.",
            {"section_id": "1.1", "chapter_id": 1, "section_heading": "1.1 Test", "start_page": 1,
             "end_page": 1, "source": "test.pdf"},
        ),
        SectionDocument(
            "A different section. Do not concatenate to the previous section.",
            {"section_id": "1.2", "chapter_id": 1, "section_heading": "1.2 Other", "start_page": 2,
             "end_page": 2, "source": "test.pdf"},
        ),
    ]
