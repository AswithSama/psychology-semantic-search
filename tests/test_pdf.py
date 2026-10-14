import pytest

from psych_search.pdf import clean_page_text, read_pdf


def test_clean_line_wraps_and_preserve_capital_start():
    assert clean_page_text("this is a\ncontinuation") == "this is a continuation"
    assert clean_page_text("Sentence.\nNext sentence") == "Sentence.\nNext sentence"
    assert clean_page_text("TITLE\nIntroduction") == "TITLE\nIntroduction"


def test_missing_pdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / "nope.pdf")


def test_reads_real_synthetic_pdf_pages(tmp_path):
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = tmp_path / "mini.pdf"
    canvas = reportlab.Canvas(str(pdf))
    canvas.drawString(50, 750, "1.1 What Is Psychology?")
    canvas.showPage()
    canvas.drawString(50, 750, "1.2 History of Psychology")
    canvas.save()
    pages = read_pdf(pdf)
    assert [p.label for p in pages] == [1, 2]
    assert "What Is Psychology?" in pages[0].text
