"""Answer matching: numeric scale tolerance, N/A, names, multi-valid."""

from eval.golden import GoldenItem
from eval.matching import is_correct, name_matches, number_matches, parse_number
from rag.generation.schema import RAGAnswer


def test_parse_number_handles_formats():
    assert parse_number("88.1") == 88.1
    assert parse_number("88,100,000") == 88100000
    assert parse_number("(4,432)") == -4432
    assert parse_number("15.63%") == 15.63
    assert parse_number("no number here") is None


def test_number_matches_across_scales():
    assert number_matches(88.1, 88100000)       # millions vs absolute
    assert number_matches(88100000, 88.1)       # symmetric
    assert not number_matches(88.1, 999)


def test_name_matches_normalizes():
    assert name_matches("MITSUI O.S.K. Lines", "MITSUI O.S.K. LINES")
    assert not name_matches("TransUnion", "Petra Diamonds")


def _ans(answer="", refused=False):
    return RAGAnswer(answer=answer, refused=refused)


def test_refusal_correct_when_na_acceptable():
    item = GoldenItem(question="q", answer_type="number", answers=["N/A"], category="hallucination")
    assert is_correct(_ans(answer="N/A", refused=True), item) is True
    # answering a number when only N/A is acceptable = wrong (hallucination)
    assert is_correct(_ans(answer="123"), item) is False


def test_numeric_answer_correct_within_scale():
    item = GoldenItem(question="q", answer_type="number", answers=[88100000], category="retrieval")
    assert is_correct(_ans(answer="88.1"), item) is True
    assert is_correct(_ans(answer="999"), item) is False


def test_multi_valid_answer_either_matches():
    # golden accepts N/A OR 15.63
    item = GoldenItem(question="q", answer_type="number", answers=["N/A", 15.63], category="tricky")
    assert is_correct(_ans(answer="15.63"), item) is True
    assert is_correct(_ans(refused=True, answer="N/A"), item) is True


def test_name_answer_matches_golden_name():
    item = GoldenItem(question="q", answer_type="name", answers=["MITSUI O.S.K. LINES"], category="compare")
    assert is_correct(_ans(answer="The answer is MITSUI O.S.K. Lines."), item) is True
