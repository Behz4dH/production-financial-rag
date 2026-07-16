"""Answer matching: numeric scale tolerance, N/A, names, multi-valid."""

from eval.golden import GoldenItem
from eval.matching import is_correct, name_matches, number_matches, parse_number, parse_numbers
from rag.generation.schema import RAGAnswer


def test_parse_number_handles_formats():
    assert parse_number("88.1") == 88.1
    assert parse_number("88,100,000") == 88100000
    assert parse_number("(4,432)") == -4432
    assert parse_number("15.63%") == 15.63
    assert parse_number("no number here") is None


def test_parse_numbers_scopes_paren_negative_to_matched_number():
    # An unrelated parenthetical elsewhere in the text must not flip the
    # sign of a plain number that isn't itself paren-wrapped.
    assert parse_numbers("Net income was 88,100,000 (see Note).") == [88100000.0]


def test_number_matches_across_scales():
    assert number_matches(88.1, 88100000)       # millions vs absolute
    assert number_matches(88100000, 88.1)       # symmetric
    assert not number_matches(88.1, 999)


def test_number_matches_upscale_can_be_disallowed():
    # Inflating a small prediction to meet a large golden needs allow_upscale;
    # a fully-written-out prediction against an abbreviated golden never does.
    assert number_matches(88.1, 88100000, allow_upscale=False) is False
    assert number_matches(88100000, 88.1, allow_upscale=False) is True
    assert number_matches(88100000, 88100000, allow_upscale=False) is True


def test_number_matches_zero_golden_no_false_positive():
    assert number_matches(0, 0) is True
    # previously true due to the 1e-9 scale fallback denom=1.0
    assert number_matches(5000, 0) is False


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
    assert is_correct(_ans(answer="$88.1 million"), item) is True
    assert is_correct(_ans(answer="999 million"), item) is False


def test_bare_number_is_not_rescued_by_scale_shift():
    # "total assets were $6,744,215" (thousands, unstated) against golden
    # 6,744,215,000 is a wrong answer as written — a scale shift is accepted
    # only when the prediction states the scale it's shifted by.
    item = GoldenItem(question="q", answer_type="number",
                      answers=[6744215000], category="retrieval")
    assert is_correct(_ans(answer="Total assets were $6,744,215."), item) is False
    assert is_correct(_ans(answer="$6,744,215 thousand"), item) is True
    assert is_correct(_ans(answer="$6.74 billion"), item) is True
    assert is_correct(_ans(answer="6,744,215,000"), item) is True  # exact, no cue needed


def test_full_digit_answer_matches_abbreviated_golden_without_cue():
    # The golden may be authored in shorthand; a fully-written-out prediction
    # is unambiguous and needs no scale word.
    item = GoldenItem(question="q", answer_type="number", answers=[88.1], category="retrieval")
    assert is_correct(_ans(answer="88,100,000"), item) is True


def test_multi_valid_answer_either_matches():
    # golden accepts N/A OR 15.63
    item = GoldenItem(question="q", answer_type="number", answers=["N/A", 15.63], category="tricky")
    assert is_correct(_ans(answer="15.63"), item) is True
    assert is_correct(_ans(refused=True, answer="N/A"), item) is True


def test_name_answer_matches_golden_name():
    item = GoldenItem(question="q", answer_type="name", answers=["MITSUI O.S.K. LINES"], category="compare")
    assert is_correct(_ans(answer="The answer is MITSUI O.S.K. Lines."), item) is True


def test_numeric_answer_matches_via_prose_not_just_first_number():
    # First number in the text (2023, the fiscal year) is not the answer;
    # the real figure (88.1 -> 88.1M) appears later in the sentence.
    item = GoldenItem(question="q", answer_type="number", answers=[88100000], category="retrieval")
    assert is_correct(_ans(answer="In fiscal year 2023, net income was $88.1 million"), item) is True


def test_numeric_answer_matches_string_typed_golden():
    # Golden value authored as a JSON string must still be coerced and compared.
    item = GoldenItem(question="q", answer_type="number", answers=["88100000"], category="retrieval")
    assert is_correct(_ans(answer="88.1 million"), item) is True


def test_name_match_bridges_aliases_of_same_entity():
    """'MOL Group' vs golden 'MITSUI O.S.K. LINES' share no tokens, but both
    are names of the same filer — alias groups from doc_metadata must bridge
    them, else the eval understates whichever pipeline answers with an alias."""
    from eval.golden import GoldenItem
    from eval.matching import is_correct
    from rag.generation.schema import RAGAnswer

    groups = [["Mitsui O.S.K. Lines, Ltd.", "MOL", "Mitsui O.S.K. Lines"]]
    item = GoldenItem(question="q?", answer_type="name",
                      answers=["MITSUI O.S.K. LINES"])
    pred = RAGAnswer(answer="MOL Group", refused=False)
    assert is_correct(pred, item) is False                      # without aliases
    assert is_correct(pred, item, alias_groups=groups) is True  # bridged

    # A name matching NO group member must not be bridged.
    stranger = RAGAnswer(answer="Tradition", refused=False)
    assert is_correct(stranger, item, alias_groups=groups) is False
