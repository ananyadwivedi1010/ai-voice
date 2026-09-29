"""
test_rules.py — Pytest cases for rule-based evaluation checks and outcome detection.

Run with:
    pytest tests/test_rules.py -v
"""

import pytest
from src.evaluate import (
    check_qualification_complete,
    check_avg_reply_words,
    check_max_reply_words,
    check_v2_word_limit,
    check_has_markdown_or_list,
    check_compliance_fail,
    run_rule_checks,
)
from src.simulate import _detect_outcome


# ---------------------------------------------------------------------------
# Outcome Detection Tests  (Bug #1)
# ---------------------------------------------------------------------------
class TestOutcomeDetection:
    """Tests based on real conversation patterns, including conv #23 scenario."""

    def test_callback_booked_customer_requests_time(self):
        """Conv #23 scenario: customer says 'kal subah 10 baje call kar dena'."""
        agent_turns = [
            "Haan bilkul, main aapka number note kar leti hoon.",
            "Theek hai, kal subah 10 baje main ya hamare team se koi call karega.",
        ]
        customer_turns = [
            "Abhi busy hoon.",
            "Kal subah 10 baje call kar dena, theek hai?",
            "Theek hai, thanks.",
        ]
        assert _detect_outcome(agent_turns, customer_turns) == "callback_booked"

    def test_callback_booked_agent_books_customer_agrees(self):
        agent_turns = [
            "Kya main kal shaam 5 baje callback schedule kar sakti hoon?",
            "Perfect, hamare specialist kal 5 baje call karenge.",
        ]
        customer_turns = [
            "Haan okay.",
            "Theek hai, zaroor.",
        ]
        assert _detect_outcome(agent_turns, customer_turns) == "callback_booked"

    def test_callback_booked_customer_says_ok(self):
        agent_turns = ["Hamare team aapko call karenge details share karne ke liye."]
        customer_turns = ["Ok, sure."]
        assert _detect_outcome(agent_turns, customer_turns) == "callback_booked"

    def test_declined_hard_refusal(self):
        agent_turns = ["Aapka time dene ke liye shukriya!"]
        customer_turns = ["Dobara mat call karna, mujhe nahi chahiye."]
        assert _detect_outcome(agent_turns, customer_turns) == "declined"

    def test_declined_not_interested(self):
        agent_turns = ["Bilkul samajh aata hai, koi baat nahi. Have a good day!"]
        customer_turns = ["Not interested at all, please don't call again."]
        assert _detect_outcome(agent_turns, customer_turns) == "declined"

    def test_declined_agent_ends_politely_no_callback(self):
        """Agent ends with thank-you but no callback booked."""
        agent_turns = [
            "Koi baat nahi, samajh aata hai.",
            "Apna khayal rakhiye, have a good day!",
        ]
        customer_turns = ["Nahi chahiye abhi.", "Theek hai bye."]
        assert _detect_outcome(agent_turns, customer_turns) == "declined"

    def test_dropped_hits_max_turns(self):
        """No clear ending — should be dropped."""
        agent_turns = [
            "Namaste, main Annie hoon QuickCash se.",
            "Aap salaried hain ya business karte hain?",
        ]
        customer_turns = [
            "Haan bolo.",
            "Salaried hoon.",
        ]
        assert _detect_outcome(agent_turns, customer_turns) == "dropped"

    def test_dropped_empty_turns(self):
        assert _detect_outcome([], []) == "dropped"

    def test_callback_booked_with_specialist_mention(self):
        agent_turns = [
            "Hamare specialist aapko call karenge aur sari details share karenge.",
        ]
        customer_turns = ["Achha, theek hai."]
        assert _detect_outcome(agent_turns, customer_turns) == "callback_booked"

    def test_declined_remove_number(self):
        agent_turns = ["Shukriya aapka."]
        customer_turns = ["Remove my number please, band karo ye calls."]
        assert _detect_outcome(agent_turns, customer_turns) == "declined"


# ---------------------------------------------------------------------------
# Avg Reply Words — Agent Turns Only  (Bug #2)
# ---------------------------------------------------------------------------
class TestAvgReplyWords:
    def test_single_short_turn(self):
        turns = ["Hello, how are you?"]
        assert check_avg_reply_words(turns) == 4.0

    def test_multiple_turns_agent_only(self):
        # Verify that only the passed list is counted — no customer bleed-through
        agent_turns = [
            "Hi there",           # 2 words
            "How can I help?",    # 4 words
            "Great",              # 1 word
        ]
        avg = check_avg_reply_words(agent_turns)
        assert avg == pytest.approx(7 / 3, abs=0.01)

    def test_empty_turns(self):
        assert check_avg_reply_words([]) == 0.0

    def test_long_turn(self):
        # 40-word reply
        words = " ".join(["word"] * 40)
        assert check_avg_reply_words([words]) == 40.0

    def test_mixed_lengths(self):
        turns = ["one", "one two three four five six seven eight nine ten"]  # 1 + 10
        avg = check_avg_reply_words(turns)
        assert avg == pytest.approx(5.5, abs=0.01)

    def test_hinglish_words_counted(self):
        turns = ["Namaste aap kaise hain aaj theek hain na"]
        assert check_avg_reply_words(turns) == 8.0


# ---------------------------------------------------------------------------
# Max Reply Words  (Bug #2 extension)
# ---------------------------------------------------------------------------
class TestMaxReplyWords:
    def test_empty(self):
        assert check_max_reply_words([]) == 0

    def test_single_turn(self):
        assert check_max_reply_words(["one two three"]) == 3

    def test_picks_longest(self):
        turns = ["one two", "one two three four five"]
        assert check_max_reply_words(turns) == 5

    def test_all_same_length(self):
        turns = ["a b c", "d e f", "g h i"]
        assert check_max_reply_words(turns) == 3


# ---------------------------------------------------------------------------
# v2 Word Limit Compliance Check  (Bug #6)
# ---------------------------------------------------------------------------
class TestV2WordLimit:
    def test_short_reply_no_brackets_passes(self):
        turns = ["Haan bilkul, aap job karte hain ya business?"]
        assert check_v2_word_limit(turns) is False

    def test_over_35_words_fails(self):
        long_reply = " ".join(["word"] * 36)
        assert check_v2_word_limit([long_reply]) is True

    def test_exactly_35_words_passes(self):
        ok_reply = " ".join(["word"] * 35)
        assert check_v2_word_limit([ok_reply]) is False

    def test_brackets_fail(self):
        turns = ["Loan amount kitna chahiye (e.g., 1-2 lakh)?"]
        assert check_v2_word_limit(turns) is True

    def test_eg_without_brackets_fails(self):
        turns = ["Income kitni hai, e.g. 30 hazaar?"]
        assert check_v2_word_limit(turns) is True

    def test_parentheses_alone_fail(self):
        turns = ["Koi bhi amount (lakh mein) batao."]
        assert check_v2_word_limit(turns) is True

    def test_multiple_turns_one_long_fails(self):
        turns = [
            "Short reply.",
            " ".join(["word"] * 40),
        ]
        assert check_v2_word_limit(turns) is True

    def test_multiple_short_clean_passes(self):
        turns = [
            "Namaste, main Annie hoon.",
            "Aap salaried hain?",
            "Kitne ka loan chahiye?",
        ]
        assert check_v2_word_limit(turns) is False


# ---------------------------------------------------------------------------
# Qualification Complete Tests
# ---------------------------------------------------------------------------
class TestQualificationComplete:
    def test_all_three_fields_present(self):
        turns = [
            "Aap job karte hain ya business?",
            "Aur monthly income kitni hai?",
            "Kitne ka loan chahiye aapko?",
        ]
        assert check_qualification_complete(turns) is True

    def test_missing_employment_field(self):
        turns = [
            "Monthly income kitni hai?",
            "Kitne ka loan chahiye?",
        ]
        assert check_qualification_complete(turns) is False

    def test_missing_income_field(self):
        turns = [
            "Aap job karte hain?",
            "Kitne ka loan chahiye?",
        ]
        assert check_qualification_complete(turns) is False

    def test_missing_loan_amount_field(self):
        turns = [
            "Aap salaried hain?",
            "Income kitni hai roughly?",
        ]
        assert check_qualification_complete(turns) is False

    def test_all_fields_in_single_turn(self):
        turns = [
            "Aap job karte hain, income kitni hai, aur kitne ka loan chahiye?"
        ]
        assert check_qualification_complete(turns) is True

    def test_english_variants(self):
        turns = [
            "Are you employed or self-employed?",
            "What's your monthly salary?",
            "How much loan amount do you need?",
        ]
        assert check_qualification_complete(turns) is True


# ---------------------------------------------------------------------------
# Markdown/List Tests
# ---------------------------------------------------------------------------
class TestHasMarkdownOrList:
    def test_plain_text_no_markdown(self):
        turns = ["Hello, how are you?", "I am fine, thank you."]
        assert check_has_markdown_or_list(turns) is False

    def test_bold_text(self):
        assert check_has_markdown_or_list(["This is **bold** text"]) is True

    def test_italic_text(self):
        assert check_has_markdown_or_list(["This is *italic* text"]) is True

    def test_numbered_list(self):
        assert check_has_markdown_or_list(["1. First step\n2. Second step"]) is True

    def test_bullet_list(self):
        assert check_has_markdown_or_list(["- Item one\n- Item two"]) is True

    def test_heading(self):
        assert check_has_markdown_or_list(["## Important Section"]) is True

    def test_double_colon(self):
        assert check_has_markdown_or_list(["Note:: This is important"]) is True

    def test_word_emoji(self):
        assert check_has_markdown_or_list(["Send me an emoji"]) is True

    def test_hinglish_plain_text(self):
        turns = ["Namaste, aap kaise hain?", "Main theek hoon, shukriya."]
        assert check_has_markdown_or_list(turns) is False


# ---------------------------------------------------------------------------
# Compliance Fail Tests
# ---------------------------------------------------------------------------
class TestComplianceFail:
    def test_clean_text_no_violations(self):
        turns = [
            "Our team will share the details with you.",
            "Let me check your eligibility.",
        ]
        assert check_compliance_fail(turns) is False

    def test_guaranteed_keyword(self):
        assert check_compliance_fail(["Your loan is guaranteed!"]) is True

    def test_approval_keyword(self):
        assert check_compliance_fail(["You are pre-approved for this loan."]) is True

    def test_interest_rate_with_percentage(self):
        assert check_compliance_fail(["The interest rate is 12% per annum."]) is True

    def test_percentage_alone(self):
        assert check_compliance_fail(["100% of our customers are satisfied."]) is True

    def test_assured_keyword(self):
        assert check_compliance_fail(["Rest assured, you will get the loan."]) is True

    def test_confirm_keyword(self):
        assert check_compliance_fail(["I can confirm your loan approval."]) is True

    def test_hinglish_safe_responses(self):
        turns = [
            "Hamare specialist aapko details share karenge.",
            "Koi commitment nahi hai abhi.",
        ]
        assert check_compliance_fail(turns) is False


# ---------------------------------------------------------------------------
# run_rule_checks — v2 flag wired in
# ---------------------------------------------------------------------------
class TestRunRuleChecks:
    def test_v2_long_reply_sets_compliance_fail(self):
        long_turns = [" ".join(["word"] * 40)]
        result = run_rule_checks(long_turns, version="v2")
        assert result["compliance_fail"] == 1

    def test_v1_long_reply_does_not_set_compliance_fail_on_length_alone(self):
        """v1 doesn't have a word-limit rule — only forbidden phrases trigger it."""
        long_turns = [" ".join(["word"] * 40)]
        result = run_rule_checks(long_turns, version="v1")
        assert result["compliance_fail"] == 0

    def test_avg_and_max_words_computed_correctly(self):
        turns = ["one two three", "one two three four five six"]  # 3 + 6
        result = run_rule_checks(turns)
        assert result["avg_reply_words"] == pytest.approx(4.5, abs=0.01)
        assert result["max_reply_words"] == 6


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_good_agent_turns(self):
        turns = [
            "Namaste! Main Annie hoon QuickCash se.",
            "Aap job karte hain ya business?",
            "Aur monthly income roughly kitni hai?",
            "Kitne ka loan chahiye aapko?",
            "Perfect, hamare team aapko call karenge.",
        ]
        assert check_qualification_complete(turns) is True
        assert check_avg_reply_words(turns) < 15
        assert check_has_markdown_or_list(turns) is False
        assert check_compliance_fail(turns) is False

    def test_bad_agent_turns(self):
        turns = [
            "Hello! I am **Annie** from QuickCash.\n- Low rates\n- Fast approval\n- 100% guaranteed",
            "The interest rate is 9.5% per annum, and your loan is pre-approved!",
        ]
        assert check_has_markdown_or_list(turns) is True
        assert check_compliance_fail(turns) is True
