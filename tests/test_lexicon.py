"""The News Hound's sentiment scorer.

It is a checked-in dictionary plus linguistic rules -- no LLM at runtime --
so its behaviour is fully testable and must stay fixed for the season.
"""

from __future__ import annotations

import pytest

from competition.strategies import lexicon

POSITIVE = [
    "Apple beats estimates and raises guidance for the fourth quarter",
    "Nvidia surges to a record high after blowout earnings",
    "Pfizer's drug met primary endpoint in phase 3 trial",
    "MICROSOFT ANNOUNCES RECORD QUARTERLY REVENUE AND RAISES DIVIDEND",
    "Board announces $10 billion share buyback program",
    "Company upgraded to buy on accelerating growth",
    "Shares rose sharply on heavy volume",
    "Regulator grants FDA approval for the new therapy",
]

NEGATIVE = [
    "Apple misses estimates, cuts guidance amid weak demand",
    "Boeing plunges as the FAA opens an investigation into production",
    "Moderna trial halted after clinical hold",
    "Company files for bankruptcy protection amid going concern doubts",
    "CFO abruptly resigns as SEC probe widens",
    "Analysts slightly lower price target on Ford",
    "Shares fell modestly in quiet trading",
    "Retailer issues a profit warning for the quarter",
]

NEUTRAL = [
    "The weather in Cupertino was pleasant today",
    "Company to present at an industry conference",
    "Board schedules the quarterly earnings call",
    "",
    "   ",
]


@pytest.mark.parametrize("text", POSITIVE)
def test_positive_headlines_score_positive(text):
    assert lexicon.score_text(text) > 0.15, text


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative_headlines_score_negative(text):
    assert lexicon.score_text(text) < -0.15, text


@pytest.mark.parametrize("text", NEUTRAL)
def test_neutral_text_scores_near_zero(text):
    assert abs(lexicon.score_text(text)) < 0.15, text


def test_scores_are_bounded():
    extreme = " ".join(["beats estimates and raises guidance"] * 20)
    assert lexicon.score_text(extreme) <= 1.0
    grim = " ".join(["files for bankruptcy amid securities fraud"] * 20)
    assert lexicon.score_text(grim) >= -1.0


# --------------------------------------------------------------------------- #
# the linguistic rules
# --------------------------------------------------------------------------- #


def test_negation_flips_a_phrase_not_just_a_word():
    """The bug this catches: negation reaching words but not phrases."""
    assert lexicon.score_text("Tesla beat expectations this quarter") > 0
    assert lexicon.score_text("Tesla did not beat expectations this quarter") < 0
    assert lexicon.score_text("Company failed to beat estimates") < 0


@pytest.mark.parametrize("neg", ["did not", "failed to", "was unable to",
                                 "never managed to"])
def test_various_negators(neg):
    assert lexicon.score_text(f"The company {neg} beat estimates") < 0


def test_negation_is_local_not_sentencewide():
    # "not" three tokens back negates; far away it must not.
    assert lexicon.score_text("Guidance was not raised") < 0.2
    far = ("There was not any doubt in the long and detailed report that the "
           "company surges to a record")
    assert lexicon.score_text(far) > 0


def test_modifiers_work_in_both_orders():
    """Market copy writes both 'sharply higher' and 'higher sharply'."""
    plain = lexicon.score_text("Shares fell in trading")
    damped = lexicon.score_text("Shares fell modestly in trading")
    amplified = lexicon.score_text("Shares fell sharply in trading")
    assert abs(damped) < abs(plain) < abs(amplified)
    pre_damped = lexicon.score_text("Shares modestly fell in trading")
    assert abs(pre_damped) < abs(plain)


def test_hedging_reduces_conviction():
    firm = lexicon.score_text("Retailer sees upside")
    hedged = lexicon.score_text("Retailer may possibly see potential upside")
    assert 0 < hedged < firm


def test_phrases_beat_their_component_words():
    """'beats estimates' must be scored once as a unit, not twice."""
    e = lexicon.explain("Company beats estimates")
    terms = [t for t, _v in e["terms"]]
    assert terms == ["beats estimates"]
    assert e["n_hits"] == 1


def test_longest_phrase_wins():
    e = lexicon.explain("Company raises guidance")
    assert [t for t, _v in e["terms"]] == ["raises guidance"]


def test_allcaps_amplifies():
    quiet = lexicon.score_text("Company announces record quarterly revenue")
    shouty = lexicon.score_text("COMPANY ANNOUNCES RECORD QUARTERLY REVENUE")
    assert shouty > quiet


def test_short_allcaps_tickers_do_not_trigger_amplification():
    # A headline that merely contains a ticker is not a shouted headline.
    a = lexicon.score_text("AAPL beats estimates")
    b = lexicon.score_text("Apple beats estimates")
    assert a == pytest.approx(b, abs=0.001)


def test_length_normalisation_keeps_a_decisive_headline_strong():
    headline = "Company cuts guidance"
    padded = headline + " " + " ".join(["the market was open today"] * 20)
    assert lexicon.score_text(padded) == pytest.approx(
        lexicon.score_text(headline), abs=0.01)


def test_a_wall_of_mild_words_still_accumulates():
    one = lexicon.score_text("growth")
    many = lexicon.score_text("growth growth growth growth growth growth")
    assert many > one


def test_mixed_signals_partially_cancel():
    mixed = lexicon.score_text(
        "Company beats estimates but cuts guidance and warns on the quarter")
    good = lexicon.score_text("Company beats estimates")
    assert mixed < good


# --------------------------------------------------------------------------- #
# mechanics
# --------------------------------------------------------------------------- #


def test_determinism_and_caching():
    text = "Company raises guidance and beats estimates"
    assert lexicon.score_text(text) == lexicon.score_text(text)


def test_punctuation_and_unicode_are_handled():
    assert lexicon.score_text("Apple's results: beats estimates!") > 0
    assert lexicon.score_text("Apple’s results — beats estimates") > 0


def test_explain_reports_the_working():
    e = lexicon.explain("Apple misses estimates, cuts guidance amid weak demand")
    assert e["score"] < 0
    terms = dict(e["terms"])
    assert "misses estimates" in terms and "cuts guidance" in terms
    assert all(isinstance(v, float) for v in terms.values())


def test_vocabulary_is_substantial():
    phrases, words = lexicon.vocabulary_size()
    assert phrases > 100 and words > 150


def test_tokenize_and_normalise():
    assert lexicon.tokenize("Don't stop!") == ["dont", "stop"]
    assert lexicon.normalise("A  B\tC") == "a b c"


def test_unknown_words_are_ignored_not_guessed():
    assert lexicon.score_text("Zorblax frobnicated the widget") == 0.0
