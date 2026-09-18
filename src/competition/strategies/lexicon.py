"""Deterministic finance sentiment scorer -- the News Hound's only "model".

The competition bans LLMs past the coding stage, so headline sentiment is
scored by a checked-in dictionary plus a handful of linguistic rules. This is
a lexicon in the VADER tradition, retuned for market copy, where the useful
signal lives in a small set of highly conventional phrases ("tops estimates",
"cuts guidance", "going concern") rather than in general affect. "Aggressive"
is negative in a novel and positive in an earnings call.

Rules applied on top of the word scores:

  * **Phrases beat words.** "beats estimates" is scored as a unit and the
    component words are then consumed, so "beats" is not double counted.
  * **Negation flips and dampens.** "did not beat" -> -0.74 x the positive
    score, within a three-token window. Financial-press negation is almost
    always local.
  * **Intensifiers and hedges scale.** "sharply higher" amplifies; "slightly
    higher" and "may rise" damp. Hedged copy is genuinely weaker signal.
  * **ALL-CAPS amplifies**, as in press-release headlines.
  * **Length normalisation.** The score is the sum of hits divided by
    sqrt(number of scored tokens), so a long article body cannot outvote a
    decisive headline, and a single strong word in a short headline counts.

Output is clamped to [-1, +1]. It is a *ranking* signal, not a probability.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# multi-word phrases (checked first, highest signal)
# --------------------------------------------------------------------------- #

PHRASES: dict[str, float] = {
    # --- earnings / guidance, the highest-signal copy in the wire ---------
    "beats estimates": 0.85, "beat estimates": 0.85, "beats expectations": 0.85,
    "beat expectations": 0.85, "tops estimates": 0.85, "topped estimates": 0.85,
    "tops expectations": 0.85, "better than expected": 0.80, "above estimates": 0.75,
    "above expectations": 0.75, "earnings beat": 0.80, "revenue beat": 0.80,
    "raises guidance": 0.95, "raised guidance": 0.95, "raises outlook": 0.90,
    "raised outlook": 0.90, "boosts outlook": 0.90, "lifts guidance": 0.90,
    "raises forecast": 0.85, "raises dividend": 0.65, "boosts dividend": 0.65,
    "increases dividend": 0.60, "initiates dividend": 0.55,
    "record revenue": 0.80, "record profit": 0.80, "record earnings": 0.80,
    "record quarter": 0.75, "record high": 0.60, "all time high": 0.60,
    "misses estimates": -0.85, "missed estimates": -0.85, "misses expectations": -0.85,
    "missed expectations": -0.85, "falls short": -0.70, "below estimates": -0.75,
    "below expectations": -0.75, "worse than expected": -0.80, "earnings miss": -0.80,
    "revenue miss": -0.80, "cuts guidance": -0.95, "cut guidance": -0.95,
    "lowers guidance": -0.95, "lowered guidance": -0.95, "cuts outlook": -0.90,
    "lowers outlook": -0.90, "slashes outlook": -1.00, "withdraws guidance": -0.95,
    "suspends guidance": -0.90, "cuts forecast": -0.85, "cuts dividend": -0.85,
    "suspends dividend": -0.90, "eliminates dividend": -0.90,
    "profit warning": -0.95, "warns on": -0.80, "guidance below": -0.85,

    # --- analyst actions --------------------------------------------------
    "price target raised": 0.70, "raises price target": 0.70, "hikes price target": 0.70,
    "upgraded to buy": 0.85, "upgrades to buy": 0.85, "upgraded to overweight": 0.80,
    "initiated with buy": 0.70, "top pick": 0.65, "outperform rating": 0.55,
    "price target cut": -0.70, "cuts price target": -0.70, "lowers price target": -0.70,
    "downgraded to sell": -0.85, "downgrades to sell": -0.85,
    "downgraded to underweight": -0.80, "downgraded to hold": -0.45,
    "underperform rating": -0.55,

    # --- corporate events -------------------------------------------------
    "share buyback": 0.65, "stock buyback": 0.65, "buyback program": 0.65,
    "repurchase program": 0.60, "stock split": 0.35,
    "acquisition of": 0.35, "to acquire": 0.40, "agrees to acquire": 0.45,
    "takeover bid": 0.70, "buyout offer": 0.70, "merger agreement": 0.45,
    "strategic partnership": 0.45, "new contract": 0.50, "contract win": 0.60,
    "wins contract": 0.60, "fda approval": 0.90, "fda approves": 0.90,
    "phase 3 success": 0.90, "positive results": 0.70, "positive data": 0.70,
    "met primary endpoint": 0.85, "breakthrough designation": 0.70,
    "added to index": 0.55, "joins s&p 500": 0.75, "added to s&p 500": 0.75,
    "fda rejects": -0.90, "complete response letter": -0.85,
    "failed to meet": -0.80, "missed primary endpoint": -0.90,
    "phase 3 failure": -0.95, "trial halted": -0.85, "clinical hold": -0.80,
    "removed from index": -0.55, "removed from s&p 500": -0.70,

    # --- distress ---------------------------------------------------------
    "class action": -0.70, "securities fraud": -0.95, "sec investigation": -0.90,
    "sec probe": -0.90, "doj investigation": -0.90, "criminal probe": -0.95,
    "accounting irregularities": -0.95, "material weakness": -0.75,
    "restates earnings": -0.85, "going concern": -0.95, "chapter 11": -1.00,
    "files for bankruptcy": -1.00, "bankruptcy protection": -1.00,
    "delisting notice": -0.90, "trading halted": -0.75, "short seller report": -0.85,
    "short report": -0.80, "ceo resigns": -0.55, "cfo resigns": -0.60,
    "cfo departure": -0.60, "abruptly resigns": -0.75, "steps down": -0.35,
    "layoffs announced": -0.30, "cuts jobs": -0.30, "product recall": -0.70,
    "data breach": -0.65, "cyberattack": -0.60, "antitrust suit": -0.65,
    "antitrust lawsuit": -0.65, "patent invalidated": -0.70,
    "supply chain disruption": -0.55, "production halt": -0.70,

    # --- hedged / neutral constructions (deliberately small) --------------
    "in line with": 0.05, "as expected": 0.05, "mixed results": -0.15,
    "no change": 0.0, "reiterates guidance": 0.15, "maintains guidance": 0.12,
    "reaffirms guidance": 0.20,
}

# --------------------------------------------------------------------------- #
# single words
# --------------------------------------------------------------------------- #

WORDS: dict[str, float] = {
    # positive
    "beats": 0.70, "beat": 0.55, "tops": 0.65, "topped": 0.60, "exceeds": 0.65,
    "exceeded": 0.60, "outperforms": 0.65, "outperformed": 0.60, "outperform": 0.50,
    "surges": 0.80, "surged": 0.80, "soars": 0.85, "soared": 0.85, "jumps": 0.70,
    "jumped": 0.70, "rallies": 0.70, "rallied": 0.70, "climbs": 0.50, "climbed": 0.50,
    "rises": 0.40, "rose": 0.40, "gains": 0.45, "gained": 0.45, "advances": 0.40,
    "higher": 0.35, "upside": 0.50, "rebounds": 0.55, "rebounded": 0.55,
    "recovers": 0.45, "upgrade": 0.75, "upgraded": 0.75, "upgrades": 0.70,
    "bullish": 0.70, "optimistic": 0.55, "confident": 0.45, "strong": 0.55,
    "strength": 0.50, "robust": 0.55, "solid": 0.45, "impressive": 0.60,
    "record": 0.60, "profitable": 0.55, "profit": 0.30, "profits": 0.30,
    "growth": 0.40, "growing": 0.35, "accelerating": 0.55, "expansion": 0.40,
    "approval": 0.70, "approved": 0.70, "approves": 0.70, "breakthrough": 0.75,
    "successful": 0.60, "success": 0.55, "wins": 0.55, "won": 0.50, "awarded": 0.50,
    "landmark": 0.55, "milestone": 0.45, "dividend": 0.25, "buyback": 0.60,
    "repurchase": 0.50, "acquires": 0.35, "partnership": 0.40, "launch": 0.35,
    "launches": 0.35, "unveils": 0.30, "demand": 0.30, "momentum": 0.40,
    "efficiency": 0.30, "turnaround": 0.50, "reinstates": 0.40, "resumes": 0.30,
    "undervalued": 0.55, "attractive": 0.45, "compelling": 0.50, "raises": 0.45,
    "boosts": 0.55, "expands": 0.35, "surpasses": 0.65, "outpaces": 0.55,
    # Past participles matter: negation ("was not raised") can only fire on a
    # term the lexicon actually knows.
    "raised": 0.45, "lifted": 0.45, "lifts": 0.45, "boosted": 0.55,
    "expanded": 0.35, "accelerated": 0.55, "improved": 0.45, "improves": 0.45,
    "beaten": 0.45, "rewarded": 0.40, "secured": 0.40, "extended": 0.20,

    # negative
    "misses": -0.70, "missed": -0.65, "miss": -0.60, "disappoints": -0.75,
    "disappointing": -0.70, "disappointment": -0.65, "underperforms": -0.65,
    "underperform": -0.50, "plunges": -0.85, "plunged": -0.85, "plummets": -0.90,
    "plummeted": -0.90, "tumbles": -0.75, "tumbled": -0.75, "sinks": -0.70,
    "sank": -0.70, "slumps": -0.70, "slumped": -0.70, "falls": -0.45, "fell": -0.45,
    "drops": -0.50, "dropped": -0.50, "declines": -0.45, "declined": -0.45,
    "slides": -0.50, "slid": -0.50, "lower": -0.35, "downside": -0.50,
    "weakness": -0.55, "weak": -0.55, "weaker": -0.55, "soft": -0.40,
    "sluggish": -0.50, "downgrade": -0.75, "downgraded": -0.75, "downgrades": -0.70,
    "bearish": -0.70, "pessimistic": -0.55, "cautious": -0.35, "concerns": -0.45,
    "concerned": -0.40, "worries": -0.50, "fears": -0.55, "warns": -0.70,
    "warning": -0.70, "warned": -0.70, "cuts": -0.55, "cut": -0.50,
    "slashes": -0.80, "slashed": -0.80, "lowers": -0.55, "lowered": -0.50,
    "reduces": -0.40, "loss": -0.55, "losses": -0.55, "deficit": -0.50,
    "writedown": -0.70, "write-down": -0.70, "impairment": -0.65,
    "restructuring": -0.35, "layoffs": -0.35, "bankruptcy": -1.00,
    "insolvency": -1.00, "default": -0.85, "defaults": -0.85, "downturn": -0.60,
    "recession": -0.60, "lawsuit": -0.60, "sued": -0.55, "litigation": -0.50,
    "probe": -0.70, "investigation": -0.65, "subpoena": -0.75, "fraud": -0.95,
    "scandal": -0.85, "recall": -0.65, "halted": -0.60, "suspended": -0.60,
    "delisted": -0.90, "dilution": -0.55, "dilutive": -0.50, "overvalued": -0.55,
    "headwinds": -0.50, "shortfall": -0.65, "delay": -0.45, "delayed": -0.45,
    "delays": -0.45, "shutdown": -0.60, "outage": -0.50, "breach": -0.60,
    "resigns": -0.45, "resignation": -0.45, "ousted": -0.65, "fired": -0.50,
    "concerning": -0.50, "troubled": -0.65, "struggling": -0.60, "stalls": -0.50,
    "reduced": -0.40, "trimmed": -0.35, "scrapped": -0.65, "scraps": -0.65,
    "abandoned": -0.60, "abandons": -0.60, "worsened": -0.60, "worsens": -0.60,
    "deteriorating": -0.65, "missing": -0.50, "halts": -0.60, "suspends": -0.60,
}

NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "cannot", "cant", "wont", "didnt", "doesnt",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "without", "fails",
    "failed", "fail", "unable", "lacks", "lacking", "denies", "denied", "rejects",
    "rejected", "declines", "refuses", "nor", "neither", "barely", "hardly",
})

INTENSIFIERS: dict[str, float] = {
    "very": 1.30, "extremely": 1.50, "sharply": 1.40, "significantly": 1.35,
    "substantially": 1.35, "dramatically": 1.45, "massively": 1.50, "hugely": 1.45,
    "strongly": 1.30, "considerably": 1.30, "markedly": 1.30, "vastly": 1.40,
    "deeply": 1.35, "steeply": 1.40, "surprisingly": 1.25, "unexpectedly": 1.30,
    "record": 1.25, "far": 1.25, "well": 1.20, "much": 1.20, "big": 1.20,
    "major": 1.25, "sheer": 1.30, "crushing": 1.45, "blowout": 1.50,
}

DIMINISHERS: dict[str, float] = {
    "slightly": 0.55, "somewhat": 0.65, "marginally": 0.50, "modestly": 0.65,
    "mildly": 0.60, "partly": 0.70, "partially": 0.70, "a": 1.0, "bit": 0.60,
    "may": 0.65, "might": 0.60, "could": 0.60, "possibly": 0.55, "perhaps": 0.55,
    "reportedly": 0.75, "rumored": 0.55, "rumoured": 0.55, "speculation": 0.55,
    "expects": 0.85, "expected": 0.85, "forecast": 0.85, "seen": 0.80,
    "likely": 0.80, "potential": 0.70, "potentially": 0.65, "if": 0.65,
}

#: Negation reach, in tokens. Financial-press negation is local.
NEGATION_WINDOW = 3
#: How far *after* a term to look for a modifier. Market copy writes both
#: "sharply fell" and "fell sharply", so the window has to be two-sided.
FORWARD_MODIFIER_WINDOW = 2
#: Multiplier applied to a negated score. Not -1.0: "did not beat" is bad but
#: less bad than "missed badly", and the asymmetry is real in market copy.
NEGATION_FACTOR = -0.74
#: Boost for SHOUTED headlines / press-release style.
ALLCAPS_FACTOR = 1.15

_TOKEN = re.compile(r"[a-z0-9&\-\.]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip punctuation that breaks phrase matching, collapse space."""
    t = text.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("'", "")                     # don't -> dont, so NEGATORS match
    t = re.sub(r"[^a-z0-9&\-\.\s]", " ", t)
    return _WS.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalise(text))


def _is_allcaps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 12:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.85


@lru_cache(maxsize=1)
def _phrase_index() -> dict[int, dict[tuple[str, ...], float]]:
    """Phrases bucketed by token length, longest-match-first at scoring time."""
    out: dict[int, dict[tuple[str, ...], float]] = {}
    for phrase, value in PHRASES.items():
        toks = tuple(_TOKEN.findall(phrase))
        if toks:
            out.setdefault(len(toks), {})[toks] = value
    return out


def _modifier(tokens: list[str], start: int, end: int) -> tuple[float, bool]:
    """(multiplier, negated) from the tokens surrounding [start, end).

    Looks backwards for negation and modifiers, and forwards for modifiers
    only -- so both "sharply higher" and "higher sharply" are caught, but a
    trailing "not" (which does not negate in English) is ignored.
    """
    multiplier = 1.0
    negated = False
    for back in range(1, NEGATION_WINDOW + 1):
        j = start - back
        if j < 0:
            break
        prev = tokens[j]
        if prev in NEGATORS:
            negated = True
            break
        if prev in INTENSIFIERS:
            multiplier *= INTENSIFIERS[prev]
        elif prev in DIMINISHERS:
            multiplier *= DIMINISHERS[prev]
    for fwd in range(FORWARD_MODIFIER_WINDOW):
        j = end + fwd
        if j >= len(tokens):
            break
        nxt = tokens[j]
        if nxt in INTENSIFIERS:
            multiplier *= INTENSIFIERS[nxt]
        elif nxt in DIMINISHERS:
            multiplier *= DIMINISHERS[nxt]
    return multiplier, negated


def _hits(text: str) -> list[tuple[str, float]]:
    """Scored terms in order, longest phrase first, each counted once."""
    tokens = tokenize(text)
    if not tokens:
        return []
    index = _phrase_index()
    max_len = max(index) if index else 1
    out: list[tuple[str, float]] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        # Longest match wins: "beats expectations" before "beats".
        for length in range(min(max_len, n - i), 0, -1):
            table = index.get(length)
            if not table:
                continue
            span = tuple(tokens[i:i + length])
            value = table.get(span)
            if value is None:
                continue
            mult, neg = _modifier(tokens, i, i + length)
            v = value * mult * (NEGATION_FACTOR if neg else 1.0)
            out.append((" ".join(span), v))
            i += length
            matched = True
            break
        if matched:
            continue
        tok = tokens[i]
        value = WORDS.get(tok)
        if value is not None:
            mult, neg = _modifier(tokens, i, i + 1)
            out.append((tok, value * mult * (NEGATION_FACTOR if neg else 1.0)))
        i += 1
    return out


@lru_cache(maxsize=4096)
def score_text(text: str) -> float:
    """Sentiment of one headline/summary in [-1, +1]. Deterministic and cached."""
    if not text or not text.strip():
        return 0.0
    hits = _hits(text)
    if not hits:
        return 0.0
    # sqrt(number of hits): one decisive phrase keeps its full weight, while a
    # pile of mild terms still accumulates but with diminishing returns, so a
    # long article body cannot outvote a decisive headline.
    score = sum(v for _t, v in hits) / math.sqrt(len(hits))
    if _is_allcaps(text):
        score *= ALLCAPS_FACTOR
    return max(min(score, 1.0), -1.0)


def explain(text: str) -> dict:
    """Which terms fired and how much -- used by `comp explain-news`."""
    hits = _hits(text)
    return {
        "score": round(score_text(text), 4),
        "terms": [(t, round(v, 4)) for t, v in hits],
        "n_hits": len(hits),
        "allcaps": _is_allcaps(text),
    }


def vocabulary_size() -> tuple[int, int]:
    return len(PHRASES), len(WORDS)
