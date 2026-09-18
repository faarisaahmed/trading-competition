"""News Hound's picker: follow the coverage.

Philosophy match: the strategy trades stories, so its picker selects names the
wire is actually talking about -- and talking about positively. It uses the
same `lexicon` scorer and the same age/exclusivity weighting the strategy
uses intraday, so the shortlist is exactly "the names my strategy would have
something to say about".

A name with no coverage is unusable to this team no matter how good it looks
on a chart, which is why this picker will happily return fewer symbols than
its cap.
"""

from __future__ import annotations

import math

from ..strategies import lexicon
from ..strategies.base import Param
from .base import Picker, PickerContext, PickResult


class NewsFlowPicker(Picker):
    DESCRIPTION = (
        "Ranks candidates by lexicon-scored news sentiment weighted by coverage volume, "
        "article age and how exclusively each story is about that ticker."
    )
    MIN_HISTORY = 20

    PARAMS = (
        Param("lookback_hours", 72, "news window", minimum=1),
        Param("min_articles", 3, "minimum distinct articles to be considered", minimum=1),
        Param("score_weight_volume", 0.35, "weight on how loud the coverage is",
              minimum=0.0),
        Param("score_weight_sentiment", 0.65, "weight on how positive it is", minimum=0.0),
        Param("require_positive", True, "drop names whose net sentiment is negative"),
        Param("half_life_hours", 12.0, "article age decay", minimum=0.1),
        Param("max_per_sector", 5, "sector concentration cap", minimum=1),
        Param("min_avg_dollar_volume", 10000000.0, "liquidity floor", minimum=0.0),
    )

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        scored: list[tuple[str, float, str]] = []
        rejected: dict[str, str] = {}

        # Normalise coverage volume against the pool so "loud" is relative to
        # the week, not to an absolute article count that drifts with the feed.
        counts = {s: len(ctx.news.get(s.upper(), ())) for s in ctx.candidates}
        busiest = max(counts.values()) if counts else 0

        for sym in ctx.candidates:
            items = ctx.news.get(sym.upper(), ())
            if len(items) < p.min_articles:
                rejected[sym] = f"only {len(items)} article(s)"
                continue
            if ctx.avg_dollar_volume(sym) < p.min_avg_dollar_volume:
                rejected[sym] = "below the liquidity floor"
                continue
            if not ctx.has_history(sym, self.MIN_HISTORY):
                rejected[sym] = "insufficient price history"
                continue

            newest = max(i.ts for i in items)
            num = den = 0.0
            for item in items:
                s = lexicon.score_text(item.text)
                if s == 0.0:
                    continue
                hours = max((newest - item.ts).total_seconds() / 3600.0, 0.0)
                age = math.exp(-math.log(2.0) * hours / max(p.half_life_hours, 0.1))
                exclusivity = 1.0 / math.sqrt(max(len(item.symbols), 1))
                w = age * exclusivity
                num += s * w
                den += w
            if den <= 0:
                rejected[sym] = "no article scored by the lexicon"
                continue
            sentiment = num / den
            if p.require_positive and sentiment <= 0:
                rejected[sym] = f"net sentiment {sentiment:+.2f}"
                continue

            volume_score = (len(items) / busiest) if busiest else 0.0
            score = (
                p.score_weight_sentiment * sentiment
                + p.score_weight_volume * volume_score
            )
            scored.append((
                sym, score,
                f"sentiment {sentiment:+.2f} across {len(items)} articles "
                f"(coverage {volume_score:.0%} of the loudest)",
            ))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        ctx.log.info(
            "newsflow_picker: %d/%d candidates had scoreable coverage -> %s",
            len(scored), len(ctx.candidates), result.describe(),
        )
        return result
