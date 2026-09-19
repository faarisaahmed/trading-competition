"""Q-Learner's picker: a LinUCB contextual bandit.

Philosophy match: the strategy learns a policy from reward rather than from a
theory of markets, so its picker does the same thing one level up. Each
candidate is an arm described by a feature vector; the bandit maintains a
ridge-regression estimate of expected reward per feature and selects by upper
confidence bound, so it explores names it knows little about and exploits the
ones whose features have paid.

LinUCB (Li et al., 2010), one shared model across arms
------------------------------------------------------
    A = lambda*I + sum x x^T        (d x d,  d = number of features)
    b = sum r x                     (d,)
    theta = A^-1 b                  (the learned reward weights)
    ucb(x)  = theta.x + alpha * sqrt(x^T A^-1 x)

The model is shared across symbols rather than per-arm, for the same reason
the Q-table is symbol-agnostic: a three-week competition never gives one
ticker enough samples. Features are standardised across the pool each call,
so the model transfers from Round 2's 300-name pool to Round 3's ten dealt
cards without retraining.

Reward is the realised forward return of a picked name over
`reward_horizon_days`, fed back by the engine through `on_fills`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..strategies.base import Param
from ..util import indicators as ind
from .base import Picker, PickerContext, PickResult

#: Feature extractors, keyed by the names usable in `feature_set`.
FEATURES: dict[str, str] = {
    "momentum_21": "21-day return",
    "momentum_63": "63-day return",
    "vol_21": "21-day realised volatility",
    "rel_volume": "recent volume vs its own 20-day average",
    "dist_from_high": "distance below the 63-day high",
    "atr_pct": "ATR as a fraction of price",
    "reversal_5": "negated 5-day return (short-term reversal)",
    "trend_r2": "linearity of the 63-day path",
}


def _bar_date(bar) -> Any:
    """A daily bar's calendar date. `Bar.ts` is the bar's open, always UTC."""
    ts = getattr(bar, "ts", None)
    if ts is None:
        raise AttributeError(
            f"{type(bar).__name__} has no `ts`; cannot align it by date"
        )
    return ts.date() if hasattr(ts, "date") else ts


class _WindowCtx:
    """The slice of `PickerContext` that `raw_features` actually touches.

    Pretraining reuses the live feature code rather than reimplementing it --
    a second implementation is a second thing to drift.
    """

    __slots__ = ("_bars",)

    def __init__(self, bars: Sequence[Any]):
        self._bars = bars

    def bars(self, _symbol: str) -> Sequence[Any]:
        return self._bars


class BanditPicker(Picker):
    DESCRIPTION = (
        "LinUCB contextual bandit over standardised price/volume features, sharing one "
        "ridge model across all symbols so the learning transfers between rounds."
    )
    MIN_HISTORY = 70

    PARAMS = (
        Param("ucb_alpha", 1.2, "exploration width; 0 = pure exploitation", minimum=0.0),
        Param("ridge_lambda", 1.0, "ridge regularisation on A", minimum=1e-6),
        Param("feature_set",
              ["momentum_21", "momentum_63", "vol_21", "rel_volume", "dist_from_high",
               "atr_pct"],
              "which features describe an arm"),
        Param("reward_horizon_days", 3, "horizon over which a pick is judged", minimum=1),
        Param("min_pulls_before_exploit", 2, "picks before UCB starts trusting theta",
              minimum=0),
        Param("max_per_sector", 5, "sector concentration cap", minimum=1),
        Param("min_avg_dollar_volume", 10000000.0, "liquidity floor", minimum=0.0),
    )

    def __init__(self, team, params=None):
        super().__init__(team, params)
        self.d = len(self.p.feature_set) + 1          # +1 for the bias term
        self.A = np.eye(self.d) * float(self.p.ridge_lambda)
        self.b = np.zeros(self.d)
        self.pulls = 0
        self.total_reward = 0.0
        #: symbol -> the feature vector it was picked with, awaiting a reward.
        self.pending: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ #
    # features
    # ------------------------------------------------------------------ #

    def raw_features(self, ctx: PickerContext, symbol: str) -> dict[str, float] | None:
        bars = ctx.bars(symbol)
        closes = [b.close for b in bars]
        if len(closes) < self.MIN_HISTORY:
            return None
        price = closes[-1]
        if price <= 0:
            return None
        vols = [b.volume for b in bars[-21:]]
        avg_vol = (sum(vols) / len(vols)) if vols else 0.0
        high63 = max(closes[-63:]) if len(closes) >= 63 else max(closes)
        return {
            "momentum_21": ind.roc(closes, 21),
            "momentum_63": ind.roc(closes, 63),
            "vol_21": ind.realized_vol(closes, 21),
            "rel_volume": (bars[-1].volume / avg_vol) if avg_vol > 0 else 1.0,
            "dist_from_high": (price / high63 - 1.0) if high63 > 0 else 0.0,
            "atr_pct": ind.atr_pct(bars, 14),
            "reversal_5": -ind.roc(closes, 5),
            "trend_r2": ind.r_squared(closes, 63),
        }

    def _standardise(self, table: dict[str, dict[str, float]]) -> dict[str, list[float]]:
        """Z-score each feature across the pool, then append a bias term.

        Standardising per call is what makes the learned weights portable: a
        21-day return of +8% means something different in a calm week than in
        a violent one, but "one sigma better than the pool" does not.
        """
        names = list(self.p.feature_set)
        out: dict[str, list[float]] = {}
        stats: dict[str, tuple[float, float]] = {}
        for feat in names:
            vals = np.array([row.get(feat, 0.0) for row in table.values()], dtype=float)
            vals = vals[np.isfinite(vals)]
            mu = float(vals.mean()) if vals.size else 0.0
            sd = float(vals.std(ddof=0)) if vals.size > 1 else 0.0
            stats[feat] = (mu, sd if sd > 1e-9 else 1.0)
        for sym, row in table.items():
            vec = []
            for feat in names:
                mu, sd = stats[feat]
                z = (float(row.get(feat, mu)) - mu) / sd
                vec.append(max(min(z, 4.0), -4.0))    # clip outliers
            vec.append(1.0)                            # bias
            out[sym] = vec
        return out

    # ------------------------------------------------------------------ #
    # LinUCB
    # ------------------------------------------------------------------ #

    @property
    def theta(self) -> np.ndarray:
        try:
            return np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def ucb(self, x: Sequence[float]) -> tuple[float, float, float]:
        """(ucb, mean, bonus) for one arm."""
        xv = np.asarray(x, dtype=float)
        try:
            a_inv_x = np.linalg.solve(self.A, xv)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0
        mean = float(self.theta @ xv)
        var = float(xv @ a_inv_x)
        bonus = float(self.p.ucb_alpha) * math.sqrt(max(var, 0.0))
        if self.pulls < int(self.p.min_pulls_before_exploit):
            # Before the model has seen anything, `mean` is noise. Rank purely
            # by uncertainty so the first rounds are honest exploration.
            return bonus, mean, bonus
        return mean + bonus, mean, bonus

    def update(self, x: Sequence[float], reward: float) -> None:
        xv = np.asarray(x, dtype=float)
        self.A += np.outer(xv, xv)
        self.b += reward * xv
        self.pulls += 1
        self.total_reward += reward

    # ------------------------------------------------------------------ #

    def pick(self, ctx: PickerContext) -> PickResult:
        p = self.p
        table: dict[str, dict[str, float]] = {}
        rejected: dict[str, str] = {}
        for sym in ctx.candidates:
            if ctx.avg_dollar_volume(sym) < p.min_avg_dollar_volume:
                rejected[sym] = "below the liquidity floor"
                continue
            row = self.raw_features(ctx, sym)
            if row is None:
                rejected[sym] = "insufficient history"
                continue
            table[sym] = row
        if not table:
            ctx.log.warning("bandit_picker: no candidate had usable features")
            return PickResult((), rejected=rejected)

        vectors = self._standardise(table)
        scored: list[tuple[str, float, str]] = []
        for sym, vec in vectors.items():
            u, mean, bonus = self.ucb(vec)
            scored.append((sym, u, f"ucb {u:+.3f} (mean {mean:+.3f} + bonus {bonus:.3f})"))

        result = self._rank(scored, ctx, max_per_sector=p.max_per_sector)
        result.rejected = rejected
        # Remember the features of what we picked so the reward can be
        # attributed to the right vector when it arrives.
        self.pending = {s: vectors[s] for s in result.symbols if s in vectors}
        result.extra["pulls"] = self.pulls
        result.extra["theta"] = [round(v, 5) for v in self.theta.tolist()]
        ctx.log.info(
            "bandit_picker: %d arms, %d pulls so far, alpha=%.2f -> %s",
            len(vectors), self.pulls, p.ucb_alpha, result.describe(),
        )
        return result

    def on_fills(self, results: Mapping[str, float]) -> None:
        """Feed realised returns back into the model."""
        for sym, reward in results.items():
            vec = self.pending.pop(sym.upper(), None)
            if vec is None:
                continue
            # Clip: one -40% day should inform the model, not dominate it.
            self.update(vec, max(min(float(reward), 0.25), -0.25))
        if results:
            self.log.info(
                "bandit_picker: %d rewards applied, %d pulls, mean reward %.4f",
                len(results), self.pulls, self.total_reward / max(self.pulls, 1),
            )

    # ------------------------------------------------------------------ #
    # offline pretraining
    # ------------------------------------------------------------------ #

    def pretrain(
        self,
        daily_bars: Mapping[str, Sequence[Any]],
        *,
        max_symbols: int = 12,
        passes: int = 1,
        horizon: int | None = None,
        log=None,
    ) -> dict[str, Any]:
        """Learn the reward weights offline, by replaying history on-policy.

        This mirrors the live loop exactly rather than fitting the ridge model
        on every symbol: on each historical date it standardises the pool,
        picks the top `max_symbols` by UCB, and updates only on *those* --
        the same bandit feedback it would have received had it been running.
        Fitting on the whole cross-section would be easier and would converge
        faster, but it would hand the picker information it could never have
        gathered itself, which is a different model than the one that then has
        to keep learning online.

        Returns a summary dict for the CLI to print.
        """
        horizon = int(horizon or self.p.reward_horizon_days)
        logger = log or self.log

        # Index each symbol by date so the pool can be walked in lockstep even
        # when one name is missing a session.
        by_date: dict[str, dict[Any, int]] = {}
        series: dict[str, list[Any]] = {}
        for sym, bars in daily_bars.items():
            rows = [b for b in bars if getattr(b, "close", 0) > 0]
            if len(rows) < self.MIN_HISTORY + horizon + 1:
                continue
            series[sym] = rows
            by_date[sym] = {_bar_date(b): i for i, b in enumerate(rows)}
        if not series:
            return {"symbols": 0, "dates": 0, "pulls": 0,
                    "note": "no symbol had enough daily history"}

        all_dates = sorted({d for idx in by_date.values() for d in idx})
        usable = all_dates[self.MIN_HISTORY:len(all_dates) - horizon]

        picks = rewards = 0
        for _pass in range(max(1, int(passes))):
            for day in usable:
                table: dict[str, dict[str, float]] = {}
                fwd: dict[str, float] = {}
                for sym, idx in by_date.items():
                    i = idx.get(day)
                    if i is None or i < self.MIN_HISTORY or i + horizon >= len(series[sym]):
                        continue
                    # A fixed tail, not the growing history: every
                    # feature looks back at most 63 bars, so this is
                    # identical output for O(1) work per date.
                    lo = max(0, i + 1 - self.MIN_HISTORY * 2)
                    window = series[sym][lo : i + 1]
                    row = self.raw_features(_WindowCtx(window), sym)
                    if row is None:
                        continue
                    here = series[sym][i].close
                    later = series[sym][i + horizon].close
                    if here <= 0:
                        continue
                    table[sym] = row
                    fwd[sym] = later / here - 1.0
                if len(table) < 2:
                    continue

                vectors = self._standardise(table)
                ranked = sorted(
                    vectors.items(), key=lambda kv: self.ucb(kv[1])[0], reverse=True
                )[:max_symbols]
                for sym, vec in ranked:
                    self.update(vec, max(min(fwd[sym], 0.25), -0.25))
                    rewards += 1
                picks += 1

        theta = self.theta
        summary = {
            "symbols": len(series),
            "dates": len(usable),
            "passes": int(passes),
            "picks_per_date": max_symbols,
            "pulls": self.pulls,
            "updates": rewards,
            "mean_reward": round(self.total_reward / max(self.pulls, 1), 6),
            "theta": {
                name: round(float(theta[i]), 5)
                for i, name in enumerate([*self.p.feature_set, "bias"])
            },
        }
        logger.info("bandit_picker: pretrained on %d symbols over %d dates, "
                    "%d pulls", summary["symbols"], summary["dates"], self.pulls)
        return summary

    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "d": self.d,
            "feature_set": list(self.p.feature_set),
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "pulls": self.pulls,
            "total_reward": round(self.total_reward, 6),
        }

    def load_state(self, blob: Mapping[str, Any]) -> None:
        if not blob:
            return
        if list(blob.get("feature_set") or []) != list(self.p.feature_set):
            self.log.warning(
                "bandit_picker: saved feature set differs from configured; starting fresh"
            )
            return
        try:
            A = np.asarray(blob["A"], dtype=float)
            b = np.asarray(blob["b"], dtype=float)
        except (KeyError, TypeError, ValueError):
            return
        if A.shape != (self.d, self.d) or b.shape != (self.d,):
            self.log.warning("bandit_picker: saved model has the wrong shape; starting fresh")
            return
        self.A, self.b = A, b
        self.pulls = int(blob.get("pulls", 0))
        self.total_reward = float(blob.get("total_reward", 0.0))
        self.log.info("bandit_picker: restored model with %d pulls", self.pulls)
