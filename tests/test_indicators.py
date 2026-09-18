"""Shared indicator library.

Two things matter: the indicators must *discriminate* (a trending series and a
mean-reverting one must not look the same), and they must never return NaN or
blow up on short or degenerate input -- a strategy that crashes on an empty
bar list forfeits its round.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from competition.types import Bar, utcnow
from competition.util import indicators as ind


def _bars(closes, sym="X"):
    out = []
    prev = float(closes[0])
    for c in closes:
        c = float(c)
        hi = max(prev, c) * 1.004
        lo = min(prev, c) * 0.996
        out.append(Bar(sym, utcnow(), prev, hi, lo, c, 1e6, 500, c))
        prev = c
    return out


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(11)
    n = 400
    # Constant drift + iid noise. Note this has a variance ratio of ~1, not
    # >1: VR measures how variance scales with aggregation, and a *constant*
    # drift adds no extra variance at longer horizons. Only autocorrelated
    # returns push VR above 1, hence `momentum` below.
    trend = 100 * np.cumprod(1 + rng.normal(0.0035, 0.012, n))
    # Positively autocorrelated returns -- what "trending" means to a
    # variance-ratio or Hurst test.
    r = [0.0]
    for _ in range(n - 1):
        r.append(0.45 * r[-1] + rng.normal(0, 0.010))
    momentum = 100 * np.cumprod(1 + np.array(r))
    mr = [100.0]
    for _ in range(n - 1):
        mr.append(mr[-1] + -0.30 * (mr[-1] - 100) + rng.normal(0, 1.4))
    walk = 100 * np.cumprod(1 + rng.normal(0, 0.012, n))
    flat = np.full(n, 50.0)
    return {"trend": list(trend), "momentum": list(momentum), "mr": mr,
            "walk": list(walk), "flat": list(flat)}


# --------------------------------------------------------------------------- #
# discrimination
# --------------------------------------------------------------------------- #


def test_half_life_separates_reverting_from_trending(series):
    assert ind.half_life(series["mr"]) < 6
    assert ind.half_life(series["trend"]) > 30


def test_variance_ratio_separates_regimes(series):
    assert ind.variance_ratio(series["mr"], 5) < 0.8
    assert ind.variance_ratio(series["momentum"], 5) > 1.3
    assert 0.6 < ind.variance_ratio(series["walk"], 5) < 1.6
    assert ind.variance_ratio(series["momentum"], 5) > ind.variance_ratio(series["mr"], 5)


def test_hurst_separates_regimes(series):
    assert ind.hurst(series["mr"]) < 0.45
    assert ind.hurst(series["momentum"]) > 0.50
    assert ind.hurst(series["momentum"]) > ind.hurst(series["mr"])


def test_adf_detects_stationarity(series):
    assert ind.adf_tstat(series["mr"]) < -3.0
    assert ind.adf_tstat(series["trend"]) > -3.0


def test_rsi_bounds_and_direction(series):
    assert ind.rsi(series["trend"]) > 60
    assert 0 <= ind.rsi(series["walk"]) <= 100
    assert ind.rsi(list(range(1, 60))) == 100.0        # monotone up


def test_adx_is_high_in_a_trend_low_in_chop(series):
    assert ind.adx(_bars(series["trend"])) > 20
    assert ind.adx(_bars(series["mr"])) < ind.adx(_bars(series["trend"]))


def test_r_squared_is_high_for_a_line():
    assert ind.r_squared([float(i) for i in range(100)]) > 0.99
    assert ind.r_squared([100.0, 1.0, 100.0, 1.0] * 25) < 0.2


def test_hedge_ratio_recovers_a_known_beta():
    x = [float(i) + 10 for i in range(200)]
    y = [2.5 * v + 7.0 for v in x]
    assert ind.hedge_ratio(y, x) == pytest.approx(2.5, abs=1e-6)


def test_spread_of_a_cointegrated_pair_is_stationary():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(0, 1, 300)) + 100
    noise = [0.0]
    for _ in range(299):
        noise.append(noise[-1] * 0.8 + rng.normal(0, 0.5))
    y = 1.3 * x + np.array(noise)
    beta = ind.hedge_ratio(list(y), list(x))
    assert beta == pytest.approx(1.3, abs=0.05)
    spread = ind.spread_series(list(y), list(x), beta)
    assert ind.adf_tstat(spread) < -4.0
    assert ind.half_life(spread) < 10


def test_bollinger_and_keltner_orderings(series):
    lo, mid, hi = ind.bollinger(series["walk"], 20, 2.0)
    assert lo < mid < hi
    klo, kmid, khi = ind.keltner(_bars(series["walk"]), 20, 1.5)
    assert klo < kmid < khi
    assert ind.bandwidth(series["walk"], 20) > 0


def test_atr_pct_is_scale_free():
    small = _bars([10.0 * (1 + 0.01 * i % 3) for i in range(60)])
    big = _bars([1000.0 * (1 + 0.01 * i % 3) for i in range(60)])
    assert ind.atr_pct(small) == pytest.approx(ind.atr_pct(big), rel=0.05)


def test_donchian_brackets_the_series(series):
    bars = _bars(series["walk"])
    hi, lo = ind.donchian(bars, 20)
    window = bars[-20:]
    assert hi == max(b.high for b in window)
    assert lo == min(b.low for b in window)


def test_percentile_rank():
    assert ind.percentile_rank([1, 2, 3, 4, 5], 3) == pytest.approx(0.6)
    assert ind.percentile_rank([1, 2, 3, 4, 5]) == 1.0


def test_performance_stats():
    equity = [100, 110, 105, 130, 90, 120]
    assert ind.max_drawdown(equity) == pytest.approx(90 / 130 - 1)
    assert ind.max_drawdown([100, 101, 102]) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    assert ind.sharpe(list(rng.normal(0.004, 0.01, 200))) > 0
    assert ind.sharpe(list(rng.normal(-0.004, 0.01, 200))) < 0
    # Zero-variance returns cannot have a defined Sharpe; the guard returns 0
    # rather than infinity so reports stay printable.
    assert ind.sharpe([0.01] * 50) == 0.0
    assert ind.calmar(0.2, equity) != 0.0


def test_macd_sign_follows_the_trend(series):
    line, signal, hist = ind.macd(series["trend"])
    assert line > 0
    down = list(reversed(series["trend"]))
    assert ind.macd(down)[0] < 0


# --------------------------------------------------------------------------- #
# degenerate input -- must never raise, never return NaN
# --------------------------------------------------------------------------- #


ALL_SERIES_FUNCS = [
    (ind.sma, (10,)), (ind.ema, (10,)), (ind.wilder, (10,)), (ind.rsi, (14,)),
    (ind.zscore, (20,)), (ind.roc, (10,)), (ind.realized_vol, (20,)),
    (ind.linreg_slope, (20,)), (ind.r_squared, (20,)), (ind.bandwidth, (20,)),
    (ind.ar1_coefficient, ()), (ind.variance_ratio, (5,)), (ind.hurst, ()),
    (ind.adf_tstat, ()), (ind.percentile_rank, ()), (ind.max_drawdown, ()),
    (ind.sharpe, ()), (ind.sortino, ()),
]


@pytest.mark.parametrize("data", [[], [1.0], [0.0, 0.0, 0.0], [5.0] * 50,
                                  [-1.0, -2.0, -3.0], [1e-12] * 30])
@pytest.mark.parametrize("func,args", ALL_SERIES_FUNCS)
def test_series_functions_survive_degenerate_input(func, args, data):
    out = func(data, *args)
    assert isinstance(out, float)
    assert not math.isnan(out)


@pytest.mark.parametrize("data", [[], [1.0], [5.0] * 50])
def test_half_life_returns_inf_not_nan(data):
    hl = ind.half_life(data)
    assert hl > 0 and not math.isnan(hl)


BAR_FUNCS = [(ind.atr, (14,)), (ind.atr_pct, (14,)), (ind.adx, (14,)),
             (ind.stochastic, (14,))]


@pytest.mark.parametrize("n", [0, 1, 2, 5])
@pytest.mark.parametrize("func,args", BAR_FUNCS)
def test_bar_functions_survive_short_input(func, args, n):
    bars = _bars([100.0] * n) if n else []
    out = func(bars, *args)
    assert isinstance(out, float) and not math.isnan(out)


def test_bollinger_and_keltner_on_empty_input():
    assert ind.bollinger([], 20) == (0.0, 0.0, 0.0)
    assert ind.keltner([], 20) == (0.0, 0.0, 0.0)
    assert ind.donchian([], 20) == (0.0, 0.0)
    assert ind.macd([]) == (0.0, 0.0, 0.0)


def test_nan_and_inf_are_filtered():
    data = [1.0, float("nan"), 2.0, float("inf"), 3.0]
    out = ind.sma(data, 3)
    assert not math.isnan(out) and math.isfinite(out)


def test_correlation_on_constant_series_is_zero_not_nan():
    assert ind.correlation([1.0] * 50, list(range(50))) == 0.0


def test_safe_and_clamp():
    assert ind.safe(float("nan"), 7.0) == 7.0
    assert ind.safe(float("inf"), 7.0) == 7.0
    assert ind.safe(2.5) == 2.5
    assert ind.clamp(5, 0, 1) == 1
    assert ind.clamp(-5, 0, 1) == 0
