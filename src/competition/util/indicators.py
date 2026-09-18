"""Technical / statistical indicators, implemented once and shared by all teams.

Fairness rule: no strategy ships its own private copy of a common indicator.
Everyone calls the same tested functions, so nobody wins or loses on an
off-by-one in an EMA. Every function:

  * takes plain sequences of floats (or `Bar`s) -- oldest first, newest last,
  * returns a float / list of floats, never NaN (uses `default` instead),
  * is pure and allocation-light enough to call on every tick.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..types import Bar

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _arr(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)] if a.size and not np.all(np.isfinite(a)) else a


def safe(x: float, default: float = 0.0) -> float:
    """Collapse NaN/inf to `default`. Called at every indicator boundary."""
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else default


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def pct_change(values: Sequence[float], periods: int = 1) -> list[float]:
    a = _arr(values)
    if a.size <= periods:
        return []
    prev, cur = a[:-periods], a[periods:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(prev > 0, cur / prev - 1.0, 0.0)
    return [safe(v) for v in out]


def log_returns(values: Sequence[float]) -> list[float]:
    a = _arr(values)
    if a.size < 2:
        return []
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.diff(np.log(np.where(a > 0, a, np.nan)))
    return [safe(v) for v in out]


# --------------------------------------------------------------------------- #
# moving averages / trend
# --------------------------------------------------------------------------- #


def sma(values: Sequence[float], period: int) -> float:
    a = _arr(values)
    if a.size < period or period <= 0:
        return safe(a[-1] if a.size else 0.0)
    return safe(a[-period:].mean())


def ema(values: Sequence[float], period: int) -> float:
    """Standard 2/(n+1) EMA seeded with the first `period` SMA."""
    a = _arr(values)
    if a.size == 0 or period <= 0:
        return 0.0
    if a.size < period:
        return safe(a.mean())
    k = 2.0 / (period + 1.0)
    val = a[:period].mean()
    for x in a[period:]:
        val = x * k + val * (1.0 - k)
    return safe(val)


def ema_series(values: Sequence[float], period: int) -> list[float]:
    a = _arr(values)
    if a.size == 0 or period <= 0:
        return []
    k = 2.0 / (period + 1.0)
    out: list[float] = []
    val = float(a[0])
    for i, x in enumerate(a):
        val = float(x) if i == 0 else float(x) * k + val * (1.0 - k)
        out.append(safe(val))
    return out


def wilder(values: Sequence[float], period: int) -> float:
    """Wilder smoothing (1/n), used by RSI/ATR/ADX."""
    a = _arr(values)
    if a.size == 0 or period <= 0:
        return 0.0
    if a.size < period:
        return safe(a.mean())
    val = a[:period].mean()
    for x in a[period:]:
        val = (val * (period - 1) + x) / period
    return safe(val)


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)."""
    a = _arr(values)
    if a.size < slow:
        return 0.0, 0.0, 0.0
    f, s = ema_series(a, fast), ema_series(a, slow)
    line = [fv - sv for fv, sv in zip(f, s, strict=True)]
    sig = ema(line[-max(signal * 3, signal):], signal)
    return safe(line[-1]), safe(sig), safe(line[-1] - sig)


def linreg_slope(values: Sequence[float], period: int | None = None) -> float:
    """Least-squares slope per bar, normalised by mean level (=> per-bar % drift)."""
    a = _arr(values)
    if period:
        a = a[-period:]
    n = a.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, a, 1)[0])
    level = float(np.mean(a))
    return safe(slope / level if level > 0 else 0.0)


def r_squared(values: Sequence[float], period: int | None = None) -> float:
    """How linear the recent path is, in [0, 1]. High = clean trend."""
    a = _arr(values)
    if period:
        a = a[-period:]
    n = a.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, a, 1)
    resid = a - np.polyval(coef, x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return safe(1.0 - ss_res / ss_tot, 0.0) if ss_tot > 0 else 0.0


def roc(values: Sequence[float], period: int) -> float:
    """Rate of change over `period` bars."""
    a = _arr(values)
    if a.size <= period or a[-period - 1] <= 0:
        return 0.0
    return safe(a[-1] / a[-period - 1] - 1.0)


# --------------------------------------------------------------------------- #
# oscillators
# --------------------------------------------------------------------------- #


def rsi(values: Sequence[float], period: int = 14) -> float:
    a = _arr(values)
    if a.size < period + 1:
        return 50.0
    d = np.diff(a)
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag, al = wilder(gains, period), wilder(losses, period)
    if al <= 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return safe(100.0 - 100.0 / (1.0 + rs), 50.0)


def stochastic(bars: Sequence[Bar], period: int = 14) -> float:
    """%K in [0, 100]: where close sits in the recent high/low range."""
    if len(bars) < period:
        return 50.0
    window = bars[-period:]
    hi = max(b.high for b in window)
    lo = min(b.low for b in window)
    if hi <= lo:
        return 50.0
    return safe((window[-1].close - lo) / (hi - lo) * 100.0, 50.0)


def zscore(values: Sequence[float], period: int = 20) -> float:
    """(last - mean) / stdev over the trailing window."""
    a = _arr(values)
    if a.size < max(period, 3):
        return 0.0
    w = a[-period:]
    sd = float(w.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return safe((float(a[-1]) - float(w.mean())) / sd)


def percentile_rank(values: Sequence[float], value: float | None = None) -> float:
    """Fraction of the window at or below `value` (default: the last value)."""
    a = _arr(values)
    if a.size == 0:
        return 0.5
    v = float(a[-1]) if value is None else float(value)
    return safe(float(np.mean(a <= v)), 0.5)


# --------------------------------------------------------------------------- #
# volatility / ranges
# --------------------------------------------------------------------------- #


def true_range(bars: Sequence[Bar]) -> list[float]:
    if len(bars) < 2:
        return [b.range for b in bars]
    out = [bars[0].range]
    for prev, cur in zip(bars, bars[1:], strict=False):
        out.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    return [safe(v) for v in out]


def atr(bars: Sequence[Bar], period: int = 14) -> float:
    if not bars:
        return 0.0
    return wilder(true_range(bars), period)


def atr_pct(bars: Sequence[Bar], period: int = 14) -> float:
    """ATR as a fraction of price -- the comparable-across-names vol measure."""
    if not bars:
        return 0.0
    px = bars[-1].close
    return safe(atr(bars, period) / px) if px > 0 else 0.0


def realized_vol(values: Sequence[float], period: int = 20, annualize: float = 1.0) -> float:
    """Stdev of log returns over the window, scaled by `annualize`."""
    lr = log_returns(values)
    if len(lr) < 3:
        return 0.0
    w = np.asarray(lr[-period:], dtype=float)
    return safe(float(w.std(ddof=1)) * math.sqrt(annualize))


def bollinger(values: Sequence[float], period: int = 20, mult: float = 2.0) -> tuple[float, float, float]:
    """Returns (lower, mid, upper)."""
    a = _arr(values)
    if a.size < max(period, 3):
        m = safe(a.mean() if a.size else 0.0)
        return m, m, m
    w = a[-period:]
    m = float(w.mean())
    sd = float(w.std(ddof=1))
    return safe(m - mult * sd), safe(m), safe(m + mult * sd)


def bandwidth(values: Sequence[float], period: int = 20, mult: float = 2.0) -> float:
    """Bollinger bandwidth / mid -- low values mean a volatility squeeze."""
    lo, mid, hi = bollinger(values, period, mult)
    return safe((hi - lo) / mid) if mid > 0 else 0.0


def donchian(bars: Sequence[Bar], period: int = 20) -> tuple[float, float]:
    """Highest high / lowest low of the last `period` bars, excluding none."""
    if not bars:
        return 0.0, 0.0
    w = bars[-period:]
    return safe(max(b.high for b in w)), safe(min(b.low for b in w))


def keltner(bars: Sequence[Bar], period: int = 20, mult: float = 1.5) -> tuple[float, float, float]:
    if not bars:
        return 0.0, 0.0, 0.0
    mid = ema([b.close for b in bars], period)
    a = atr(bars, period)
    return safe(mid - mult * a), safe(mid), safe(mid + mult * a)


def adx(bars: Sequence[Bar], period: int = 14) -> float:
    """Average Directional Index in [0, 100]. >25 conventionally = trending."""
    if len(bars) < period + 2:
        return 0.0
    plus_dm, minus_dm = [], []
    for prev, cur in zip(bars, bars[1:], strict=False):
        up = cur.high - prev.high
        dn = prev.low - cur.low
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
    tr = true_range(bars)[1:]
    atr_v = wilder(tr, period)
    if atr_v <= 0:
        return 0.0
    pdi = 100.0 * wilder(plus_dm, period) / atr_v
    mdi = 100.0 * wilder(minus_dm, period) / atr_v
    denom = pdi + mdi
    return safe(100.0 * abs(pdi - mdi) / denom) if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# mean-reversion statistics (used by the reverter, the stat-arb team and both
# of their pickers)
# --------------------------------------------------------------------------- #


def ar1_coefficient(values: Sequence[float]) -> float:
    """OLS phi from x_t = a + phi * x_{t-1}. phi -> 1 is a random walk."""
    a = _arr(values)
    if a.size < 10:
        return 1.0
    x, y = a[:-1], a[1:]
    vx = float(np.var(x))
    if vx <= 1e-15:
        return 1.0
    phi = float(np.cov(x, y, bias=True)[0, 1] / vx)
    return safe(phi, 1.0)


def half_life(values: Sequence[float]) -> float:
    """OU half-life in bars. Short = snaps back fast. inf = no reversion."""
    phi = ar1_coefficient(values)
    if not (0.0 < phi < 1.0):
        return math.inf
    return safe(-math.log(2.0) / math.log(phi), math.inf)


def variance_ratio(values: Sequence[float], lag: int = 5) -> float:
    """Var(r_lag)/(lag*Var(r_1)). <1 mean-reverting, ~1 random walk, >1 trending."""
    lr = log_returns(values)
    if len(lr) < lag * 4:
        return 1.0
    r1 = np.asarray(lr, dtype=float)
    agg = np.add.reduceat(r1, np.arange(0, r1.size - r1.size % lag, lag))
    v1 = float(np.var(r1, ddof=1))
    vk = float(np.var(agg, ddof=1))
    if v1 <= 1e-18 or agg.size < 3:
        return 1.0
    return safe(vk / (lag * v1), 1.0)


def hurst(values: Sequence[float], max_lag: int = 20) -> float:
    """Rescaled-range Hurst exponent. <0.5 reverting, >0.5 trending."""
    a = _arr(values)
    if a.size < max_lag * 3:
        return 0.5
    lags = range(2, max_lag)
    tau, ok = [], []
    for lag in lags:
        d = a[lag:] - a[:-lag]
        sd = float(np.std(d))
        if sd > 1e-15:
            tau.append(sd)
            ok.append(lag)
    if len(tau) < 4:
        return 0.5
    h = float(np.polyfit(np.log(ok), np.log(tau), 1)[0])
    return safe(clamp(h, 0.0, 1.0), 0.5)


def hedge_ratio(y: Sequence[float], x: Sequence[float]) -> float:
    """OLS beta of y on x with no intercept shrinkage -- the pair hedge ratio."""
    ya, xa = _arr(y), _arr(x)
    n = min(ya.size, xa.size)
    if n < 10:
        return 1.0
    ya, xa = ya[-n:], xa[-n:]
    vx = float(np.var(xa))
    if vx <= 1e-15:
        return 1.0
    beta = float(np.cov(xa, ya, bias=True)[0, 1] / vx)
    return safe(beta, 1.0)


def spread_series(y: Sequence[float], x: Sequence[float], beta: float) -> list[float]:
    ya, xa = _arr(y), _arr(x)
    n = min(ya.size, xa.size)
    if n == 0:
        return []
    return [safe(float(yv - beta * xv)) for yv, xv in zip(ya[-n:], xa[-n:], strict=True)]


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = _arr(a), _arr(b)
    n = min(aa.size, bb.size)
    if n < 10:
        return 0.0
    aa, bb = aa[-n:], bb[-n:]
    if float(np.std(aa)) <= 1e-15 or float(np.std(bb)) <= 1e-15:
        return 0.0
    return safe(float(np.corrcoef(aa, bb)[0, 1]))


def adf_tstat(values: Sequence[float]) -> float:
    """Dickey-Fuller t-stat on dx_t = a + g*x_{t-1} + e.

    Rough but dependency-free. More negative = stronger evidence of
    stationarity; -2.9 is the conventional 5% critical value.
    """
    a = _arr(values)
    n = a.size
    if n < 20:
        return 0.0
    x, dx = a[:-1], np.diff(a)
    X = np.column_stack([np.ones(x.size), x])
    try:
        beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    resid = dx - X @ beta
    dof = x.size - 2
    if dof <= 0:
        return 0.0
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    se = math.sqrt(max(float(cov[1, 1]), 1e-18))
    return safe(float(beta[1]) / se)


# --------------------------------------------------------------------------- #
# performance stats (scoring + reporting)
# --------------------------------------------------------------------------- #


def max_drawdown(equity: Sequence[float]) -> float:
    a = _arr(equity)
    if a.size < 2:
        return 0.0
    peak = np.maximum.accumulate(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, a / peak - 1.0, 0.0)
    return safe(float(dd.min()))


def sharpe(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    a = _arr(returns)
    if a.size < 3:
        return 0.0
    sd = float(a.std(ddof=1))
    if sd <= 1e-15:
        # Zero-variance returns have no defined Sharpe (it is +/-infinity).
        # Reports have to stay printable and sortable, so return 0. This
        # cannot arise from real price data, only from a flat equity curve.
        return 0.0
    return safe(float(a.mean()) / sd * math.sqrt(periods_per_year))


def sortino(returns: Sequence[float], periods_per_year: float = 252.0) -> float:
    a = _arr(returns)
    if a.size < 3:
        return 0.0
    downside = a[a < 0]
    dd = float(downside.std(ddof=1)) if downside.size >= 2 else 0.0
    if dd <= 1e-15:
        return 0.0
    return safe(float(a.mean()) / dd * math.sqrt(periods_per_year))


def calmar(total_return: float, equity: Sequence[float]) -> float:
    mdd = abs(max_drawdown(equity))
    return safe(total_return / mdd) if mdd > 1e-9 else 0.0
