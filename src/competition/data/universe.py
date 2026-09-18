"""Universe construction: the Round 3 top-500 pool and Round 2's candidates.

Round 3 needs a stable, ranked list of "the 500 most valuable public
companies". That ranking is the spine of the fairness constraint (every dealt
hand's ranks sum to the same number), so it must be:

  * **explicit** -- checked into `data/top500.csv`, not fetched at draft time,
  * **auditable** -- the snapshot carries its own date and source,
  * **refreshable** -- `comp refresh-universe` can re-rank before a round.

Two ranking metrics are supported, because "most expensive" is ambiguous:
  * `market_cap`  (default) -- total company value. The usual reading.
  * `share_price`           -- literal price per share, computed live from
                               Alpaca with zero external dependencies.
"""

from __future__ import annotations

import contextlib
import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..config import REPO_ROOT
from ..types import utcnow

log = logging.getLogger("competition.universe")

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "top500.csv"

#: Substrings that mark a ticker as a leveraged/inverse ETF or similar
#: instrument we never want in an equity-only competition.
_ETF_MARKERS = ("2X", "3X", "ULTRA", "INVERSE", "BULL ", "BEAR ", "LEVERAGED")


@dataclass(frozen=True)
class ValuationRow:
    """One company in the ranked pool."""

    rank: int                 # 1 = most valuable
    symbol: str
    name: str
    sector: str = "Unknown"
    market_cap: float = 0.0   # USD
    share_price: float = 0.0

    @property
    def decile(self) -> int:
        """1..10 -- which tenth of the pool this rank falls in (needs pool size)."""
        return max(1, min(10, (self.rank - 1) // 50 + 1))

    def metric(self, name: str) -> float:
        return self.share_price if name == "share_price" else self.market_cap


@dataclass(frozen=True)
class Asset:
    """Tradability metadata from Alpaca's /v2/assets."""

    symbol: str
    name: str = ""
    exchange: str = ""
    tradable: bool = False
    fractionable: bool = False
    shortable: bool = False
    easy_to_borrow: bool = False
    status: str = "active"
    asset_class: str = "us_equity"

    @property
    def is_active(self) -> bool:
        return self.status == "active" and self.tradable

    @property
    def looks_like_leveraged_etf(self) -> bool:
        up = self.name.upper()
        return any(m in up for m in _ETF_MARKERS)


@dataclass
class UniverseSnapshot:
    """A ranked pool plus the provenance needed to audit a draft."""

    rows: tuple[ValuationRow, ...]
    metric: str = "market_cap"
    as_of: str = ""
    source: str = "bundled"

    def __post_init__(self) -> None:
        if not self.as_of:
            self.as_of = date.today().isoformat()

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(r.symbol for r in self.rows)

    def by_symbol(self, symbol: str) -> ValuationRow | None:
        s = symbol.upper()
        for r in self.rows:
            if r.symbol == s:
                return r
        return None

    def by_rank(self, rank: int) -> ValuationRow | None:
        for r in self.rows:
            if r.rank == rank:
                return r
        return None

    def rank_of(self, symbol: str) -> int | None:
        row = self.by_symbol(symbol)
        return row.rank if row else None

    def sector_of(self, symbol: str) -> str:
        row = self.by_symbol(symbol)
        return row.sector if row else "Unknown"

    def head(self, n: int) -> UniverseSnapshot:
        return UniverseSnapshot(self.rows[:n], self.metric, self.as_of, self.source)

    def reranked(self, metric: str) -> UniverseSnapshot:
        """Re-sort by a different metric, renumbering ranks 1..N."""
        ordered = sorted(self.rows, key=lambda r: (-r.metric(metric), r.symbol))
        rows = tuple(
            ValuationRow(i + 1, r.symbol, r.name, r.sector, r.market_cap, r.share_price)
            for i, r in enumerate(ordered)
        )
        return UniverseSnapshot(rows, metric, self.as_of, self.source)

    def filtered(self, keep: Iterable[str]) -> UniverseSnapshot:
        """Drop anything not in `keep` (e.g. not tradable), then renumber."""
        allow = {s.upper() for s in keep}
        rows = tuple(r for r in self.rows if r.symbol in allow)
        renumbered = tuple(
            ValuationRow(i + 1, r.symbol, r.name, r.sector, r.market_cap, r.share_price)
            for i, r in enumerate(rows)
        )
        return UniverseSnapshot(renumbered, self.metric, self.as_of, self.source)

    def sectors(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.sector] = out.get(r.sector, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # -- io ---------------------------------------------------------------- #

    def to_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# as_of", self.as_of, "metric", self.metric, "source", self.source])
            w.writerow(["rank", "symbol", "name", "sector", "market_cap_usd", "share_price"])
            for r in self.rows:
                w.writerow([r.rank, r.symbol, r.name, r.sector,
                            f"{r.market_cap:.0f}", f"{r.share_price:.2f}"])
        return p

    @classmethod
    def from_csv(cls, path: str | Path) -> UniverseSnapshot:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"universe snapshot not found: {p}")
        as_of, metric, source = "", "market_cap", str(p.name)
        rows: list[ValuationRow] = []
        with p.open(newline="") as fh:
            for raw in csv.reader(fh):
                if not raw:
                    continue
                if raw[0].startswith("#"):
                    meta = raw[1:]
                    as_of = meta[0] if meta else as_of
                    if "metric" in raw:
                        i = raw.index("metric")
                        metric = raw[i + 1] if i + 1 < len(raw) else metric
                    if "source" in raw:
                        i = raw.index("source")
                        source = raw[i + 1] if i + 1 < len(raw) else source
                    continue
                if raw[0].strip().lower() == "rank":
                    continue
                try:
                    rows.append(
                        ValuationRow(
                            rank=int(raw[0]),
                            symbol=raw[1].strip().upper(),
                            name=raw[2].strip(),
                            sector=(raw[3].strip() or "Unknown") if len(raw) > 3 else "Unknown",
                            market_cap=float(raw[4]) if len(raw) > 4 and raw[4] else 0.0,
                            share_price=float(raw[5]) if len(raw) > 5 and raw[5] else 0.0,
                        )
                    )
                except (ValueError, IndexError):
                    log.warning("skipping malformed universe row: %r", raw)
        rows.sort(key=lambda r: r.rank)
        return cls(tuple(rows), metric=metric, as_of=as_of, source=source)


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #


class UniverseProvider:
    """Loads, filters and refreshes the ranked pool + Round 2 candidates."""

    def __init__(
        self,
        *,
        snapshot_path: str | Path = DEFAULT_SNAPSHOT,
        reader=None,
        trading_client=None,
    ):
        self.snapshot_path = Path(snapshot_path)
        self.reader = reader              # AlpacaDataReader (optional)
        self.trading_client = trading_client  # AlpacaClient (optional)
        self._assets: dict[str, Asset] | None = None
        self._snapshot: UniverseSnapshot | None = None

    # -- ranked pool ------------------------------------------------------- #

    def snapshot(self, *, metric: str = "market_cap", size: int | None = None) -> UniverseSnapshot:
        if self._snapshot is None:
            self._snapshot = UniverseSnapshot.from_csv(self.snapshot_path)
        snap = self._snapshot
        if metric != snap.metric:
            if metric == "share_price" and all(r.share_price <= 0 for r in snap.rows):
                snap = self._price_snapshot(snap)
            snap = snap.reranked(metric)
        return snap.head(size) if size else snap

    def _price_snapshot(self, snap: UniverseSnapshot) -> UniverseSnapshot:
        """Fill in live share prices so `share_price` ranking works."""
        if self.reader is None:
            raise RuntimeError(
                "metric=share_price needs a live data reader; run "
                "`comp refresh-universe --metric share_price` first"
            )
        prices = self.latest_prices(snap.symbols)
        rows = tuple(
            ValuationRow(r.rank, r.symbol, r.name, r.sector, r.market_cap,
                         prices.get(r.symbol, r.share_price))
            for r in snap.rows
        )
        return UniverseSnapshot(rows, snap.metric, utcnow().date().isoformat(), "alpaca-live-prices")

    def tradable_pool(
        self, *, metric: str = "market_cap", size: int = 500, verify: bool = True
    ) -> UniverseSnapshot:
        """The ranked pool with untradable names dropped and ranks renumbered.

        Dropping *before* the draft matters: a hand containing a delisted
        ticker is not a fair hand, and renumbering keeps the rank-sum target
        exactly achievable.
        """
        snap = self.snapshot(metric=metric)
        if not verify or self.trading_client is None:
            return snap.head(size)
        assets = self.assets()
        keep = [
            r.symbol for r in snap.rows
            if (a := assets.get(r.symbol)) is not None and a.is_active
            and not a.looks_like_leveraged_etf
        ]
        filtered = snap.filtered(keep)
        dropped = len(snap) - len(filtered)
        if dropped:
            log.info("dropped %d untradable names from the pool", dropped)
        if len(filtered) < size:
            log.warning(
                "tradable pool is %d names, smaller than requested %d; "
                "refresh data/top500.csv", len(filtered), size
            )
        return filtered.head(size)

    # -- asset metadata ---------------------------------------------------- #

    def assets(self, *, refresh: bool = False) -> dict[str, Asset]:
        if self._assets is not None and not refresh:
            return self._assets
        out: dict[str, Asset] = {}
        if self.trading_client is not None:
            try:
                for raw in self.trading_client.assets():
                    sym = str(raw.get("symbol", "")).upper()
                    if not sym:
                        continue
                    out[sym] = Asset(
                        symbol=sym,
                        name=str(raw.get("name", "") or ""),
                        exchange=str(raw.get("exchange", "") or ""),
                        tradable=bool(raw.get("tradable", False)),
                        fractionable=bool(raw.get("fractionable", False)),
                        shortable=bool(raw.get("shortable", False)),
                        easy_to_borrow=bool(raw.get("easy_to_borrow", False)),
                        status=str(raw.get("status", "active")),
                        asset_class=str(raw.get("class", "us_equity")),
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("asset list unavailable (%s); treating all names as tradable", e)
        self._assets = out
        return out

    def is_tradable(self, symbol: str) -> bool:
        assets = self.assets()
        if not assets:
            return True
        a = assets.get(symbol.upper())
        return bool(a and a.is_active)

    def is_fractionable(self, symbol: str) -> bool:
        a = self.assets().get(symbol.upper())
        return bool(a and a.fractionable)

    # -- Round 2 candidate pool ------------------------------------------- #

    def latest_prices(self, symbols: Sequence[str]) -> dict[str, float]:
        if self.reader is None:
            return {}
        out: dict[str, float] = {}
        try:
            snaps = self.reader.snapshots(symbols)
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot fetch failed: %s", e)
            return out
        for sym, snap in snaps.items():
            px = 0.0
            trade = snap.get("latestTrade") or {}
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            for candidate in (trade.get("p"), daily.get("c")):
                try:
                    if candidate and float(candidate) > 0:
                        px = float(candidate)
                        break
                except (TypeError, ValueError):
                    continue
            if px > 0:
                out[sym.upper()] = px
        return out

    def candidate_pool(
        self,
        *,
        size: int = 300,
        liquidity=None,
        mode: str = "top_active",
        extra: Sequence[str] = (),
    ) -> list[str]:
        """The screening universe every Round 2 picker starts from.

        Identical for all teams -- the pickers differ in what they *choose*,
        not in what they are shown. Built from Alpaca's most-actives screener
        (deep, liquid, survivorship-free) unioned with the top-cap snapshot,
        then filtered on the round's liquidity rules.
        """
        seeds: list[str] = []
        if mode in ("top_active", "hybrid") and self.reader is not None:
            try:
                for row in self.reader.most_active(top=100, by="volume"):
                    sym = str(row.get("symbol", "")).upper()
                    if sym:
                        seeds.append(sym)
                for row in self.reader.most_active(top=100, by="trades"):
                    sym = str(row.get("symbol", "")).upper()
                    if sym:
                        seeds.append(sym)
            except Exception as e:  # noqa: BLE001
                log.warning("most-actives screener unavailable: %s", e)
        with contextlib.suppress(FileNotFoundError):
            seeds.extend(self.snapshot().symbols)
        seeds.extend(s.upper() for s in extra)

        pool = list(dict.fromkeys(s for s in seeds if s and s.isalpha()))
        assets = self.assets()
        if assets:
            pool = [
                s for s in pool
                if (a := assets.get(s)) is not None and a.is_active
                and a.asset_class == "us_equity"
                and not (liquidity and liquidity.exclude_leveraged_etf and a.looks_like_leveraged_etf)
            ]
        if liquidity is not None:
            pool = self._apply_liquidity(pool, liquidity)
        return pool[:size]

    def _apply_liquidity(self, symbols: Sequence[str], liq) -> list[str]:
        """Price / dollar-volume / trade-count screen from daily bars."""
        if self.reader is None or not symbols:
            return list(symbols)
        try:
            bars = self.reader.bars(symbols, "1Day", limit=25)
        except Exception as e:  # noqa: BLE001
            log.warning("liquidity screen skipped (%s)", e)
            return list(symbols)
        keep: list[str] = []
        for sym in symbols:
            rows = bars.get(sym) or []
            if len(rows) < 5:
                continue
            recent = rows[-20:]
            px = recent[-1].close
            adv = sum(b.dollar_volume for b in recent) / len(recent)
            trades = sum(b.trade_count for b in recent) / len(recent)
            if px < liq.min_price or px > liq.max_price:
                continue
            if adv < liq.min_avg_dollar_volume:
                continue
            if trades < liq.min_avg_trade_count:
                continue
            keep.append(sym)
        return keep

    # -- refresh ----------------------------------------------------------- #

    def refresh(self, *, metric: str = "market_cap", size: int = 500,
                out_path: str | Path | None = None) -> UniverseSnapshot:
        """Re-rank the bundled snapshot using live data and write it back.

        `share_price` is fully self-contained (prices come from Alpaca).
        `market_cap` needs share counts, which Alpaca does not publish; the
        bundled caps are used as the base and only the ordering of names whose
        prices have moved materially is refreshed, with a warning. For a hard
        re-rank, drop a fresh CSV into data/top500.csv -- the format is
        documented in docs/universe.md.
        """
        snap = self.snapshot()
        prices = self.latest_prices(snap.symbols)
        if not prices:
            log.warning("no live prices available; snapshot unchanged")
            return snap

        rows = []
        for r in snap.rows:
            px = prices.get(r.symbol, r.share_price)
            cap = r.market_cap
            if metric == "market_cap" and r.share_price > 0 and px > 0 and cap > 0:
                # Scale the stored cap by the price change since the snapshot:
                # share counts move slowly, prices do not.
                cap = cap * (px / r.share_price)
            rows.append(ValuationRow(r.rank, r.symbol, r.name, r.sector, cap, px))
        refreshed = UniverseSnapshot(
            tuple(rows), metric=snap.metric,
            as_of=utcnow().date().isoformat(),
            source="refreshed-from-alpaca-prices",
        ).reranked(metric).head(size)
        refreshed.to_csv(out_path or self.snapshot_path)
        return refreshed

    def audit(self, *, metric: str = "market_cap", size: int = 500) -> dict:
        """Facts a referee would want before running the Round 3 draft."""
        snap = self.snapshot(metric=metric, size=size)
        assets = self.assets()
        untradable = [r.symbol for r in snap.rows if assets and not self.is_tradable(r.symbol)]
        return {
            "path": str(self.snapshot_path),
            "as_of": snap.as_of,
            "source": snap.source,
            "metric": metric,
            "rows": len(snap),
            "sectors": snap.sectors(),
            "top_10": [r.symbol for r in snap.rows[:10]],
            "bottom_10": [r.symbol for r in snap.rows[-10:]],
            "untradable": untradable,
            "checked_tradability": bool(assets),
        }
