"""Alpaca REST client and `Broker` implementation.

Written directly against Alpaca's documented HTTP API rather than the SDK:
the wire format is stable, it keeps the dependency surface to `requests`, and
it lets the engine control retry/rate-limit behaviour precisely (eight teams
polling at once will hit the shared 200 req/min budget otherwise).

Endpoints used
--------------
Trading  (paper-api.alpaca.markets)
    GET    /v2/account
    GET    /v2/positions            DELETE /v2/positions[/{symbol}]
    GET    /v2/orders               POST   /v2/orders
    DELETE /v2/orders[/{id}]
    GET    /v2/assets[/{symbol}]
    GET    /v2/clock, /v2/calendar
Market data (data.alpaca.markets)
    GET    /v2/stocks/bars, /v2/stocks/quotes/latest, /v2/stocks/snapshots
    GET    /v2/stocks/trades/latest, /v2/stocks/meta/... , /v1beta1/news
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from ..types import (
    UTC,
    Account,
    Bar,
    NewsItem,
    Order,
    OrderIntent,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    TimeInForce,
    utcnow,
)
from .base import Broker, BrokerError, InsufficientFunds, OrderRejected

log = logging.getLogger("competition.alpaca")

DEFAULT_TRADING_URL = "https://paper-api.alpaca.markets"
DEFAULT_DATA_URL = "https://data.alpaca.markets"

#: Alpaca's REST budget is per *account*, so every client throttles itself.
#: 180/min leaves headroom under the documented 200/min ceiling.
DEFAULT_RATE_LIMIT_PER_MIN = 180

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class AlpacaHTTPError(BrokerError):
    """Non-retryable HTTP failure from Alpaca, with the response body attached."""

    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")
        self.status = status
        self.url = url
        self.body = body


class _RateLimiter:
    """Simple thread-safe token bucket over a rolling 60s window."""

    def __init__(self, per_minute: int):
        self.per_minute = max(int(per_minute), 1)
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._stamps = [t for t in self._stamps if t > cutoff]
                if len(self._stamps) < self.per_minute:
                    self._stamps.append(now)
                    return
                sleep_for = 60.0 - (now - self._stamps[0]) + 0.01
            time.sleep(min(max(sleep_for, 0.01), 5.0))


@dataclass(frozen=True)
class AlpacaCredentials:
    key_id: str
    secret_key: str
    trading_url: str = DEFAULT_TRADING_URL
    data_url: str = DEFAULT_DATA_URL
    feed: str = "iex"

    @staticmethod
    def from_env(prefix: str, *, feed: str | None = None) -> AlpacaCredentials:
        """Build credentials from `<PREFIX>_KEY_ID` / `<PREFIX>_SECRET_KEY`."""
        kid = os.environ.get(f"{prefix}_KEY_ID", "").strip()
        sec = os.environ.get(f"{prefix}_SECRET_KEY", "").strip()
        if not kid or not sec:
            raise BrokerError(
                f"missing Alpaca credentials for {prefix}: set {prefix}_KEY_ID and "
                f"{prefix}_SECRET_KEY in .env (see .env.example)"
            )
        return AlpacaCredentials(
            key_id=kid,
            secret_key=sec,
            trading_url=os.environ.get("ALPACA_TRADING_BASE_URL", DEFAULT_TRADING_URL).rstrip("/"),
            data_url=os.environ.get("ALPACA_DATA_BASE_URL", DEFAULT_DATA_URL).rstrip("/"),
            feed=(feed or os.environ.get("ALPACA_DATA_FEED", "iex")).strip(),
        )

    @property
    def is_paper(self) -> bool:
        return "paper" in self.trading_url

    def redacted(self) -> str:
        return f"{self.key_id[:4]}…{self.key_id[-2:]}" if len(self.key_id) > 6 else "…"


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class AlpacaClient:
    """Thin, retrying JSON client for one Alpaca account."""

    def __init__(
        self,
        creds: AlpacaCredentials,
        *,
        timeout: float = 15.0,
        max_retries: int = 4,
        rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
        session: requests.Session | None = None,
    ):
        self.creds = creds
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiter = _RateLimiter(rate_limit_per_min)
        self._session = session or requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": creds.key_id,
            "APCA-API-SECRET-KEY": creds.secret_key,
            "Accept": "application/json",
            "User-Agent": "trading-competition/1.0",
        })

    # -- plumbing ---------------------------------------------------------- #

    def _request(self, method: str, base: str, path: str, **kw) -> Any:
        url = f"{base}{path}"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:  # network blip
                last = e
                if attempt >= self.max_retries:
                    raise BrokerError(f"network error calling {url}: {e}") from e
                self._backoff(attempt)
                continue

            if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                log.debug("alpaca %s %s -> %s, retrying", method, path, resp.status_code)
                self._backoff(attempt, delay)
                continue

            if resp.status_code == 204 or not (resp.content or b"").strip():
                return {}
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError as e:
                    raise BrokerError(f"non-JSON response from {url}: {resp.text[:200]}") from e
            raise AlpacaHTTPError(resp.status_code, url, resp.text)
        raise BrokerError(f"exhausted retries for {url}: {last}")

    def _backoff(self, attempt: int, explicit: float | None = None) -> None:
        delay = explicit if explicit is not None else min(0.5 * (2**attempt), 8.0)
        time.sleep(delay + random.uniform(0, 0.25))

    # -- trading API ------------------------------------------------------- #

    def trading_get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", self.creds.trading_url, path, params=params)

    def trading_post(self, path: str, body: dict) -> Any:
        return self._request("POST", self.creds.trading_url, path, json=body)

    def trading_delete(self, path: str, params: dict | None = None) -> Any:
        return self._request("DELETE", self.creds.trading_url, path, params=params)

    # -- data API ---------------------------------------------------------- #

    def data_get(self, path: str, params: dict | None = None) -> Any:
        return self._request("GET", self.creds.data_url, path, params=params)

    def data_paginate(self, path: str, params: dict, container: str) -> Iterator[dict]:
        """Follow `next_page_token` and yield each page's `container` payload."""
        page = dict(params)
        seen = 0
        while True:
            payload = self.data_get(path, page)
            yield payload.get(container) or {}
            token = payload.get("next_page_token")
            seen += 1
            if not token or seen > 200:
                return
            page["page_token"] = token

    # -- convenience ------------------------------------------------------- #

    def clock(self) -> dict:
        return self.trading_get("/v2/clock")

    def calendar(self, start: str, end: str) -> list[dict]:
        return self.trading_get("/v2/calendar", {"start": start, "end": end}) or []

    def assets(self, *, status: str = "active", asset_class: str = "us_equity") -> list[dict]:
        return self.trading_get("/v2/assets", {"status": status, "asset_class": asset_class}) or []

    def asset(self, symbol: str) -> dict:
        return self.trading_get(f"/v2/assets/{symbol.upper()}")

    def verify(self) -> dict:
        """Cheap credential check; raises if the keys are bad."""
        acct = self.trading_get("/v2/account")
        return {
            "account_number": acct.get("account_number", "?"),
            "status": acct.get("status"),
            "equity": float(acct.get("equity") or 0.0),
            "cash": float(acct.get("cash") or 0.0),
            "paper": self.creds.is_paper,
            "currency": acct.get("currency", "USD"),
            "pattern_day_trader": bool(acct.get("pattern_day_trader", False)),
            "shorting_enabled": bool(acct.get("shorting_enabled", False)),
            "trading_blocked": bool(acct.get("trading_blocked", False)),
        }


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if not v:
        return utcnow()
    s = str(v).replace("Z", "+00:00")
    # Alpaca returns nanosecond precision; fromisoformat wants <= 6 digits.
    if "." in s:
        head, _, tail = s.partition(".")
        digits = "".join(c for c in tail if c.isdigit())[:6].ljust(6, "0")
        offset = tail[len(digits):] if not tail[len(digits):].isdigit() else ""
        for marker in ("+", "-"):
            if marker in tail:
                offset = marker + tail.split(marker, 1)[1]
                break
        s = f"{head}.{digits}{offset or '+00:00'}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return utcnow()
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "held": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "pending_replace": OrderStatus.PENDING,
    "replaced": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.PENDING,
    "calculated": OrderStatus.PENDING,
    "stopped": OrderStatus.CANCELED,
    "done_for_day": OrderStatus.CANCELED,
}


def parse_order(d: dict) -> Order:
    return Order(
        id=str(d.get("id", "")),
        client_order_id=str(d.get("client_order_id", "")),
        symbol=str(d.get("symbol", "")).upper(),
        side=Side(str(d.get("side", "buy")).lower()),
        qty=_f(d.get("qty") or d.get("notional")),
        filled_qty=_f(d.get("filled_qty")),
        filled_avg_price=_f(d.get("filled_avg_price")),
        order_type=OrderType(str(d.get("type", "market")).lower())
        if str(d.get("type", "market")).lower() in ("market", "limit")
        else OrderType.MARKET,
        limit_price=_f(d.get("limit_price")) or None,
        tif=TimeInForce(str(d.get("time_in_force", "day")).lower())
        if str(d.get("time_in_force", "day")).lower() in ("day", "gtc", "ioc", "fok")
        else TimeInForce.DAY,
        status=_STATUS_MAP.get(str(d.get("status", "new")).lower(), OrderStatus.PENDING),
        submitted_at=_ts(d.get("submitted_at") or d.get("created_at")),
        updated_at=_ts(d.get("updated_at") or d.get("submitted_at")),
    )


def parse_position(d: dict) -> Position:
    return Position(
        symbol=str(d.get("symbol", "")).upper(),
        qty=_f(d.get("qty")),
        avg_entry_price=_f(d.get("avg_entry_price")),
        current_price=_f(d.get("current_price")) or _f(d.get("avg_entry_price")),
    )


def parse_bar(symbol: str, d: dict) -> Bar:
    return Bar(
        symbol=symbol,
        ts=_ts(d.get("t")),
        open=_f(d.get("o")),
        high=_f(d.get("h")),
        low=_f(d.get("l")),
        close=_f(d.get("c")),
        volume=_f(d.get("v")),
        trade_count=int(_f(d.get("n"))),
        vwap=_f(d.get("vw")),
    )


def parse_quote(symbol: str, d: dict) -> Quote:
    return Quote(
        symbol=symbol,
        ts=_ts(d.get("t")),
        bid=_f(d.get("bp")),
        ask=_f(d.get("ap")),
        bid_size=_f(d.get("bs")),
        ask_size=_f(d.get("as")),
    )


def parse_news(d: dict) -> NewsItem:
    return NewsItem(
        id=str(d.get("id", "")),
        ts=_ts(d.get("updated_at") or d.get("created_at")),
        headline=str(d.get("headline", "")),
        summary=str(d.get("summary", "") or ""),
        source=str(d.get("source", "") or ""),
        symbols=tuple(str(s).upper() for s in (d.get("symbols") or ())),
        url=str(d.get("url", "") or ""),
    )


# --------------------------------------------------------------------------- #
# Broker implementation
# --------------------------------------------------------------------------- #


class AlpacaBroker(Broker):
    """One Alpaca paper account, presented as a `Broker`."""

    def __init__(
        self,
        client: AlpacaClient,
        *,
        name: str = "alpaca",
        cache_seconds: float = 1.0,
        allow_notional: bool = True,
    ):
        self.client = client
        self.name = name
        self.cache_seconds = cache_seconds
        self.allow_notional = allow_notional
        self._acct_cache: tuple[float, Account] | None = None
        self._baseline_equity: float | None = None
        self._asset_cache: dict[str, dict] = {}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_env(cls, env_prefix: str, *, name: str | None = None, **kw) -> AlpacaBroker:
        creds = AlpacaCredentials.from_env(env_prefix)
        return cls(AlpacaClient(creds), name=name or env_prefix.lower(), **kw)

    # -- account ----------------------------------------------------------- #

    def account(self, *, fresh: bool = False) -> Account:
        now = time.monotonic()
        if not fresh and self._acct_cache and now - self._acct_cache[0] < self.cache_seconds:
            return self._acct_cache[1]
        raw = self.client.trading_get("/v2/account")
        positions = tuple(parse_position(p) for p in (self.client.trading_get("/v2/positions") or []))
        acct = Account(
            cash=_f(raw.get("cash")),
            equity=_f(raw.get("equity")),
            buying_power=_f(raw.get("buying_power")),
            positions=positions,
        )
        self._acct_cache = (now, acct)
        return acct

    def positions(self) -> list[Position]:
        return list(self.account().positions)

    def invalidate(self) -> None:
        self._acct_cache = None

    @property
    def baseline_equity(self) -> float | None:
        """Equity recorded at the round's start; round P&L is measured off it."""
        return self._baseline_equity

    # -- orders ------------------------------------------------------------ #

    def submit(self, intent: OrderIntent, *, client_order_id: str | None = None) -> Order:
        body: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": intent.side.value,
            "type": intent.order_type.value,
            "time_in_force": intent.tif.value,
            "client_order_id": client_order_id or f"comp-{uuid.uuid4().hex[:24]}",
        }
        if intent.qty is not None:
            # Alpaca wants fractional quantities as strings to avoid float drift.
            body["qty"] = f"{intent.qty:.9f}".rstrip("0").rstrip(".")
        else:
            if not self.allow_notional:
                raise OrderRejected(f"{self.name}: notional orders disabled; convert to qty first")
            body["notional"] = f"{intent.notional:.2f}"
        if intent.order_type is OrderType.LIMIT:
            body["limit_price"] = f"{intent.limit_price:.2f}"
            # Notional + limit is not supported by Alpaca; the engine sizes
            # limit orders in shares, but guard anyway.
            if "notional" in body:
                raise OrderRejected(f"{self.name}: limit orders require qty, not notional")

        try:
            raw = self.client.trading_post("/v2/orders", body)
        except AlpacaHTTPError as e:
            self.invalidate()
            blob = e.body.lower()
            if "insufficient" in blob or "buying power" in blob:
                raise InsufficientFunds(f"{self.name}: {intent.describe()} -> {e.body[:200]}") from e
            if e.status in (403, 422):
                raise OrderRejected(f"{self.name}: {intent.describe()} -> {e.body[:200]}") from e
            raise
        self.invalidate()
        order = parse_order(raw)
        order.reason = intent.reason
        return order

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        params: dict[str, Any] = {"status": "open", "limit": 500, "nested": "false"}
        if symbol:
            params["symbols"] = symbol.upper()
        return [parse_order(o) for o in (self.client.trading_get("/v2/orders", params) or [])]

    def closed_orders(self, *, after: datetime | None = None, limit: int = 500) -> list[Order]:
        params: dict[str, Any] = {"status": "closed", "limit": limit, "direction": "desc"}
        if after:
            params["after"] = after.astimezone(UTC).isoformat()
        return [parse_order(o) for o in (self.client.trading_get("/v2/orders", params) or [])]

    def cancel(self, order_id: str) -> None:
        try:
            self.client.trading_delete(f"/v2/orders/{order_id}")
        except AlpacaHTTPError as e:
            # 404/422 == already terminal. Not an error worth propagating.
            if e.status not in (404, 422):
                raise
        self.invalidate()

    def cancel_all(self, symbol: str | None = None) -> int:
        if symbol is None:
            try:
                resp = self.client.trading_delete("/v2/orders")
                self.invalidate()
                return len(resp) if isinstance(resp, list) else 0
            except AlpacaHTTPError as e:
                if e.status == 404:
                    return 0
                raise
        n = 0
        for o in self.open_orders(symbol):
            self.cancel(o.id)
            n += 1
        return n

    # -- bulk -------------------------------------------------------------- #

    def close_position(self, symbol: str) -> Order | None:
        try:
            raw = self.client.trading_delete(f"/v2/positions/{symbol.upper()}")
        except AlpacaHTTPError as e:
            if e.status in (404, 422):  # nothing held / not closable right now
                return None
            raise
        self.invalidate()
        return parse_order(raw) if isinstance(raw, dict) and raw.get("id") else None

    def close_all_positions(self, *, cancel_orders: bool = True) -> list[Order]:
        try:
            resp = self.client.trading_delete(
                "/v2/positions", {"cancel_orders": "true" if cancel_orders else "false"}
            )
        except AlpacaHTTPError as e:
            if e.status == 404:
                return []
            raise
        self.invalidate()
        out: list[Order] = []
        for item in resp if isinstance(resp, list) else []:
            body = item.get("body") if isinstance(item, dict) else None
            if isinstance(body, dict) and body.get("id"):
                out.append(parse_order(body))
        return out

    # -- lifecycle --------------------------------------------------------- #

    def sync(self, now: datetime | None = None) -> None:
        self.invalidate()

    def reset_for_round(self, starting_cash: float) -> None:
        """Flatten and record the baseline equity for this round.

        A live account's balance cannot be set, so the engine measures each
        round's return against `baseline_equity` instead of `starting_cash`.
        If the account has drifted from the intended bankroll, that is logged
        loudly -- an unequal bankroll is the one thing that breaks fairness,
        and `comp doctor` refuses to start a round when it is out of tolerance.
        """
        self.cancel_all()
        self.close_all_positions(cancel_orders=True)
        # Market-sell settlement is not instant; poll briefly for a flat book.
        for _ in range(10):
            acct = self.account(fresh=True)
            if not acct.held_symbols:
                break
            time.sleep(1.0)
        acct = self.account(fresh=True)
        self._baseline_equity = acct.equity
        drift = acct.equity - starting_cash
        if abs(drift) > max(0.01 * starting_cash, 5.0):
            log.warning(
                "%s: account equity %.2f differs from intended bankroll %.2f (drift %+.2f); "
                "round P&L will be measured from the actual baseline",
                self.name, acct.equity, starting_cash, drift,
            )

    @property
    def supports_fractional(self) -> bool:
        return True

    @property
    def supports_notional_orders(self) -> bool:
        return self.allow_notional

    # -- asset metadata ---------------------------------------------------- #

    def asset_info(self, symbol: str) -> dict:
        sym = symbol.upper()
        if sym not in self._asset_cache:
            try:
                self._asset_cache[sym] = self.client.asset(sym) or {}
            except AlpacaHTTPError:
                self._asset_cache[sym] = {}
        return self._asset_cache[sym]

    def is_tradable(self, symbol: str) -> bool:
        info = self.asset_info(symbol)
        return bool(info.get("tradable", False)) and info.get("status") != "inactive"

    def is_fractionable(self, symbol: str) -> bool:
        return bool(self.asset_info(symbol).get("fractionable", False))


# --------------------------------------------------------------------------- #
# market data reader (shared across teams -- see data/feed.py)
# --------------------------------------------------------------------------- #


class AlpacaDataReader:
    """Market-data access for one client. Used by the single shared data hub."""

    def __init__(self, client: AlpacaClient, *, feed: str | None = None):
        self.client = client
        self.feed = feed or client.creds.feed

    def bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        limit: int = 500,
        start: datetime | None = None,
        end: datetime | None = None,
        adjustment: str = "split",
        batch: int = 100,
    ) -> dict[str, list[Bar]]:
        """Multi-symbol bars, oldest first. Batches to stay inside URL limits."""
        out: dict[str, list[Bar]] = {s.upper(): [] for s in symbols}
        syms = [s.upper() for s in symbols]
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "timeframe": timeframe,
                "limit": min(max(limit, 1) * len(chunk), 10000),
                "feed": self.feed,
                "adjustment": adjustment,
                "sort": "asc",
            }
            if start:
                params["start"] = start.astimezone(UTC).isoformat()
            if end:
                params["end"] = end.astimezone(UTC).isoformat()
            for page in self.client.data_paginate("/v2/stocks/bars", params, "bars"):
                for sym, rows in (page or {}).items():
                    out.setdefault(sym.upper(), []).extend(parse_bar(sym.upper(), r) for r in rows or [])
        for sym, rows in out.items():
            rows.sort(key=lambda b: b.ts)
            if limit and len(rows) > limit:
                out[sym] = rows[-limit:]
        return out

    def latest_quotes(self, symbols: Sequence[str], *, batch: int = 200) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        syms = [s.upper() for s in symbols]
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            payload = self.client.data_get(
                "/v2/stocks/quotes/latest", {"symbols": ",".join(chunk), "feed": self.feed}
            )
            for sym, q in (payload.get("quotes") or {}).items():
                out[sym.upper()] = parse_quote(sym.upper(), q)
        return out

    def snapshots(self, symbols: Sequence[str], *, batch: int = 200) -> dict[str, dict]:
        """Raw snapshot payloads (latest trade/quote, daily + prev daily bar)."""
        out: dict[str, dict] = {}
        syms = [s.upper() for s in symbols]
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            payload = self.client.data_get(
                "/v2/stocks/snapshots", {"symbols": ",".join(chunk), "feed": self.feed}
            )
            snaps = payload.get("snapshots") if isinstance(payload, dict) else None
            for sym, snap in (snaps or payload or {}).items():
                if isinstance(snap, dict):
                    out[sym.upper()] = snap
        return out

    def news(
        self,
        symbols: Iterable[str] | None = None,
        *,
        hours: int = 72,
        limit: int = 50,
        include_content: bool = True,
    ) -> list[NewsItem]:
        params: dict[str, Any] = {
            "start": (utcnow() - timedelta(hours=hours)).isoformat(),
            "limit": min(max(limit, 1), 50),
            "sort": "desc",
            "include_content": "true" if include_content else "false",
            "exclude_contentless": "true",
        }
        syms = [s.upper() for s in (symbols or ())]
        if syms:
            params["symbols"] = ",".join(syms[:50])
        items: list[NewsItem] = []
        seen: set[str] = set()
        page = dict(params)
        for _ in range(20):
            payload = self.client.data_get("/v1beta1/news", page)
            for raw in payload.get("news") or []:
                item = parse_news(raw)
                if item.id and item.id not in seen:
                    seen.add(item.id)
                    items.append(item)
            token = payload.get("next_page_token")
            if not token or len(items) >= limit * max(len(syms), 1):
                break
            page["page_token"] = token
        items.sort(key=lambda n: n.ts, reverse=True)
        return items

    def most_active(self, *, top: int = 100, by: str = "volume") -> list[dict]:
        """Alpaca's screener: the day's most active names. Used by pickers."""
        payload = self.client.data_get(
            "/v1beta1/screener/stocks/most-actives", {"by": by, "top": min(max(top, 1), 100)}
        )
        return payload.get("most_actives") or []
