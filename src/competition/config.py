"""Typed configuration loader.

The YAML in `config/` is the rulebook. This module parses it into frozen
dataclasses, validates it hard (a bad rulebook should fail at startup, not at
3pm on day four), and computes a `rules_hash` that gets stamped into the
ledger so results can always be tied back to the exact rules in force.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

UniverseMode = Literal["fixed", "picker", "draft"]


class ConfigError(RuntimeError):
    """Raised for any malformed or self-contradictory rulebook."""


# --------------------------------------------------------------------------- #
# .env loading (no external dependency)
# --------------------------------------------------------------------------- #

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read a `.env` file into os.environ. Returns what it loaded.

    Deliberately minimal: `KEY=value`, `#` comments, optional quotes. Values
    are never logged anywhere in this codebase.
    """
    p = Path(path) if path else REPO_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        val = val.split(" #", 1)[0].strip() if not val.startswith("#") else ""
        if not val:
            continue
        if override or key not in os.environ:
            os.environ[key] = val
        loaded[key] = val
    return loaded


# --------------------------------------------------------------------------- #
# risk / fairness / data blocks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RiskConfig:
    allow_short: bool = False
    max_gross_leverage: float = 1.0
    max_position_pct: float = 1.0
    min_order_notional: float = 1.0
    max_order_notional: float = 5000.0
    max_orders_per_tick: int = 12
    max_orders_per_day: int = 2000
    daily_loss_kill_switch_pct: float = 0.35
    max_limit_deviation_pct: float = 0.10
    max_quote_age_seconds: float = 120.0
    liquidate_at_round_end: bool = True
    liquidate_minutes_before_close: int = 10

    def validate(self) -> None:
        if not 0 < self.max_gross_leverage <= 4:
            raise ConfigError("risk.max_gross_leverage must be in (0, 4]")
        if not 0 < self.max_position_pct <= 1.0 * self.max_gross_leverage:
            raise ConfigError("risk.max_position_pct must be in (0, max_gross_leverage]")
        if self.min_order_notional <= 0 or self.max_order_notional <= self.min_order_notional:
            raise ConfigError("risk order notional bounds are inconsistent")
        if not 0 < self.daily_loss_kill_switch_pct < 1:
            raise ConfigError("risk.daily_loss_kill_switch_pct must be in (0, 1)")
        if self.max_orders_per_tick < 1:
            raise ConfigError("risk.max_orders_per_tick must be >= 1")


@dataclass(frozen=True)
class FairnessConfig:
    shared_data_snapshot: bool = True
    strategy_timeout_seconds: float = 5.0
    rotate_execution_order: bool = True
    competition_seed: int = 20260101
    sim_slippage_bps: float = 2.0
    sim_commission_per_share: float = 0.0

    def validate(self) -> None:
        if self.strategy_timeout_seconds <= 0:
            raise ConfigError("fairness.strategy_timeout_seconds must be > 0")
        if self.sim_slippage_bps < 0:
            raise ConfigError("fairness.sim_slippage_bps must be >= 0")


@dataclass(frozen=True)
class DataConfig:
    feed: str = "iex"
    primary_timeframe: str = "5Min"
    fast_timeframe: str = "1Min"
    slow_timeframe: str = "1Day"
    history_bars_primary: int = 500
    history_bars_fast: int = 240
    history_days_slow: int = 400
    news_lookback_hours: int = 72
    news_limit_per_symbol: int = 50

    def validate(self) -> None:
        if self.feed not in ("iex", "sip", "otc"):
            raise ConfigError(f"data.feed must be iex|sip|otc, got {self.feed!r}")
        for name in ("primary_timeframe", "fast_timeframe", "slow_timeframe"):
            tf = getattr(self, name)
            if not re.fullmatch(r"\d+(Min|Hour|Day|Week|Month)", tf):
                raise ConfigError(f"data.{name}={tf!r} is not an Alpaca timeframe (e.g. 5Min, 1Day)")


# --------------------------------------------------------------------------- #
# rounds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiquidityFilter:
    min_price: float = 5.0
    max_price: float = 5000.0
    min_avg_dollar_volume: float = 20_000_000.0
    min_avg_trade_count: int = 5_000
    require_tradable: bool = True
    require_fractionable: bool = False
    exclude_leveraged_etf: bool = True


@dataclass(frozen=True)
class PickerRoundConfig:
    max_symbols: int = 12
    refresh: str = "daily"
    candidate_pool: str = "top_active"
    candidate_pool_size: int = 300
    liquidity_filter: LiquidityFilter = field(default_factory=LiquidityFilter)


@dataclass(frozen=True)
class DraftConfig:
    pool_size: int = 500
    picks_per_team: int = 10
    metric: str = "market_cap"
    rank_sum_target: int | None = None
    rank_sum_exact: bool = True
    unique_across_teams: bool = True
    max_sector_share: float = 0.40
    min_rank_deciles: int = 4
    #: "exchange" -- sample freely, then repair the sum by swapping ranks.
    #: "complement" -- deal (r, pool_size+1-r) pairs, so the sum is exact by
    #: construction. Requires an even picks_per_team.
    method: str = "exchange"
    #: Attempts before the dealer gives up and reports infeasibility.
    max_attempts: int = 400

    @property
    def fair_rank_sum(self) -> int:
        """The only target that gives every hand the same mean rank.

        With ranks 1..N, the expected sum of k draws is k*(N+1)/2. For
        N=500, k=10 that is exactly 2505, so it is achievable as an integer.
        """
        if self.rank_sum_target is not None:
            return int(self.rank_sum_target)
        num = self.picks_per_team * (self.pool_size + 1)
        if num % 2 != 0:
            # Not integral -- nudge to the nearest achievable sum and say so.
            return num // 2
        return num // 2

    def validate(self, n_teams: int) -> None:
        if self.picks_per_team < 2:
            raise ConfigError("draft.picks_per_team must be >= 2")
        if self.pool_size < self.picks_per_team * n_teams and self.unique_across_teams:
            raise ConfigError(
                f"draft.pool_size={self.pool_size} too small for {n_teams} unique hands "
                f"of {self.picks_per_team}"
            )
        k, n = self.picks_per_team, self.pool_size
        lo, hi = k * (k + 1) // 2, k * (2 * n - k + 1) // 2
        target = self.fair_rank_sum
        if not lo <= target <= hi:
            raise ConfigError(f"draft.rank_sum_target={target} unreachable; must be in [{lo}, {hi}]")
        if not 0 < self.max_sector_share <= 1.0:
            raise ConfigError("draft.max_sector_share must be in (0, 1]")
        if not 1 <= self.min_rank_deciles <= 10:
            raise ConfigError("draft.min_rank_deciles must be in [1, 10]")
        if self.method not in ("exchange", "complement"):
            raise ConfigError("draft.method must be exchange|complement")
        if self.method == "complement":
            if k % 2 != 0:
                raise ConfigError("draft.method=complement requires an even picks_per_team")
            if self.rank_sum_target is not None and target != k // 2 * (n + 1):
                raise ConfigError(
                    f"draft.method=complement forces rank_sum_target="
                    f"{k // 2 * (n + 1)}, not {target}"
                )
        if self.min_rank_deciles > k:
            raise ConfigError(
                f"draft.min_rank_deciles ({self.min_rank_deciles}) cannot exceed "
                f"picks_per_team ({k})"
            )
        if int(self.max_sector_share * k) < 1:
            raise ConfigError(
                f"draft.max_sector_share ({self.max_sector_share}) allows fewer than one "
                f"name per sector for a hand of {k}"
            )


@dataclass(frozen=True)
class RoundConfig:
    id: int
    name: str
    universe_mode: UniverseMode
    blurb: str = ""
    symbols: tuple[str, ...] = ()
    picker_enabled: bool = False
    picker: PickerRoundConfig = field(default_factory=PickerRoundConfig)
    draft: DraftConfig | None = None

    def validate(self, n_teams: int) -> None:
        if self.universe_mode == "fixed":
            if not self.symbols:
                raise ConfigError(f"round {self.id}: universe_mode=fixed requires symbols")
        elif self.universe_mode == "draft":
            if self.draft is None:
                raise ConfigError(f"round {self.id}: universe_mode=draft requires a draft block")
            self.draft.validate(n_teams)
            if self.picker.max_symbols > self.draft.picks_per_team:
                raise ConfigError(
                    f"round {self.id}: picker.max_symbols ({self.picker.max_symbols}) exceeds the "
                    f"dealt hand size ({self.draft.picks_per_team})"
                )
        elif self.universe_mode == "picker":
            if not self.picker_enabled:
                raise ConfigError(f"round {self.id}: universe_mode=picker requires picker_enabled")
        else:
            raise ConfigError(f"round {self.id}: unknown universe_mode {self.universe_mode!r}")

    @property
    def max_symbols(self) -> int:
        if self.universe_mode == "fixed":
            return len(self.symbols)
        return self.picker.max_symbols


# --------------------------------------------------------------------------- #
# teams
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TeamConfig:
    key: str
    name: str
    strategy: str
    picker: str
    env_prefix: str
    scored: bool = True
    tagline: str = ""
    philosophy: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    picker_params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", self.key):
            raise ConfigError(f"team key {self.key!r} must be lower_snake_case")
        for spec, label in ((self.strategy, "strategy"), (self.picker, "picker")):
            if ":" not in spec:
                raise ConfigError(f"team {self.key}: {label} must be 'module.path:ClassName'")

    def credentials(self) -> tuple[str | None, str | None]:
        """(key_id, secret) from the environment, or (None, None) if unset."""
        kid = os.environ.get(f"{self.env_prefix}_KEY_ID") or None
        sec = os.environ.get(f"{self.env_prefix}_SECRET_KEY") or None
        return kid, sec

    @property
    def has_credentials(self) -> bool:
        return all(self.credentials())


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompetitionConfig:
    name: str
    season: int
    timezone: str
    starting_cash: float
    round_length_days: int
    points_table: tuple[int, ...]
    tie_policy: str
    tiebreakers: tuple[str, ...]
    risk: RiskConfig
    fairness: FairnessConfig
    data: DataConfig
    rounds: tuple[RoundConfig, ...]
    teams: tuple[TeamConfig, ...]
    source_files: tuple[Path, ...] = ()

    # -- lookups ----------------------------------------------------------- #

    @property
    def scored_teams(self) -> tuple[TeamConfig, ...]:
        return tuple(t for t in self.teams if t.scored)

    @property
    def team_keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.teams)

    def team(self, key: str) -> TeamConfig:
        for t in self.teams:
            if t.key == key:
                return t
        raise KeyError(f"no such team: {key!r} (have {', '.join(self.team_keys)})")

    def round(self, rid: int) -> RoundConfig:
        for r in self.rounds:
            if r.id == rid:
                return r
        raise KeyError(f"no such round: {rid} (have {[r.id for r in self.rounds]})")

    def points_for_place(self, place: int) -> int:
        """1-indexed place -> points. Places past the table score 0."""
        if 1 <= place <= len(self.points_table):
            return self.points_table[place - 1]
        return 0

    # -- validation / provenance ------------------------------------------- #

    def validate(self) -> None:
        if self.starting_cash <= 0:
            raise ConfigError("competition.starting_cash must be > 0")
        if self.round_length_days < 1:
            raise ConfigError("competition.round_length_days must be >= 1")
        if not self.rounds:
            raise ConfigError("no rounds defined")
        if len({r.id for r in self.rounds}) != len(self.rounds):
            raise ConfigError("duplicate round ids")
        if not self.teams:
            raise ConfigError("no teams defined")
        if len(set(self.team_keys)) != len(self.teams):
            raise ConfigError("duplicate team keys")
        prefixes = [t.env_prefix for t in self.teams]
        if len(set(prefixes)) != len(prefixes):
            raise ConfigError("two teams share an env_prefix -- they would trade the same account")
        n_scored = len(self.scored_teams)
        if len(self.points_table) < n_scored:
            raise ConfigError(
                f"points_table has {len(self.points_table)} entries but there are "
                f"{n_scored} scored teams; last places would score 0"
            )
        if self.tie_policy not in ("average", "best", "worst"):
            raise ConfigError("tie_policy must be average|best|worst")
        if list(self.points_table) != sorted(self.points_table, reverse=True):
            raise ConfigError("points_table must be non-increasing (1st place first)")
        self.risk.validate()
        self.fairness.validate()
        self.data.validate()
        for t in self.teams:
            t.validate()
        for r in self.rounds:
            r.validate(n_scored)

    @property
    def rules_hash(self) -> str:
        """Stable hash of the effective rules, stamped into every run."""
        payload = json.dumps(_to_jsonable(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def team_seed(self, team_key: str, round_id: int) -> int:
        """Deterministic per-(team, round) RNG seed. Equal treatment, reproducible."""
        h = hashlib.sha256(f"{self.fairness.competition_seed}:{team_key}:{round_id}".encode())
        return int.from_bytes(h.digest()[:8], "big") % (2**63 - 1)

    def round_window(self, round_id: int, start: date) -> tuple[date, date]:
        """Inclusive [start, end] calendar window for a round."""
        return start, start + timedelta(days=self.round_length_days - 1)


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: _to_jsonable(getattr(obj, k))
            for k in obj.__dataclass_fields__
            if k != "source_files"
        }
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (Path, datetime, date, time)):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _sub(cls, data: dict[str, Any] | None, *, where: str):
    """Build a frozen dataclass from a dict, rejecting unknown keys loudly."""
    data = dict(data or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    return cls(**data)


def _parse_round(raw: dict[str, Any]) -> RoundConfig:
    rid = raw.get("id")
    if rid is None:
        raise ConfigError("a round is missing its `id`")
    where = f"round {rid}"
    picker_raw = dict(raw.get("picker") or {})
    liq = picker_raw.pop("liquidity_filter", None)
    picker = _sub(PickerRoundConfig, picker_raw, where=f"{where}.picker")
    if liq is not None:
        picker = PickerRoundConfig(
            max_symbols=picker.max_symbols,
            refresh=picker.refresh,
            candidate_pool=picker.candidate_pool,
            candidate_pool_size=picker.candidate_pool_size,
            liquidity_filter=_sub(LiquidityFilter, liq, where=f"{where}.picker.liquidity_filter"),
        )
    draft = _sub(DraftConfig, raw.get("draft"), where=f"{where}.draft") if raw.get("draft") else None
    return RoundConfig(
        id=int(rid),
        name=str(raw.get("name", f"Round {rid}")),
        universe_mode=str(raw.get("universe_mode", "fixed")),  # type: ignore[arg-type]
        blurb=str(raw.get("blurb", "")).strip(),
        symbols=tuple(s.upper().strip() for s in (raw.get("symbols") or ())),
        picker_enabled=bool(raw.get("picker_enabled", False)),
        picker=picker,
        draft=draft,
    )


def _parse_team(raw: dict[str, Any]) -> TeamConfig:
    try:
        key = raw["key"]
        return TeamConfig(
            key=str(key),
            name=str(raw.get("name", key)),
            strategy=str(raw["strategy"]),
            picker=str(raw["picker"]),
            env_prefix=str(raw.get("env_prefix", f"ALPACA_{str(key).upper()}")),
            scored=bool(raw.get("scored", True)),
            tagline=str(raw.get("tagline", "")).strip(),
            philosophy=" ".join(str(raw.get("philosophy", "")).split()),
            params=dict(raw.get("params") or {}),
            picker_params=dict(raw.get("picker_params") or {}),
        )
    except KeyError as e:
        raise ConfigError(f"team entry missing required field {e}") from e


def load_config(
    config_dir: str | Path | None = None,
    *,
    env_file: str | Path | None = None,
    validate: bool = True,
) -> CompetitionConfig:
    """Load `competition.yaml` + `teams.yaml` from `config_dir`."""
    cdir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    comp_path, teams_path = cdir / "competition.yaml", cdir / "teams.yaml"
    for p in (comp_path, teams_path):
        if not p.exists():
            raise ConfigError(f"missing config file: {p}")
    load_dotenv(env_file)

    comp_raw = yaml.safe_load(comp_path.read_text()) or {}
    teams_raw = yaml.safe_load(teams_path.read_text()) or {}
    block = comp_raw.get("competition") or {}

    cfg = CompetitionConfig(
        name=str(block.get("name", "Trading Competition")),
        season=int(block.get("season", datetime.now().year)),
        timezone=str(block.get("timezone", "America/New_York")),
        starting_cash=float(block.get("starting_cash", 5000.0)),
        round_length_days=int(block.get("round_length_days", 7)),
        points_table=tuple(int(x) for x in block.get("points_table", [15, 11, 8, 6, 5, 4, 3, 2])),
        tie_policy=str(block.get("tie_policy", "average")),
        tiebreakers=tuple(str(x) for x in block.get("tiebreakers", ["total_points", "total_return"])),
        risk=_sub(RiskConfig, block.get("risk"), where="competition.risk"),
        fairness=_sub(FairnessConfig, block.get("fairness"), where="competition.fairness"),
        data=_sub(DataConfig, block.get("data"), where="competition.data"),
        rounds=tuple(_parse_round(r) for r in (comp_raw.get("rounds") or [])),
        teams=tuple(_parse_team(t) for t in (teams_raw.get("teams") or [])),
        source_files=(comp_path, teams_path),
    )
    if validate:
        cfg.validate()
    return cfg
