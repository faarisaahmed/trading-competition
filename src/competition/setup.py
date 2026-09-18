"""Credential wrangling: parse, verify and write nine key pairs safely.

Setting up this competition means pasting eighteen secrets into a file in the
right order. Doing that by hand is where the mistakes happen -- a key pasted
into the wrong team, a secret truncated on a line wrap, a live account
mistaken for a paper one. So it is a command instead:

    comp setup-accounts --from-file keys.txt     # bulk
    comp setup-accounts                          # prompt per team
    comp setup-accounts --verify                 # check what is already there

It parses, checks each pair against Alpaca, refuses live accounts, warns when
balances differ (unequal bankrolls are the one thing that invalidates a
round), and only then writes `.env` -- backing up any existing one first.

Secrets are never logged, printed or echoed. Key ids are shown truncated.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .broker.alpaca import AlpacaClient, AlpacaCredentials
from .broker.base import BrokerError
from .config import CompetitionConfig, TeamConfig

#: A pasted pair, however the dashboard formatted it.
_PAIR = re.compile(
    r"^\s*(?:(?P<team>[a-z][a-z0-9_]*)\s*[:=]\s*)?"
    r"(?P<key>[A-Za-z0-9]{12,})\s*[,;:\s]\s*(?P<secret>[A-Za-z0-9/+_-]{20,})\s*$"
)
_BARE = re.compile(r"^\s*(?P<value>[A-Za-z0-9/+_-]{12,})\s*$")


class SetupError(RuntimeError):
    """Malformed input, or credentials that must not be used."""


def mask(key_id: str) -> str:
    """A key id, safe to print."""
    if len(key_id) <= 8:
        return "…"
    return f"{key_id[:4]}…{key_id[-4:]}"


@dataclass
class Pair:
    key_id: str
    secret: str
    team: str | None = None

    def masked(self) -> str:
        return mask(self.key_id)


@dataclass
class Checked:
    """One verified (or failed) credential pair, bound to a team."""

    team: TeamConfig
    pair: Pair
    ok: bool = False
    account_number: str = ""
    equity: float = 0.0
    cash: float = 0.0
    paper: bool = False
    status: str = ""
    blocked: bool = False
    shorting: bool = False
    error: str = ""

    @property
    def fatal(self) -> str:
        """A reason this pair must not be used, or "" if it is fine."""
        if not self.ok:
            return self.error or "verification failed"
        if not self.paper:
            return ("this is a LIVE account, not paper -- refusing to trade "
                    "real money")
        if self.blocked:
            return "trading is blocked on this account"
        if self.status and self.status.upper() not in ("ACTIVE", "ACCOUNT_UPDATED"):
            return f"account status is {self.status}"
        return ""


def parse_pairs(text: str) -> list[Pair]:
    """Parse pasted credentials in any of the shapes a dashboard gives you.

    Accepted, one account per line:
        KEY,SECRET        KEY SECRET        KEY:SECRET
        team_key=KEY,SECRET          (explicit assignment)
    Or alternating bare lines:
        KEY
        SECRET
    Blank lines and `#` comments are ignored.
    """
    lines = [ln for ln in (raw.strip() for raw in text.splitlines())
             if ln and not ln.startswith("#")]
    pairs: list[Pair] = []
    bare: list[str] = []

    for ln in lines:
        m = _PAIR.match(ln)
        if m:
            if bare:
                raise SetupError(
                    f"mixed formats: {len(bare)} bare line(s) before "
                    f"{mask(m.group('key'))}. Use one style throughout."
                )
            pairs.append(Pair(m.group("key"), m.group("secret"), m.group("team")))
            continue
        b = _BARE.match(ln)
        if b:
            bare.append(b.group("value"))
            continue
        raise SetupError(f"cannot parse line: {ln[:24]}…")

    if bare:
        if len(bare) % 2:
            raise SetupError(
                f"{len(bare)} bare values is odd -- alternating KEY/SECRET lines "
                f"need an even count. Did a secret wrap onto two lines?"
            )
        for i in range(0, len(bare), 2):
            pairs.append(Pair(bare[i], bare[i + 1]))

    seen: dict[str, int] = {}
    for i, p in enumerate(pairs, 1):
        if p.key_id in seen:
            raise SetupError(
                f"key {p.masked()} appears twice (entries {seen[p.key_id]} and "
                f"{i}). Each team needs its own account."
            )
        seen[p.key_id] = i
    return pairs


def assign(cfg: CompetitionConfig, pairs: Sequence[Pair],
           *, only: Sequence[str] | None = None) -> list[tuple[TeamConfig, Pair]]:
    """Bind pairs to teams -- explicitly where given, else in roster order."""
    teams = [t for t in cfg.teams if not only or t.key in set(only)]
    explicit = {p.team: p for p in pairs if p.team}
    unknown = set(explicit) - {t.key for t in cfg.teams}
    if unknown:
        raise SetupError(f"unknown team key(s): {', '.join(sorted(unknown))}")

    positional = [p for p in pairs if not p.team]
    out: list[tuple[TeamConfig, Pair]] = []
    cursor = 0
    for team in teams:
        if team.key in explicit:
            out.append((team, explicit[team.key]))
        elif cursor < len(positional):
            out.append((team, positional[cursor]))
            cursor += 1
    if cursor < len(positional):
        raise SetupError(
            f"{len(positional) - cursor} more credential pair(s) than teams to "
            f"assign them to ({len(teams)})."
        )
    return out


def verify(bound: Iterable[tuple[TeamConfig, Pair]]) -> list[Checked]:
    """Hit Alpaca once per pair and report what each account actually is."""
    out: list[Checked] = []
    for team, pair in bound:
        checked = Checked(team=team, pair=pair)
        try:
            creds = AlpacaCredentials(key_id=pair.key_id, secret_key=pair.secret)
            info = AlpacaClient(creds).verify()
            checked.ok = True
            checked.account_number = str(info.get("account_number", "?"))
            checked.equity = float(info.get("equity") or 0.0)
            checked.cash = float(info.get("cash") or 0.0)
            checked.paper = bool(info.get("paper"))
            checked.status = str(info.get("status") or "")
            checked.blocked = bool(info.get("trading_blocked"))
            checked.shorting = bool(info.get("shorting_enabled"))
        except BrokerError as e:
            # Deliberately truncated: an Alpaca error body can echo a key.
            checked.error = str(e)[:160]
        out.append(checked)
    return out


def report(results: Sequence[Checked], *, bankroll: float) -> tuple[list[str], list[str]]:
    """(lines, problems). Never includes a secret."""
    lines: list[str] = []
    problems: list[str] = []
    width = max((len(c.team.key) for c in results), default=12)

    for c in results:
        if c.fatal:
            lines.append(f"  FAIL  {c.team.key:<{width}}  {c.pair.masked()}  {c.fatal}")
            problems.append(f"{c.team.key}: {c.fatal}")
            continue
        flags = []
        if c.shorting:
            flags.append("shorting enabled (unused: the competition is long-only)")
        lines.append(
            f"  ok    {c.team.key:<{width}}  {c.pair.masked()}  "
            f"#{c.account_number:<12} equity ${c.equity:>12,.2f}"
            + ("  " + "; ".join(flags) if flags else "")
        )

    good = [c for c in results if not c.fatal]
    if len(good) > 1:
        equities = [c.equity for c in good]
        spread = max(equities) - min(equities)
        tolerance = max(0.01 * bankroll, 5.0)
        lines.append("")
        lines.append(
            f"  balances: ${min(equities):,.2f} .. ${max(equities):,.2f} "
            f"(spread ${spread:,.2f}, tolerance ${tolerance:,.2f})"
        )
        if spread > tolerance:
            problems.append(
                f"account balances differ by ${spread:,.2f}. Every team must start "
                f"from the same bankroll or the round is not comparable -- reset "
                f"the paper accounts to a common value."
            )
        off = [c for c in good if abs(c.equity - bankroll) > tolerance]
        if off and spread <= tolerance:
            lines.append(
                f"  note: balances are ${good[0].equity:,.2f}, not the "
                f"${bankroll:,.2f} in the rulebook. That is fine -- every strategy "
                f"sizes by weight of equity and the order caps scale with the "
                f"bankroll -- but set `starting_cash` to match so the reports read "
                f"correctly."
            )
    return lines, problems


def render_env(
    results: Sequence[Checked],
    *,
    data_from: str | None = None,
    feed: str = "iex",
    trading_url: str = "https://paper-api.alpaca.markets",
    data_url: str = "https://data.alpaca.markets",
) -> str:
    """The `.env` body. `data_from` names the team whose keys feed market data."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    shared = next((c for c in results if c.team.key == data_from), None)
    if shared is None:
        shared = next((c for c in results if not c.fatal), None)

    out = [
        "# ---------------------------------------------------------------------",
        f"# Written by `comp setup-accounts` on {stamp}.",
        "# Secrets live here and nowhere else. This file is gitignored --",
        "# keep it that way.",
        "# ---------------------------------------------------------------------",
        "",
        "# Market data is read ONCE per tick from this one account and shared",
        "# with every team, so all strategies see a byte-identical tape.",
    ]
    if shared is not None:
        out += [
            f"ALPACA_DATA_KEY_ID={shared.pair.key_id}",
            f"ALPACA_DATA_SECRET_KEY={shared.pair.secret}",
        ]
    else:
        out += ["ALPACA_DATA_KEY_ID=", "ALPACA_DATA_SECRET_KEY="]
    out += [
        "",
        f"ALPACA_TRADING_BASE_URL={trading_url}",
        f"ALPACA_DATA_BASE_URL={data_url}",
        f"ALPACA_DATA_FEED={feed}",
        "",
        "# One paper account per team.",
    ]
    for c in results:
        note = f"   # {c.fatal}" if c.fatal else f"   # account #{c.account_number}"
        out += [
            "",
            f"# {c.team.name}{note}",
            f"{c.team.env_prefix}_KEY_ID={c.pair.key_id}",
            f"{c.team.env_prefix}_SECRET_KEY={c.pair.secret}",
        ]
    return "\n".join(out) + "\n"


def render_template(cfg: CompetitionConfig) -> str:
    """A `keys.txt` skeleton with one labelled line per team.

    Every line carries an explicit `team=` prefix, so the pairs bind by name
    rather than by position. That removes the one setup error that is both easy
    to make and nearly invisible afterwards: pasting nine keys one row out, so
    every team trades the account labelled for its neighbour.
    """
    lines = [
        "# Alpaca PAPER credentials, one account per team.",
        "#",
        "# Each line is:   <team>=<KEY_ID>,<SECRET>",
        "# The team= prefix binds by name, so the ORDER of these lines does not",
        "# matter -- you cannot paste them one row out.",
        "#",
        "# The Alpaca dashboard's Nickname field takes the env prefix shown",
        "# per team -- naming the account after the variable it fills removes",
        "# any doubt about which account belongs to which strategy.",
        "# Delete this file once `comp setup-accounts --from-file keys.txt` has",
        "# written .env.",
        "",
    ]
    for i, team in enumerate(cfg.teams, 1):
        note = "" if team.scored else "   (unscored reference)"
        lines += [
            f"# {i}. {team.name}{note}",
            f"#    Alpaca Nickname: {team.env_prefix}    Set Funds: "
            f"{cfg.starting_cash:,.0f}",
            f"{team.key}=",
            "",
        ]
    return "\n".join(lines)


def write_env(body: str, path: str | Path) -> tuple[Path, Path | None]:
    """Write `.env` with 0600 permissions, backing up any existing file."""
    target = Path(path)
    backup: Path | None = None
    if target.exists():
        backup = target.with_suffix(
            target.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}"
        )
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    target.chmod(0o600)
    return target, backup
