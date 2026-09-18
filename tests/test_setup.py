"""Credential setup: parsing, assignment, verification and writing .env.

Setting this competition up means pasting eighteen secrets in the right
order. These tests cover the ways that goes wrong -- a wrapped secret, a key
pasted twice, a live account mistaken for paper, unequal balances -- and the
one rule that must never break: a secret is never printed.
"""

from __future__ import annotations

import stat

import pytest

from competition.broker.base import BrokerError
from competition.setup import (
    Checked,
    Pair,
    SetupError,
    assign,
    mask,
    parse_pairs,
    render_env,
    report,
    verify,
    write_env,
)

KEY1 = "PKAAAAAAAAAA1111111111"
KEY2 = "PKBBBBBBBBBB2222222222"
SEC1 = "secretsecretsecretsecretaaa1"
SEC2 = "secretsecretsecretsecretbbb2"


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [
    f"{KEY1},{SEC1}",
    f"{KEY1} {SEC1}",
    f"{KEY1}:{SEC1}",
    f"{KEY1};{SEC1}",
    f"  {KEY1} , {SEC1}  ",
])
def test_accepts_every_single_line_shape(text):
    pairs = parse_pairs(text)
    assert len(pairs) == 1
    assert pairs[0].key_id == KEY1 and pairs[0].secret == SEC1


def test_accepts_alternating_bare_lines():
    pairs = parse_pairs(f"{KEY1}\n{SEC1}\n{KEY2}\n{SEC2}")
    assert [p.key_id for p in pairs] == [KEY1, KEY2]
    assert [p.secret for p in pairs] == [SEC1, SEC2]


def test_accepts_explicit_team_assignment():
    pairs = parse_pairs(f"gambler={KEY1},{SEC1}\nscalper:{KEY2},{SEC2}")
    assert [p.team for p in pairs] == ["gambler", "scalper"]


def test_ignores_comments_and_blank_lines():
    pairs = parse_pairs(f"# from the dashboard\n\n{KEY1},{SEC1}\n\n# end\n")
    assert len(pairs) == 1


def test_rejects_an_odd_number_of_bare_lines():
    """The usual cause is a secret that wrapped onto two lines."""
    with pytest.raises(SetupError, match="odd"):
        parse_pairs(f"{KEY1}\n{SEC1}\n{KEY2}")


def test_rejects_a_duplicated_key():
    with pytest.raises(SetupError, match="twice"):
        parse_pairs(f"{KEY1},{SEC1}\n{KEY1},{SEC2}")


def test_rejects_mixed_formats():
    with pytest.raises(SetupError, match="mixed formats"):
        parse_pairs(f"{KEY1}\n{SEC1}\n{KEY2},{SEC2}")


def test_rejects_unparseable_input():
    with pytest.raises(SetupError, match="cannot parse"):
        parse_pairs("this is not a key")


def test_empty_input_is_empty_not_an_error():
    assert parse_pairs("") == []
    assert parse_pairs("# only a comment\n") == []


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #


def test_assigns_positionally_in_roster_order(cfg):
    pairs = [Pair(f"PK{i:020d}", f"secret{i:022d}") for i in range(len(cfg.teams))]
    bound = assign(cfg, pairs)
    assert [t.key for t, _p in bound] == list(cfg.team_keys)


def test_explicit_assignment_wins(cfg):
    pairs = [Pair(KEY1, SEC1, "gambler"), Pair(KEY2, SEC2)]
    bound = dict((t.key, p) for t, p in assign(cfg, pairs))
    assert bound["gambler"].key_id == KEY1
    # The positional pair went to the first team that was not claimed.
    assert bound[cfg.team_keys[0]].key_id == KEY2


def test_rejects_an_unknown_team_key(cfg):
    with pytest.raises(SetupError, match="unknown team"):
        assign(cfg, [Pair(KEY1, SEC1, "nonexistent")])


def test_rejects_more_pairs_than_teams(cfg):
    pairs = [Pair(f"PK{i:020d}", f"secret{i:022d}")
             for i in range(len(cfg.teams) + 2)]
    with pytest.raises(SetupError, match="more credential pair"):
        assign(cfg, pairs)


def test_can_limit_to_specific_teams(cfg):
    bound = assign(cfg, [Pair(KEY1, SEC1)], only=["gambler"])
    assert [t.key for t, _p in bound] == ["gambler"]


def test_fewer_pairs_than_teams_binds_a_prefix(cfg):
    bound = assign(cfg, [Pair(KEY1, SEC1), Pair(KEY2, SEC2)])
    assert len(bound) == 2
    assert [t.key for t, _p in bound] == list(cfg.team_keys[:2])


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def _fake_verify(monkeypatch, responses):
    """Patch AlpacaClient.verify to return canned account info per key id."""
    calls = []

    def fake(self):
        calls.append(self.creds.key_id)
        result = responses.get(self.creds.key_id)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("competition.setup.AlpacaClient.verify", fake)
    return calls


def _good(equity=5000.0, **over):
    base = dict(account_number="PA123", status="ACTIVE", equity=equity,
                cash=equity, paper=True, currency="USD",
                pattern_day_trader=False, shorting_enabled=False,
                trading_blocked=False)
    base.update(over)
    return base


def test_verifies_a_good_pair(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good()})
    team = cfg.team("gambler")
    result = verify([(team, Pair(KEY1, SEC1))])[0]
    assert result.ok and result.paper and not result.fatal
    assert result.equity == 5000.0


def test_a_live_account_is_fatal(cfg, monkeypatch):
    """The single most important check: never trade real money."""
    _fake_verify(monkeypatch, {KEY1: _good(paper=False)})
    result = verify([(cfg.team("gambler"), Pair(KEY1, SEC1))])[0]
    assert "LIVE account" in result.fatal


def test_blocked_trading_is_fatal(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good(trading_blocked=True)})
    result = verify([(cfg.team("gambler"), Pair(KEY1, SEC1))])[0]
    assert "blocked" in result.fatal


def test_a_non_active_status_is_fatal(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good(status="ACCOUNT_CLOSED")})
    result = verify([(cfg.team("gambler"), Pair(KEY1, SEC1))])[0]
    assert "ACCOUNT_CLOSED" in result.fatal


def test_a_bad_key_is_reported_not_raised(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: BrokerError("HTTP 401 unauthorized")})
    result = verify([(cfg.team("gambler"), Pair(KEY1, SEC1))])[0]
    assert not result.ok and "401" in result.fatal


def test_error_text_is_truncated(cfg, monkeypatch):
    """An Alpaca error body can echo a key, so it is clipped."""
    _fake_verify(monkeypatch, {KEY1: BrokerError("x" * 900)})
    result = verify([(cfg.team("gambler"), Pair(KEY1, SEC1))])[0]
    assert len(result.error) <= 160


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _checked(cfg, key, equity, **over):
    base = dict(team=cfg.team(key), pair=Pair(KEY1, SEC1), ok=True,
                account_number="PA1", equity=equity, cash=equity, paper=True,
                status="ACTIVE")
    base.update(over)
    return Checked(**base)


def test_report_never_prints_a_secret(cfg):
    results = [_checked(cfg, "gambler", 5000.0)]
    lines, _ = report(results, bankroll=5000.0)
    body = "\n".join(lines)
    assert SEC1 not in body
    assert KEY1 not in body           # the full key id is masked too
    assert mask(KEY1) in body


def test_report_flags_unequal_balances(cfg):
    results = [_checked(cfg, "gambler", 5000.0),
               _checked(cfg, "scalper", 7500.0)]
    _lines, problems = report(results, bankroll=5000.0)
    assert any("differ" in p for p in problems)


def test_report_accepts_equal_balances(cfg):
    results = [_checked(cfg, "gambler", 5000.0),
               _checked(cfg, "scalper", 5000.0)]
    _lines, problems = report(results, bankroll=5000.0)
    assert problems == []


def test_report_notes_a_non_matching_but_equal_bankroll(cfg):
    """$100k accounts are fine -- sizing is by weight -- but say so."""
    results = [_checked(cfg, "gambler", 100_000.0),
               _checked(cfg, "scalper", 100_000.0)]
    lines, problems = report(results, bankroll=5000.0)
    assert problems == []
    assert any("not the $5,000" in ln for ln in lines)


def test_report_surfaces_fatal_rows(cfg):
    results = [_checked(cfg, "gambler", 5000.0, paper=False)]
    lines, problems = report(results, bankroll=5000.0)
    assert any("FAIL" in ln for ln in lines)
    assert problems and "LIVE" in problems[0]


# --------------------------------------------------------------------------- #
# writing .env
# --------------------------------------------------------------------------- #


def test_env_body_contains_every_team(cfg):
    results = [_checked(cfg, t.key, 5000.0) for t in cfg.teams]
    body = render_env(results)
    for team in cfg.teams:
        assert f"{team.env_prefix}_KEY_ID={KEY1}" in body
        assert f"{team.env_prefix}_SECRET_KEY={SEC1}" in body
    assert "ALPACA_DATA_KEY_ID=" in body
    assert "ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets" in body


def test_shared_data_keys_can_be_chosen(cfg):
    results = [_checked(cfg, "gambler", 5000.0),
               Checked(team=cfg.team("scalper"), pair=Pair(KEY2, SEC2), ok=True,
                       paper=True, status="ACTIVE", equity=5000.0)]
    body = render_env(results, data_from="scalper")
    assert f"ALPACA_DATA_KEY_ID={KEY2}" in body


def test_written_env_is_owner_only(cfg, tmp_path):
    results = [_checked(cfg, t.key, 5000.0) for t in cfg.teams]
    target, backup = write_env(render_env(results), tmp_path / ".env")
    assert backup is None
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600, f"mode {oct(mode)} -- secrets must not be world-readable"


def test_existing_env_is_backed_up(cfg, tmp_path):
    path = tmp_path / ".env"
    path.write_text("OLD=content\n")
    results = [_checked(cfg, t.key, 5000.0) for t in cfg.teams]
    target, backup = write_env(render_env(results), path)
    assert backup is not None and backup.exists()
    assert backup.read_text() == "OLD=content\n"
    assert "OLD=content" not in target.read_text()


def test_written_env_round_trips_through_the_loader(cfg, tmp_path, monkeypatch):
    """What we write must be what `load_dotenv` reads back."""
    from competition.config import load_dotenv

    results = [_checked(cfg, t.key, 5000.0) for t in cfg.teams]
    target, _ = write_env(render_env(results), tmp_path / ".env")
    for team in cfg.teams:
        monkeypatch.delenv(f"{team.env_prefix}_KEY_ID", raising=False)
        monkeypatch.delenv(f"{team.env_prefix}_SECRET_KEY", raising=False)
    loaded = load_dotenv(target, override=True)
    for team in cfg.teams:
        assert loaded[f"{team.env_prefix}_KEY_ID"] == KEY1
        assert loaded[f"{team.env_prefix}_SECRET_KEY"] == SEC1
        assert team.has_credentials


def test_masking():
    assert mask(KEY1).startswith("PKAA") and mask(KEY1).endswith("1111")
    assert KEY1 not in mask(KEY1)
    assert mask("short") == "…"
