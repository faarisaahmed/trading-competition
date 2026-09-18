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
from competition.config import AccountsConfig
from competition.setup import (
    Checked,
    Pair,
    SetupError,
    assign,
    mask,
    parse_pairs,
    render_env,
    render_template,
    report,
    targets_for,
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
# account targets: one credential pair per REAL account
# --------------------------------------------------------------------------- #


@pytest.fixture
def per_team_cfg(cfg):
    """The same rulebook, but with one Alpaca account per team."""
    import dataclasses
    return dataclasses.replace(
        cfg, accounts=AccountsConfig(mode="per_team", groups=()))


@pytest.fixture
def shared_cfg(cfg):
    """Three accounts of three teams -- the shipped default."""
    assert cfg.accounts.is_shared, "the rulebook is expected to ship shared"
    return cfg


def test_per_team_mode_needs_one_account_per_team(per_team_cfg):
    targets = targets_for(per_team_cfg)
    assert len(targets) == len(per_team_cfg.teams)
    assert all(not t.is_group for t in targets)
    assert all(t.capital == per_team_cfg.starting_cash for t in targets)


def test_shared_mode_needs_one_account_per_group(shared_cfg):
    targets = targets_for(shared_cfg)
    assert len(targets) == len(shared_cfg.accounts.groups) == 3
    assert all(t.is_group for t in targets)
    # Each account must hold the whole group's capital.
    for target in targets:
        assert target.capital == pytest.approx(
            shared_cfg.starting_cash * len(target.teams))
    # And every team is covered exactly once.
    covered = [k for t in targets for k in t.teams]
    assert sorted(covered) == sorted(shared_cfg.team_keys)


def test_shared_targets_total_the_whole_field(shared_cfg):
    targets = targets_for(shared_cfg)
    total = sum(t.capital for t in targets)
    assert total == pytest.approx(shared_cfg.starting_cash * len(shared_cfg.teams))


# --------------------------------------------------------------------------- #
# assignment
# --------------------------------------------------------------------------- #


def _pairs(n):
    return [Pair(f"PK{i:020d}", f"secret{i:022d}") for i in range(n)]


def test_assigns_positionally_in_target_order(shared_cfg):
    targets = targets_for(shared_cfg)
    bound = assign(shared_cfg, _pairs(len(targets)))
    assert [t.key for t, _p in bound] == [t.key for t in targets]


def test_assigns_per_team_when_configured(per_team_cfg):
    bound = assign(per_team_cfg, _pairs(len(per_team_cfg.teams)))
    assert [t.key for t, _p in bound] == list(per_team_cfg.team_keys)


def test_explicit_labels_bind_by_name(shared_cfg):
    pairs = parse_pairs(f"group-c={KEY1},{SEC1}\ngroup-a={KEY2},{SEC2}")
    bound = dict((t.key, p.key_id) for t, p in assign(shared_cfg, pairs))
    assert bound["group-c"] == KEY1
    assert bound["group-a"] == KEY2


def test_labelled_pairs_bind_by_name_not_position(shared_cfg):
    """The error the template exists to prevent: everything one row out."""
    pairs = parse_pairs(
        f"group-b={KEY1},{SEC1}\n"
        f"group-a={KEY2},{SEC2}\n"      # deliberately reversed
    )
    bound = dict((t.key, p.key_id) for t, p in assign(shared_cfg, pairs))
    assert bound["group-b"] == KEY1
    assert bound["group-a"] == KEY2


def test_rejects_an_unknown_account_key(shared_cfg):
    with pytest.raises(SetupError, match="unknown account"):
        assign(shared_cfg, [Pair(KEY1, SEC1, "nonexistent")])


def test_rejects_more_pairs_than_accounts(shared_cfg):
    n = len(targets_for(shared_cfg))
    with pytest.raises(SetupError, match="more credential pair"):
        assign(shared_cfg, _pairs(n + 2))


def test_fewer_pairs_than_accounts_binds_a_prefix(shared_cfg):
    bound = assign(shared_cfg, _pairs(2))
    assert len(bound) == 2
    assert [t.key for t, _p in bound] == [t.key for t in targets_for(shared_cfg)][:2]


def test_can_limit_to_specific_teams(shared_cfg):
    """Limiting to one team narrows to the account that team lives in."""
    bound = assign(shared_cfg, [Pair(KEY1, SEC1)], only=["gambler"])
    assert len(bound) == 1
    assert "gambler" in bound[0][0].teams


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def _target(cfg, key=None):
    targets = targets_for(cfg)
    if key is None:
        return targets[0]
    return next(t for t in targets if t.key == key)


def _fake_verify(monkeypatch, responses):
    """Patch AlpacaClient.verify to return canned account info per key id."""
    def fake(self):
        result = responses.get(self.creds.key_id)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("competition.setup.AlpacaClient.verify", fake)


def _good(equity=15000.0, **over):
    base = dict(account_number="PA123", status="ACTIVE", equity=equity,
                cash=equity, paper=True, currency="USD",
                pattern_day_trader=False, shorting_enabled=False,
                trading_blocked=False)
    base.update(over)
    return base


def test_verifies_a_good_pair(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good()})
    result = verify([(_target(cfg), Pair(KEY1, SEC1))])[0]
    assert result.ok and result.paper and not result.fatal
    assert result.equity == 15000.0


def test_a_live_account_is_fatal(cfg, monkeypatch):
    """The single most important check: never trade real money."""
    _fake_verify(monkeypatch, {KEY1: _good(paper=False)})
    result = verify([(_target(cfg), Pair(KEY1, SEC1))])[0]
    assert "LIVE account" in result.fatal


def test_blocked_trading_is_fatal(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good(trading_blocked=True)})
    assert "blocked" in verify([(_target(cfg), Pair(KEY1, SEC1))])[0].fatal


def test_a_non_active_status_is_fatal(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: _good(status="ACCOUNT_CLOSED")})
    assert "ACCOUNT_CLOSED" in verify([(_target(cfg), Pair(KEY1, SEC1))])[0].fatal


def test_a_bad_key_is_reported_not_raised(cfg, monkeypatch):
    _fake_verify(monkeypatch, {KEY1: BrokerError("HTTP 401 unauthorized")})
    result = verify([(_target(cfg), Pair(KEY1, SEC1))])[0]
    assert not result.ok and "401" in result.fatal


def test_error_text_is_truncated(cfg, monkeypatch):
    """An Alpaca error body can echo a key, so it is clipped."""
    _fake_verify(monkeypatch, {KEY1: BrokerError("x" * 900)})
    assert len(verify([(_target(cfg), Pair(KEY1, SEC1))])[0].error) <= 160


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _checked(cfg, equity, key=None, **over):
    base = dict(team=_target(cfg, key), pair=Pair(KEY1, SEC1), ok=True,
                account_number="PA1", equity=equity, cash=equity, paper=True,
                status="ACTIVE")
    base.update(over)
    return Checked(**base)


def test_report_never_prints_a_secret(cfg):
    lines, _ = report([_checked(cfg, 15000.0)], bankroll=cfg.starting_cash)
    body = "\n".join(lines)
    assert SEC1 not in body
    assert KEY1 not in body           # the full key id is masked too
    assert mask(KEY1) in body


def test_report_accepts_correctly_funded_accounts(shared_cfg):
    results = [_checked(shared_cfg, t.capital, t.key)
               for t in targets_for(shared_cfg)]
    _lines, problems = report(results, bankroll=shared_cfg.starting_cash)
    assert problems == []


def test_report_flags_an_underfunded_shared_account(shared_cfg):
    """A group account funded for one team starves the other two."""
    results = [_checked(shared_cfg, shared_cfg.starting_cash, t.key)
               for t in targets_for(shared_cfg)]
    lines, problems = report(results, bankroll=shared_cfg.starting_cash)
    assert problems, "an underfunded group account was accepted"
    assert any("need" in p for p in problems)
    assert any("FUNDS" in ln for ln in lines)


def test_report_flags_an_overfunded_account(shared_cfg):
    results = [_checked(shared_cfg, t.capital * 3, t.key)
               for t in targets_for(shared_cfg)]
    _lines, problems = report(results, bankroll=shared_cfg.starting_cash)
    assert problems


def test_report_flags_unequal_per_team_accounts(per_team_cfg):
    targets = targets_for(per_team_cfg)
    results = [_checked(per_team_cfg, per_team_cfg.starting_cash, targets[0].key),
               _checked(per_team_cfg, per_team_cfg.starting_cash * 2,
                        targets[1].key)]
    _lines, problems = report(results, bankroll=per_team_cfg.starting_cash)
    assert problems and any("same bankroll" in p for p in problems)


def test_report_surfaces_fatal_rows(cfg):
    lines, problems = report([_checked(cfg, 15000.0, paper=False)],
                             bankroll=cfg.starting_cash)
    assert any("FAIL" in ln for ln in lines)
    assert problems and "LIVE" in problems[0]


# --------------------------------------------------------------------------- #
# writing .env
# --------------------------------------------------------------------------- #


def test_env_body_contains_every_account(cfg):
    results = [_checked(cfg, t.capital, t.key) for t in targets_for(cfg)]
    body = render_env(results)
    for target in targets_for(cfg):
        assert f"{target.env_prefix}_KEY_ID={KEY1}" in body
        assert f"{target.env_prefix}_SECRET_KEY={SEC1}" in body
    assert "ALPACA_DATA_KEY_ID=" in body
    assert "ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets" in body


def test_env_records_which_teams_share_an_account(shared_cfg):
    results = [_checked(shared_cfg, t.capital, t.key)
               for t in targets_for(shared_cfg)]
    body = render_env(results)
    assert "teams" in body and "holds $" in body


def test_shared_data_keys_can_be_chosen(cfg):
    targets = targets_for(cfg)
    results = [
        _checked(cfg, targets[0].capital, targets[0].key),
        Checked(team=targets[1], pair=Pair(KEY2, SEC2), ok=True, paper=True,
                status="ACTIVE", equity=targets[1].capital),
    ]
    body = render_env(results, data_from=targets[1].key)
    assert f"ALPACA_DATA_KEY_ID={KEY2}" in body


def test_written_env_is_owner_only(cfg, tmp_path):
    results = [_checked(cfg, t.capital, t.key) for t in targets_for(cfg)]
    target, backup = write_env(render_env(results), tmp_path / ".env")
    assert backup is None
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600, f"mode {oct(mode)} -- secrets must not be world-readable"


def test_existing_env_is_backed_up(cfg, tmp_path):
    path = tmp_path / ".env"
    path.write_text("OLD=content\n")
    results = [_checked(cfg, t.capital, t.key) for t in targets_for(cfg)]
    written, backup = write_env(render_env(results), path)
    assert backup is not None and backup.read_text() == "OLD=content\n"
    assert "OLD=content" not in written.read_text()


def test_written_env_round_trips_through_the_loader(cfg, tmp_path, monkeypatch):
    """What we write must be what `load_dotenv` reads back."""
    import os

    from competition.config import load_dotenv

    results = [_checked(cfg, t.capital, t.key) for t in targets_for(cfg)]
    written, _ = write_env(render_env(results), tmp_path / ".env")
    for target in targets_for(cfg):
        monkeypatch.delenv(f"{target.env_prefix}_KEY_ID", raising=False)
        monkeypatch.delenv(f"{target.env_prefix}_SECRET_KEY", raising=False)
    loaded = load_dotenv(written, override=True)
    for target in targets_for(cfg):
        assert loaded[f"{target.env_prefix}_KEY_ID"] == KEY1
        assert os.environ[f"{target.env_prefix}_SECRET_KEY"] == SEC1


# --------------------------------------------------------------------------- #
# the labelled template
# --------------------------------------------------------------------------- #


def test_template_lists_every_account(cfg):
    body = render_template(cfg)
    for i, target in enumerate(targets_for(cfg), 1):
        assert f"{i}. {target.name}" in body
        assert f"\n{target.key}=" in body
        assert f"Nickname: {target.env_prefix}" in body
        assert f"Set Funds: {target.capital:,.0f}" in body


def test_template_warns_when_accounts_are_shared(shared_cfg):
    body = render_template(shared_cfg)
    assert "SHARED" in body
    assert "virtual book" in body
    assert "UNCHECKED" in body


def test_template_parses_back_once_filled(cfg):
    """The skeleton must be valid input after the keys are pasted in."""
    body = render_template(cfg)
    filled = []
    for i, line in enumerate(body.splitlines()):
        if line.endswith("=") and not line.startswith("#"):
            filled.append(f"{line}PK{i:020d},secret{i:022d}")
        else:
            filled.append(line)
    pairs = parse_pairs("\n".join(filled))
    targets = targets_for(cfg)
    assert len(pairs) == len(targets)
    assert {p.team for p in pairs} == {t.key for t in targets}
    assert [t.key for t, _p in assign(cfg, pairs)] == [t.key for t in targets]


def test_masking():
    assert mask(KEY1).startswith("PKAA") and mask(KEY1).endswith("1111")
    assert KEY1 not in mask(KEY1)
    assert mask("short") == "\u2026"
