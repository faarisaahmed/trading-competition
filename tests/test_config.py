"""Rulebook loading and validation."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from competition.config import (
    CompetitionConfig,
    ConfigError,
    RiskConfig,
    load_config,
    load_dotenv,
)


def test_loads_the_shipped_rulebook(cfg):
    assert cfg.starting_cash == 5000.0
    assert cfg.round_length_days == 7
    assert len(cfg.rounds) == 3
    assert len(cfg.scored_teams) == 8
    assert len(cfg.teams) == 9          # + the unscored benchmark


def test_rounds_are_the_three_described_in_the_brief(cfg):
    r1, r2, r3 = (cfg.round(i) for i in (1, 2, 3))
    assert r1.universe_mode == "fixed"
    assert set(r1.symbols) == {"AAPL", "GOOGL", "MSFT"}
    assert not r1.picker_enabled
    assert r2.universe_mode == "picker" and r2.picker_enabled
    assert r3.universe_mode == "draft" and r3.picker_enabled
    assert r3.draft.pool_size == 500
    assert r3.draft.picks_per_team == 10
    assert r3.draft.fair_rank_sum == 2505


def test_points_table_covers_every_scored_team(cfg):
    assert len(cfg.points_table) >= len(cfg.scored_teams)
    assert list(cfg.points_table) == sorted(cfg.points_table, reverse=True)
    assert cfg.points_for_place(1) > cfg.points_for_place(8)
    assert cfg.points_for_place(99) == 0


def test_every_team_has_a_distinct_account_prefix(cfg):
    prefixes = [t.env_prefix for t in cfg.teams]
    assert len(set(prefixes)) == len(prefixes)


def test_rules_hash_is_stable_and_sensitive(cfg):
    assert cfg.rules_hash == load_config().rules_hash
    tweaked = CompetitionConfig(**{**cfg.__dict__, "starting_cash": 6000.0})
    assert tweaked.rules_hash != cfg.rules_hash


def test_team_seeds_are_deterministic_and_distinct(cfg):
    a = cfg.team_seed("trend_rider", 1)
    assert a == cfg.team_seed("trend_rider", 1)
    assert a != cfg.team_seed("trend_rider", 2)
    assert a != cfg.team_seed("gambler", 1)


def test_round_window_is_seven_calendar_days(cfg):
    start, end = cfg.round_window(1, date(2026, 9, 7))
    assert start == date(2026, 9, 7)
    assert end == date(2026, 9, 13)
    assert (end - start).days == 6


def test_unknown_team_lookup_is_explicit(cfg):
    with pytest.raises(KeyError, match="no such team"):
        cfg.team("does_not_exist")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _write(tmp_path, comp: dict, teams: dict):
    (tmp_path / "competition.yaml").write_text(yaml.safe_dump(comp))
    (tmp_path / "teams.yaml").write_text(yaml.safe_dump(teams))
    return tmp_path


def _base(cfg_dir):
    comp = yaml.safe_load((cfg_dir / "competition.yaml").read_text())
    teams = yaml.safe_load((cfg_dir / "teams.yaml").read_text())
    return comp, teams


def test_rejects_unknown_config_key(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    comp["competition"]["risk"]["nonsense_key"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_ascending_points_table(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    comp["competition"]["points_table"] = [1, 2, 3, 4, 5, 6, 7, 8]
    with pytest.raises(ConfigError, match="non-increasing"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_too_short_points_table(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    comp["competition"]["points_table"] = [10, 5]
    with pytest.raises(ConfigError, match="points_table has"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_shared_env_prefix(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    teams["teams"][1]["env_prefix"] = teams["teams"][0]["env_prefix"]
    with pytest.raises(ConfigError, match="share an env_prefix"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_draft_pool_too_small(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    for r in comp["rounds"]:
        if r.get("draft"):
            r["draft"]["pool_size"] = 20
    with pytest.raises(ConfigError, match="too small"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_picker_cap_above_hand_size(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    for r in comp["rounds"]:
        if r.get("draft"):
            r["picker"]["max_symbols"] = 20
    with pytest.raises(ConfigError, match="exceeds the dealt hand"):
        load_config(_write(tmp_path, comp, teams))


def test_rejects_bad_timeframe(tmp_path, cfg):
    comp, teams = _base(cfg.source_files[0].parent)
    comp["competition"]["data"]["primary_timeframe"] = "5minutes"
    with pytest.raises(ConfigError, match="Alpaca timeframe"):
        load_config(_write(tmp_path, comp, teams))


def test_risk_validation_catches_inconsistencies():
    with pytest.raises(ConfigError, match="max_gross_leverage"):
        RiskConfig(max_gross_leverage=0).validate()
    with pytest.raises(ConfigError, match="max_position_pct"):
        RiskConfig(max_position_pct=2.0).validate()
    with pytest.raises(ConfigError, match="kill_switch"):
        RiskConfig(daily_loss_kill_switch_pct=1.5).validate()


# --------------------------------------------------------------------------- #
# dotenv
# --------------------------------------------------------------------------- #


def test_dotenv_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "PLAIN=value\n"
        'QUOTED="quoted value"\n'
        "SQ='single'\n"
        "export EXPORTED=yes\n"
        "EMPTY=\n"
        "\n"
        "SPACED  =  padded\n"
    )
    for k in ("PLAIN", "QUOTED", "SQ", "EXPORTED", "SPACED", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    loaded = load_dotenv(env)
    assert loaded["PLAIN"] == "value"
    assert loaded["QUOTED"] == "quoted value"
    assert loaded["SQ"] == "single"
    assert loaded["EXPORTED"] == "yes"
    assert loaded["SPACED"] == "padded"
    assert "EMPTY" not in loaded          # blank values are skipped, not set to ""


def test_dotenv_does_not_override_by_default(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PRESET=from_file\n")
    monkeypatch.setenv("PRESET", "from_env")
    load_dotenv(env)
    import os
    assert os.environ["PRESET"] == "from_env"


def test_missing_dotenv_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}
