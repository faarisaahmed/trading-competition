"""The `comp` command line. Smoke-level, but it covers every subcommand."""

from __future__ import annotations

import json

import pytest

from competition.cli import build_parser, main


def run(argv, ledger=None):
    args = list(argv)
    if ledger is not None:
        args = ["--ledger", str(ledger)] + args
    return main(args)


def test_parser_exposes_every_documented_command():
    parser = build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)
    assert {"doctor", "teams", "universe", "refresh-universe", "draft",
            "verify-draft", "pretrain", "backtest", "run", "score",
            "leaderboard", "report", "lexicon", "explain-news"} <= choices


def test_doctor_succeeds_without_credentials(capsys, tmp_path):
    # No keys configured is a warning, not an error -- sim mode still works.
    assert run(["--quiet", "doctor"], tmp_path / "l.sqlite") == 0
    out = capsys.readouterr().out
    assert "rules hash" in out
    assert "strategies and pickers" in out
    assert "FAIL" not in out


def test_teams_describes_the_field(capsys, tmp_path, cfg):
    assert run(["--quiet", "teams"], tmp_path / "l.sqlite") == 0
    out = capsys.readouterr().out
    for team in cfg.teams:
        assert team.name in out
    assert "STRATEGY" in out and "PICKER" in out


def test_teams_can_filter(capsys, tmp_path):
    assert run(["--quiet", "teams", "--team", "gambler"], tmp_path / "l.sqlite") == 0
    out = capsys.readouterr().out
    assert "The Gambler" in out and "Trend Rider" not in out


def test_unknown_team_is_an_error(capsys, tmp_path):
    assert run(["--quiet", "teams", "--team", "nope"], tmp_path / "l.sqlite") == 2
    assert "unknown team" in capsys.readouterr().out


def test_universe_audit(capsys, tmp_path):
    assert run(["--quiet", "universe", "--size", "100"], tmp_path / "l.sqlite") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 100
    assert payload["metric"] == "market_cap"
    assert len(payload["top_10"]) == 10
    assert payload["sectors"]


def test_universe_by_share_price(capsys, tmp_path):
    assert run(["--quiet", "universe", "--metric", "share_price", "--size", "50"],
               tmp_path / "l.sqlite") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "share_price"


def test_lexicon_stats(capsys, tmp_path):
    assert run(["--quiet", "lexicon"], tmp_path / "l.sqlite") == 0
    assert "phrases" in capsys.readouterr().out


def test_explain_news(capsys, tmp_path):
    assert run(["--quiet", "explain-news", "Apple beats estimates and raises guidance"],
               tmp_path / "l.sqlite") == 0
    out = capsys.readouterr().out
    assert "score:" in out and "beats estimates" in out


def test_draft_prints_a_fairness_proof(capsys, tmp_path):
    ledger = tmp_path / "l.sqlite"
    assert run(["--quiet", "draft", "--round", "3", "--seed", "5", "--write"],
               ledger) == 0
    out = capsys.readouterr().out
    assert "target sum=2505" in out
    assert "fairness proof" in out
    assert "every hand's rank sum : [2505]" in out
    assert "no overlap" in out
    assert "independent verification: CLEAN" in out
    # And it is now recoverable from the ledger.
    assert run(["--quiet", "verify-draft", "--round", "3"], ledger) == 0
    assert "VERIFIED" in capsys.readouterr().out


def test_draft_can_write_json(tmp_path, capsys):
    out_json = tmp_path / "draft.json"
    assert run(["--quiet", "draft", "--round", "3", "--seed", "9",
                "--json", str(out_json)], tmp_path / "l.sqlite") == 0
    payload = json.loads(out_json.read_text())
    assert payload["target_sum"] == 2505
    # All nine entries are dealt a hand, including the unscored benchmark --
    # it needs ten names of its own to be a Round 3 reference line.
    assert len(payload["hands"]) == 9
    assert all(sum(h["ranks"]) == 2505 for h in payload["hands"])
    assert all(len(h["symbols"]) == 10 for h in payload["hands"])
    dealt = [s for h in payload["hands"] for s in h["symbols"]]
    assert len(dealt) == len(set(dealt)) == 90


def test_draft_on_a_non_draft_round_is_refused(capsys, tmp_path):
    assert run(["--quiet", "draft", "--round", "1"], tmp_path / "l.sqlite") == 1
    assert "no draft" in capsys.readouterr().out


def test_verify_draft_with_nothing_recorded(capsys, tmp_path):
    assert run(["--quiet", "verify-draft", "--round", "3"], tmp_path / "l.sqlite") == 1
    assert "no draft recorded" in capsys.readouterr().out


def test_pretrain_on_synthetic_data(capsys, tmp_path):
    ledger = tmp_path / "l.sqlite"
    rc = run(["--quiet", "pretrain", "--source", "synthetic", "--passes", "1",
              "--days", "25", "--symbols", "AAPL,MSFT,GOOGL",
              "--start", "2026-08-10", "--end", "2026-09-04", "--seed", "3"], ledger)
    assert rc == 0
    out = capsys.readouterr().out
    assert '"states"' in out and "saved learned state" in out
    # The state is now in the ledger for the engine to pick up.
    from competition.engine import Ledger
    led = Ledger(ledger)
    blob = led.load_learned_state("q_learner", "strategy")
    led.close()
    assert blob.get("q")


def test_pretrain_dry_run_saves_nothing(capsys, tmp_path):
    ledger = tmp_path / "l.sqlite"
    assert run(["--quiet", "pretrain", "--source", "synthetic", "--passes", "1",
                "--days", "20", "--symbols", "AAPL,MSFT",
                "--start", "2026-08-14", "--end", "2026-09-04", "--dry-run"],
               ledger) == 0
    assert "--dry-run: nothing saved" in capsys.readouterr().out
    from competition.engine import Ledger
    led = Ledger(ledger)
    assert led.load_learned_state("q_learner", "strategy") == {}
    led.close()


def test_pretrain_on_a_non_learning_team_is_refused(capsys, tmp_path):
    assert run(["--quiet", "pretrain", "--team-key", "gambler"],
               tmp_path / "l.sqlite") == 1
    assert "no pretrain step" in capsys.readouterr().out


@pytest.mark.parametrize("rnd", [1, 2, 3])
def test_backtest_runs_every_round(capsys, tmp_path, rnd, cfg):
    ledger = tmp_path / f"r{rnd}.sqlite"
    rc = run(["--quiet", "backtest", "--round", str(rnd),
              "--start", "2026-09-08", "--end", "2026-09-10",
              "--source", "synthetic", "--seed", "11", "--pool-size", "14",
              "--daily-history", "150", "--warmup-sessions", "25",
              "--tick-seconds", "300"], ledger)
    assert rc == 0
    out = capsys.readouterr().out
    assert f"Round {rnd}" in out
    assert "place" in out and "points" in out
    # Every scored team appears with a place.
    for team in cfg.scored_teams:
        assert team.name in out
    if rnd == 3:
        assert "sum=2505" in out or "Round 3 hands" in out
    # Results are queryable afterwards.
    assert run(["--quiet", "score", "--round", str(rnd)], ledger) == 0
    assert f"Round {rnd}" in capsys.readouterr().out


def test_backtest_then_leaderboard_and_report(capsys, tmp_path):
    ledger = tmp_path / "season.sqlite"
    for rnd in (1, 2):
        assert run(["--quiet", "backtest", "--round", str(rnd),
                    "--start", "2026-09-08", "--end", "2026-09-10",
                    "--source", "synthetic", "--seed", "21", "--pool-size", "12",
                    "--daily-history", "140", "--warmup-sessions", "25",
                    "--tick-seconds", "300"], ledger) == 0
    capsys.readouterr()

    assert run(["--quiet", "leaderboard"], ledger) == 0
    out = capsys.readouterr().out
    assert "FINAL STANDINGS" in out and "Champion:" in out

    assert run(["--quiet", "leaderboard", "--markdown"], ledger) == 0
    md = capsys.readouterr().out
    assert md.startswith("## ") and "| Points |" in md

    assert run(["--quiet", "leaderboard", "--json"], ledger) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rounds"] == [1, 2] and payload["standings"]

    results_md = tmp_path / "RESULTS.md"
    assert run(["--quiet", "leaderboard", "--write", str(results_md)], ledger) == 0
    capsys.readouterr()
    assert "Standings" in results_md.read_text()

    assert run(["--quiet", "report", "--trades", "5"], ledger) == 0
    rep = capsys.readouterr().out
    assert "ledger:" in rep and "fills" in rep


def test_score_and_leaderboard_with_an_empty_ledger(capsys, tmp_path):
    ledger = tmp_path / "empty.sqlite"
    assert run(["--quiet", "score"], ledger) == 1
    assert "no recorded results" in capsys.readouterr().out
    assert run(["--quiet", "leaderboard"], ledger) == 1
    assert "no recorded results" in capsys.readouterr().out
    assert run(["--quiet", "report"], ledger) == 1
    assert "no runs recorded" in capsys.readouterr().out


def test_run_without_credentials_fails_cleanly(capsys, tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("ALPACA_"):
            monkeypatch.delenv(key, raising=False)
    rc = run(["--quiet", "run", "--round", "1", "--dry-run"], tmp_path / "l.sqlite")
    assert rc == 1
    assert "credentials" in capsys.readouterr().out


def test_refresh_universe_without_credentials_fails_cleanly(capsys, tmp_path,
                                                            monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("ALPACA_"):
            monkeypatch.delenv(key, raising=False)
    assert run(["--quiet", "refresh-universe"], tmp_path / "l.sqlite") == 1
    assert "credentials" in capsys.readouterr().out


def test_bad_date_is_reported(capsys, tmp_path):
    rc = run(["--quiet", "backtest", "--round", "1", "--start", "not-a-date"],
             tmp_path / "l.sqlite")
    assert rc == 2
    assert "bad date" in capsys.readouterr().out


def test_backtest_with_alpaca_source_and_no_keys(capsys, tmp_path, monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("ALPACA_"):
            monkeypatch.delenv(key, raising=False)
    rc = run(["--quiet", "backtest", "--round", "1", "--source", "alpaca",
              "--start", "2026-09-08", "--end", "2026-09-10"],
             tmp_path / "l.sqlite")
    assert rc == 1
    assert "synthetic" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# where the credentials live
# --------------------------------------------------------------------------- #


def test_shared_mode_finds_credentials_on_the_group(cfg, monkeypatch):
    """The bug that would have stopped the season at the opening bell.

    In shared mode the keys belong to the account group, not to the team.
    Asking each team whether it has credentials reports all nine unfunded
    even when all three accounts are set up perfectly.
    """
    from competition.cli import _funded_teams

    assert cfg.accounts.is_shared, "this test is about shared mode"
    for group in cfg.accounts.groups:
        monkeypatch.setenv(f"{group.env_prefix}_KEY_ID", "PK" + "A" * 18)
        monkeypatch.setenv(f"{group.env_prefix}_SECRET_KEY", "s" * 40)

    funded = _funded_teams(cfg, cfg.teams)
    assert funded == set(cfg.team_keys), "every team in a funded group can trade"


def test_shared_mode_an_unfunded_group_grounds_only_its_own_teams(cfg, monkeypatch):
    from competition.cli import _funded_teams

    groups = list(cfg.accounts.groups)
    for group in groups[:-1]:
        monkeypatch.setenv(f"{group.env_prefix}_KEY_ID", "PK" + "A" * 18)
        monkeypatch.setenv(f"{group.env_prefix}_SECRET_KEY", "s" * 40)

    funded = _funded_teams(cfg, cfg.teams)
    assert funded and not (set(groups[-1].teams) & funded)
    for group in groups[:-1]:
        assert set(group.teams) <= funded


def test_shared_mode_half_a_key_pair_does_not_count(cfg, monkeypatch):
    from competition.cli import _funded_teams

    group = cfg.accounts.groups[0]
    monkeypatch.setenv(f"{group.env_prefix}_KEY_ID", "PK" + "A" * 18)
    # secret deliberately absent
    assert _funded_teams(cfg, cfg.teams) == set()


def test_per_team_mode_still_reads_the_team_prefix(cfg, monkeypatch):
    import dataclasses

    from competition.cli import _funded_teams
    from competition.config import AccountsConfig

    per_team = dataclasses.replace(
        cfg, accounts=AccountsConfig(mode="per_team", groups=()))
    team = per_team.teams[0]
    monkeypatch.setenv(f"{team.env_prefix}_KEY_ID", "PK" + "A" * 18)
    monkeypatch.setenv(f"{team.env_prefix}_SECRET_KEY", "s" * 40)

    assert _funded_teams(per_team, per_team.teams) == {team.key}
