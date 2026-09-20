"""The background service definition.

The plist is the thing that decides whether the season is actually running at
9:30 on a Monday, so its contents are worth asserting rather than eyeballing.
launchctl itself is never invoked here -- these tests must not install
anything on the machine running them.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from competition import service
from competition.service import LABEL, ServiceError, build_plist, paths

#: The real helpers, grabbed at import time. The autouse fixture below swaps
#: the module attributes out, so a test that wants to exercise the helper
#: itself has to hold its own reference.
_REAL_LAUNCHCTL = service._launchctl
_REAL_SYSTEMCTL = service._systemctl


def _ok():
    """A successful subprocess result."""
    class P:
        returncode = 0
        stdout = ""
        stderr = ""
    return P()


@pytest.fixture(autouse=True)
def never_touch_the_supervisor(monkeypatch):
    """No test may invoke a real `launchctl` or `systemctl`.

    Not merely for isolation: without this, running the suite on Linux
    genuinely attempted to install a systemd unit on the machine under test,
    because only the macOS helper was being mocked. Both are blocked by
    default and individual tests opt in to a recorder.
    """
    def refuse(name):
        def boom(*args, **kwargs):
            raise AssertionError(
                f"a test tried to run real {name} {args!r}. Mock it."
            )
        return boom

    monkeypatch.setattr(service, "_launchctl", refuse("launchctl"))
    monkeypatch.setattr(service, "_systemctl", refuse("systemctl"))
    monkeypatch.setattr(service.subprocess, "run", refuse("subprocess.run"))


@pytest.fixture
def macos(monkeypatch):
    """Force the launchd path, whatever platform the suite runs on."""
    monkeypatch.setattr(service, "PLATFORM", "launchd")


@pytest.fixture
def no_launchctl(monkeypatch, macos):
    """Record supervisor calls instead of making them."""
    calls = []

    def fake(*args, check=True):
        calls.append(args)
        return _ok()

    monkeypatch.setattr(service, "_launchctl", fake)
    monkeypatch.setattr(service, "_systemctl", fake)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: _ok())
    return calls


# --------------------------------------------------------------------------- #
# the plist
# --------------------------------------------------------------------------- #


def test_the_agent_runs_the_season(tmp_path):
    d = build_plist(tmp_path)
    assert d["Label"] == LABEL
    assert d["ProgramArguments"][-1] == "season"
    assert d["WorkingDirectory"] == str(tmp_path)


def test_it_holds_the_machine_awake_but_not_the_display(tmp_path):
    """A Mac asleep at 10am is lost trading time; a dark screen is not."""
    args = build_plist(tmp_path)["ProgramArguments"]
    assert args[0] == "/usr/bin/caffeinate"
    flags = args[1]
    assert "i" in flags and "s" in flags, "must prevent idle and system sleep"
    assert "d" not in flags, "no reason to hold the display on for three weeks"


def test_it_restarts_itself(tmp_path):
    """The engine checkpoints every tick, so a restart rejoins the round."""
    d = build_plist(tmp_path)
    assert d["KeepAlive"] is True
    assert d["RunAtLoad"] is True
    assert d["ThrottleInterval"] >= 10, "do not hammer launchd on a fatal error"


def test_logs_go_somewhere_findable(tmp_path):
    d = build_plist(tmp_path)
    assert str(tmp_path) in d["StandardOutPath"]
    assert str(tmp_path) in d["StandardErrorPath"]
    assert d["StandardOutPath"] != d["StandardErrorPath"]


def test_extra_arguments_are_passed_through(tmp_path):
    d = build_plist(tmp_path, extra_args=["--no-publish"])
    assert d["ProgramArguments"][-2:] == ["season", "--no-publish"]


def test_the_plist_is_serialisable(tmp_path):
    """launchd rejects a plist it cannot parse, silently enough to miss."""
    body = plistlib.dumps(build_plist(tmp_path))
    assert plistlib.loads(body)["Label"] == LABEL


def test_path_does_not_rely_on_a_shell_profile(tmp_path):
    """launchd starts with a minimal environment and no login shell."""
    env = build_plist(tmp_path)["EnvironmentVariables"]
    comp = build_plist(tmp_path)["ProgramArguments"][2]
    assert str(Path(comp).parent) in env["PATH"]
    assert "/usr/bin" in env["PATH"]


def test_the_executable_is_not_a_pyenv_shim(tmp_path, monkeypatch):
    """A shim resolves through the shell; launchd would never find it."""
    shim = tmp_path / "shims" / "comp"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\n")
    real_dir = tmp_path / "versions" / "3.10.13" / "bin"
    real_dir.mkdir(parents=True)
    (real_dir / "comp").write_text("#!/bin/sh\n")

    monkeypatch.setattr(service.shutil, "which", lambda _n: str(shim))
    monkeypatch.setattr(service.sys, "executable", str(real_dir / "python3"))
    assert service._comp_executable() == str(real_dir / "comp")


def test_a_missing_entry_point_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(service.shutil, "which", lambda _n: None)
    monkeypatch.setattr(service.sys, "executable", str(tmp_path / "python3"))
    with pytest.raises(ServiceError, match="pip install"):
        service._comp_executable()


# --------------------------------------------------------------------------- #
# install / uninstall
# --------------------------------------------------------------------------- #


def test_install_writes_the_plist_and_bootstraps(tmp_path, monkeypatch,
                                                 no_launchctl):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    p = service.install(tmp_path)

    assert p.plist.exists()
    assert plistlib.loads(p.plist.read_bytes())["Label"] == LABEL
    assert p.stdout.parent.exists(), "log directory must exist before launchd starts"
    verbs = [c[0] for c in no_launchctl]
    assert "bootout" in verbs and "bootstrap" in verbs
    assert verbs.index("bootout") < verbs.index("bootstrap"), (
        "a previous agent must be removed before the new one is loaded")


def test_install_is_repeatable(tmp_path, monkeypatch, no_launchctl):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    service.install(tmp_path)
    service.install(tmp_path)          # must not raise on an existing agent
    assert paths(tmp_path).plist.exists()


def test_uninstall_removes_it(tmp_path, monkeypatch, no_launchctl):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    service.install(tmp_path)
    assert service.uninstall(tmp_path) is True
    assert not paths(tmp_path).plist.exists()


def test_uninstall_when_nothing_is_installed(tmp_path, monkeypatch, no_launchctl):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    assert service.uninstall(tmp_path) is False


def test_status_reports_not_installed(tmp_path, monkeypatch, macos):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    class P:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(service, "_launchctl", lambda *a, **k: P())
    st = service.status(tmp_path)
    assert st["installed"] is False and st["running"] is False


def test_status_parses_a_running_agent(tmp_path, monkeypatch, macos):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    class P:
        returncode = 0
        stdout = "  pid = 4242\n  last exit code = 0\n"
        stderr = ""

    monkeypatch.setattr(service, "_launchctl", lambda *a, **k: P())
    st = service.status(tmp_path)
    assert st["running"] is True and st["pid"] == 4242


def test_launchctl_failure_is_reported(monkeypatch):
    """The helper must surface launchctl's own complaint, not swallow it."""
    class P:
        returncode = 1
        stdout = ""
        stderr = "Load failed: 5: Input/output error"

    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: P())
    with pytest.raises(ServiceError, match="Load failed"):
        _REAL_LAUNCHCTL("bootstrap", "gui/501", "/x.plist")


def test_systemctl_failure_is_reported(monkeypatch):
    class P:
        returncode = 1
        stdout = ""
        stderr = "Failed to enable unit: No such file or directory"

    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: P())
    with pytest.raises(ServiceError, match="Failed to enable"):
        _REAL_SYSTEMCTL("enable", "x.service")


def test_the_suite_cannot_install_a_real_service():
    """A guard on the guard.

    Running the tests on Linux once genuinely tried to `systemctl enable` a
    unit on the machine, because only the macOS helper was mocked. If the
    autouse fixture ever stops covering both, this fails loudly.
    """
    with pytest.raises(AssertionError, match="Mock it"):
        service._systemctl("enable", "anything")
    with pytest.raises(AssertionError, match="Mock it"):
        service._launchctl("bootstrap", "anything")


# --------------------------------------------------------------------------- #
# the Linux path
# --------------------------------------------------------------------------- #


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(service, "PLATFORM", "systemd")


@pytest.fixture
def no_systemctl(monkeypatch, linux):
    calls = []

    def fake(*args, check=True):
        calls.append(args)
        return _ok()

    monkeypatch.setattr(service, "_systemctl", fake)
    monkeypatch.setattr(service, "_launchctl", fake)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: _ok())
    return calls


def test_the_unit_runs_the_season(tmp_path, linux):
    unit = service.build_unit(tmp_path)
    assert "ExecStart=" in unit
    assert unit.rstrip().endswith("WantedBy=default.target")
    assert " season" in unit
    assert f"WorkingDirectory={tmp_path}" in unit


def test_the_unit_does_not_wrap_caffeinate(tmp_path, linux):
    """A server does not sleep, and the binary is not there anyway."""
    assert "caffeinate" not in service.build_unit(tmp_path)


def test_the_unit_restarts_and_waits_for_the_network(tmp_path, linux):
    unit = service.build_unit(tmp_path)
    assert "Restart=always" in unit
    assert "RestartSec=60" in unit
    assert "network-online.target" in unit, (
        "Alpaca is unreachable before the network is up")


def test_the_unit_sets_an_explicit_path(tmp_path, linux):
    """systemd gives a service almost no environment."""
    assert "Environment=PATH=" in service.build_unit(tmp_path)


def test_linux_uses_a_systemd_unit_path(tmp_path, linux, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    p = service.paths(tmp_path)
    assert p.plist.suffix == ".service"
    assert ".config/systemd/user" in str(p.plist)


def test_linux_install_enables_and_lingers(tmp_path, monkeypatch, no_systemctl):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ran = []
    monkeypatch.setattr(service.subprocess, "run",
                        lambda cmd, **k: ran.append(cmd) or _ok())

    p = service.install(tmp_path)
    assert p.plist.exists()
    verbs = [c[0] for c in no_systemctl]
    assert "daemon-reload" in verbs and "enable" in verbs
    assert any("enable-linger" in " ".join(c) for c in ran), (
        "without lingering the service dies when the SSH session closes")


def test_linux_uninstall_removes_the_unit(tmp_path, monkeypatch, no_systemctl):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    service.install(tmp_path)
    assert service.uninstall(tmp_path) is True
    assert not service.paths(tmp_path).plist.exists()


def test_linux_status_parses_systemctl(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    class P:
        returncode = 0
        stdout = "MainPID=8123\nActiveState=active\nExecMainStatus=0\n"
        stderr = ""

    monkeypatch.setattr(service, "_systemctl", lambda *a, **k: P())
    st = service.status(tmp_path)
    assert st["running"] is True and st["pid"] == 8123
    assert st["platform"] == "systemd"


def test_linux_status_knows_when_it_is_stopped(tmp_path, monkeypatch, linux):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))

    class P:
        returncode = 0
        stdout = "MainPID=0\nActiveState=inactive\nExecMainStatus=1\n"
        stderr = ""

    monkeypatch.setattr(service, "_systemctl", lambda *a, **k: P())
    st = service.status(tmp_path)
    assert st["running"] is False and st["pid"] is None
