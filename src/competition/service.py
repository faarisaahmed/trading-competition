"""Run the season as a background service, so nobody has to mind a terminal.

Three weeks of trading is too long to hold a shell open. This installs a
supervised service -- a LaunchAgent on macOS, a systemd user unit on Linux --
that starts the season, restarts it if it dies, and on a laptop keeps the
machine awake through the session. The engine checkpoints every tick, which is
what makes a restart *rejoin* a round rather than restart it.

Both are per-user rather than system-wide. Root is a needless privilege for
something whose only secret is a paper-trading key.

On macOS the process is wrapped in `caffeinate` because a desktop sleeps; on a
Linux server there is nothing to keep awake, so it is not.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Which supervisor this platform uses.
PLATFORM = "launchd" if sys.platform == "darwin" else "systemd"

#: Reverse-DNS label, the macOS convention. Also the plist's filename.
LABEL = "com.trading-competition.season"

#: systemd prefers a plain name.
UNIT_NAME = "trading-competition"


class ServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServicePaths:
    plist: Path
    stdout: Path
    stderr: Path

    @property
    def target(self) -> str:
        return f"gui/{os.getuid()}/{LABEL}"


def paths(repo: Path) -> ServicePaths:
    logs = repo / "runs" / "logs"
    if PLATFORM == "launchd":
        unit = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    else:
        unit = (Path.home() / ".config" / "systemd" / "user"
                / f"{UNIT_NAME}.service")
    return ServicePaths(
        plist=unit,
        stdout=logs / "season.out.log",
        stderr=logs / "season.err.log",
    )


def _comp_executable() -> str:
    """The `comp` entry point, resolved to a real path.

    launchd runs with a minimal PATH and no shell profile, so a pyenv *shim*
    would not resolve. The real interpreter's script directory is what has to
    go in the plist.
    """
    found = shutil.which("comp")
    if found:
        real = Path(found)
        # A pyenv shim lives in `.../shims/`; the actual script sits beside
        # the interpreter that will run it.
        if "shims" in real.parts:
            candidate = Path(sys.executable).parent / "comp"
            if candidate.exists():
                return str(candidate)
        return str(real)
    candidate = Path(sys.executable).parent / "comp"
    if candidate.exists():
        return str(candidate)
    raise ServiceError(
        "cannot find the `comp` entry point. Install the package first: "
        "`pip install -e .`"
    )


def build_plist(repo: Path, *, extra_args: list[str] | None = None) -> dict:
    """The LaunchAgent definition."""
    p = paths(repo)
    comp = _comp_executable()
    args = ["season", *(extra_args or [])]

    program = [
        # Prevent idle, disk and system sleep -- but NOT display sleep. The
        # screen may switch off; the machine may not, or it sleeps mid-session
        # and simply misses the market. Three weeks is too long to hold a
        # display awake for nothing.
        "/usr/bin/caffeinate", "-ims",
        comp, *args,
    ]
    return {
        "Label": LABEL,
        "ProgramArguments": program,
        "WorkingDirectory": str(repo),
        "RunAtLoad": True,
        # Restart if it exits for any reason. The engine checkpoints every
        # tick, so a restart rejoins the round rather than restarting it.
        "KeepAlive": True,
        # Do not hammer launchd if something is fatally wrong at startup.
        "ThrottleInterval": 60,
        "StandardOutPath": str(p.stdout),
        "StandardErrorPath": str(p.stderr),
        "ProcessType": "Interactive",
        "EnvironmentVariables": {
            "PATH": f"{Path(comp).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1",
        },
    }


def build_unit(repo: Path, *, extra_args: list[str] | None = None) -> str:
    """The systemd user unit. No `caffeinate`: a server does not sleep."""
    comp = _comp_executable()
    args = " ".join(["season", *(extra_args or [])])
    return f"""[Unit]
Description=2026 Model Trading Competition -- season runner
Documentation=https://github.com/faarisaahmed/trading-competition
# Alpaca and GitHub are both unreachable before the network is up, and the
# first thing the season does on a restart is read the accounts.
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={repo}
ExecStart={comp} {args}
# The engine checkpoints every tick, so a restart rejoins the round.
Restart=always
RestartSec=60
Environment=PYTHONUNBUFFERED=1
Environment=PATH={Path(comp).parent}:/usr/local/bin:/usr/bin:/bin
StandardOutput=append:{paths(repo).stdout}
StandardError=append:{paths(repo).stderr}

[Install]
WantedBy=default.target
"""


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["systemctl", "--user", *args],
                          capture_output=True, text=True)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ServiceError(f"systemctl {args[0]} failed: {detail[:200]}")
    return proc


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["launchctl", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ServiceError(f"launchctl {args[0]} failed: {detail[:200]}")
    return proc


def install(repo: Path, *, extra_args: list[str] | None = None) -> ServicePaths:
    """Write the unit and start the service. Safe to re-run."""
    p = paths(repo)
    p.plist.parent.mkdir(parents=True, exist_ok=True)
    p.stdout.parent.mkdir(parents=True, exist_ok=True)

    if PLATFORM == "launchd":
        p.plist.write_bytes(plistlib.dumps(build_plist(repo, extra_args=extra_args)))
        # Replace any previous incarnation rather than erroring on a duplicate.
        _launchctl("bootout", p.target, check=False)
        _launchctl("bootstrap", f"gui/{os.getuid()}", str(p.plist))
        _launchctl("enable", p.target, check=False)
        return p

    p.plist.write_text(build_unit(repo, extra_args=extra_args))
    _systemctl("daemon-reload")
    _systemctl("enable", "--now", f"{UNIT_NAME}.service")
    # Without lingering, a user service stops the moment you log out -- which
    # on a VPS is the moment you close the SSH session.
    subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")],
                   capture_output=True, text=True)
    return p


def uninstall(repo: Path) -> bool:
    """Stop and remove the service. True if there was one."""
    p = paths(repo)
    existed = p.plist.exists()
    if PLATFORM == "launchd":
        _launchctl("bootout", p.target, check=False)
    else:
        _systemctl("disable", "--now", f"{UNIT_NAME}.service", check=False)
    if existed:
        p.plist.unlink()
        if PLATFORM == "systemd":
            _systemctl("daemon-reload", check=False)
    return existed


def status(repo: Path) -> dict:
    """What the supervisor currently thinks of the service."""
    p = paths(repo)
    out = {"installed": p.plist.exists(),
           "label": LABEL if PLATFORM == "launchd" else UNIT_NAME,
           "platform": PLATFORM,
           "plist": str(p.plist), "running": False, "pid": None,
           "last_exit": None}
    if PLATFORM == "systemd":
        proc = _systemctl("show", f"{UNIT_NAME}.service",
                          "--property=MainPID,ActiveState,ExecMainStatus",
                          check=False)
        if proc.returncode != 0:
            return out
        fields = dict(
            line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
        )
        pid = int(fields.get("MainPID", "0") or 0)
        out["running"] = fields.get("ActiveState") == "active" and pid > 0
        out["pid"] = pid or None
        out["last_exit"] = fields.get("ExecMainStatus")
        return out

    proc = _launchctl("print", p.target, check=False)
    if proc.returncode != 0:
        return out
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("pid = "):
            out["pid"] = int(line.split("=")[1])
            out["running"] = True
        elif line.startswith("last exit code = "):
            out["last_exit"] = line.split("=", 1)[1].strip()
    return out
