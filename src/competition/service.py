"""Run the season as a background service, so nobody has to mind a terminal.

Three weeks of trading is too long to hold a shell open. This installs a macOS
LaunchAgent that starts the season at login, restarts it if it dies, and keeps
the machine awake through the session -- with the engine's own checkpointing
making a restart safe rather than merely survivable.

Deliberately a LaunchAgent (per-user) rather than a LaunchDaemon (system): the
daemon would run as root, which is a needless privilege for something whose
only secret is a paper-trading key, and it would run outside the user session
where `caffeinate` cannot hold the display awake.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Reverse-DNS label, the macOS convention. Also the plist's filename.
LABEL = "com.trading-competition.season"


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
    return ServicePaths(
        plist=Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist",
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


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["launchctl", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise ServiceError(f"launchctl {args[0]} failed: {detail[:200]}")
    return proc


def install(repo: Path, *, extra_args: list[str] | None = None) -> ServicePaths:
    """Write the plist and start the agent. Safe to re-run."""
    p = paths(repo)
    p.plist.parent.mkdir(parents=True, exist_ok=True)
    p.stdout.parent.mkdir(parents=True, exist_ok=True)

    data = build_plist(repo, extra_args=extra_args)
    p.plist.write_bytes(plistlib.dumps(data))

    # Replace any previous incarnation rather than erroring on a duplicate.
    _launchctl("bootout", p.target, check=False)
    _launchctl("bootstrap", f"gui/{os.getuid()}", str(p.plist))
    _launchctl("enable", p.target, check=False)
    return p


def uninstall(repo: Path) -> bool:
    """Stop and remove the agent. True if there was one."""
    p = paths(repo)
    existed = p.plist.exists()
    _launchctl("bootout", p.target, check=False)
    if existed:
        p.plist.unlink()
    return existed


def status(repo: Path) -> dict:
    """What launchd currently thinks of the agent."""
    p = paths(repo)
    out = {"installed": p.plist.exists(), "label": LABEL,
           "plist": str(p.plist), "running": False, "pid": None,
           "last_exit": None}
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
