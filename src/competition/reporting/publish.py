"""Publish the live dashboard to GitHub Pages.

The dashboard is a single self-contained HTML file, which makes publishing it
almost trivial: put it on an orphan branch as `index.html` and let Pages serve
it. Two details that are not trivial:

**History.** A round pushes every few minutes for three weeks. Accumulating
~1,500 commits of a generated artefact would bury the project's real history
in noise, so the branch is kept as a *single* commit that is replaced each
time. Pages only ever serves the tip, so nothing is lost.

**Secrets.** This is a public page. Nothing here reads `.env`, and the payload
is checked for anything that looks like an Alpaca key before it is pushed --
a belt-and-braces check, since the dashboard never had access to them anyway.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

#: Alpaca key ids are `PK`/`AK` + 18 uppercase alphanumerics. Secrets are 40+
#: mixed-case base64-ish characters, which is too common a shape to match on
#: safely, so the key id is the reliable tell -- and a leaked secret without
#: its id is not usable against the API.
_KEY_RE = re.compile(r"\b[AP]K[A-Z0-9]{16,22}\b")


class PublishError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishResult:
    url: str | None
    commit: str
    pushed_at: datetime
    bytes_written: int


def scrub(html: str) -> str:
    """Remove anything key-shaped. Returns the text unchanged if clean."""
    return _KEY_RE.sub("[redacted]", html)


def contains_secret(html: str) -> bool:
    return bool(_KEY_RE.search(html))


def _git(*args: str, cwd: Path, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        # Never echo the remote URL: it can carry a token in the userinfo.
        err = (proc.stderr or proc.stdout).strip().splitlines()
        tail = err[-1] if err else f"exit {proc.returncode}"
        raise PublishError(f"git {args[0]} failed: {tail[:200]}")
    return proc.stdout.strip()


def publish(
    html: str,
    *,
    repo: Path,
    branch: str = "gh-pages",
    message: str | None = None,
    extra_files: dict[str, str] | None = None,
    remote: str = "origin",
) -> PublishResult:
    """Force `branch` to a single commit containing `index.html`.

    Uses a temporary worktree so the caller's checkout -- which during a live
    round has an engine running against it -- is never touched, and a failure
    here can never leave the working tree dirty or on another branch.
    """
    repo = Path(repo).resolve()
    if contains_secret(html):
        # Defensive: refuse rather than scrub silently, so a leak is loud.
        raise PublishError(
            "refusing to publish: the rendered page contains something shaped "
            "like an Alpaca key id"
        )

    stamp = datetime.now().astimezone()
    message = message or f"dashboard {stamp:%Y-%m-%d %H:%M:%S %Z}"

    # A throwaway branch name per push. Reusing `branch` itself would work
    # exactly once: `checkout --orphan` refuses a name that already exists
    # locally, so the second publish of a round would fail.
    scratch = f"pages-build-{uuid.uuid4().hex[:12]}"

    with tempfile.TemporaryDirectory(prefix="comp-pages-") as tmp:
        tree = Path(tmp) / "site"
        _git("worktree", "add", "--detach", "--no-checkout", str(tree), cwd=repo)
        try:
            _git("checkout", "--orphan", scratch, cwd=tree)
            _git("reset", "--hard", cwd=tree)
            (tree / "index.html").write_text(html, encoding="utf-8")
            # Pages runs Jekyll by default, which ignores files beginning with
            # an underscore and can rewrite markup. We serve raw HTML.
            (tree / ".nojekyll").write_text("", encoding="utf-8")
            for name, body in (extra_files or {}).items():
                dest = tree / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body, encoding="utf-8")
            _git("add", "-A", cwd=tree)
            _git("-c", "user.name=competition-bot",
                 "-c", "user.email=bot@localhost",
                 "commit", "-q", "-m", message, cwd=tree)
            commit = _git("rev-parse", "HEAD", cwd=tree)
            _git("push", "--force", remote, f"HEAD:refs/heads/{branch}", cwd=tree)
        finally:
            _git("worktree", "remove", "--force", str(tree), cwd=repo, check=False)
            # The scratch ref has served its purpose; leaving one behind per
            # push would litter the repo with hundreds of branches.
            _git("branch", "-D", scratch, cwd=repo, check=False)

    return PublishResult(
        url=pages_url(repo, branch),
        commit=commit[:12],
        pushed_at=stamp,
        bytes_written=len(html.encode("utf-8")),
    )


def push_state(
    files: Mapping[str, bytes],
    *,
    repo: Path,
    branch: str,
    message: str | None = None,
    remote: str = "origin",
) -> str:
    """Force `branch` to a single commit holding `files` (path -> bytes).

    Used to carry the ledger between runs on a host with no persistent disk.
    Single-commit for the same reason the dashboard is: this is state, not
    history, and a commit per session would be thousands of them.
    """
    repo = Path(repo).resolve()
    scratch = f"state-build-{uuid.uuid4().hex[:12]}"
    stamp = datetime.now().astimezone()
    message = message or f"state {stamp:%Y-%m-%d %H:%M:%S %Z}"

    with tempfile.TemporaryDirectory(prefix="comp-state-") as tmp:
        tree = Path(tmp) / "state"
        _git("worktree", "add", "--detach", "--no-checkout", str(tree), cwd=repo)
        try:
            _git("checkout", "--orphan", scratch, cwd=tree)
            _git("reset", "--hard", cwd=tree)
            for name, body in files.items():
                dest = tree / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
            _git("add", "-A", "-f", cwd=tree)      # -f: state is gitignored
            _git("-c", "user.name=competition-bot",
                 "-c", "user.email=bot@localhost",
                 "commit", "-q", "-m", message, cwd=tree)
            commit = _git("rev-parse", "HEAD", cwd=tree)
            _git("push", "--force", remote, f"HEAD:refs/heads/{branch}", cwd=tree)
        finally:
            _git("worktree", "remove", "--force", str(tree), cwd=repo, check=False)
            _git("branch", "-D", scratch, cwd=repo, check=False)
    return commit[:12]


def pull_state(
    *,
    repo: Path,
    branch: str,
    dest: Path,
    remote: str = "origin",
) -> list[str]:
    """Restore whatever `push_state` last wrote. [] if the branch is absent."""
    repo = Path(repo).resolve()
    probe = subprocess.run(
        ["git", "ls-remote", "--heads", remote, branch],
        cwd=repo, capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return []                                  # first ever run

    _git("fetch", "--depth", "1", remote, f"{branch}:refs/remotes/{remote}/{branch}",
         cwd=repo, check=False)
    listing = _git("ls-tree", "-r", "--name-only", f"{remote}/{branch}", cwd=repo)
    restored = []
    dest = Path(dest)
    for name in listing.splitlines():
        name = name.strip()
        if not name:
            continue
        blob = subprocess.run(
            ["git", "show", f"{remote}/{branch}:{name}"],
            cwd=repo, capture_output=True,
        )
        if blob.returncode != 0:
            continue
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob.stdout)
        restored.append(name)
    return restored


def pages_url(repo: Path, branch: str = "gh-pages") -> str | None:
    """Best-effort https URL for the published site."""
    try:
        origin = _git("remote", "get-url", "origin", cwd=Path(repo))
    except PublishError:
        return None
    m = re.search(r"github\.com[:/]+([^/]+)/([^/.]+)", origin)
    if not m:
        return None
    owner, name = m.group(1), m.group(2)
    return f"https://{owner.lower()}.github.io/{name}/"


def enable_pages(owner_repo: str, branch: str = "gh-pages") -> str | None:
    """Turn on Pages for the repo via `gh`. Idempotent; None if unavailable."""
    import json
    probe = subprocess.run(
        ["gh", "api", f"repos/{owner_repo}/pages"],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        try:
            return json.loads(probe.stdout).get("html_url")
        except json.JSONDecodeError:
            return None
    create = subprocess.run(
        ["gh", "api", "-X", "POST", f"repos/{owner_repo}/pages",
         "-f", f"source[branch]={branch}", "-f", "source[path]=/"],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        log.warning("could not enable GitHub Pages automatically: %s",
                    (create.stderr or "").strip()[:200])
        return None
    try:
        return json.loads(create.stdout).get("html_url")
    except json.JSONDecodeError:
        return None
