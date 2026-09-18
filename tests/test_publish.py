"""Publishing the dashboard to GitHub Pages.

The publisher force-pushes a generated branch from inside a live trading run,
so the two things worth proving are that it cannot leak a credential and
cannot disturb the working tree it is called from.
"""

from __future__ import annotations

import subprocess

import pytest

from competition.reporting.publish import (
    PublishError,
    contains_secret,
    pages_url,
    publish,
    scrub,
)

KEY = "PKJH2N6C2F7NQJLV35EQ"


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


class _Repo:
    """The working checkout plus the bare remote it pushes to."""

    def __init__(self, work, bare):
        self.work, self.bare = work, bare

    def __fspath__(self):
        return str(self.work)

    def published(self, path="index.html"):
        """Read a file as it exists on the pushed branch, not locally."""
        return subprocess.run(["git", "show", f"gh-pages:{path}"],
                              cwd=self.bare, capture_output=True,
                              text=True).stdout

    def published_files(self):
        return subprocess.run(["git", "ls-tree", "-r", "--name-only", "gh-pages"],
                              cwd=self.bare, capture_output=True,
                              text=True).stdout.split()

    def commit_count(self):
        return subprocess.run(["git", "rev-list", "--count", "gh-pages"],
                              cwd=self.bare, capture_output=True,
                              text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A local repo with a bare 'remote' to push at."""
    bare = tmp_path / "remote.git"
    _run("git", "init", "--bare", "-q", str(bare), cwd=tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=work)
    _run("git", "config", "user.email", "t@example.com", cwd=work)
    _run("git", "config", "user.name", "T", cwd=work)
    (work / "README.md").write_text("hello\n")
    _run("git", "add", "-A", cwd=work)
    _run("git", "commit", "-qm", "init", cwd=work)
    _run("git", "remote", "add", "origin", str(bare), cwd=work)
    _run("git", "push", "-q", "origin", "main", cwd=work)
    return _Repo(work, bare)


# --------------------------------------------------------------------------- #
# the secret guard
# --------------------------------------------------------------------------- #


def test_a_key_id_is_detected():
    assert contains_secret(f"<p>{KEY}</p>")


def test_ordinary_html_is_not_flagged():
    html = "<p>AAPL GOOGL MSFT +8.71% $5,435.69 The Scalper</p>"
    assert not contains_secret(html)
    assert scrub(html) == html


def test_scrub_removes_the_key():
    out = scrub(f"key={KEY} rest")
    assert KEY not in out and "[redacted]" in out


def test_publishing_a_page_with_a_key_is_refused(repo):
    """Loud failure beats a silent scrub: a leak should stop the push."""
    with pytest.raises(PublishError, match="Alpaca key"):
        publish(f"<html>{KEY}</html>", repo=repo.work)
    # And nothing was pushed.
    out = subprocess.run(["git", "ls-remote", "--heads", "origin", "gh-pages"],
                         cwd=repo.work, capture_output=True, text=True)
    assert out.stdout.strip() == ""


# --------------------------------------------------------------------------- #
# pushing
# --------------------------------------------------------------------------- #


def test_publish_creates_the_branch(repo):
    result = publish("<html>hi</html>", repo=repo.work, branch="gh-pages")
    assert result.commit and result.bytes_written == len("<html>hi</html>")
    listing = repo.published_files()
    assert "index.html" in listing
    # Jekyll would otherwise mangle raw HTML and drop underscore-prefixed files.
    assert ".nojekyll" in listing


def test_the_branch_stays_a_single_commit(repo):
    """Three weeks of five-minute pushes must not bury the real history."""
    for i in range(3):
        publish(f"<html>{i}</html>", repo=repo.work)
    assert repo.commit_count() == "1"


def test_the_latest_content_wins(repo):
    publish("<html>old</html>", repo=repo.work)
    publish("<html>new</html>", repo=repo.work)
    assert repo.published() == "<html>new</html>"


def test_the_working_tree_is_untouched(repo):
    """This runs mid-round; it must not move HEAD or dirty the checkout."""
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo.work,
                            capture_output=True, text=True).stdout
    publish("<html>hi</html>", repo=repo.work)
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo.work,
                           capture_output=True, text=True).stdout
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo.work,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo.work,
                            capture_output=True, text=True).stdout
    assert before == after and branch == "main" and status == ""


def test_no_worktree_is_left_behind(repo):
    publish("<html>hi</html>", repo=repo.work)
    out = subprocess.run(["git", "worktree", "list"], cwd=repo.work,
                         capture_output=True, text=True).stdout
    assert out.count("\n") == 1, f"a stray worktree survived:\n{out}"


def test_extra_files_are_published(repo):
    publish("<html>hi</html>", repo=repo.work, extra_files={"data/x.json": "{}"})
    assert repo.published("data/x.json") == "{}"


def test_a_missing_remote_fails_cleanly(tmp_path):
    work = tmp_path / "solo"
    work.mkdir()
    _run("git", "init", "-q", "-b", "main", cwd=work)
    _run("git", "config", "user.email", "t@example.com", cwd=work)
    _run("git", "config", "user.name", "T", cwd=work)
    (work / "a.txt").write_text("a")
    _run("git", "add", "-A", cwd=work)
    _run("git", "commit", "-qm", "init", cwd=work)
    with pytest.raises(PublishError):
        publish("<html>hi</html>", repo=work)


# --------------------------------------------------------------------------- #
# the URL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("remote,want", [
    ("https://github.com/Owner/Repo.git", "https://owner.github.io/Repo/"),
    ("git@github.com:Owner/Repo.git", "https://owner.github.io/Repo/"),
])
def test_pages_url_from_either_remote_form(tmp_path, remote, want):
    work = tmp_path / "u"
    work.mkdir()
    _run("git", "init", "-q", cwd=work)
    _run("git", "remote", "add", "origin", remote, cwd=work)
    assert pages_url(work) == want


def test_pages_url_is_none_without_a_github_remote(tmp_path):
    work = tmp_path / "v"
    work.mkdir()
    _run("git", "init", "-q", cwd=work)
    _run("git", "remote", "add", "origin", "/somewhere/local.git", cwd=work)
    assert pages_url(work) is None


def test_no_scratch_branches_accumulate(repo):
    """Each push uses a throwaway branch; none may survive it."""
    for _ in range(3):
        publish("<html>x</html>", repo=repo.work)
    branches = subprocess.run(["git", "branch", "--list"], cwd=repo.work,
                              capture_output=True, text=True).stdout
    assert "pages-build-" not in branches, branches
