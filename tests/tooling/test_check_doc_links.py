"""``scripts/check_doc_links.py`` reports broken relative links and nothing else.

Each test builds a throwaway git repository under ``tmp_path``, so the suite
never reads this repository's own documents.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_doc_links import find_broken_links


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a git repository at ``tmp_path`` with ``files`` tracked."""
    _ = subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(body, encoding="utf-8")
    _ = subprocess.run(["git", "-C", str(tmp_path), "add", "--", *files], check=True)
    return tmp_path


def test_resolving_links_pass(tmp_path: Path) -> None:
    """Relative links to existing files, with or without anchors, are clean."""
    root = _repo(
        tmp_path,
        {
            "README.md": "[a](docs/a.md) [b](docs/a.md#part) ![i](img.png)\n",
            "docs/a.md": "[up](../README.md)\n",
            "img.png": "",
        },
    )
    assert find_broken_links(root) == ([], 2)


def test_missing_target_is_reported_with_location(tmp_path: Path) -> None:
    """A link to a missing file is reported as ``file:line: target``."""
    root = _repo(tmp_path, {"docs/a.md": "intro\n\nsee [gone](gone.md#x)\n"})
    assert find_broken_links(root) == (["docs/a.md:3: gone.md#x"], 1)


def test_external_escaping_and_code_links_are_ignored(tmp_path: Path) -> None:
    """URLs, links leaving the repository, code spans and fences are not checked."""
    body = (
        "[web](https://example.com/missing.md) [out](../../elsewhere.md)\n"
        "`[T](missing.md)` and [mail](mailto:a@b.c)\n"
        "```\n[fenced](missing.md)\n```\n"
    )
    root = _repo(tmp_path, {"README.md": body})
    assert find_broken_links(root) == ([], 1)


def test_untracked_markdown_is_not_checked(tmp_path: Path) -> None:
    """Only tracked Markdown is in scope."""
    root = _repo(tmp_path, {"README.md": "clean\n"})
    _ = (root / "notes.md").write_text("[gone](gone.md)\n", encoding="utf-8")
    assert find_broken_links(root) == ([], 1)
