#!/usr/bin/env python3
"""Check that relative Markdown links in this repository's tracked docs resolve.

Run from anywhere in the repository; standard library only, so it needs no
``uv sync``. CI runs it as the ``docs-links`` job. Exit 0 means every checked
link resolves; exit 1 lists each broken link as ``file:line: target``.

Scope:

- checked: inline Markdown links and images with relative-path targets, in
  every tracked ``.md`` file present in the working tree, against that tree;
- skipped: tracked Markdown files deleted in the working tree, links that
  escape the repository root, and external links (anything with a URI scheme
  — checking those in CI is flaky and nothing here rests on one);
- not validated: ``#heading`` anchors and reference-style link definitions;
- fenced code blocks and inline code spans are excluded from matching
  (``[T](...)`` in a signature is not a link).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)<>\s]+?)>?\s*\)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
FENCE = re.compile(r"^\s*(```|~~~)")
CODE_SPAN = re.compile(r"`[^`]*`")


def find_broken_links(root: Path) -> tuple[list[str], int]:
    """Return ``(broken, checked_file_count)`` for the git repository at ``root``.

    Each broken entry reads ``file:line: target`` with ``file`` repo-relative.
    """
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    files = [f for f in listed.split("\0") if f and (root / f).is_file()]

    broken: list[str] = []
    for f in files:
        in_fence = False
        with (root / f).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for m in LINK.finditer(CODE_SPAN.sub("", line)):
                    target = urllib.parse.unquote(m.group(1)).split("#", 1)[0]
                    if not target or SCHEME.match(target):
                        continue
                    rel = os.path.normpath(os.path.join(os.path.dirname(f), target))
                    if rel == ".." or rel.startswith(".." + os.sep):
                        continue  # escapes the repository: not ours to check
                    if not os.path.lexists(root / rel):
                        broken.append(f"{f}:{lineno}: {m.group(1)}")
    return broken, len(files)


def main() -> int:
    toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    broken, checked = find_broken_links(Path(toplevel))
    if broken:
        print("\n".join(broken))
        print(
            f"\n{len(broken)} broken relative link(s) in {checked} tracked .md files",
            file=sys.stderr,
        )
        return 1
    print(f"all relative links resolve ({checked} tracked .md files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
