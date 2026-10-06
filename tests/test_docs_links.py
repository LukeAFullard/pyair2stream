"""
Every relative link and image in the documentation points at a file that exists, and every
link to a section (#anchor) points at a heading that exists (GitHub's anchor rules).
"""

import glob
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = sorted({os.path.join(REPO, p) for p in ("README.md", "USER_GUIDE.md", "CHANGELOG.md")}
              | set(glob.glob(os.path.join(REPO, "docs", "*.md")))
              | set(glob.glob(os.path.join(REPO, "examples", "*", "README.md")))
              | {os.path.join(REPO, "examples", "README.md"), os.path.join(REPO, "validation", "README.md")})
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _anchor(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation removed, spaces as hyphens."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        text = re.sub(r"```.*?```", "", f.read(), flags=re.S)
    return {_anchor(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", text, flags=re.M)}


def _links(path: str):
    with open(path, encoding="utf-8") as f:
        text = re.sub(r"```.*?```", "", f.read(), flags=re.S)
    for target in LINK.findall(text):
        if not re.match(r"[a-z]+:", target):
            yield target


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: os.path.relpath(p, REPO))
def test_relative_links_and_anchors_resolve(doc):
    problems = []
    for target in _links(doc):
        file_part, _, anchor = target.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(doc), file_part)) if file_part else doc
        if not os.path.exists(dest):
            problems.append(f"missing file: {target}")
        elif anchor and dest.endswith(".md") and anchor not in _anchors(dest):
            problems.append(f"missing section: {target}")
    assert not problems, f"{os.path.relpath(doc, REPO)}: " + "; ".join(problems)
