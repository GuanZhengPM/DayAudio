from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from dayaudio.paths import filesystem_path, filesystem_tree_path


@pytest.fixture
def long_path_root(tmp_path: Path):
    """Create and reliably clean up a path beyond legacy Windows MAX_PATH."""

    anchor = tmp_path / "long-path-root"
    current = anchor
    index = 0
    while len(str(current)) < 280:
        current /= f"segment-{index:02d}-xxxxxxxxxxxxxxxxxxxxxxxx"
        index += 1
    filesystem_current = filesystem_path(current)
    filesystem_current.mkdir(parents=True, exist_ok=True)
    cleanup_anchor = filesystem_current
    for _ in range(index):
        cleanup_anchor = cleanup_anchor.parent
    try:
        yield current
    finally:
        shutil.rmtree(cleanup_anchor, ignore_errors=False)


@pytest.fixture
def near_path_root():
    """Create a conventional root whose descendants cross legacy MAX_PATH."""

    # tempfile may return an alias such as /var on macOS or RUNNER~1 on
    # Windows.  Production path handling resolves those aliases, so build the
    # threshold fixture from the canonical spelling as well.
    anchor = Path(tempfile.mkdtemp(prefix="dayaudio-near-")).resolve()
    current = anchor
    while True:
        length = len(os.path.abspath(current).encode("utf-16-le")) // 2
        component_length = min(40, 235 - length - 1)
        if component_length <= 0:
            break
        current /= "n" * component_length
    length = len(os.path.abspath(current).encode("utf-16-le")) // 2
    assert 230 <= length < 248
    filesystem_path(current).mkdir(parents=True, exist_ok=True)
    try:
        yield current
    finally:
        shutil.rmtree(filesystem_tree_path(anchor), ignore_errors=False)
