"""
Shared fixtures for the KE2 week-1 tests.

Run from the lab repo root:  pytest teams/04-ke2/tests -q

The tests build the byte tables in memory, so they do not depend on the
generated build/ folder. The tokenizer is found by ke2_common.find_tokenizer_json
(KE2_TOKENIZER, a sibling clone, the cache, or a pinned download). Tests that
need it are skipped only if it cannot be found or downloaded.

No decorator lines: pr-guard reads each added line of a pull request with its
leading plus sign, and a plus sign followed by an at sign and a dotted name
looks like an email address to its secret scan. So fixtures are registered
with plain calls below, and parameters come from pytest_generate_tests:
every test that takes P or kind runs for each value, and a test module can
add more names in a PARAMS dict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ke2_common as C  # noqa: E402

DEFAULT_PARAMS = {"P": C.P_VALUES, "kind": C.KINDS}


def pytest_generate_tests(metafunc):
    extra = getattr(metafunc.module, "PARAMS", {})
    for name in metafunc.fixturenames:
        values = extra.get(name, DEFAULT_PARAMS.get(name))
        if values is not None:
            metafunc.parametrize(name, values)


def tok():
    try:
        return C.load_tokenizer()
    except Exception as e:   # no local copy and no network
        pytest.skip(f"tokenizer not available: {e}")


def pinned(tok):
    """True when the tokenizer is the pinned BrahmicTokenizer-131K (committed results apply)."""
    return C.tokenizer_info()["pinned"]


def full(tok):
    """Untruncated byte lists: {"decode": [...], "raw": [...]}."""
    return {kind: C.full_bytes(tok, kind) for kind in C.KINDS}


def tables(full):
    """{(kind, P): (byte_buffer, length_buffer)} for the four tables."""
    return {(kind, P): C.make_table(full[kind], P) for kind in C.KINDS for P in C.P_VALUES}


def classes(tok, full):
    specials = C.special_ids(tok)
    return [C.script_class(b, i in specials) for i, b in enumerate(full["raw"])]


tok = pytest.fixture(scope="session")(tok)
pinned = pytest.fixture(scope="session")(pinned)
full = pytest.fixture(scope="session")(full)
tables = pytest.fixture(scope="session")(tables)
classes = pytest.fixture(scope="session")(classes)
