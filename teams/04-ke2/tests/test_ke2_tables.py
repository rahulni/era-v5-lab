"""The decode table, the raw byte table, and the fix for the U+FFFD issue.

Tests that take P or kind run for every value (conftest.py)."""

from __future__ import annotations

import numpy as np
import pytest

import ke2_common as C
from kronecker_embeddings import build_byte_buffer

U_FFFD = C.U_FFFD


def test_table_layout(tok, tables, kind, P):
    buf, lens = tables[(kind, P)]
    V = C.vocab_size(tok)
    assert buf.shape == (V, P) and buf.dtype == np.uint8
    assert lens.shape == (V,) and lens.dtype == np.int16
    assert int(lens.max()) <= P and int(lens.min()) >= 0


def test_decode_table_is_the_repo_buffer(tok, tables, P):
    repo_buf, repo_lens = build_byte_buffer(tok, pos_dim=P)
    buf, lens = tables[("decode", P)]
    assert np.array_equal(buf, repo_buf.numpy()) and np.array_equal(lens, repo_lens.numpy())


def test_decode_is_raw_with_invalid_utf8_replaced(full):
    for r, d in zip(full["raw"], full["decode"]):
        assert r.decode("utf-8", "replace").encode("utf-8") == d


def test_known_issue_decode_rows_shared_only_through_u_fffd(tables, P):
    rows = C.table_rows(*tables[("decode", P)])
    counts = {}
    for r in rows:
        counts[r] = counts.get(r, 0) + 1
    shared = [r for r, c in counts.items() if c > 1]
    assert all(U_FFFD in r for r in shared)


def test_raw_table_fixes_the_issue(full, tables, P):
    """No raw row is shared because of U+FFFD: a raw row holds U+FFFD only if the token's bytes really do."""
    rows = C.table_rows(*tables[("raw", P)])
    for r, f in zip(rows, full["raw"]):
        if U_FFFD in r:
            assert f.startswith(r) and r.decode("utf-8", "strict")


def test_raw_table_gives_every_token_a_unique_row(pinned, full, tables, P):
    if not pinned:
        pytest.skip("uniqueness is a property of the pinned tokenizer; other tokenizers rely on test_ke2_collisions")
    rows = C.table_rows(*tables[("raw", P)])
    assert len(set(rows)) == len(rows)
    assert max(map(len, full["raw"])) <= P                  # nothing is truncated


def test_tables_match_committed_manifest(pinned, tables):
    if not pinned:
        pytest.skip("results/ were produced with the pinned tokenizer")
    manifest = C.read_json(C.RESULTS / "tables_manifest.json")
    for (kind, P), (buf, lens) in tables.items():
        assert C.sha256_table(buf, lens) == manifest["tables"][f"{kind}_P{P}"]["sha256"]
