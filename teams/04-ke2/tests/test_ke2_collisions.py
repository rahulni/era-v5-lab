"""Each token has a unique code or a known collision cause; results/collisions.csv agrees."""

from __future__ import annotations

import csv
from collections import defaultdict

import hashlib

import numpy as np
import pytest

import ke2_common as C
import synthetic as S
from collisions import KNOWN, explain
from decode_ref import encode_rows, pack_rows
from kronecker_embeddings import utf8_safe_truncate


def groups_of(rows):
    g = defaultdict(list)
    for i, r in enumerate(rows):
        g[r].append(i)
    return sorted((ids for ids in g.values() if len(ids) > 1), key=lambda x: (-len(x), x[0]))


def recomputed(full, tables):
    """{table: [(ids, causes)]} from the in-memory tables."""
    out = {}
    for (kind, P), (buf, lens) in tables.items():
        rows = C.table_rows(buf, lens)
        out[f"{kind}_P{P}"] = [(ids, explain(ids, rows[ids[0]], kind, P, full["raw"], full["decode"])[0])
                               for ids in groups_of(rows)]
    return out


recomputed = pytest.fixture(scope="module")(recomputed)   # plain call, see conftest.py


def test_every_token_is_unique_or_in_a_group_with_a_known_cause(recomputed):
    for table, groups in recomputed.items():
        for ids, causes in groups:
            assert set(causes) <= KNOWN, f"{table}: ids {ids[:5]} have causes {causes}"


def test_synthetic_vocabulary_every_cause_is_found_and_checked(P):
    """Theorem 1 (iii) on a vocabulary where all five causes are planted (synthetic.py)."""
    vocab = S.synthetic_vocabulary()
    raw = [r for r, _ in vocab]
    dec = [S.lam(r) for r in raw]
    found = set()
    for kind, full in (("raw", raw), ("decode", dec)):
        rows = [utf8_safe_truncate(f, P) if len(f) > P else f for f in full]
        cause_of = {}
        for ids in groups_of(rows):
            causes = explain(ids, rows[ids[0]], kind, P, raw, dec)[0]
            assert set(causes) <= KNOWN, (kind, ids[:3], causes)
            found.update(causes)
            cause_of.update({t: set(causes) for t in ids})
        for t, (_, planted) in enumerate(vocab):
            want = planted[kind]
            assert (t not in cause_of) if want is None else (t not in cause_of or want in cause_of[t]), (kind, t)
        buf, lens = pack_rows(rows, P)
        codes = [hashlib.sha256(e.tobytes()).digest() for _, E in encode_rows(buf, lens, P) for e in np.ascontiguousarray(E.numpy())]
        assert sorted(map(sorted, groups_of(codes))) == sorted(map(sorted, groups_of(rows)))   # equal codes = equal rows
    assert found == KNOWN


def test_raw_tables_have_no_u_fffd_collisions(recomputed):
    for table in ("raw_P32", "raw_P64"):
        assert all("UTF8_REPLACEMENT" not in causes for _, causes in recomputed[table])


def test_collisions_csv_lists_exactly_these_groups(pinned, recomputed):
    if not pinned:
        pytest.skip("results/ were produced with the pinned tokenizer")
    with open(C.RESULTS / "collisions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and set(rows[0]) >= {"table", "group", "size", "cause", "script_classes", "member_ids", "reason"}
    from_csv = sorted((r["table"], tuple(map(int, r["member_ids"].split())), r["cause"]) for r in rows)
    expect = sorted((t, tuple(ids), "+".join(c)) for t, groups in recomputed.items() for ids, c in groups)
    assert from_csv == expect
    for r in rows:
        assert int(r["size"]) == len(r["member_ids"].split())


def test_collisions_by_script_adds_up(pinned, tok, recomputed, classes):
    if not pinned:
        pytest.skip("results/ were produced with the pinned tokenizer")
    with open(C.RESULTS / "collisions_by_script.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    V = C.vocab_size(tok)
    for table, groups in recomputed.items():
        mine = [r for r in rows if r["table"] == table]
        assert sum(int(r["tokens"]) for r in mine) == V
        assert sum(int(r["colliding_tokens"]) for r in mine) == sum(len(ids) for ids, _ in groups)
        colliding = defaultdict(int)
        for ids, _ in groups:
            for t in ids:
                colliding[classes[t]] += 1
        assert {r["script_class"]: int(r["colliding_tokens"]) for r in mine if int(r["colliding_tokens"])} == dict(colliding)
