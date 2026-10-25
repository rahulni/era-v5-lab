"""
Theorem 1 on the full vocabulary: the forward encode of every token id decodes
back to its table row, so equal codes <=> equal rows. Uses the repo's
kronecker_codec on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import ke2_common as C


def test_every_token_code_decodes_to_its_row(tables, kind, P):
    buf, lens = tables[(kind, P)]
    for s, E in C.encode_chunks(buf, lens, P, chunk=4096):
        db, dl, ok = C.decode_codes(E, P)
        assert bool(ok.all()), f"{kind}_P{P}: malformed code near id {s}"
        assert np.array_equal(db.numpy(), buf[s:s + len(E)]), f"{kind}_P{P}: byte mismatch near id {s}"
        assert np.array_equal(dl.numpy(), lens[s:s + len(E)]), f"{kind}_P{P}: length mismatch near id {s}"


def test_committed_proof_checks_all_pass(pinned):
    if not pinned:
        pytest.skip("results/ were produced with the pinned tokenizer")
    report = C.read_json(C.RESULTS / "proof_checks.json")
    failed = [f"{c['id']} {c['scope']}" for c in report["checks"] if not c["pass"]]
    assert report["all_pass"] and not failed, failed
    assert {c["id"] for c in report["checks"]} == {"C0", "L1-a", "L1-b", "L2-a", "L2-b", "L2-c", "L2-d", "L2-e",
                                                    "T1-a", "T1-b", "T1-c", "T1-d", "T1-e", "T1-f"}
