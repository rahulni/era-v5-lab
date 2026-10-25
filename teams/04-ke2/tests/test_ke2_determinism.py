"""results/decode_cpu_sha256.txt holds three matching hashes, and this machine reproduces them."""

from __future__ import annotations

import re

import pytest

import ke2_common as C
from determinism import decode_hash


def hash_lines():
    text = (C.RESULTS / "decode_cpu_sha256.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_hash_file_has_three_matching_hashes():
    lines = hash_lines()
    assert len(lines) == 3
    hashes = [line.split()[0] for line in lines]
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes)
    assert len(set(hashes)) == 1


def test_decode_is_repeatable_in_one_process(tok):
    assert decode_hash(tok) == decode_hash(tok)


def test_this_machine_reproduces_the_committed_hash(pinned, tok):
    if not pinned:
        pytest.skip("the committed hash is for the pinned tokenizer")
    assert decode_hash(tok) == hash_lines()[0].split()[0]
