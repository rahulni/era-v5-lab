"""
Synthetic inputs for the cases the real vocabulary never reaches.

BrahmicTokenizer-131K has no token longer than 32 bytes and no duplicate
bytes, so on real data the cut rule never cuts and only one collision cause
occurs (U+FFFD replacement). These builders exercise everything else, for
the checks in proof_checks.py and the tests:

  CUT_EXAMPLE                 a 36-byte token that the repository cut rule
                              empties at P = 32 (it starts with a stray
                              continuation byte, so no prefix is valid UTF-8)
  cut_rule_samples(seed, n)   byte strings for the cut-rule properties C1-C6:
                              random bytes, half of them continuation bytes,
                              and valid UTF-8 in many scripts, up to 320 bytes
  synthetic_vocabulary(seed)  a small vocabulary in which every collision
                              cause is planted; each token carries the cause
                              it must produce in the raw and decode tables
  lam(raw)                    the printout model: invalid UTF-8 -> U+FFFD
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

LETTERS = list("aZ5-_ ") + ["é", "ñ", "ಕ", "ನ", "್", "ಡ", "ಾ", "ि", "र", "ा", "日", "😀", "ᱚ", "ꯃ"]
KANNADA = "ಕನ್ನಡ".encode("utf-8")                  # 5 letters, 15 bytes
CUT_EXAMPLE = b"\x95" + b"a" * 20 + KANNADA       # 36 bytes


def lam(raw: bytes) -> bytes:
    """lambda in docs/proof.md: decode with U+FFFD for invalid sequences, encode again."""
    return raw.decode("utf-8", errors="replace").encode("utf-8")


def _word(rng: random.Random, lo: int, hi: int) -> bytes:
    return "".join(rng.choice(LETTERS) for _ in range(rng.randrange(lo, hi + 1))).encode("utf-8")


def cut_rule_samples(seed: int = 0, n: int = 10000) -> List[bytes]:
    rng = random.Random(seed)
    out = [b"", CUT_EXAMPLE, b"\x80" * 70, "😀".encode() * 20]
    for k in range(n):
        if k % 2:
            out.append(_word(rng, 0, 80))
        else:
            L = rng.randrange(0, 200)
            out.append(bytes(rng.randrange(0x80, 0xC0) if rng.random() < 0.5 else rng.randrange(256)
                             for _ in range(L)))
    return out


Planted = Dict[str, Optional[str]]   # {"raw": cause or None, "decode": cause or None}


def synthetic_vocabulary(seed: int = 7) -> List[Tuple[bytes, Planted]]:
    """
    (raw bytes, planted causes), one cause per table kind. A planted cause
    says: if this token shares a row in that table, the group's causes must
    include it. None says: this token must have a unique row in that table.
    At P = 32 and at P = 64 every one of the five causes occurs.
    """
    rng = random.Random(seed)
    vocab: List[Tuple[bytes, Planted]] = []

    def add(raw: bytes, raw_cause: Optional[str], dec_cause: Optional[str]) -> None:
        vocab.append((raw, {"raw": raw_cause, "decode": dec_cause}))

    ordinary = set()
    while len(ordinary) < 1500:                       # at most 8 letters, so at most 32 bytes: never cut
        ordinary.add(_word(rng, 1, 8))
    for w in sorted(ordinary):
        add(w, None, None)
    for f in range(30):                               # long, same first 64+ bytes: truncation at P = 32 and 64
        base = f"family{f:02d}-".encode() + KANNADA * 4
        for k in range(4):
            add(base + f"#{k}".encode() + _word(rng, 1, 5), "TRUNCATION_P", "TRUNCATION_P")
    for n in range(33, 81):                           # runs of one character
        add(b"-" * n, "TRUNCATION_P", "TRUNCATION_P")
    for k in range(20):                               # exact duplicates
        w = f"dup{k}-".encode() + _word(rng, 1, 4)
        add(w, "DUPLICATE_BYTES", "DUPLICATE_BYTES")
        add(w, "DUPLICATE_BYTES", "DUPLICATE_BYTES")
    add(b"<s>", "DUPLICATE_BYTES", "DUPLICATE_BYTES")  # special token and plain text, same bytes
    add(b"<s>", "DUPLICATE_BYTES", "DUPLICATE_BYTES")
    for frag in (b"\xe1", b"\xff", b"\xe0\xb2", b"\xa8", b"\xb2\x95", b"\x80\x80"):
        add(frag, None, "UTF8_REPLACEMENT")           # pieces of letters: they print as U+FFFD
    for k in range(6):                                # long and starting with a stray tail byte
        add(b"\x95" + f"cut{k}-".encode() + KANNADA * 5, "CUT_TO_EMPTY", None)
    add(b"", "EMPTY", "EMPTY")                        # empty tokens
    add(b"", "EMPTY", "EMPTY")
    return vocab
