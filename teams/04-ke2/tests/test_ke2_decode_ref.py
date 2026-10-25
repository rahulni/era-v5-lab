"""decode_ref (for G6): token bytes, text from ids, codes back to bytes and to token ids."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ke2_common as C
from decode_ref import CodeIndex, decode_code_bytes, decode_ids, encode_rows, pack_rows, token_bytes

TEXTS = [
    "भारत एक देश है", "বাংলা ভাষা", "ਪੰਜਾਬੀ", "ગુજરાતી", "ଓଡ଼ିଆ", "தமிழ் மொழி", "తెలుగు", "ಕನ್ನಡ",
    "മലയാളം", "Hello, world!", "def f(x): return x**2", "😀🙂", "naïve café",
]
# Text whose characters fall back to byte fragments, so per-token decode([id]) breaks.
FRAGMENT_TEXTS = ["ᱚᱞ ᱪᱤᱠᱤ", "ꯃꯤꯇꯩ ꯃꯌꯦꯛ", "අකුරු", "日本語のテキスト", "한국어"]
# Parameters by argument name, applied by conftest.py (P runs for 32 and 64 there).
PARAMS = {"text": TEXTS + FRAGMENT_TEXTS, "frag_text": FRAGMENT_TEXTS}


def test_token_bytes_is_the_raw_table(tok, full):
    rng = np.random.default_rng(0)
    for i in rng.choice(len(full["raw"]), 2000, replace=False).tolist() + sorted(C.special_ids(tok)):
        assert token_bytes(tok, i) == full["raw"][i]


def test_decode_ids_gives_back_the_text(tok, text):
    assert decode_ids(tok, tok.encode(text, add_special_tokens=False)) == text


def test_per_token_decode_breaks_where_decode_ref_does_not(tok, frag_text):
    ids = tok.encode(frag_text, add_special_tokens=False)
    assert "".join(tok.decode([i]) for i in ids) != frag_text     # the known issue
    assert decode_ids(tok, ids) == frag_text                      # the fix


def test_codes_map_back_to_token_ids(tok, full, tables, P):
    rows = C.table_rows(*tables[("raw", P)])
    index = CodeIndex(rows, P)
    rng = np.random.default_rng(P)
    fragments = [i for i, r in enumerate(full["raw"]) if C.U_FFFD in full["decode"][i] and r != full["decode"][i]]
    ids = sorted(set(rng.choice(len(rows), 3000, replace=False).tolist() + fragments + sorted(C.special_ids(tok))))
    buf, lens = pack_rows([rows[i] for i in ids], P)
    E = torch.cat([e for _, e in encode_rows(buf, lens, P)])
    assert index.lookup(E).tolist() == ids
    assert decode_code_bytes(E[:5], P) == [rows[i] for i in ids[:5]]


def test_code_index_refuses_the_decode_table(tables):
    with pytest.raises(ValueError):
        CodeIndex(C.table_rows(*tables[("decode", 32)]), 32)


def test_bytes_outside_the_vocab_give_minus_one(tables):
    index = CodeIndex(C.table_rows(*tables[("raw", 32)]), 32)
    buf, lens = pack_rows([b"\xff\xfe not a token \xff"], 32)
    E = torch.cat([e for _, e in encode_rows(buf, lens, 32)])
    assert index.lookup(E).tolist() == [-1]
