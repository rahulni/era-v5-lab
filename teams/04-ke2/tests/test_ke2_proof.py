"""
Lemma 1 and Lemma 2 on synthetic byte strings (no tokenizer needed).

Every encode is the repo's kronecker_codec. The decoder is decode_ref.decode_codes.
Tests that take P run for P = 32 and P = 64 (conftest.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch

import synthetic as S
from decode_ref import ROBUST_THRESHOLD, decode_codes, pack_rows
from kronecker_embeddings import kronecker_codec, utf8_safe_truncate
from proof_checks import analytic_bounds

EPS = 1e-6


def closed_form(L, P):
    D = 256 * P
    mu = math.sqrt(L) / D
    sigma = math.sqrt((1 - L / D) / (D - 1))
    return mu, sigma, (1 / math.sqrt(L) - mu) / (sigma + EPS), -mu / (sigma + EPS)


def random_rows(P, n, seed):
    """n random byte strings, every length 0..P represented, all 256 byte values in play."""
    rng = np.random.default_rng(seed)
    lengths = np.concatenate([np.arange(P + 1), rng.integers(0, P + 1, n - P - 1)])
    return [rng.integers(0, 256, L, dtype=np.uint8).tobytes() for L in lengths]


def encode(rows, P, z=True):
    buf, lens = pack_rows(rows, P)
    E = kronecker_codec(torch.from_numpy(buf), torch.from_numpy(lens), pos_dim=P, z_normalize=z)
    return buf, lens, E


def test_lemma1_support_is_exactly_byte_times_P_plus_position(P):
    rows = random_rows(P, 3000, seed=P)
    _, lens, K = encode(rows, P, z=False)
    for i, r in enumerate(rows):
        nz = torch.nonzero(K[i]).flatten().tolist()
        assert nz == sorted(b * P + p for p, b in enumerate(r))
        if r:
            assert torch.all(K[i][nz] == torch.rsqrt(torch.tensor(float(len(r)))))


def test_lemma1_gram_identity(P):
    rng = np.random.default_rng(1)
    rows_a, rows_b = [], []
    for _ in range(4000):
        L1, L2 = rng.integers(1, P + 1, 2)
        a = rng.integers(0, 4, L1, dtype=np.uint8)          # small alphabet: many matching positions
        b = rng.integers(0, 4, L2, dtype=np.uint8)
        rows_a.append(a.tobytes())
        rows_b.append(b.tobytes())
    _, _, Ka = encode(rows_a, P, z=False)
    _, _, Kb = encode(rows_b, P, z=False)
    dot = (Ka.double() * Kb.double()).sum(1)
    expect = torch.tensor([sum(x == y for x, y in zip(a, b)) / math.sqrt(len(a) * len(b))
                           for a, b in zip(rows_a, rows_b)], dtype=torch.float64)
    assert float((dot - expect).abs().max()) < 1e-6


def test_lemma2_two_values_match_closed_form_for_every_length(P):
    rows = [bytes((37 * p + 11) % 256 for p in range(L)) for L in range(P + 1)]
    _, _, K = encode(rows, P, z=False)
    _, _, E = encode(rows, P)
    assert torch.all(E[0] == 0)                              # empty string -> zero vector
    for L in range(1, P + 1):
        mu, sigma, h, l = closed_form(L, P)
        S = K[L] != 0
        assert abs(float(K[L].double().mean()) - mu) / mu < 1e-4
        assert abs(float(K[L].double().std()) - sigma) / sigma < 1e-4
        assert torch.unique(E[L][S]).numel() == 1 and torch.unique(E[L][~S]).numel() == 1
        assert abs(float(E[L][S][0]) - h) / h < 1e-4 and abs(float(E[L][~S][0]) - l) / abs(l) < 1e-4
        assert float(E[L][S][0]) > 0 > float(E[L][~S][0])   # signs give the support


def test_decoder_is_exact_on_every_string_of_length_at_most_2():
    P = 32
    rows = [b""] + [bytes([x]) for x in range(256)] + [bytes([x, y]) for x in range(256) for y in range(256)]
    buf, lens = pack_rows(rows, P)
    seen = set()
    for s in range(0, len(rows), 4096):
        E = kronecker_codec(torch.from_numpy(buf[s:s + 4096]), torch.from_numpy(lens[s:s + 4096]), pos_dim=P)
        db, dl, ok = decode_codes(E, P)
        assert bool(ok.all())
        assert np.array_equal(db.numpy(), buf[s:s + 4096]) and np.array_equal(dl.numpy(), lens[s:s + 4096])
        seen.update(r.tobytes() for r in E.numpy())
    assert len(seen) == len(rows)                            # all 65,793 codes distinct


def test_decoder_is_exact_on_random_strings(P):
    rows = random_rows(P, 5000, seed=7 + P)
    buf, lens, E = encode(rows, P)
    db, dl, ok = decode_codes(E, P)
    assert bool(ok.all())
    assert np.array_equal(db.numpy(), buf) and np.array_equal(dl.numpy(), lens)


def test_decoder_flags_vectors_that_are_not_codes():
    P = 32
    E = torch.full((3, 256 * P), -0.01)
    E[0, 5 * P + 0] = E[0, 9 * P + 0] = 16.0                 # two bytes in one position
    E[1, 5 * P + 1] = 16.0                                   # position 1 used, position 0 empty
    E[2, 5 * P + 0] = 16.0                                   # a valid one-byte code
    for threshold in (0.0, ROBUST_THRESHOLD):
        _, _, ok = decode_codes(E, P, threshold)
        assert ok.tolist() == [False, False, True]


def _valid(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def test_cut_rule_properties_c1_to_c6(P):
    emptied = 0
    for r in S.cut_rule_samples(seed=P, n=4000):
        t = utf8_safe_truncate(r, P)
        assert r.startswith(t)                                       # C1 prefix
        assert len(r) > P or t == r                                  # C2 unchanged if short
        assert len(t) <= P                                           # C3 at most P bytes ...
        if len(r) > P:
            assert _valid(t) and (not t or P - 3 <= len(t) <= P)     # ... valid, empty or P-3..P bytes
            assert not (_valid(r) and not t)                         # C4 valid UTF-8 is never emptied (C6)
            emptied += not t
        assert utf8_safe_truncate(t, P) == t                         # C5 idempotent
    assert emptied > 0                                               # the empty case really happens
    assert len(utf8_safe_truncate(S.CUT_EXAMPLE, P)) == (0 if P < len(S.CUT_EXAMPLE) else len(S.CUT_EXAMPLE))


def test_lemma2_uniform_bounds_for_every_length(P):
    """Lemma 2 (4)-(5): h_L >= 15.93 > 8, -0.0627 <= l_L < 0, gap >= 15.99, for every L = 1..P."""
    rows = [bytes((53 * p + 7) % 256 for p in range(L)) for L in range(1, P + 1)]
    _, _, E = encode(rows, P)
    bh, bl, bg = analytic_bounds(P)
    assert bh > ROBUST_THRESHOLD
    for k, r in enumerate(rows):
        mark = torch.zeros(256 * P, dtype=torch.bool)
        mark[[b * P + p for p, b in enumerate(r)]] = True
        h, lo_max, lo_min = float(E[k][mark].min()), float(E[k][~mark].max()), float(E[k][~mark].min())
        assert h >= bh * (1 - 1e-5) and lo_max < 0 and lo_min >= -bl * (1 + 1e-5) and h - lo_min >= bg * (1 - 1e-5)


def test_decoders_survive_bf16_fp16_and_noise(P):
    """Lemma 2 (6)-(7): thresholds 0 and 8 decode exactly after casting; threshold 8 survives noise of 7.9."""
    rows = random_rows(P, 1500, seed=11 + P)
    buf, lens, E = encode(rows, P)
    l1 = abs(float(encode([b"a"], P)[2][0].min()))                 # |l_1|, the sign rule's margin
    g = torch.Generator().manual_seed(P)
    noise = lambda a: (torch.rand(E.shape, generator=g) * 2 - 1) * a
    cases = [(E.to(torch.bfloat16), 0.0), (E.to(torch.bfloat16), ROBUST_THRESHOLD),
             (E.to(torch.float16), 0.0), (E.to(torch.float16), ROBUST_THRESHOLD),
             (E + noise(7.9), ROBUST_THRESHOLD)]
    for Ex, threshold in cases:                                    # every row, the empty one included
        db, dl, ok = decode_codes(Ex, P, threshold)
        assert bool(ok.all()) and np.array_equal(db.numpy(), buf) and np.array_equal(dl.numpy(), lens)
    # The sign rule's margin is |l_1| for a non-empty row but 0 for the empty row (its code is all zeros).
    db, dl, ok = decode_codes(E + noise(0.9 * l1), P, 0.0)
    full_rows = torch.from_numpy(lens > 0)
    assert bool(ok[full_rows].all()) and np.array_equal(db.numpy()[lens > 0], buf[lens > 0])
    assert not bool((dl[~full_rows] == 0).all())                   # noisy empty codes fool the sign rule ...
    db, dl, ok = decode_codes(E + noise(0.9 * l1), P, ROBUST_THRESHOLD)
    assert bool(ok.all()) and np.array_equal(db.numpy(), buf)     # ... but not threshold 8
