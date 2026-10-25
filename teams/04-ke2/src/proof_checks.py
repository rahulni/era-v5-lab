"""
Numerical checks for the cut rule, Lemma 1, Lemma 2 and Theorem 1 (docs/proof.md).

Every check has an id that proof.md cites. All encodes go through the repo's
``kronecker_codec`` on CPU, so the proofs are checked against the starting
code itself.

  C0    cut rule C1-C6 on 10,000+ synthetic strings per P, and the 36-byte
        example that the rule empties
  L1-a  support of kappa (no z-norm): exactly L nonzeros, each 1/sqrt(L), at
        byte*P + p; every token of all 4 tables
  L1-b  Gram identity <k(a),k(b)> = m(a,b)/sqrt(|a||b|) on 200k token pairs
  L2-a  closed forms mu, sigma_L, h_L, l_L against the codec for every L = 0..P
  L2-b  every token code is two-valued (h_L on S, l_L off S) and {E>0} = S
  L2-c  uniform bounds for every L: h_L >= 15.93, -0.0627 <= l_L < 0, gap >= 15.99
  L2-d  decoders with threshold 0 and 8 recover every row of all 4 tables, in
        float32 and after casting the codes to bf16 and fp16
  L2-e  noise: threshold 8 survives +-7.9 per entry; the sign rule only +-0.9|l_1|
  T1-a  decoder D(E_P(t)) == T_P(t) for every token id, all 4 tables
  T1-b  ids with bit-identical codes == ids with identical rows
  T1-c  decode bytes == raw.decode("utf-8", "replace") for every token id
  T1-d  exhaustive: all 65,793 byte strings of length <= 2 decode back exactly
        and get pairwise distinct codes
  T1-e  raw tables: no token is truncated and no two share raw bytes, so by
        Theorem 1 (iv) every token id has a unique code
  T1-f  synthetic vocabulary with all five causes planted: equal codes = equal
        rows, and every group gets its planted, checked cause

Output: results/proof_checks.json

Run from the lab repo root:  python teams/04-ke2/src/proof_checks.py   (after build_tables.py)
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter

import numpy as np
import torch

import ke2_common as C
import synthetic as S
from collisions import KNOWN, explain
from decode_ref import ROBUST_THRESHOLD, pack_rows
from kronecker_embeddings import utf8_safe_truncate

EPS = 1e-6
checks = []


def record(cid, statement, scope, passed, **values):
    checks.append({"id": cid, "statement": statement, "scope": scope, "pass": bool(passed), **values})
    vals = ", ".join(f"{k}={v}" for k, v in values.items())
    print(f"  [{'PASS' if passed else 'FAIL'}] {cid:5s} {scope:11s} {vals}")


def closed_form(L: int, P: int):
    """Lemma 2 (1)-(2) in float64: (mu, sigma_L, h_L, l_L)."""
    D = C.CHAR_DIM * P
    if L == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = math.sqrt(L) / D
    sigma = math.sqrt((1 - L / D) / (D - 1))
    return mu, sigma, (1 / math.sqrt(L) - mu) / (sigma + EPS), -mu / (sigma + EPS)


def expected_support(buf: torch.Tensor, lens: torch.Tensor, P: int) -> torch.Tensor:
    """Dense bool mask of S(b) = {b_p * P + p : p < L} per row (Lemma 1 (1))."""
    B = buf.shape[0]
    pos = torch.arange(P).unsqueeze(0).expand(B, -1)
    valid = pos < lens.long().unsqueeze(1)
    mask = torch.zeros((B, C.CHAR_DIM * P), dtype=torch.bool)
    # Each position p writes a distinct index (remainder p), so a False written
    # for a padded slot never overwrites a real support entry.
    mask.scatter_(1, buf.long() * P + pos, valid)
    return mask


def row_hashes(E: torch.Tensor):
    return [hashlib.sha256(r.tobytes()).digest() for r in np.ascontiguousarray(E.numpy())]


def groups_of(keys):
    g = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    return g


def canon(keys):
    """The partition of ids induced by keys, in a comparable form."""
    return sorted(sorted(g) for g in groups_of(keys).values())


def analytic_bounds(P: int):
    """Lemma 2 (4)-(5): lower bound on every h_L, upper bound on every |l_L|, lower bound on every gap."""
    D = 256 * P
    h = math.sqrt((D - 1) / P) * (1 - P / D) / (1 + EPS * math.sqrt(D))
    l = (math.sqrt(P) / D) / math.sqrt((1 - P / D) / (D - 1))
    gap = math.sqrt((D - 1) / P) / (1 + EPS * math.sqrt(D - 1))
    return h, l, gap


def check_table(kind: str, P: int):
    name = f"{kind}_P{P}"
    buf_np, lens_np = C.load_table(kind, P)
    buf, lens = torch.from_numpy(buf_np), torch.from_numpy(lens_np)
    V = len(lens_np)

    bad_support = bad_value = 0
    for s, K in C.encode_chunks(buf_np, lens_np, P, z_normalize=False):
        b, l = buf[s:s + len(K)], lens[s:s + len(K)]
        mask = expected_support(b, l, P)
        bad_support += int(((K != 0) != mask).any(dim=1).sum())
        scale = torch.rsqrt(l.long().clamp_min(1).float()).unsqueeze(1).expand_as(K)
        bad_value += int(((K != scale) & mask).any(dim=1).sum())
    record("L1-a", "kappa has exactly L nonzeros, each 1/sqrt(L), at {b_p*P+p}", name,
           bad_support == 0 and bad_value == 0, tokens=V, rows_wrong_support=bad_support, rows_wrong_value=bad_value)

    cf = {L: closed_form(L, P) for L in range(P + 1)}
    not_two = mismatch_support = mismatch_decode = ill_formed = 0
    max_rel_h = max_rel_l = 0.0
    min_pos, max_neg = float("inf"), -float("inf")
    hashes = []
    inf = torch.tensor(float("inf"))
    dec_bad = Counter()
    cast_min_h = {"bf16": float("inf"), "fp16": float("inf")}
    cast_max_l = {"bf16": -float("inf"), "fp16": -float("inf")}
    for s, E in C.encode_chunks(buf_np, lens_np, P):
        b, l = buf[s:s + len(E)], lens[s:s + len(E)]
        mask = expected_support(b, l, P)
        mismatch_support += int(((E > 0) != mask).any(dim=1).sum())
        hi_max = torch.where(mask, E, -inf).max(1).values
        hi_min = torch.where(mask, E, inf).min(1).values
        lo_max = torch.where(~mask, E, -inf).max(1).values
        lo_min = torch.where(~mask, E, inf).min(1).values
        ne = l > 0
        not_two += int(((hi_max != hi_min) | (lo_max != lo_min))[ne].sum())
        not_two += int((E[~ne] != 0).any(dim=1).sum())   # L = 0 must give the zero vector
        for L in torch.unique(l[ne]).tolist():
            sel = l == L
            _, _, h, lo = cf[L]
            max_rel_h = max(max_rel_h, float(((hi_max[sel].double() - h).abs() / abs(h)).max()))
            max_rel_l = max(max_rel_l, float(((lo_max[sel].double() - lo).abs() / abs(lo)).max()))
        if ne.any():
            min_pos = min(min_pos, float(hi_min[ne].min()))
            max_neg = max(max_neg, float(lo_max[ne].max()))
        db, dl, wf = C.decode_codes(E, P)
        ill_formed += int((~wf).sum())
        mismatch_decode += int(((db != b).any(dim=1) | (dl != l)).sum())
        hashes.extend(row_hashes(E))
        for dname, Ex in (("fp32", E), ("bf16", E.to(torch.bfloat16)), ("fp16", E.to(torch.float16))):
            for th in (0.0, ROBUST_THRESHOLD):
                xb, xl, xw = C.decode_codes(Ex, P, th)
                dec_bad[f"{dname}_threshold_{th:g}"] += int(((xb != b).any(dim=1) | (xl != l) | ~xw).sum())
            if dname != "fp32":
                Ef = Ex.float()
                cast_min_h[dname] = min(cast_min_h[dname], float(torch.where(mask, Ef, inf).min()))
                cast_max_l[dname] = max(cast_max_l[dname], float(torch.where(~mask, Ef, -inf).max()))

    record("L2-b", "each code is two-valued (h_L on S, l_L off S) and {E>0} = S", name,
           not_two == 0 and mismatch_support == 0 and max_rel_h < 1e-4 and max_rel_l < 1e-4,
           rows_not_two_valued=not_two, rows_support_mismatch=mismatch_support,
           max_rel_err_h=f"{max_rel_h:.2e}", max_rel_err_l=f"{max_rel_l:.2e}",
           min_positive_entry=f"{min_pos:.6g}", max_negative_entry=f"{max_neg:.6g}")
    record("L2-d", "decoders with threshold 0 and 8 recover every row, in float32 and after casting to bf16 and fp16",
           name, sum(dec_bad.values()) == 0 and min(cast_min_h.values()) > ROBUST_THRESHOLD
           and max(cast_max_l.values()) < 0,
           tokens=V, mismatches=sum(dec_bad.values()), decoders_checked=len(dec_bad),
           min_h_bf16=f"{cast_min_h['bf16']:.6g}", max_l_bf16=f"{cast_max_l['bf16']:.6g}",
           min_h_fp16=f"{cast_min_h['fp16']:.6g}", max_l_fp16=f"{cast_max_l['fp16']:.6g}")
    record("T1-a", "D(E_P(t)) == T_P(t) for every token id (threshold 8)", name,
           mismatch_decode == 0 and ill_formed == 0,
           tokens=V, decode_mismatches=mismatch_decode, ill_formed_codes=ill_formed)

    rows = C.table_rows(buf_np, lens_np)
    same = canon(hashes) == canon(rows)
    record("T1-b", "ids with bit-identical codes == ids with identical table rows", name,
           same and len(set(hashes)) == len(set(rows)),
           distinct_codes=len(set(hashes)), distinct_rows=len(set(rows)), partitions_equal=same)


def check_gram(P: int, n_pairs: int = 100_000):
    buf_np, lens_np = C.load_table("raw", P)
    rng = np.random.default_rng(0)
    nonempty = np.flatnonzero(lens_np > 0)
    a_uni, b_uni = rng.choice(nonempty, n_pairs), rng.choice(nonempty, n_pairs)
    key = lens_np.astype(np.int64) * 256 + buf_np[:, 0]      # same length and first byte: many matches
    pool = [np.array(v) for v in groups_of(key.tolist()).values() if len(v) > 1]
    picks = rng.integers(0, len(pool), n_pairs)
    a_hard = np.array([rng.choice(pool[k]) for k in picks])
    b_hard = np.array([rng.choice(pool[k]) for k in picks])
    a, b = np.concatenate([a_uni, a_hard]), np.concatenate([b_uni, b_hard])
    dot = torch.cat([(ka.double() * kb.double()).sum(1) for (_, ka), (_, kb) in zip(
        C.encode_chunks(buf_np[a], lens_np[a], P, z_normalize=False),
        C.encode_chunks(buf_np[b], lens_np[b], P, z_normalize=False))])
    ba, bb = torch.from_numpy(buf_np[a]), torch.from_numpy(buf_np[b])
    la, lb = torch.from_numpy(lens_np[a]).long(), torch.from_numpy(lens_np[b]).long()
    m = ((ba == bb) & (torch.arange(P).unsqueeze(0) < torch.minimum(la, lb).unsqueeze(1))).sum(1).double()
    err = float((dot - m / torch.sqrt((la * lb).double())).abs().max())
    record("L1-b", "<k(a),k(b)> = m(a,b)/sqrt(|a||b|)", f"raw_P{P}", err < 1e-6,
           pairs=len(a), pairs_with_m_ge_2=int((m >= 2).sum()), max_abs_err=f"{err:.2e}")


def check_closed_form(P: int):
    D = C.CHAR_DIM * P
    buf = np.zeros((P + 1, P), dtype=np.uint8)
    lens = np.arange(P + 1, dtype=np.int16)
    for L in range(P + 1):
        buf[L, :L] = [(37 * p + 11) % 256 for p in range(L)]
    K = next(C.encode_chunks(buf, lens, P, z_normalize=False))[1].double()
    E = next(C.encode_chunks(buf, lens, P))[1]
    worst = {"mu": 0.0, "sigma": 0.0, "h": 0.0, "l": 0.0}
    signs_ok = True
    for L in range(1, P + 1):
        mu, sigma, h, lo = closed_form(L, P)
        S = K[L] != 0
        worst["mu"] = max(worst["mu"], abs(float(K[L].mean()) - mu) / mu)
        worst["sigma"] = max(worst["sigma"], abs(float(K[L].std()) - sigma) / sigma)
        worst["h"] = max(worst["h"], float((E[L][S].double() - h).abs().max()) / h)
        worst["l"] = max(worst["l"], float((E[L][~S].double() - lo).abs().max()) / abs(lo))
        signs_ok &= h > 0 > lo
    zero_ok = bool((E[0] == 0).all())
    record("L2-a", "mu=sqrt(L)/D, sigma_L=sqrt((1-L/D)/(D-1)), h_L>0>l_L, e(empty)=0", f"P={P}",
           signs_ok and zero_ok and all(v < 1e-4 for v in worst.values()),
           lengths_checked=f"0..{P}", D=D, **{f"max_rel_err_{k}": f"{v:.2e}" for k, v in worst.items()},
           h_1=f"{closed_form(1, P)[2]:.6g}", l_1=f"{closed_form(1, P)[3]:.6g}",
           h_P=f"{closed_form(P, P)[2]:.6g}", l_P=f"{closed_form(P, P)[3]:.6g}", empty_is_zero=zero_ok)


def check_bounds(P: int):
    """L2-c: h_L, l_L and the gap for every L = 1..P, from the codec, against Lemma 2 (4)-(5)."""
    D = 256 * P
    buf = np.zeros((P, P), dtype=np.uint8)
    lens = np.arange(1, P + 1, dtype=np.int16)
    for k in range(P):
        buf[k, :k + 1] = [(53 * p + 7) % 256 for p in range(k + 1)]
    E = next(C.encode_chunks(buf, lens, P))[1].double()
    hs, l_hi, l_lo = [], [], []
    for k in range(P):
        mark = torch.zeros(D, dtype=torch.bool)
        mark[[int(buf[k, p]) * P + p for p in range(k + 1)]] = True
        hs.append(float(E[k][mark].min()))
        l_hi.append(float(E[k][~mark].max()))
        l_lo.append(float(E[k][~mark].min()))
    gap = min(h - l for h, l in zip(hs, l_lo))
    ah, al, ag = analytic_bounds(P)
    tol = 1e-5   # float32 output against exact bounds
    record("L2-c", "for every L: h_L >= 15.93, -0.0627 <= l_L < 0, gap h_L - l_L >= 15.99", f"P={P}",
           min(hs) >= ah * (1 - tol) and ah > ROBUST_THRESHOLD and max(l_hi) < 0 and min(l_lo) >= -al * (1 + tol)
           and gap >= ag * (1 - tol),
           min_h=f"{min(hs):.6g}", bound_h=f"{ah:.6g}", max_l=f"{max(l_hi):.6g}", min_l=f"{min(l_lo):.6g}",
           bound_l=f"{-al:.6g}", min_gap=f"{gap:.6g}", bound_gap=f"{ag:.6g}")


def check_noise(P: int, n: int = 2000, chunk: int = 250):
    """L2-e: decoding noisy codes of real tokens with threshold 8 and with the sign rule."""
    buf_np, lens_np = C.load_table("raw", P)
    ids = np.random.default_rng(P).choice(len(lens_np), n, replace=False)
    l1 = abs(closed_form(1, P)[3])
    g = torch.Generator().manual_seed(P)
    bad = Counter()
    for s in range(0, n, chunk):
        sel = ids[s:s + chunk]
        b, l = torch.from_numpy(buf_np[sel]), torch.from_numpy(lens_np[sel])
        E = next(C.encode_chunks(buf_np[sel], lens_np[sel], P, chunk=chunk))[1]

        def errors(Ex, th):
            xb, xl, xw = C.decode_codes(Ex, P, th)
            return int(((xb != b).any(dim=1) | (xl != l) | ~xw).sum())

        noise = lambda a: (torch.rand(E.shape, generator=g) * 2 - 1) * a
        bad["threshold_8_at_7.9"] += errors(E + noise(7.9), ROBUST_THRESHOLD)
        bad["threshold_8_at_0.05"] += errors(E + noise(0.05), ROBUST_THRESHOLD)
        bad["sign_rule_at_0.9_l1"] += errors(E + noise(0.9 * l1), 0.0)
        bad["sign_rule_at_0.05"] += errors(E + noise(0.05), 0.0)
    record("L2-e", "threshold 8 survives noise up to 7.9 per entry; the sign rule only up to |l_1|", f"raw_P{P}",
           bad["threshold_8_at_7.9"] == 0 and bad["threshold_8_at_0.05"] == 0 and bad["sign_rule_at_0.9_l1"] == 0,
           codes=n, l1=f"{l1:.6g}", **{k: v for k, v in bad.items()})


def check_cut_rule(P: int):
    """C0: properties C1-C6 of utf8_safe_truncate (docs/proof.md section 0)."""
    def valid(x):
        try:
            x.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    v = Counter()
    samples = S.cut_rule_samples(seed=P)
    n_long = n_valid_long = n_empty = 0
    for r in samples:
        t = utf8_safe_truncate(r, P)
        if not r.startswith(t):
            v["C1"] += 1
        if len(r) <= P and t != r:
            v["C2"] += 1
        if len(t) > P or (len(r) > P and (not valid(t) or (t and not P - 3 <= len(t) <= P))):
            v["C3"] += 1
        if len(r) > P:
            n_long += 1
            if valid(r):
                n_valid_long += 1
                if not t:
                    v["C4"] += 1
            if not t:
                n_empty += 1
                if valid(r):
                    v["C6"] += 1
        if utf8_safe_truncate(t, P) != t:
            v["C5"] += 1
    kept = len(utf8_safe_truncate(S.CUT_EXAMPLE, P))
    expect = 0 if len(S.CUT_EXAMPLE) > P else len(S.CUT_EXAMPLE)
    record("C0", "cut rule: prefix; unchanged if short; at most P bytes; valid UTF-8 never emptied; idempotent; "
           "empty only for invalid input", f"P={P}", not v and kept == expect and n_empty > 0,
           strings=len(samples), longer_than_P=n_long, valid_longer_than_P=n_valid_long, cut_to_empty=n_empty,
           violations=sum(v.values()), example_36_bytes_kept=kept)


def check_synthetic(P: int):
    """T1-f: every cause, planted in a synthetic vocabulary, is found and checked."""
    vocab = S.synthetic_vocabulary()
    raw = [r for r, _ in vocab]
    dec = [S.lam(r) for r in raw]
    found, problems = set(), 0
    for kind, full in (("raw", raw), ("decode", dec)):
        rows = [utf8_safe_truncate(f, P) if len(f) > P else f for f in full]
        cause_of = {}
        for ids in (g for g in groups_of(rows).values() if len(g) > 1):
            causes, _ = explain(ids, rows[ids[0]], kind, P, raw, dec)
            problems += not set(causes) <= KNOWN
            found.update(causes)
            for t in ids:
                cause_of[t] = set(causes)
        for t, (_, planted) in enumerate(vocab):
            want = planted[kind]
            problems += (want is None and t in cause_of) or (want is not None and t in cause_of and want not in cause_of[t])
        buf, lens = pack_rows(rows, P)
        hashes = [h for _, E in C.encode_chunks(buf, lens, P) for h in row_hashes(E)]
        problems += canon(hashes) != canon(rows)
    record("T1-f", "synthetic vocabulary, all five causes planted: equal codes = equal rows, every group gets its "
           "planted, checked cause", f"P={P}", problems == 0 and found >= KNOWN,
           tokens=len(vocab), causes_found=" ".join(sorted(found)), problems=problems)


def check_exhaustive(P: int):
    strings = [b""] + [bytes([x]) for x in range(256)] + [bytes([x, y]) for x in range(256) for y in range(256)]
    buf = np.zeros((len(strings), P), dtype=np.uint8)
    lens = np.array([len(s) for s in strings], dtype=np.int16)
    for i, s in enumerate(strings):
        buf[i, :len(s)] = list(s)
    mismatch, hashes = 0, set()
    for s, E in C.encode_chunks(buf, lens, P):
        db, dl, wf = C.decode_codes(E, P)
        mismatch += int(((db != torch.from_numpy(buf[s:s + len(E)])).any(dim=1)
                         | (dl != torch.from_numpy(lens[s:s + len(E)])) | ~wf).sum())
        hashes.update(row_hashes(E))
    record("T1-d", "all byte strings of length <= 2: D(e_P(b)) == b, codes pairwise distinct", f"P={P}",
           mismatch == 0 and len(hashes) == len(strings),
           strings=len(strings), decode_mismatches=mismatch, distinct_codes=len(hashes))


def check_decode_is_lossy_raw(raw, dec):
    bad = sum(1 for r, d in zip(raw, dec) if r.decode("utf-8", "replace").encode("utf-8") != d)
    record("T1-c", "decode(t) == raw(t).decode('utf-8','replace') for every id", "all ids", bad == 0,
           tokens=len(raw), ids_where_decode_differs_from_raw=sum(r != d for r, d in zip(raw, dec)),
           ids_violating=bad)


def check_raw_unique(raw, P: int):
    buf, lens = C.load_table("raw", P)
    truncated = sum(len(r) > P for r in raw)
    duplicates = len(raw) - len(set(raw))
    distinct_rows = len(set(C.table_rows(buf, lens)))
    record("T1-e", "raw table: no truncation, no duplicate raw strings => every token id has a unique code",
           f"raw_P{P}", truncated == 0 and duplicates == 0 and distinct_rows == len(raw),
           tokens=len(raw), max_raw_length=max(map(len, raw)), tokens_truncated=truncated,
           duplicate_raw_strings=duplicates, distinct_rows=distinct_rows)


def main() -> int:
    t0 = time.time()
    torch.set_grad_enabled(False)
    tok = C.load_tokenizer()
    raw, dec = C.raw_byte_table(tok), C.decode_full_bytes(tok)
    print("Cut rule / Lemma 1 / Lemma 2 / Theorem 1 numerical checks (CPU)")
    for P in C.P_VALUES:
        check_cut_rule(P)
        check_closed_form(P)
        check_bounds(P)
        check_gram(P)
    for kind in C.KINDS:
        for P in C.P_VALUES:
            check_table(kind, P)
    check_decode_is_lossy_raw(raw, dec)
    for P in C.P_VALUES:
        check_noise(P)
        check_exhaustive(P)
        check_raw_unique(raw, P)
        check_synthetic(P)

    order = ["C0", "L1-a", "L1-b", "L2-a", "L2-b", "L2-c", "L2-d", "L2-e",
             "T1-a", "T1-b", "T1-c", "T1-d", "T1-e", "T1-f"]
    checks.sort(key=lambda c: (order.index(c["id"]), c["scope"]))
    n_pass = sum(c["pass"] for c in checks)
    C.write_json(C.RESULTS / "proof_checks.json", {
        "provenance": C.provenance(), "all_pass": n_pass == len(checks), "passed": n_pass,
        "total": len(checks), "elapsed_s": round(time.time() - t0, 1), "checks": checks})
    print(f"{n_pass}/{len(checks)} checks pass ({time.time() - t0:.0f} s) -> results/proof_checks.json")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
