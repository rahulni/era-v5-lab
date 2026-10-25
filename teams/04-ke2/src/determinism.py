"""
Decode every token id on CPU three times; the three hashes must match.

The parent launches three fresh Python processes, one after the other. Each
run differs in what could leak nondeterminism:

  run 1: PYTHONHASHSEED=0          torch threads = 1
  run 2: PYTHONHASHSEED=12345      torch threads = 4
  run 3: PYTHONHASHSEED=987654321  torch threads = library default

Each run, from scratch, on CPU:
  decode       tokenizer.decode([id]) for every id (repo token_id_to_bytes)
  raw          the raw byte table for every id (decode_ref.raw_byte_table)
  table:*      the four byte tables
  encode:*     the forward encode of every token (repo kronecker_codec), float32
  roundtrip:*  every code decoded back to bytes (decode_ref.decode_codes)

Hash format: SHA-256. Byte lists: for each id in order, 4-byte little-endian
length, then the bytes. Tables: byte_buffer bytes, then length_buffer as
little-endian int16. Encodes: the float32 rows in id order.

Outputs:
  results/decode_cpu_sha256.txt   the three decode hashes, one line per run
  results/determinism.json        all 14 hashes per run, run settings, cross-checks

Run from the lab repo root:  python teams/04-ke2/src/determinism.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import ke2_common as C

RUNS = [("0", 1), ("12345", 4), ("987654321", None)]


def decode_hash(tok) -> str:
    """The hash written to decode_cpu_sha256.txt: tokenizer.decode([id]) bytes for every id."""
    return C.sha256_bytes_list(C.decode_full_bytes(tok))


def worker(threads) -> dict:
    t0 = time.time()
    if threads:
        torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    tok = C.load_tokenizer()
    full = {"decode": C.decode_full_bytes(tok), "raw": C.raw_byte_table(tok)}
    hashes = {k: C.sha256_bytes_list(v) for k, v in full.items()}
    t_decode = time.time() - t0
    for kind in C.KINDS:
        for P in C.P_VALUES:
            name = f"{kind}_P{P}"
            buf, lens = C.make_table(full[kind], P)
            hashes[f"table:{name}"] = C.sha256_table(buf, lens)
            h = hashlib.sha256()
            back_buf, back_len = np.zeros_like(buf), np.zeros_like(lens)
            for s, E in C.encode_chunks(buf, lens, P):
                h.update(np.ascontiguousarray(E.numpy()).tobytes())
                db, dl, _ = C.decode_codes(E, P)
                back_buf[s:s + len(E)], back_len[s:s + len(E)] = db.numpy(), dl.numpy()
            hashes[f"encode:{name}"] = h.hexdigest()
            hashes[f"roundtrip:{name}"] = C.sha256_table(back_buf, back_len)
    return {"pid": os.getpid(), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "torch_threads": torch.get_num_threads(), "device": str(C.DEVICE),
            "token_ids_decoded": len(full["decode"]), "decode_seconds": round(t_decode, 2),
            "total_seconds": round(time.time() - t0, 2), "hashes": hashes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-out")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.worker_out:
        C.write_json(Path(args.worker_out), worker(args.threads))
        return 0

    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (seed, threads) in enumerate(RUNS, 1):
            out = Path(tmp) / f"run{i}.json"
            env = {**os.environ, "PYTHONHASHSEED": seed, "CUDA_VISIBLE_DEVICES": ""}
            print(f"run {i}: PYTHONHASHSEED={seed} threads={threads or 'default'} ...", flush=True)
            subprocess.run([sys.executable, __file__, "--worker-out", str(out), "--threads", str(threads or 0)],
                           env=env, check=True)
            runs.append({"run": i, **C.read_json(out)})
            print(f"  decoded {runs[-1]['token_ids_decoded']} ids, {runs[-1]['torch_threads']} threads, "
                  f"{runs[-1]['total_seconds']} s")

    manifest = C.read_json(C.RESULTS / "tables_manifest.json")
    compare = {}
    for key in runs[0]["hashes"]:
        vals = [r["hashes"][key] for r in runs]
        e = {"match": len(set(vals)) == 1, "sha256": vals[0] if len(set(vals)) == 1 else vals}
        if key.startswith("table:"):
            e["equals_tables_manifest"] = vals[0] == manifest["tables"][key[6:]]["sha256"]
        if key.startswith("roundtrip:"):
            e["equals_table_hash"] = vals[0] == runs[0]["hashes"]["table:" + key[10:]]
        compare[key] = e
    all_match = all(e["match"] for e in compare.values())
    cross_ok = all(e.get("equals_tables_manifest", True) and e.get("equals_table_hash", True)
                   for e in compare.values())

    info = C.tokenizer_info()
    lines = [
        "# KE2 week 1: tokenizer.decode([id]) for every token id, on CPU, in three fresh processes.",
        f"# Tokenizer: {info['name']}, tokenizer.json sha256 {info['tokenizer_json_sha256']}, "
        f"{runs[0]['token_ids_decoded']} ids.",
        "# Hash: SHA-256 over each id in order: 4-byte little-endian length, then the decoded bytes.",
        "# The raw byte table, the tables, the encodes and the roundtrips are in determinism.json.",
    ] + [f"{r['hashes']['decode']}  run{r['run']}  PYTHONHASHSEED={r['PYTHONHASHSEED']} "
         f"torch_threads={r['torch_threads']}" for r in runs]
    (C.RESULTS / "decode_cpu_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    C.write_json(C.RESULTS / "determinism.json", {
        "provenance": C.provenance(),
        "runs": [{k: v for k, v in r.items() if k != "hashes"} for r in runs],
        "all_three_match": all_match, "tables_match_manifest_and_roundtrip": cross_ok, "hashes": compare})

    w = max(map(len, compare))
    for key, e in compare.items():
        print(f"  {'OK ' if e['match'] else 'BAD'} {key:<{w}}  {e['sha256'][:16] if e['match'] else e['sha256']}")
    print(f"three runs match: {all_match}; tables == manifest and roundtrip == table: {cross_ok}")
    return 0 if all_match and cross_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
