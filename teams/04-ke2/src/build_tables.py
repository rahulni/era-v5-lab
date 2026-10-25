"""
Build the decode table and the raw byte table for P = 32 and P = 64.

  decode table   tokenizer.decode([id]) through the repo's token_id_to_bytes.
                 This is what the repo feeds the codec today. Known issue:
                 a token that holds part of a character decodes to U+FFFD,
                 so all such tokens share one code.
  raw table      the token's true bytes from its vocab piece (decode_ref).
                 This fixes the issue: every raw row is distinct.

Each table maps token id -> at most P bytes (repo utf8_safe_truncate), in the
repo's layout: byte_buffer (V, P) uint8, length_buffer (V,) int16. Pass the raw
pair to KroneckerEmbedding(byte_buffer=..., length_buffer=...) to use it.

Outputs:
  build/tables/{decode,raw}_P{32,64}.npz   the tables (generated, not committed)
  results/tables_manifest.json             sha256 and stats per table, and the
                                           check that the decode table equals
                                           the repo's build_byte_buffer

Run from the lab repo root:  python teams/04-ke2/src/build_tables.py
"""

from __future__ import annotations

import time

import numpy as np

import ke2_common as C
from kronecker_embeddings import build_byte_buffer


def build_all(tok):
    """Return {kind: full byte list} and {(kind, P): (buf, lens)} for the four tables."""
    full = {kind: C.full_bytes(tok, kind) for kind in C.KINDS}
    tables = {(kind, P): C.make_table(full[kind], P) for kind in C.KINDS for P in C.P_VALUES}
    return full, tables


def main() -> int:
    t0 = time.time()
    tok = C.load_tokenizer()
    V = C.vocab_size(tok)
    specials = C.special_ids(tok)
    full, tables = build_all(tok)
    print(f"{C.TOKENIZER_NAME}: V={V}, special tokens={len(specials)}")

    C.TABLES.mkdir(parents=True, exist_ok=True)
    manifest = {"provenance": C.provenance(), "vocab_size": V, "special_tokens": len(specials), "tables": {}}
    for P in C.P_VALUES:
        repo_buf, repo_lens = build_byte_buffer(tok, pos_dim=P)
        for kind in C.KINDS:
            buf, lens = tables[(kind, P)]
            full_len = np.array([len(b) for b in full[kind]], dtype=np.int32)
            np.savez_compressed(C.table_path(kind, P), byte_buffer=buf, length_buffer=lens, full_length=full_len)
            rows = C.table_rows(buf, lens)
            info = {
                "file": f"build/tables/{C.table_path(kind, P).name}",
                "shape": {"byte_buffer": list(buf.shape), "length_buffer": list(lens.shape)},
                "dtype": {"byte_buffer": "uint8", "length_buffer": "int16"},
                "sha256": C.sha256_table(buf, lens),
                "distinct_rows": len(set(rows)),
                "ids_sharing_a_row": V - len(set(rows)),
                "rows_with_u_fffd": sum(C.U_FFFD in r for r in rows),
                "n_truncated": int((full_len > lens).sum()),
                "n_empty": int((lens == 0).sum()),
                "max_full_length": int(full_len.max()),
            }
            if kind == "decode":
                info["matches_repo_build_byte_buffer"] = bool(
                    np.array_equal(buf, repo_buf.numpy()) and np.array_equal(lens, repo_lens.numpy()))
            manifest["tables"][f"{kind}_P{P}"] = info
            print(f"  {kind}_P{P}: distinct={info['distinct_rows']} sharing={info['ids_sharing_a_row']} "
                  f"truncated={info['n_truncated']} sha256={info['sha256'][:16]}"
                  + (f" repo_match={info['matches_repo_build_byte_buffer']}" if kind == "decode" else ""))

    manifest["full_sha256"] = {k: C.sha256_bytes_list(v) for k, v in full.items()}
    classes = [C.script_class(b, i in specials) for i, b in enumerate(full["raw"])]
    manifest["script_class_counts"] = {s: classes.count(s) for s in C.ordered_classes(classes)}
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    C.write_json(C.RESULTS / "tables_manifest.json", manifest)
    print(f"wrote {C.TABLES} and results/tables_manifest.json ({manifest['elapsed_s']} s)")
    ok = all(t.get("matches_repo_build_byte_buffer", True) for t in manifest["tables"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
