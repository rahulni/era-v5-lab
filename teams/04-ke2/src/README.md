# KE2 week 1: code

Proves the forward encode of Kronecker Embedding 2 and checks that each token has a unique code
or a collision with a known cause. Everything runs on CPU. There is no new encoder code: the
encoder is `kronecker_codec` from the kronecker-embeddings repository, with `pos_dim` = P
(32 by default, 64 by setting `pos_dim=64`).

Tokenizer: BrahmicTokenizer-131K, pinned by commit and by the SHA-256 of `tokenizer.json`. It is
found in this order:
1. the `KE2_TOKENIZER` env var (a `tokenizer.json` or its folder; use it for G3's v0);
2. a clone next to this repo (`../BrahmicTokenizer-131K`);
3. `~/.cache/ke2/`;
4. a download of the pinned commit.

## Run

From the lab repo root:

```bash
pip install -r teams/04-ke2/src/requirements.txt

python teams/04-ke2/src/build_tables.py    # tables -> build/tables/, results/tables_manifest.json   ~45 s
python teams/04-ke2/src/collisions.py      # results/collisions*.csv, collisions_summary.json      ~10 s
python teams/04-ke2/src/proof_checks.py    # results/proof_checks.json (37 checks)                 ~4 min
python teams/04-ke2/src/proof_figures.py   # docs/figures/*.svg, the figures of docs/proof.md      ~10 s
python teams/04-ke2/src/determinism.py     # results/decode_cpu_sha256.txt, determinism.json      ~5 min
pytest teams/04-ke2/tests -q               # 72 tests                                             ~1 min
```

Each script exits non-zero if its own check fails.

## Files

| file | what it does |
|---|---|
| `build_tables.py` | Builds the decode table (`decode([id])`, the repo's current path) and the raw byte table (the fix) for P = 32 and P = 64 |
| `collisions.py` | Groups ids with equal rows, gives each group a checked cause, and counts per script class |
| `proof_checks.py` | Numerical checks for the cut rule, Lemma 1, Lemma 2 and Theorem 1 (`../docs/proof.md`) |
| `proof_figures.py` | Draws the 8 figures of `../docs/proof.md` from the codec's outputs, and stops if they disagree with Lemma 2 |
| `synthetic.py` | Synthetic strings and a vocabulary with every collision cause planted, for cases real data never reaches |
| `determinism.py` | Decodes every token id in three fresh CPU processes and compares the hashes |
| `decode_ref.py` | Decode reference for G6: token bytes, text from ids, codes back to bytes and to token ids |
| `ke2_common.py` | Shared definitions: tokenizer pin, tables, script classes, hashing, provenance |

## Results (`../results/`)

| file | content |
|---|---|
| `collisions.csv` | one row per collision group (all four tables): cause, script classes, every member id, the reason |
| `collisions_by_script.csv` | per table and script class: ids, colliding ids, groups, colliding ids per cause |
| `collisions_summary.json` | totals per table, and whether each P = 64 group sits inside one P = 32 group |
| `decode_cpu_sha256.txt` | the three decode hashes, one line per run |
| `determinism.json` | all 14 hashes per run (decode, raw, tables, float32 codes, roundtrips) |
| `proof_checks.json` | the 37 numerical checks |
| `tables_manifest.json` | SHA-256 and stats of the four tables |

The tables themselves are written to `../build/tables/` and are not committed. They exceed the
5 MB pull-request limit, and the manifest hashes identify them.

## Results in one line each

- decode table: 129,931 distinct codes for 131,072 ids. 10 groups, all `UTF8_REPLACEMENT`.
- raw byte table: 131,072 distinct codes at P = 32 and at P = 64. Every token is unique.
- 37/37 proof checks pass. The three CPU decode hashes match.
