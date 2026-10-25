# KE2 output shape (for G5)

KE2 replaces `nn.Embedding(V, d_model)` with a fixed byte codec followed by one trainable
`nn.Linear(D, d_model, bias=False)`. The input and output contract is the same as `nn.Embedding`.
Paths below are relative to `teams/04-ke2/`; the proof is `docs/proof.md`.

Tokenizer: BrahmicTokenizer-131K, so **V = 131,072** (ids 0 … 131,071). $P$ = the codec's
`pos_dim`, and **D = 256 · P**. The shapes below are formulas in V and P, so they still hold if
G3's tokenizer v0 (28 Oct) changes V.

## Shapes and dtypes

| stage | shape | dtype | trainable | P = 32 | P = 64 |
|---|---|---|---|---|---|
| input token ids | (B, T) | int64 | – | | |
| byte buffer (raw byte table) | (V, P) | uint8 | no (buffer) | 4 MiB | 8 MiB |
| length buffer | (V,) | int16 | no (buffer) | 256 KiB | 256 KiB |
| codec output E | (B, T, D) | float32 | no (no grad) | D = 8,192 | D = 16,384 |
| projection weight | (d_model, D) | model dtype | **yes** | 8,192 · d_model | 16,384 · d_model |
| **KE2 output** | **(B, T, d_model)** | projection weight dtype | – | | |

- The output dtype follows the projection weight. The codec output is cast to it
  (`embedding.py:203`).
- `state_dict()` holds only `projection.weight`, shape (d_model, D). The byte buffers are
  rebuilt from the tokenizer.
- There is no bias, no normalization and no activation after the projection.

## Sizes to plan for

| | P = 32 | P = 64 |
|---|---|---|
| trainable params, d_model = 1024 | 8.4 M | 16.8 M |
| trainable params, d_model = 2048 | 16.8 M | 33.6 M |
| `nn.Embedding(131072, d)` for comparison | 134.2 M (d = 1024), 268.4 M (d = 2048) | same |
| codec output per token (float32) | 32 KiB | 64 KiB |
| codec output for 32,768 tokens per micro-batch (e.g. B = 8, T = 4096) | 1 GiB | 2 GiB |
| `mode="cached"` full table (V × D float32) | 4 GiB | 8 GiB |

- **The codec output must be saved for the backward pass** of the projection
  (dW = gradᵀ · E). Budget for it as an activation.
- `mode="dynamic"`, the repo default, keeps only the byte buffers and recomputes E each step.

## How to build it

Use the **raw byte table**, not the repo's default `KroneckerEmbedding(tokenizer=...)`. The
default builds the table with `decode([id])`, and there 1,151 ids share 10 codes
(`results/collisions.csv`). With the raw table every id has a unique code (`docs/proof.md`, Theorem 1).

```python
import torch
from kronecker_embeddings import KroneckerEmbedding          # pip install kronecker-embeddings==0.1.1
from decode_ref import pack_rows, raw_byte_table             # teams/04-ke2/src
from ke2_common import load_tokenizer

tok = load_tokenizer()                                      # BrahmicTokenizer-131K, pinned
P, d_model = 32, 1024
buf, lens = pack_rows(raw_byte_table(tok), P)               # (V, P) uint8, (V,) int16
emb = KroneckerEmbedding(vocab_size=buf.shape[0], d_model=d_model, pos_dim=P,
                         byte_buffer=torch.from_numpy(buf), length_buffer=torch.from_numpy(lens))
x = emb(input_ids)                                          # (B, T) int64 -> (B, T, d_model)
```

This snippet was run with the pinned tokenizer (d_model = 64): output (1, 4, 64) float32,
`state_dict` = {`projection.weight`: (64, 8192)}.

## What E looks like

These properties are useful for kernels and for debugging. They are proven in `docs/proof.md` (Lemma 2).

- Each token's row of E has exactly two values: $h_L>0$ at its $L$ byte positions (index
  `byte * P + position`), and $\ell_L<0$ everywhere else. Both depend only on the byte length
  $L$ ($1\le L\le 32$ for this tokenizer).
- For every length, $h_L\ge15.93$ and $-0.063<\ell_L<0$. Casting E to bf16 or fp16 keeps the two
  values and their signs. A fixed threshold of 8 finds the marks even when every entry is off by
  up to 7.93 (`decode_ref.ROBUST_THRESHOLD`; `docs/proof.md`, Lemma 2).
- So $E\,W^\top = (h_L-\ell_L)\sum_{s\in S} W_{:,s} + \ell_L \sum_i W_{:,i}$. That is $L$ column
  gathers plus one shared column sum, instead of a D-wide matmul. It agrees with the dense
  product to $3.6\times10^{-15}$ in float64.
- E is deterministic on CPU. Its float32 bits are identical across 1, 4 and 20 threads
  (`results/determinism.json`).

## Choosing P

For this tokenizer, **P = 32 already gives every token a unique code**: the longest token is 32
bytes, so nothing is truncated. P = 64 doubles D, parameters and activation memory, and gains no
uniqueness. The KE2 decision on 14 Nov can take this as given.

## Open inputs

- **G3 tokenizer v0 (28 Oct):** V and the byte table may change; the formulas above do not. Set
  `KE2_TOKENIZER` and re-run `src/build_tables.py`.
- **G10 special-token list (31 Oct):** the 356 special tokens are encoded from their literal text
  (at most 20 bytes). Their handling may change; the shapes do not.
