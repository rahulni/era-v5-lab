"""
KE2 decode reference (for G6).

Four exact decodes, CPU only, no learned parameters:

  token id -> bytes      token_bytes(tok, i), raw_byte_table(tok)
      The token's true bytes, read from its vocab piece through the
      byte-level BPE map. This replaces tokenizer.decode([id]), which turns
      every token that holds part of a character into U+FFFD, so those
      tokens would share one code.

  ids      -> text       decode_ids(tok, ids)
      Join the raw bytes of all ids, then UTF-8 decode once. Characters
      split across tokens come back whole.

  code     -> bytes      decode_codes(E, P, threshold=8.0)
      Inverse of the KE2 forward encode (``kronecker_codec`` in the
      kronecker-embeddings repo). By Lemma 2 in docs/proof.md every code has
      two values, h >= 15.93 on its marks (indices byte * P + position) and
      -0.063 < l < 0 elsewhere, for every token length. So any threshold
      between them finds the marks exactly. The default, 8, sits in the
      middle: each entry may be off by up to 7.93 and decoding is still
      exact (threshold=0, the plain sign rule, tolerates only 0.0078).
      Exact in float32 and after casting codes to bf16 or fp16.

  code     -> token id   CodeIndex(rows).lookup(E)
      Decode the code to bytes, then look the bytes up in the raw byte
      table. Exact because every raw row is unique (results/collisions.csv).

Dependencies: numpy, torch; a Hugging Face tokenizer for the token-level
functions. The forward encode itself is only needed by ``encode_rows``.

CLI examples:
    python decode_ref.py --ids 66526 2420 13092 732
    python decode_ref.py --text "भारत एक देश है"
    python decode_ref.py --roundtrip 32
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

CHAR_DIM = 256
ROBUST_THRESHOLD = 8.0   # Lemma 2 (6): midway between max l (< 0) and min h (>= 15.93)


# ---------------------------------------------------------------- token id -> bytes

def bytes_to_unicode() -> Dict[int, str]:
    """
    The reversible byte -> printable-character map used by byte-level BPE
    (GPT-2, o200k_base, BrahmicTokenizer-131K). Vocab pieces are written in
    these characters; the inverse map gives a piece's true bytes.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_UNICODE_TO_BYTE = {c: b for b, c in bytes_to_unicode().items()}


def special_ids(tok) -> set:
    """Added / special token ids. Their bytes are their literal text, as in the repo."""
    ids = set(getattr(tok, "added_tokens_decoder", {}).keys())
    ids.update(getattr(tok, "all_special_ids", []) or [])
    return ids


def _check_byte_level(tok) -> None:
    decoder = type(tok.backend_tokenizer.decoder).__name__
    if decoder != "ByteLevel":
        raise NotImplementedError(f"raw bytes need a byte-level BPE tokenizer; decoder is {decoder}")


def token_bytes(tok, token_id: int, specials: Optional[set] = None) -> bytes:
    """The true bytes of one token id."""
    specials = special_ids(tok) if specials is None else specials
    piece = tok.convert_ids_to_tokens(int(token_id))
    if token_id in specials:
        return piece.encode("utf-8")
    return bytes(_UNICODE_TO_BYTE[c] for c in piece)


def raw_byte_table(tok) -> List[bytes]:
    """True bytes for every token id 0 .. V-1 (the RAW byte table, untruncated)."""
    _check_byte_level(tok)
    specials = special_ids(tok)
    V = max(tok.get_vocab().values()) + 1
    return [token_bytes(tok, i, specials) for i in range(V)]


def decode_ids(tok, ids: Iterable[int], errors: str = "strict") -> str:
    """Text of a sequence of token ids: join raw bytes, decode UTF-8 once."""
    specials = special_ids(tok)
    data = b"".join(token_bytes(tok, int(i), specials) for i in ids)
    return data.decode("utf-8", errors=errors)


def pack_rows(rows: Sequence[bytes], P: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pack byte rows (each at most P bytes) into the repo's (V, P) uint8 + (V,) int16 layout."""
    buf = np.zeros((len(rows), P), dtype=np.uint8)
    lens = np.zeros((len(rows),), dtype=np.int16)
    for i, r in enumerate(rows):
        if len(r) > P:
            raise ValueError(f"row {i} has {len(r)} bytes > P={P}; truncate first")
        if r:
            buf[i, : len(r)] = np.frombuffer(r, dtype=np.uint8)
        lens[i] = len(r)
    return buf, lens


# ---------------------------------------------------------------- code -> bytes

def decode_codes(E: torch.Tensor, P: int, threshold: float = ROBUST_THRESHOLD
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Invert the forward encode for a batch of codes E of shape (B, 256 * P).

    Support S = {i : E_i > threshold}. Any threshold in (-0.0078, 15.93)
    decodes every clean code exactly (Lemma 2); the default 8 also survives
    errors of up to 7.93 per entry. Index i = byte * P + position, so in the
    (256, P) view each column holds at most one support entry, whose row is
    the byte at that position.

    Returns (bytes (B, P) uint8, lengths (B,) int16, well_formed (B,) bool).
    well_formed: every column has <= 1 support entry and the occupied
    columns are exactly 0 .. L-1. Every output of the forward encode is
    well formed (Lemma 1); a code that is not did not come from the encoder.
    """
    if E.dim() != 2 or E.shape[1] != CHAR_DIM * P:
        raise ValueError(f"E must be (B, {CHAR_DIM * P}); got {tuple(E.shape)}")
    B = E.shape[0]
    support = (E.float() > threshold).view(B, CHAR_DIM, P)
    per_col = support.sum(dim=1)
    occupied = per_col > 0
    L = occupied.sum(dim=1)
    prefix = torch.arange(P, device=E.device).unsqueeze(0) < L.unsqueeze(1)
    well_formed = (per_col <= 1).all(dim=1) & (occupied == prefix).all(dim=1)
    byte = support.to(torch.uint8).argmax(dim=1)
    byte = torch.where(occupied, byte, torch.zeros_like(byte))
    return byte.to(torch.uint8), L.to(torch.int16), well_formed


def decode_code_bytes(E: torch.Tensor, P: int, threshold: float = ROBUST_THRESHOLD) -> List[bytes]:
    """decode_codes, returned as Python bytes objects. Raises on a malformed code."""
    buf, lens, ok = decode_codes(E, P, threshold)
    if not bool(ok.all()):
        raise ValueError(f"{int((~ok).sum())} code(s) are not outputs of the forward encode")
    b = buf.cpu().numpy()
    return [b[i, : int(n)].tobytes() for i, n in enumerate(lens.tolist())]


def encode_rows(buf: np.ndarray, lens: np.ndarray, P: int, chunk: int = 4096):
    """Forward encode (repo ``kronecker_codec``, z-normalized), yielded in chunks: (start, E)."""
    from kronecker_embeddings import kronecker_codec
    bb, ll = torch.from_numpy(buf), torch.from_numpy(lens)
    for s in range(0, bb.shape[0], chunk):
        yield s, kronecker_codec(bb[s:s + chunk], ll[s:s + chunk], char_dim=CHAR_DIM, pos_dim=P)


# ---------------------------------------------------------------- code -> token id

class CodeIndex:
    """
    Map codes back to token ids through a byte table.

    rows: the byte row of each token id (e.g. the raw table truncated to P).
    The raw table has no duplicate rows, so lookup is exact. A table with
    duplicate rows (such as the decode table) is refused unless
    allow_collisions=True, in which case the smallest id of a group is used.
    """

    def __init__(self, rows: Sequence[bytes], P: int, allow_collisions: bool = False):
        self.P = P
        self.index: Dict[bytes, int] = {}
        dup = 0
        for i, r in enumerate(rows):
            if r in self.index:
                dup += 1
                continue
            self.index[r] = i
        if dup and not allow_collisions:
            raise ValueError(f"{dup} token ids share a row with another id; codes would be ambiguous")

    def lookup(self, E: torch.Tensor, threshold: float = ROBUST_THRESHOLD) -> torch.Tensor:
        """Token id for each code in E (B, 256 * P); -1 where the bytes are not a vocab token."""
        return torch.tensor([self.index.get(b, -1) for b in decode_code_bytes(E, self.P, threshold)],
                            dtype=torch.long)


# ---------------------------------------------------------------- CLI

def _load_default_tokenizer():
    from ke2_common import load_tokenizer
    return load_tokenizer()


def main() -> int:
    ap = argparse.ArgumentParser(description="KE2 decode reference")
    ap.add_argument("--ids", type=int, nargs="*", help="token ids to decode")
    ap.add_argument("--text", help="encode this text, then decode it back from raw bytes and from codes")
    ap.add_argument("--roundtrip", type=int, metavar="P", help="encode every token id at this P and decode back")
    args = ap.parse_args()
    tok = _load_default_tokenizer()

    if args.ids:
        for i in args.ids:
            b = token_bytes(tok, i)
            print(f"{i:>7}  {b.hex(' '):<40}  {b.decode('utf-8', 'backslashreplace')!r}")
        print("joined:", repr(decode_ids(tok, args.ids, errors="replace")))

    if args.text is not None:
        ids = tok.encode(args.text, add_special_tokens=False)
        rows = [token_bytes(tok, i) for i in ids]
        buf, lens = pack_rows(rows, 32)
        E = torch.cat([e for _, e in encode_rows(buf, lens, 32)])
        index = CodeIndex(raw_byte_table(tok), 32)
        back = index.lookup(E).tolist()
        print("ids:          ", ids)
        print("ids from codes:", back, "(match)" if back == ids else "(MISMATCH)")
        print("text from raw bytes:", repr(decode_ids(tok, back)))

    if args.roundtrip:
        P = args.roundtrip
        raw = raw_byte_table(tok)
        if max(map(len, raw)) > P:
            raise SystemExit(f"some tokens exceed {P} bytes; truncate with utf8_safe_truncate first")
        buf, lens = pack_rows(raw, P)
        bad = 0
        for s, E in encode_rows(buf, lens, P):
            db, dl, ok = decode_codes(E, P)
            bad += int(((db.numpy() != buf[s:s + len(E)]).any(1) | (dl.numpy() != lens[s:s + len(E)]) | ~ok.numpy()).sum())
        print(f"P={P}: {len(raw)} token ids encoded and decoded back, {bad} mismatches")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
