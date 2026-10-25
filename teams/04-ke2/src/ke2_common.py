"""
Shared definitions for the KE2 week-1 work. Each is defined once and cited
from proof.md:

  tokenizer and its pin                    TOKENIZER_*, find_tokenizer_json(), load_tokenizer()
  decode table / raw byte table            decode_full_bytes(), raw_byte_table() (decode_ref)
  P-truncation into (V, P) buffers         make_table()  (repo utf8_safe_truncate)
  forward encode, CPU                      encode_chunks()  (repo kronecker_codec)
  inverse of the forward encode            decode_codes()  (decode_ref, Lemma 2)
  script class of a token                  script_class()
  hashing and provenance                   sha256_*(), provenance()

The forward encode is never re-implemented: it is the repository's
``kronecker_codec`` (P = pos_dim; 32 is the repo default, 64 uses pos_dim=64).

Tokenizer: BrahmicTokenizer-131K. Set KE2_TOKENIZER to a tokenizer.json (or
its folder) to use another one, for example G3's tokenizer v0.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # week 1 is CPU only
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch

import kronecker_embeddings
from kronecker_embeddings import token_id_to_bytes, utf8_safe_truncate

from decode_ref import CHAR_DIM, decode_codes, encode_rows, raw_byte_table, special_ids  # noqa: F401

# ---------------------------------------------------------------- config

TOKENIZER_NAME = "BrahmicTokenizer-131K"
TOKENIZER_REPO = "theschoolofai/BrahmicTokenizer-131K"
TOKENIZER_COMMIT = "7dfcf398172c825c6689ad578e8b484ca5f57bc9"
TOKENIZER_SHA256 = "01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff"
TOKENIZER_URL = f"https://raw.githubusercontent.com/{TOKENIZER_REPO}/{TOKENIZER_COMMIT}/tokenizer.json"

P_VALUES = (32, 64)
KINDS = ("decode", "raw")
DEVICE = torch.device("cpu")

TEAM = Path(__file__).resolve().parents[1]          # teams/04-ke2
RESULTS = TEAM / "results"
BUILD = TEAM / "build"                               # generated, not committed
TABLES = BUILD / "tables"

U_FFFD = "�".encode("utf-8")


# ---------------------------------------------------------------- tokenizer

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_tokenizer_json() -> Tuple[Path, str]:
    """
    Locate tokenizer.json. Returns (path, source label). Order:
      1. $KE2_TOKENIZER (file or folder): used as given, not pinned.
      2. a sibling clone: <lab>/../BrahmicTokenizer-131K/tokenizer.json
      3. the cache: ~/.cache/ke2/BrahmicTokenizer-131K-<commit>/tokenizer.json
      4. download from GitHub at the pinned commit into the cache.
    Sources 2-4 must match the pinned SHA-256.
    """
    env = os.environ.get("KE2_TOKENIZER")
    if env:
        p = Path(env)
        p = p / "tokenizer.json" if p.is_dir() else p
        if not p.is_file():
            raise FileNotFoundError(f"KE2_TOKENIZER={env} has no tokenizer.json")
        return p, "KE2_TOKENIZER"

    sibling = TEAM.parents[2] / "BrahmicTokenizer-131K" / "tokenizer.json"
    cache = Path.home() / ".cache" / "ke2" / f"BrahmicTokenizer-131K-{TOKENIZER_COMMIT[:12]}" / "tokenizer.json"
    for p, label in ((sibling, "sibling clone"), (cache, "cache")):
        if p.is_file():
            if _sha256_file(p) != TOKENIZER_SHA256:
                raise ValueError(f"{label} tokenizer.json does not match the pinned SHA-256")
            return p, label

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".part")
    urllib.request.urlretrieve(TOKENIZER_URL, tmp)
    if _sha256_file(tmp) != TOKENIZER_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError("downloaded tokenizer.json does not match the pinned SHA-256")
    tmp.replace(cache)
    return cache, "download"


def load_tokenizer():
    from transformers import PreTrainedTokenizerFast
    path, _ = find_tokenizer_json()
    return PreTrainedTokenizerFast(tokenizer_file=str(path))


def tokenizer_info() -> dict:
    path, source = find_tokenizer_json()
    sha = _sha256_file(path)
    return {
        "name": TOKENIZER_NAME if sha == TOKENIZER_SHA256 else "custom (KE2_TOKENIZER)",
        "repo": TOKENIZER_REPO,
        "commit": TOKENIZER_COMMIT if sha == TOKENIZER_SHA256 else None,
        "tokenizer_json_sha256": sha,
        "pinned": sha == TOKENIZER_SHA256,
        "source": source,
    }


def vocab_size(tok) -> int:
    return max(tok.get_vocab().values()) + 1   # same rule as the repo's build_byte_buffer


# ---------------------------------------------------------------- byte tables

def decode_full_bytes(tok) -> List[bytes]:
    """DECODE table before truncation: the repo's token_id_to_bytes, i.e. tokenizer.decode([id])."""
    specials = set(getattr(tok, "all_special_ids", []) or [])
    return [token_id_to_bytes(tok, i, specials) for i in range(vocab_size(tok))]


def full_bytes(tok, kind: str) -> List[bytes]:
    return decode_full_bytes(tok) if kind == "decode" else raw_byte_table(tok)


def make_table(full: Sequence[bytes], P: int) -> Tuple[np.ndarray, np.ndarray]:
    """T_P: repo utf8_safe_truncate to at most P bytes, packed as (V, P) uint8 + (V,) int16."""
    V = len(full)
    buf = np.zeros((V, P), dtype=np.uint8)
    lens = np.zeros((V,), dtype=np.int16)
    for i, b in enumerate(full):
        t = utf8_safe_truncate(b, P) if len(b) > P else b
        if t:
            buf[i, : len(t)] = np.frombuffer(t, dtype=np.uint8)
        lens[i] = len(t)
    return buf, lens


def row_bytes(buf: np.ndarray, lens: np.ndarray, i: int) -> bytes:
    return buf[i, : int(lens[i])].tobytes()


def table_rows(buf: np.ndarray, lens: np.ndarray) -> List[bytes]:
    return [row_bytes(buf, lens, i) for i in range(len(lens))]


def table_path(kind: str, P: int) -> Path:
    return TABLES / f"{kind}_P{P}.npz"


def load_table(kind: str, P: int) -> Tuple[np.ndarray, np.ndarray]:
    path = table_path(kind, P)
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not built yet; run src/build_tables.py")
    z = np.load(path)
    return z["byte_buffer"], z["length_buffer"]


def encode_chunks(buf: np.ndarray, lens: np.ndarray, P: int, chunk: int = 2048,
                  z_normalize: bool = True) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield (start, E[start:start+chunk]) from the repo's kronecker_codec on CPU."""
    if z_normalize:
        yield from encode_rows(buf, lens, P, chunk)
        return
    from kronecker_embeddings import kronecker_codec
    bb, ll = torch.from_numpy(buf), torch.from_numpy(lens)
    for s in range(0, bb.shape[0], chunk):
        yield s, kronecker_codec(bb[s:s + chunk], ll[s:s + chunk], char_dim=CHAR_DIM, pos_dim=P, z_normalize=False)


# ---------------------------------------------------------------- script class

# Unicode blocks of the Brahmic scripts and neighbours. A letter or mark in one
# of these blocks gets the block's script.
_BLOCKS = [
    (0x0900, 0x097F, "Devanagari"), (0xA8E0, 0xA8FF, "Devanagari"), (0x1CD0, 0x1CFF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"), (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"), (0x0B80, 0x0BFF, "Tamil"), (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"), (0x0D00, 0x0D7F, "Malayalam"), (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"), (0x0F00, 0x0FFF, "Tibetan"), (0x1C50, 0x1C7F, "Ol Chiki"),
    (0xAAE0, 0xAAFF, "Meetei Mayek"), (0xABC0, 0xABFF, "Meetei Mayek"),
]
_NAME_WORDS = {"LATIN": "Latin", "GREEK": "Greek", "CYRILLIC": "Cyrillic", "ARMENIAN": "Armenian",
               "HEBREW": "Hebrew", "ARABIC": "Arabic", "CJK": "CJK-Han", "IDEOGRAPHIC": "CJK-Han",
               "HIRAGANA": "Kana", "KATAKANA": "Kana", "KATAKANA-HIRAGANA": "Kana", "HANGUL": "Hangul"}

BRAHMIC = ["Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Odia", "Tamil", "Telugu", "Kannada",
           "Malayalam", "Sinhala", "Ol Chiki", "Meetei Mayek", "Tibetan", "Thai"]
_LETTER_SCRIPTS = (["Latin (ASCII)", "Latin (extended)"] + BRAHMIC
                   + ["Greek", "Cyrillic", "Armenian", "Hebrew", "Arabic", "CJK-Han", "Kana", "Hangul",
                      "Other-script", "Mixed-script"])
SCRIPT_ORDER = (_LETTER_SCRIPTS + ["Digit", "Whitespace", "Punct/Symbol", "Control"]
                + [f"Fragment: {s}" for s in _LETTER_SCRIPTS + ["Punct/Symbol"]]
                + ["Fragment: undetermined", "Special", "Empty"])


def _letter_script(cp: int) -> Optional[str]:
    for lo, hi, name in _BLOCKS:
        if lo <= cp <= hi:
            return name
    name = unicodedata.name(chr(cp), "")
    for word in name.split()[:2]:
        if word in _NAME_WORDS:
            s = _NAME_WORDS[word]
            if s == "Latin":
                return "Latin (ASCII)" if cp < 128 else "Latin (extended)"
            return s
    if name.startswith("COMBINING") or unicodedata.category(chr(cp)) == "Mn" and cp < 0x0370:
        return None   # inherited: takes the script of its base letter
    return "Other-script"


def _merge_scripts(scripts: set) -> Optional[str]:
    if not scripts:
        return None
    if scripts == {"Latin (ASCII)", "Latin (extended)"}:
        return "Latin (extended)"
    return next(iter(scripts)) if len(scripts) == 1 else "Mixed-script"


def _text_scripts(s: str) -> Tuple[set, set]:
    scripts, kinds = set(), set()
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in "LM":
            sc = _letter_script(ord(ch))
            if sc:
                scripts.add(sc)
        elif cat == "Cf":
            continue   # ZWJ / ZWNJ and other format characters carry no script
        elif cat[0] == "N":
            kinds.add("Digit")
        elif ch.isspace():
            kinds.add("Whitespace")
        elif cat == "Cc":
            kinds.add("Control")
        else:
            kinds.add("Punct/Symbol")
    return scripts, kinds


def _window_script(lo: int, size: int) -> Optional[str]:
    """Script of the code points [lo, lo+size) that a partial UTF-8 sequence can still become."""
    scripts, symbols = set(), False
    for cp in range(lo, min(lo + size, 0x110000)):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        cat = unicodedata.category(chr(cp))
        if cat[0] in "LM":
            sc = _letter_script(cp)
            if sc:
                scripts.add(sc)
        elif cat[0] in "PS":
            symbols = True
    merged = _merge_scripts(scripts)
    return merged or ("Punct/Symbol" if symbols else None)


def script_class(b: bytes, special: bool = False) -> str:
    """
    Script class of one token's RAW bytes (its true identity).

    Valid UTF-8: the script of its letters (Brahmic scripts by Unicode block).
    Invalid UTF-8 is a fragment of a multi-byte character. Its script is
    taken from its complete characters plus the code points the first
    partial sequence can still become: a lead byte with one continuation
    byte fixes a 64-code-point window (e.g. e0 a4 -> U+0900..093F,
    Devanagari). A lone lead byte e0 or a run of continuation bytes does not
    fix a script, so those are "Fragment: undetermined".
    """
    if special:
        return "Special"
    if not b:
        return "Empty"
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError as e:
        scripts, _ = _text_scripts(b.decode("utf-8", errors="ignore"))
        x = b[e.start]
        nxt = b[e.start + 1] if e.start + 1 < len(b) else None
        cont = nxt is not None and 0x80 <= nxt <= 0xBF
        win = None
        if 0xC2 <= x <= 0xDF:
            win = _window_script((x & 0x1F) << 6, 64)
        elif 0xE0 <= x <= 0xEF and cont:
            win = _window_script(((x & 0x0F) << 12) | ((nxt & 0x3F) << 6), 64)
        elif 0xF0 <= x <= 0xF4 and cont:
            win = _window_script(((x & 0x07) << 18) | ((nxt & 0x3F) << 12), 4096)
        if win and win != "Punct/Symbol":
            scripts.add(win)
        merged = _merge_scripts(scripts) or win
        return f"Fragment: {merged}" if merged else "Fragment: undetermined"
    scripts, kinds = _text_scripts(s)
    merged = _merge_scripts(scripts)
    if merged:
        return merged
    if "Control" in kinds:
        return "Control"
    if kinds == {"Whitespace"}:
        return "Whitespace"
    if kinds <= {"Digit", "Whitespace"} and kinds:
        return "Digit"
    return "Punct/Symbol"


def ordered_classes(classes: Sequence[str]) -> List[str]:
    present = set(classes)
    return [c for c in SCRIPT_ORDER if c in present] + sorted(present - set(SCRIPT_ORDER))


# ---------------------------------------------------------------- hashing / provenance

def sha256_bytes_list(items: Sequence[bytes]) -> str:
    """Canonical hash of a per-token byte list: for each id in order, 4-byte LE length + bytes."""
    h = hashlib.sha256()
    for b in items:
        h.update(len(b).to_bytes(4, "little"))
        h.update(b)
    return h.hexdigest()


def sha256_table(buf: np.ndarray, lens: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(buf, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(lens, dtype="<i2").tobytes())
    return h.hexdigest()


def show(b: bytes) -> str:
    """One-line readable form of a byte string: valid UTF-8 as text, other bytes as \\xNN."""
    s = b.decode("utf-8", errors="backslashreplace")
    out = []
    for ch in s:
        if ch == "\\" or (ch.isprintable() and ch not in "\t\r\n"):
            out.append(ch)
        else:
            out.append(ch.encode("unicode_escape").decode("ascii"))
    return "".join(out)


def provenance() -> dict:
    """What produced an output. No local paths or user names."""
    import tokenizers
    import transformers
    pkg = Path(kronecker_embeddings.__file__).parent
    return {
        "tokenizer": tokenizer_info(),
        "kronecker_embeddings": {
            "version": getattr(kronecker_embeddings, "__version__", "unknown"),
            "codec_py_sha256": _sha256_file(pkg / "codec.py"),
            "tokenizer_utils_py_sha256": _sha256_file(pkg / "tokenizer_utils.py"),
        },
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "unicode_version": unicodedata.unidata_version,
        "platform": platform.system() + " " + platform.release(),
        "device": str(DEVICE),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
