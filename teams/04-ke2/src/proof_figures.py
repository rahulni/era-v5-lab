"""
Draw the figures of docs/proof.md as SVG files in docs/figures/.

Every number on a figure is computed here, from the repository's kronecker_codec
and the closed forms of Lemma 2, and the script stops if the two disagree. The
example tokens (A = 2420, B = 39329, C = 74642) are read from the pinned
tokenizer, so a figure can only show bytes the tokenizer really has.

  fig1_pipeline.svg     token -> bytes -> row -> marks -> code -> decoder, and which result covers which step
  fig2_proof_map.svg    which result uses which, with the check ids
  fig3_cut_rule.svg     the cut rule on a valid and on a broken long token (C1-C6)
  fig4_marks_grid.svg   Lemma 1: the marks of A, and of B and C, in the full 256 x 32 grid
  fig5_two_values.svg   Lemma 2 (2)-(5): h_L and l_L for every length, P = 32 and 64
  fig6_margins.svg      Lemma 2 (6): room for error of threshold 8 and of the sign rule
  fig7_cause_tree.svg   Theorem 1 (3): the five causes as a decision tree
  fig8_theorem_BC.svg   Theorem 1 on B and C, raw table against decode table

The output is deterministic: the same inputs give byte-identical files.

Run from the lab repo root:  python teams/04-ke2/src/proof_figures.py
"""

from __future__ import annotations

import math
from html import escape
from typing import List, Sequence

import torch
from kronecker_embeddings.codec import kronecker_codec

import ke2_common as C

OUT = C.TEAM / "docs" / "figures"
EPS = 1e-6
A_ID, B_ID, C_ID = 2420, 39329, 74642

INK, INK2, MUTED = "#0f2230", "#3f5260", "#667884"
RULE, RULE2, PAPER = "#d8e0e4", "#b7c4cb", "#f3f6f7"
RAW, RAW_W, DEC, DEC_W = "#2a78d6", "#e2edfb", "#eb6834", "#fde8dd"
MARK, MARK_E = "#ffd675", "#c98d0c"
GOOD, GOOD_W, BAD = "#0a8a0a", "#e1f3e1", "#b42318"
SANS = "'Segoe UI', 'Helvetica Neue', Arial, 'Nirmala UI', 'Noto Sans', sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"


# ------------------------------------------------------------------ numbers
def closed_form(L: int, P: int):
    """(h_L, l_L) of Lemma 2 (2)."""
    D = 256 * P
    sigma = math.sqrt((1 - L / D) / (D - 1))
    mu = math.sqrt(L) / D
    return (1 / math.sqrt(L) - mu) / (sigma + EPS), -mu / (sigma + EPS)


def codes(rows: Sequence[bytes], P: int) -> torch.Tensor:
    buf = torch.zeros((len(rows), P), dtype=torch.long)
    for k, r in enumerate(rows):
        buf[k, :len(r)] = torch.tensor(list(r), dtype=torch.long)
    return kronecker_codec(buf, torch.tensor([len(r) for r in rows]), pos_dim=P)


def cut(r: bytes, P: int) -> int:
    """Length the cut rule keeps (the repository's utf8_safe_truncate, written out to report the tries)."""
    if len(r) <= P:
        return len(r)
    for c in range(P, P - 4, -1):
        try:
            r[:c].decode("utf-8")
            return c
        except UnicodeDecodeError:
            pass
    return 0


def facts() -> dict:
    tok = C.load_tokenizer()
    raw, dec = C.full_bytes(tok, "raw"), C.full_bytes(tok, "decode")
    f = {"A": raw[A_ID], "B": raw[B_ID], "C": raw[C_ID], "Bd": dec[B_ID], "Cd": dec[C_ID]}
    assert f["A"] == bytes.fromhex("20e0a48fe0a495") and dec[A_ID] == f["A"]
    assert f["B"] == bytes.fromhex("b8e0af8d") and f["C"] == bytes.fromhex("aee0af8d")
    assert f["Bd"] == f["Cd"] == bytes.fromhex("efbfbde0af8d")
    E = codes([f["A"], f["B"], f["C"], f["Bd"]], 32)
    for k, row in enumerate((f["A"], f["B"], f["C"], f["Bd"])):
        h, l = closed_form(len(row), 32)
        assert set((E[k] > 8).nonzero().flatten().tolist()) == {v * 32 + p for p, v in enumerate(row)}, "Lemma 1"
        assert abs(E[k].max().item() - h) < 1e-4 and abs(E[k].min().item() - l) < 1e-6, "Lemma 2 (2)"
    f["hA"], f["lA"] = E[0].max().item(), E[0].min().item()
    f["diffBC"] = (E[1] != E[2]).nonzero().flatten().tolist()
    assert f["diffBC"] == [0xAE * 32, 0xB8 * 32]
    assert torch.equal(codes([f["Bd"]], 32)[0], codes([f["Cd"]], 32)[0])
    f["curves"] = {P: [closed_form(L, P) for L in range(1, P + 1)] for P in (32, 64)}
    for P in (32, 64):                                   # the codec agrees with the closed forms at every length
        E = codes([b"a" * L for L in range(1, P + 1)], P)
        for L in range(1, P + 1):
            h, l = f["curves"][P][L - 1]
            assert abs(E[L - 1].max().item() - h) / h < 1e-5 and abs(E[L - 1].min().item() - l) / -l < 1e-5
    return f


# ------------------------------------------------------------------ drawing helpers
def text(x, y, s, size=13, anchor="start", weight=400, fill=INK, mono=False) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def lines(x, y, rows: List[str], size=13, lh=None, **kw) -> str:
    lh = lh or size * 1.35
    return "".join(text(x, y + i * lh, r, size, **kw) for i, r in enumerate(rows))


def rect(x, y, w, h, fill="#ffffff", stroke=RULE2, rx=8, sw=1.2, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=INK2, sw=1.5, arrow=False, dash=None, both=False) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += ' marker-end="url(#ah)"' if arrow else ""
    extra += ' marker-start="url(#ah)"' if both else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'


def path(d, stroke=INK2, sw=1.5) -> str:
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#ah)"/>'


def circle(x, y, r, fill, stroke=INK, sw=1.2) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def tag(x, y, s, fill=PAPER) -> str:
    w = 10 + 7 * len(s)
    return rect(x, y, w, 20, fill, RULE2, 10, 1) + text(x + w / 2, y + 14, s, 11.5, "middle", 600, INK2, mono=True)


def byte_box(x, y, s: str, fill="#ffffff", stroke=RULE2, w=33, h=26, ink=INK, sw=1.2) -> str:
    return rect(x, y, w, h, fill, stroke, 4, sw) + text(x + w / 2, y + h / 2 + 4.5, s, 12, "middle", 600, ink, mono=True)


def svg(w, h, title, body: List[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{escape(title)}">\n<title>{escape(title)}</title>\n'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12" fill="#ffffff" stroke="{RULE}"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def num(v: float, d: int) -> str:
    return f"{v:.{d}f}".replace("-", "−")


# ------------------------------------------------------------------ figure 1: the pipeline
def fig1(f) -> str:
    b = [text(24, 34, "From a token to its code, and back", 18, weight=700),
         text(24, 56, "Token A (id 2420) at P = 32. Each box is real code; the brackets show which result covers which step.", 13, fill=MUTED)]
    xs = [24 + i * 164 for i in range(6)]
    boxes = [("TOKEN ID", [("2420", 20, True), ("\" एक\"", 15, False)], "#ffffff", RULE2),
             ("FULL BYTES", [("20 e0 a4 8f", 14, True), ("e0 a4 95", 14, True), ("7 bytes", 12, False)], RAW_W, RAW),
             ("ROW (CUT RULE)", [("7 of 32 slots", 14, False), ("kept whole,", 12, False), ("since 7 ≤ 32", 12, False)], RAW_W, RAW),
             ("MARKS", [("7 entries of 0.378", 13, False), ("8,185 zeros", 13, False)], "#ffffff", RULE2),
             ("CODE (NORMALIZED)", [(f"7 × {f['hA']:.2f}", 13, True), (f"8,185 × {num(f['lA'], 4)}", 13, True)], MARK, MARK_E),
             ("DECODER", [("entries above 8", 13, False), ("→ the 7 bytes", 12, False), ("→ token 2420", 12, False)], GOOD_W, GOOD)]
    for i, (lab, rows, fill, stroke) in enumerate(boxes):
        x = xs[i]
        b.append(rect(x, 76, 140, 96, fill, stroke))
        b.append(text(x + 70, 96, lab, 10.5, "middle", 700, MUTED))
        for k, (r, s, m) in enumerate(rows):
            b.append(text(x + 70, 121 + k * 19, r, s, "middle", 700 if k == 0 else 400, INK, mono=m))
        if i < 5:
            b.append(line(x + 142, 124, x + 162, 124, arrow=True))
    b.append(path(f"M{xs[5] + 70},172 V192 H{xs[2] + 70} V175", stroke=GOOD, sw=1.8))
    b.append(text((xs[2] + xs[5]) / 2 + 70, 210, "the decoder gives back the row exactly", 12.5, "middle", 600, GOOD))
    for i, lab in ((1, "Task 1: raw or decode table"), (2, "Setup: cut rule C1–C6"), (3, "Lemma 1"),
                   (4, "Lemma 2 (1)–(5)"), (5, "Lemma 2 (6)–(7)")):
        x = xs[i]
        b.append(line(x + 6, 228, x + 134, 228, stroke=RULE2) + line(x + 6, 223, x + 6, 233, stroke=RULE2)
                 + line(x + 134, 223, x + 134, 233, stroke=RULE2))
        b.append(text(x + 70, 248, lab, 12.5, "middle", 600, INK2))
    b.append(line(xs[2] + 6, 268, xs[5] + 134, 268, stroke=INK, sw=1.8) + line(xs[2] + 6, 262, xs[2] + 6, 274, stroke=INK, sw=1.8)
             + line(xs[5] + 134, 262, xs[5] + 134, 274, stroke=INK, sw=1.8))
    b.append(text((xs[2] + xs[5] + 140) / 2, 292, "Theorem 1: two tokens get the same code exactly when they have the same row",
                  13.5, "middle", 700))
    return svg(1000, 312, "The forward encode of token A and the result that covers each step", b)


# ------------------------------------------------------------------ figure 2: proof map
def fig2(f) -> str:
    b = [text(24, 32, "How the proof is built", 18, weight=700)]
    top = [(24, "SETUP 0.2 · THE CUT RULE", ["C1–C6: keeps a start of the token,", "never splits a letter, and makes", "every row fit in P slots"], ["C0"]),
           (350, "LEMMA 1 · THE MARKS", ["each byte puts one mark in its", "own column, and the marks spell", "out the row"], ["L1-a", "L1-b"]),
           (676, "LEMMA 2 · NORMALIZING", ["two values: marks above 15.93,", "the rest between −0.063 and 0;", "threshold 8 reads the row back"], ["L2-a … L2-e"])]
    for x, head, rows, tags in top:
        b.append(rect(x, 84, 300, 116))
        b.append(text(x + 14, 106, head, 11, weight=700, fill=MARK_E))
        b.append(lines(x + 14, 128, rows, 13))
        tx = x + 14
        for t in tags:
            b.append(tag(tx, 170, t))
            tx += 18 + 7 * len(t)
    b.append(path("M174,84 C174,62 440,62 440,82"))
    b.append(text(307, 58, "C3: every row fits in P", 12, "middle", 600, INK2))
    b.append(path("M560,84 C560,62 826,62 826,82"))
    b.append(text(693, 58, "(4): the marks have Σ x² = 1", 12, "middle", 600, INK2))
    b.append(rect(190, 252, 610, 104, MARK, MARK_E, sw=1.6))
    b.append(text(206, 274, "THEOREM 1 · THE FORWARD ENCODE IS EXACT", 11, weight=700))
    b.append(lines(206, 296, ["(1) same code exactly when same row    (2) exact decoding in float32, bf16, fp16",
                               "(3) every shared row has one of five causes    (4) a uniqueness criterion",
                               "(5) Gate 1 for BrahmicTokenizer-131K"], 13))
    b.append(tag(690, 326, "T1-a … T1-f", fill="#ffffff"))
    b.append(line(174, 200, 300, 250, arrow=True))
    b.append(text(40, 232, "C1, C2, C6: the causes", 12, "start", 600, INK2))
    b.append(line(500, 200, 500, 250, arrow=True))
    b.append(text(508, 230, "(3): read the row back", 12, "start", 600, INK2))
    b.append(line(826, 200, 700, 250, arrow=True))
    b.append(text(772, 232, "(6), (7): the decoder", 12, "start", 600, INK2))
    b.append(rect(826, 262, 150, 84, "#ffffff", RULE2, dash="5 4"))
    b.append(lines(838, 282, ["CHECKED FACT", "printout = λ(raw)", "for all 131,072 ids"], 12))
    b.append(tag(838, 318, "T1-c"))
    b.append(line(826, 304, 802, 304, arrow=True))
    b.append(rect(330, 392, 340, 40, GOOD_W, GOOD, sw=1.6))
    b.append(text(500, 417, "Gate 1 holds", 15, "middle", 700, GOOD))
    b.append(line(500, 356, 500, 390, arrow=True))
    return svg(1000, 452, "Dependency map of the cut rule, Lemma 1, Lemma 2, Theorem 1 and Gate 1", b)


# ------------------------------------------------------------------ figure 3: the cut rule
def cut_row(b: List[str], y: int, data: bytes, P: int, title: str, letters, note: str, stray0=False) -> int:
    end = cut(data, P)
    b.append(text(24, y, title, 14, weight=700))
    show = [0, 1, 2] + list(range(24, len(data)))
    pos_x, x = {}, 40
    for i in show:
        if i == 24:
            b.append(text(x + 18, y + 59, "…", 18, "middle", 700, MUTED))
            x += 40
        pos_x[i] = x
        x += 37
    for i in show:
        kept, stray = i < end, stray0 and i == 0
        b.append(byte_box(pos_x[i], y + 40, f"{data[i]:02x}", RAW_W if kept else PAPER, BAD if stray else (RAW if kept else RULE2),
                          ink=INK if kept else MUTED, sw=2.2 if stray else 1.2))
        if i in (0, 24, 29, 31, 35, 41) and i < len(data):
            b.append(text(pos_x[i] + 16.5, y + 82, str(i), 10.5, "middle", fill=MUTED))
    for lo, hi, lab in letters:
        xa, xb = pos_x[lo] + 2, pos_x[hi] + 31
        b.append(line(xa, y + 32, xb, y + 32, stroke=MARK_E, sw=1.5))
        b.append(text((xa + xb) / 2, y + 26, lab, 13, "middle", 600, MARK_E))
    xP = pos_x[P - 1] + 35
    b.append(line(xP, y + 34, xP, y + 72, stroke=BAD, sw=2, dash="4 3"))
    for c in range(P, P - 4, -1):
        if end and c < end:
            break
        xc, ok = pos_x[c - 1] + 35, c == end
        b.append(line(xc, y + 88, xc, y + 96, stroke=GOOD if ok else BAD, sw=2))
        b.append(text(xc, y + 112, f"{c} {'✓' if ok else '✗'}", 12, "middle", 700, GOOD if ok else BAD))
    b.append(text(24, y + 138, note, 13, fill=INK2))
    return end


def fig3(f) -> str:
    b = [text(24, 32, "The cut rule at P = 32: try 32, 31, 30 and 29 bytes, keep the first that is valid text", 17, weight=700),
         text(24, 54, "Blue bytes are kept. The red dashed line is P = 32. Each tick below a row is a place the rule tries to cut.", 13, fill=MUTED)]
    e1 = cut_row(b, 96, b"a" * 30 + "भारत".encode(), 32, "A valid token of 42 bytes: 30 × a, then भारत",
                 [(30, 32, "भ"), (33, 35, "ा"), (36, 38, "र"), (39, 41, "त")],
                 "Cutting at 32 or 31 would split भ, so those starts are not valid text. 30 is valid: the row keeps 30 bytes (C3, C4).")
    e2 = cut_row(b, 266, b"\x95" + b"a" * 20 + "ಕನ್ನಡ".encode(), 32, "A broken token of 36 bytes: 95, then 20 × a, then ಕನ್ನಡ",
                 [(24, 26, "ನ"), (27, 29, "್"), (30, 32, "ನ"), (33, 35, "ಡ")],
                 "Every start begins with 95, a continuation byte with nothing before it. No cut is valid, so the row is empty (C6).",
                 stray0=True)
    assert (e1, e2) == (30, 0)
    b.append(text(24, 440, "No BrahmicTokenizer token is longer than 32 bytes, so at P = 32 and P = 64 the rule never cuts anything (C2). "
                  "Both cases matter only for another tokenizer or a smaller P.", 12, fill=MUTED))
    return svg(1000, 460, "The cut rule on a valid 42-byte token and on a broken 36-byte token", b)


# ------------------------------------------------------------------ figure 4: the marks in the grid
def grid_panel(b: List[str], ox: float, title: List[str], marks, labels):
    gx, gy, cw, rh, P = ox + 46, 104, 9, 2.4, 32
    b.append(text(ox, 34, title[0], 15, weight=700))
    b.append(text(ox, 54, title[1], 12.5, fill=MUTED))
    b.append(rect(gx, gy, cw * P, rh * 256, PAPER, RULE2, rx=2, sw=1))
    for p in range(1, P):
        b.append(line(gx + p * cw, gy, gx + p * cw, gy + rh * 256, stroke="#e2e8eb", sw=0.6))
    for v in range(16, 256, 16):
        b.append(line(gx, gy + v * rh, gx + cw * P, gy + v * rh, stroke="#d5dde1", sw=0.6))
    for v in range(0, 256, 32):
        b.append(text(gx - 6, gy + v * rh + 4, f"{v:02x}", 11, "end", fill=MUTED, mono=True))
    for p in (0, 8, 16, 24, 31):
        b.append(text(gx + p * cw + cw / 2, gy - 8, str(p), 11, "middle", fill=MUTED))
    b.append(text(gx, gy - 26, "position (column) →", 11.5, fill=INK2))
    b.append(text(ox, gy + rh * 256 + 24, "byte value (row) ↓, 256 rows", 11.5, fill=INK2))
    for v, p, _ in marks:
        y = gy + v * rh + rh / 2
        b.append(line(gx + p * cw + cw / 2 + 6, y, gx + cw * P + 6, y, stroke=MARK_E, sw=0.8, dash="2 2"))
    for v, p, kind in marks:
        b.append(circle(gx + p * cw + cw / 2, gy + v * rh + rh / 2, 5.5, {"both": MARK, "B": INK, "C": "#ffffff"}[kind],
                        MARK_E if kind == "both" else INK, 1.6))
    for v, dy, s in labels:
        b.append(text(gx + cw * P + 10, gy + v * rh + rh / 2 + 4 + dy, s, 12, mono=True))


def fig4(f) -> str:
    b: List[str] = []
    grid_panel(b, 24, ["Token A: 7 bytes, 7 marks", "one mark per column; e0 and a4 appear twice, in different columns"],
               [(v, p, "both") for p, v in enumerate(f["A"])],
               [(0x20, 0, "20 · col 0"), (0x8f, -3, "8f · col 3"), (0x95, 4, "95 · col 6"), (0xa4, 0, "a4 · col 2, 5"),
                (0xe0, 0, "e0 · col 1, 4")])
    grid_panel(b, 520, ["Tokens B and C: 4 marks each, 3 shared", "they differ only in column 0: b8 (B) against ae (C)"],
               [(0xB8, 0, "B"), (0xAE, 0, "C")] + [(v, p, "both") for p, v in enumerate(f["B"]) if p > 0],
               [(0x8d, 0, "8d · col 3"), (0xae, -6, "ae · col 0 (C)"), (0xaf, 9, "af · col 2"), (0xb8, 5, "b8 · col 0 (B)"),
                (0xe0, 0, "e0 · col 1")])
    ly = 104 + 2.4 * 256 + 52
    b.append(text(24, ly, "Entry number = byte × 32 + column, so each mark gives back its byte and its position.", 12, fill=MUTED))
    b.append(circle(566, ly - 4, 5.5, MARK, MARK_E, 1.6) + text(578, ly, "mark (B and C: shared)", 12))
    b.append(circle(762, ly - 4, 5.5, INK, INK, 1.6) + text(774, ly, "only B", 12))
    b.append(circle(852, ly - 4, 5.5, "#ffffff", INK, 1.6) + text(864, ly, "only C", 12))
    return svg(1000, int(ly + 22), "The marks of tokens A, B and C in the 256 by 32 grid", b)


# ------------------------------------------------------------------ figure 5: the two values
def chart(b: List[str], x0, x1, y0, y1, ymin, ymax, yticks, fy, title, series, refs, rings, caption):
    X = lambda L: x0 + (L - 1) / 63 * (x1 - x0)
    Y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
    b.append(text(x0 - 50, y0 - 22, title, 14.5, weight=700))
    for t in yticks:
        b.append(line(x0, Y(t), x1, Y(t), stroke="#e6ecef", sw=1))
        b.append(text(x0 - 8, Y(t) + 4, fy(t), 11, "end", fill=MUTED))
    for L in (1, 8, 16, 24, 32, 40, 48, 56, 64):
        b.append(text(X(L), y1 + 18, str(L), 11, "middle", fill=MUTED))
    b.append(line(x0, y1, x1, y1, stroke=RULE2, sw=1.2))
    b.append(text((x0 + x1) / 2, y1 + 38, "token length L (bytes)", 12, "middle", fill=INK2))
    for v, lab, col, at in refs:
        b.append(line(x0, Y(v), x1, Y(v), stroke=col, sw=1.6, dash="6 4"))
        if isinstance(at, tuple):                        # a key in an empty corner, where no curve passes
            kx, ky = X(at[0]), Y(at[1])
            b.append(line(kx, ky - 4, kx + 22, ky - 4, stroke=col, sw=1.6, dash="6 4"))
            b.append(text(kx + 28, ky, lab, 11.5, "start", 700, col))
        else:
            b.append(text(x0 + 6, Y(v) + at, lab, 11.5, "start", 700, col))
    for vals, col in series:
        pts = " ".join(f"{X(L):.1f},{Y(v):.2f}" for L, v in enumerate(vals, start=1))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        b.extend(circle(X(L), Y(v), 2.2, col, col, 0.5) for L, v in enumerate(vals, start=1))
    for L, v in rings:
        b.append(circle(X(L), Y(v), 6.5, "none", INK, 1.6))
    b.append(text(x0 - 50, y1 + 62, caption, 12, weight=600))


def fig5(f) -> str:
    b = [text(24, 30, "Lemma 2: every code has just two values, and they depend only on the token length L", 17, weight=700)]
    c32, c64 = f["curves"][32], f["curves"][64]
    chart(b, 84, 470, 92, 330, 0, 140, [0, 20, 40, 60, 80, 100, 120, 140], str, "On each of the L marks: h(L)",
          [([h for h, _ in c32], RAW), ([h for h, _ in c64], MARK_E)],
          [(15.93, "floor 15.93, for every L", BAD, (30, 124)), (8, "threshold 8", GOOD, (30, 110))],
          [(7, c32[6][0]), (4, c32[3][0])],
          f"Ringed at P = 32: A (L = 7) {c32[6][0]:.2f}, and B, C (L = 4) {c32[3][0]:.2f}")
    chart(b, 584, 970, 92, 330, -0.07, 0.005, [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0], lambda t: num(t, 2),
          "On every other entry: ℓ(L)",
          [([l for _, l in c32], RAW), ([l for _, l in c64], MARK_E)],
          [(0, "0, the sign rule's threshold", BAD, -6), (-0.06262, "floor −0.0627, for every L", BAD, 15)],
          [(7, c32[6][1]), (4, c32[3][1])],
          f"Ringed at P = 32: A {num(c32[6][1], 4)}, and B, C {num(c32[3][1], 4)}")
    b.append(line(24, 428, 44, 428, stroke=RAW, sw=2.4) + text(50, 432, "P = 32 (L up to 32, D = 8,192)", 12))
    b.append(line(290, 428, 310, 428, stroke=MARK_E, sw=2.4) + text(316, 432, "P = 64 (L up to 64, D = 16,384)", 12))
    b.append(text(560, 432, "The two values never meet: the gap is always above 15.99.", 12, fill=INK2))
    return svg(1000, 452, "The two values of a normalized code for every token length, at P = 32 and P = 64", b)


# ------------------------------------------------------------------ figure 6: room for error
def fig6(f) -> str:
    b = [text(24, 30, "Lemma 2 (6): how much error each decoder survives", 17, weight=700),
         text(24, 52, "Every value of a code lies in one of two zones. A decoder works as long as noise cannot push a value across its threshold.",
              13, fill=MUTED)]
    X = lambda v: 60 + (v + 2) * 36
    ya = 176
    b.append(line(X(-2), ya, X(22), ya, stroke=INK2, sw=1.2))
    for t in range(-2, 23, 2):
        b.append(line(X(t), ya, X(t), ya + 5, stroke=INK2, sw=1) + text(X(t), ya + 20, str(t).replace("-", "−"), 11, "middle", fill=MUTED))
    b.append(rect(X(15.93), 128, X(22) - X(15.93), 48, MARK, MARK_E, rx=3))
    b.append(text(X(15.93) + 10, 148, "every mark ≥ 15.93", 12.5, weight=700))
    b.append(text(X(15.93) + 10, 166, "(up to 128) →", 12, fill=INK2))
    b.append(rect(X(-0.0627), 128, X(0) - X(-0.0627), 48, INK, INK, rx=0, sw=0.5))
    b.append(line(X(-0.03), 182, X(-1.2), 214, stroke=INK2, sw=1))
    b.append(text(X(-2), 230, "every other entry: between −0.063 and 0", 12.5, weight=700))
    b.append(line(X(8), 84, X(8), 186, stroke=GOOD, sw=2.5))
    b.append(text(X(8), 78, "threshold 8 (the default)", 12.5, "middle", 700, GOOD))
    b.append(line(X(0) + 4, 112, X(8) - 3, 112, stroke=GOOD, sw=1.5, arrow=True, both=True))
    b.append(text((X(0) + X(8)) / 2, 106, "room > 8", 12.5, "middle", 700, GOOD))
    b.append(line(X(8) + 3, 112, X(15.93) - 3, 112, stroke=GOOD, sw=1.5, arrow=True, both=True))
    b.append(text((X(8) + X(15.93)) / 2, 106, "room ≥ 7.93", 12.5, "middle", 700, GOOD))
    b.append(line(X(0), 84, X(0), 186, stroke=BAD, sw=2))
    b.append(text(X(0) + 6, 78, "sign rule (threshold 0)", 12.5, "middle", 700, BAD))
    Z = lambda v: 60 + (v + 0.08) * 8800
    yz = 352
    b.append(text(24, 270, "Zoomed in 1,000 times around 0: the sign rule's room is tiny", 14, weight=700))
    b.append(line(Z(-0.08), yz, Z(0.02), yz, stroke=INK2, sw=1.2))
    for t in (-0.08, -0.06, -0.04, -0.02, 0, 0.02):
        b.append(line(Z(t), yz, Z(t), yz + 5, stroke=INK2, sw=1) + text(Z(t), yz + 20, num(t, 2), 11, "middle", fill=MUTED))
    b.append(rect(Z(-0.0627), 310, Z(-0.0078) - Z(-0.0627), 42, RULE2, INK2, rx=3, sw=1))
    b.append(text((Z(-0.0627) + Z(-0.0078)) / 2, 336, "ℓ of every non-empty row: −0.0627 to −0.0078", 12.5, "middle", 700))
    b.append(line(Z(0), 296, Z(0), yz + 4, stroke=BAD, sw=2))
    b.append(line(Z(-0.0078) + 2, 302, Z(0) - 2, 302, stroke=BAD, sw=1.5, arrow=True, both=True))
    b.append(text((Z(-0.0078) + Z(0)) / 2, 296, "0.0078", 12, "middle", 700, BAD))
    b.append(circle(Z(0), yz, 5, BAD, BAD))
    b.append(lines(Z(0) + 12, 318, ["empty row: every entry is", "exactly 0, so the sign rule", "has no room at all"], 12, fill=BAD))
    return svg(1000, 394, "Room for error of the threshold-8 decoder and of the sign rule", b)


# ------------------------------------------------------------------ figure 7: the cause tree
def qbox(b, cx, y, w, s):
    b.append(rect(cx - w / 2, y, w, 40, PAPER, INK2, rx=20, sw=1.4))
    b.append(text(cx, y + 24.5, s, 13, "middle", 600))


def leaf(b, cx, y, name, rows, real, hit=False):
    w, h = 180, 122
    b.append(rect(cx - w / 2, y, w, h, GOOD_W if hit else "#ffffff", GOOD if hit else RULE2, sw=1.8 if hit else 1.2))
    b.append(text(cx, y + 22, name, 12.5, "middle", 700, mono=True))
    b.append(lines(cx, y + 44, rows, 11.5, 15.5, anchor="middle", fill=INK2))
    b.append(text(cx, y + h - 11, real, 11.5, "middle", 700, GOOD if hit else MUTED))


def edge(b, x1, y1, x2, y2, lab, lx, ly, anchor="middle"):
    b.append(line(x1, y1, x2, y2, arrow=True))
    b.append(text(lx, ly, lab, 12, anchor, 700, INK2))


def fig7(f) -> str:
    b = [text(24, 30, "Theorem 1 (3): two tokens share a row for exactly one of five reasons", 17, weight=700),
         text(24, 52, "Follow the questions from the top. Green: the only case BrahmicTokenizer-131K produces (decode table).", 13, fill=MUTED)]
    b.append(rect(320, 70, 360, 38, INK, INK, rx=8))
    b.append(text(500, 94, "Tokens s ≠ t have the same row r", 14, "middle", 700, "#ffffff"))
    b.append(line(500, 108, 500, 126, arrow=True))
    qbox(b, 500, 128, 440, "Were their full byte strings f(s) and f(t) different?")
    qbox(b, 203, 222, 270, "Is the shared row r empty?")
    edge(b, 390, 168, 240, 220, "yes: the cut made them equal", 300, 190, "end")
    qbox(b, 625, 222, 280, "Are both full strings empty?")
    edge(b, 600, 168, 620, 220, "no: equal before any cut", 622, 190, "start")
    qbox(b, 785, 306, 250, "Are the raw bytes the same?")
    edge(b, 690, 262, 760, 304, "no", 740, 280)
    leaf(b, 106, 400, "CUT_TO_EMPTY", ["both longer than P and", "broken, so the cut kept", "nothing · proof: C2, C6"], "real: 0 groups")
    edge(b, 160, 262, 112, 398, "yes", 120, 330, "end")
    leaf(b, 300, 400, "TRUNCATION_P", ["one is longer than P, and", "r is the start they share,", "e.g. 40 and 48 dashes"], "real: 0 groups")
    edge(b, 246, 262, 296, 398, "no", 282, 330, "start")
    leaf(b, 494, 400, "EMPTY", ["two tokens with no bytes:", "both get the all-zero", "code"], "real: 0 groups")
    edge(b, 590, 262, 510, 398, "yes", 548, 330, "end")
    leaf(b, 688, 400, "DUPLICATE_BYTES", ["the same bytes twice, e.g.", "<s> as a special token", "and as plain text"], "real: 0 groups")
    edge(b, 745, 346, 700, 398, "yes", 712, 372, "end")
    leaf(b, 882, 400, "UTF8_REPLACEMENT", ["decode table only: broken", "bytes print as U+FFFD,", "e.g. B and C · uses T1-c"],
         "real: 10 groups", hit=True)
    edge(b, 825, 346, 868, 398, "no", 858, 372, "start")
    b.append(text(24, 552, "Each question splits the remaining cases in two, so exactly one leaf applies. "
                  "All five are planted in a synthetic vocabulary and found (check T1-f).", 12, fill=MUTED))
    return svg(1000, 572, "Decision tree of the five collision causes of Theorem 1", b)


# ------------------------------------------------------------------ figure 8: Theorem 1 on B and C
def fig8(f) -> str:
    b = [text(24, 30, "Theorem 1 on tokens B and C at P = 32", 17, weight=700),
         text(24, 52, "The same two tokens in two tables. The code follows the row: different rows give different codes, equal rows one code.",
              13, fill=MUTED)]
    d0, d1 = f["diffBC"]
    lanes = [(76, "RAW BYTE TABLE", RAW_W, RAW, f["B"], f["C"], True,
              ["the marks differ only in column 0:", f"B marks entry {d1:,} (b8 × 32),", f"C marks entry {d0:,} (ae × 32)"],
              ["the codes differ in 2 of 8,192", "numbers; decoding gives back two", "different rows: unique codes"]),
             (216, "DECODE TABLE", DEC_W, DEC, f["Bd"], f["Cd"], False,
              ["both rows are ef bf bd e0 af 8d,", "so the marks are identical"],
              ["one code (0 of 8,192 numbers", "differ), shared for a known cause:", "UTF8_REPLACEMENT, checked"])]
    for y, name, fill, stroke, rb, rc, raw, mid, verdict in lanes:
        b.append(rect(24, y, 952, 124, PAPER, RULE, rx=10, sw=1))
        b.append(text(40, y + 24, name, 11.5, weight=700, fill=stroke))
        for k, (lab, row) in enumerate((("B", rb), ("C", rc))):
            yy = y + 38 + k * 38
            b.append(text(40, yy + 18, lab, 14, weight=700))
            for p, v in enumerate(row):
                hi = raw and p == 0
                b.append(byte_box(60 + p * 34, yy, f"{v:02x}", MARK if hi else fill, MARK_E if hi else stroke, w=31, sw=1.8 if hi else 1.2))
        b.append(line(276, y + 64, 316, y + 64, arrow=True))
        b.append(lines(328, y + 46, mid, 13))
        b.append(line(566, y + 64, 604, y + 64, arrow=True))
        b.append(rect(614, y + 24, 346, 78, GOOD_W, GOOD, sw=1.4))
        b.append(lines(630, y + 47, verdict, 13, weight=600))
    return svg(1000, 360, "Theorem 1 applied to tokens B and C in the raw and in the decode table", b)


def main() -> int:
    f = facts()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"fig1_pipeline.svg": fig1, "fig2_proof_map.svg": fig2, "fig3_cut_rule.svg": fig3, "fig4_marks_grid.svg": fig4,
            "fig5_two_values.svg": fig5, "fig6_margins.svg": fig6, "fig7_cause_tree.svg": fig7, "fig8_theorem_BC.svg": fig8}
    for name, draw in figs.items():
        (OUT / name).write_text(draw(f), encoding="utf-8", newline="\n")
        print(f"wrote docs/figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
