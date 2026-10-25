"""
Count collisions per script class, with a cause for each group.

A collision group in table T_P is a set of >= 2 token ids with the same byte
row. By Theorem 1 (proof.md) these are exactly the ids that share a
forward-encode code, so this accounts for every non-unique code.

Each group gets one or more causes, and each cause is checked against its
mechanism. A cause that fails its check becomes UNEXPLAINED (a test fails):

  UTF8_REPLACEMENT  decode table only. Members have different raw bytes, but
                    tokenizer.decode() gives the same text. Check: for every
                    member raw.decode("utf-8", "replace") == decoded text, and
                    the text contains U+FFFD (ef bf bd).
  TRUNCATION_P      Members have different full byte strings that share the
                    same first <= P bytes (a non-empty row). Check: every full
                    string starts with the row and at least one is longer than P.
  CUT_TO_EMPTY      The shared row is empty although a member is not: the cut
                    rule found no valid UTF-8 prefix of a long token. Check:
                    every non-empty member is longer than P and not valid UTF-8
                    (property C6 in docs/proof.md).
  DUPLICATE_BYTES   Members have identical raw bytes in the vocab.
  EMPTY             Members all have zero bytes.

These five are all the ways two rows can be equal (Theorem 1 (3)).

Script class comes from each token's raw bytes (ke2_common.script_class).

Outputs (results/):
  collisions.csv            one row per group, all four tables, every member id
  collisions_by_script.csv  per table and script class: ids, colliding ids,
                            groups, colliding ids per cause
  collisions_summary.json   totals per table, and whether each P=64 group sits
                            inside one P=32 group

Run from the lab repo root:  python teams/04-ke2/src/collisions.py   (after build_tables.py)
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

import ke2_common as C

CAUSES = ["UTF8_REPLACEMENT", "TRUNCATION_P", "CUT_TO_EMPTY", "DUPLICATE_BYTES", "EMPTY", "UNEXPLAINED"]
KNOWN = set(CAUSES) - {"UNEXPLAINED"}


def _valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def partition(ids, key):
    parts = defaultdict(list)
    for t in ids:
        parts[key(t)].append(t)
    return parts


def explain(ids, row, kind, P, raw, dec):
    """Return (causes, reason) for one group, checking each cause."""
    full = dec if kind == "decode" else raw
    causes, reasons = [], []

    by_full = partition(ids, lambda t: full[t])
    if len(by_full) > 1 and row == b"":
        ok = all(not f or (len(f) > P and not _valid_utf8(f)) for f in by_full)
        causes.append("CUT_TO_EMPTY" if ok else "UNEXPLAINED")
        reasons.append(f"{len(by_full)} different {kind} byte strings (lengths {sorted({len(f) for f in by_full})}) "
                       f"all end up empty: each non-empty one is longer than {P} bytes with no valid UTF-8 prefix "
                       f"of length {P - 3} to {P}, so utf8_safe_truncate returns nothing")
    elif len(by_full) > 1:
        ok = all(full[t].startswith(row) for t in ids) and any(len(full[t]) > P for t in ids)
        causes.append("TRUNCATION_P" if ok else "UNEXPLAINED")
        reasons.append(f"{len(by_full)} different {kind} byte strings (lengths {sorted({len(f) for f in by_full})}) "
                       f"share their first {len(row)} bytes; utf8_safe_truncate(., {P}) drops the rest")

    for f, members in by_full.items():
        by_raw = partition(members, lambda t: raw[t])
        if len(by_raw) > 1:
            ok = C.U_FFFD in f and all(raw[t].decode("utf-8", "replace").encode("utf-8") == f for t in members)
            causes.append("UTF8_REPLACEMENT" if ok else "UNEXPLAINED")
            literal = [t for t in members if raw[t] == f]
            ex = [r for r in by_raw if r != f][:3]
            reasons.append(f"{len(by_raw)} different raw byte strings decode to the same text '{C.show(f)}': "
                           f"decode replaces each invalid UTF-8 sequence with U+FFFD (efbfbd); "
                           f"e.g. raw {', '.join(r.hex() for r in ex)} -> {f.hex()}"
                           + (f"; id {literal} holds this text literally" if literal else ""))
        for r, same in by_raw.items():
            if len(same) > 1:
                causes.append("EMPTY" if r == b"" else "DUPLICATE_BYTES")
                reasons.append(f"ids {same} have identical raw bytes {r.hex() or '(empty)'}")

    causes = [c for c in CAUSES if c in causes] or ["UNEXPLAINED"]
    return causes, "; ".join(reasons)


def analyse(kind, P, raw, dec, classes):
    buf, lens = C.load_table(kind, P)
    rows = C.table_rows(buf, lens)
    V = len(rows)
    by_row = partition(range(V), lambda t: rows[t])
    groups = sorted((g for g in by_row.values() if len(g) > 1), key=lambda g: (-len(g), g[0]))

    out, token_group = [], {}
    for gid, ids in enumerate(groups, 1):
        causes, reason = explain(ids, rows[ids[0]], kind, P, raw, dec)
        for t in ids:
            token_group[t] = (gid, causes)
        out.append({"group": gid, "size": len(ids), "causes": causes, "row": rows[ids[0]], "ids": ids,
                    "classes": C.ordered_classes([classes[t] for t in ids]), "reason": reason})

    per_script = []
    totals = Counter(classes)
    for sc in C.ordered_classes(classes):
        members = [t for t in token_group if classes[t] == sc]
        per_script.append({
            "script_class": sc, "tokens": totals[sc], "colliding_tokens": len(members),
            "groups": len({token_group[t][0] for t in members}),
            "by_cause": {c: sum(c in token_group[t][1] for t in members) for c in CAUSES},
        })
    summary = {
        "table": f"{kind}_P{P}", "kind": kind, "P": P, "vocab_size": V,
        "distinct_codes": len(by_row), "unique_tokens": V - len(token_group),
        "colliding_tokens": len(token_group), "groups": len(groups), "codes_lost": V - len(by_row),
        "groups_by_cause": dict(Counter("+".join(g["causes"]) for g in out)),
        "unexplained_groups": sum("UNEXPLAINED" in g["causes"] for g in out),
    }
    return summary, out, per_script, rows


def write_csvs(results):
    with open(C.RESULTS / "collisions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["table", "kind", "P", "group", "size", "cause", "script_classes", "row_len",
                    "row_hex", "row_text", "member_ids", "reason"])
        for s, groups, _, _ in results:
            for g in groups:
                w.writerow([s["table"], s["kind"], s["P"], g["group"], g["size"], "+".join(g["causes"]),
                            "; ".join(g["classes"]), len(g["row"]), g["row"].hex(), C.show(g["row"]),
                            " ".join(map(str, g["ids"])), g["reason"]])
    with open(C.RESULTS / "collisions_by_script.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["table", "script_class", "tokens", "colliding_tokens", "groups"] + CAUSES)
        for s, _, per_script, _ in results:
            for r in per_script:
                w.writerow([s["table"], r["script_class"], r["tokens"], r["colliding_tokens"], r["groups"]]
                           + [r["by_cause"][c] for c in CAUSES])


def main() -> int:
    tok = C.load_tokenizer()
    specials = C.special_ids(tok)
    raw, dec = C.raw_byte_table(tok), C.decode_full_bytes(tok)
    classes = [C.script_class(b, i in specials) for i, b in enumerate(raw)]
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    results = [analyse(kind, P, raw, dec, classes) for kind in C.KINDS for P in C.P_VALUES]
    for s, _, _, _ in results:
        print(f"{s['table']}: groups={s['groups']} colliding={s['colliding_tokens']} codes_lost={s['codes_lost']} "
              f"by_cause={s['groups_by_cause']} unexplained={s['unexplained_groups']}")
    rows = {s["table"]: r for s, _, _, r in results}
    groups = {s["table"]: g for s, g, _, _ in results}
    refinement = {kind: all(len({rows[f"{kind}_P32"][t] for t in g["ids"]}) == 1 for g in groups[f"{kind}_P64"])
                  for kind in C.KINDS}

    write_csvs(results)
    C.write_json(C.RESULTS / "collisions_summary.json", {
        "provenance": C.provenance(), "causes": CAUSES, "p64_refines_p32": refinement,
        "tables": [s for s, _, _, _ in results]})
    print(f"P64 refines P32: {refinement}\nwrote results/collisions.csv, collisions_by_script.csv, "
          "collisions_summary.json")
    return 0 if all(s["unexplained_groups"] == 0 for s, _, _, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
