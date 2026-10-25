"""
Refresh docs/ke2_week1_explainer.html, the team's interactive explainer of the
week-1 work, from the files in this folder. The page is one self-contained HTML
file: open it in any browser.

The page's text is written by hand. Its playgrounds run on this data block,
which holds:
  - the raw bytes of all token ids (and the ids whose decode([id]) differs)
  - the script class of every token
  - real tokenizations of the example sentences
  - results/: collisions, per-script counts, proof checks, determinism, manifest
  - the test ids collected from teams/04-ke2/tests

Before writing, it checks that the embedded raw and decode bytes hash to the
values in results/determinism.json, so the playgrounds run on audited data.

It also inlines the figures of docs/proof.md (drawn by src/proof_figures.py)
into every <div class="fig-scroll" data-fig="NAME"> slot. Their fixed colours
become the page's theme tokens, so the figures follow light and dark mode.

Run from the lab repo root:  python teams/04-ke2/src/build_explainer.py
(after the other scripts; it reads their outputs and rewrites the page in place)
"""

from __future__ import annotations

import base64
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import ke2_common as C

TEAM = C.TEAM
LAB = TEAM.parents[1]
PAGE = TEAM / "docs" / "ke2_week1_explainer.html"
BLOCK = re.compile(r'(<script type="application/json" id="ke2-data">)(.*?)(</script>)', re.S)
FIGS = TEAM / "docs" / "figures"
FIG_SLOT = re.compile(r'(<div class="fig-scroll" data-fig="(\w+)">)(.*?)(</div>)', re.S)
TAG = re.compile(r'<(\w+)((?:\s+[\w:-]+="[^"]*")*)\s*(/?>)')
# proof_figures.py colour -> page token (defined for light and dark in the page's :root blocks)
THEME = {"#ffffff": "--fig-bg", "#0f2230": "--ink", "#3f5260": "--ink-2", "#667884": "--muted",
         "#d8e0e4": "--rule", "#b7c4cb": "--rule-strong", "#f3f6f7": "--paper", "#2a78d6": "--raw",
         "#e2edfb": "--raw-wash", "#eb6834": "--dec", "#fde8dd": "--dec-wash", "#ffd675": "--mark",
         "#c98d0c": "--mark-edge", "#0a8a0a": "--good", "#e1f3e1": "--good-bg", "#b42318": "--bad-ink",
         "#e2e8eb": "--fig-grid", "#d5dde1": "--fig-grid", "#e6ecef": "--fig-grid"}

EXAMPLES = [
    ("Hindi (token A)", "Devanagari", "भारत एक देश है"),
    ("Tamil (tokens B and C)", "Tamil", "மிஸ் ம்ம்"),
    ("Bengali", "Bengali", "আমি বাংলায় গান গাই"),
    ("Tamil", "Tamil", "தமிழ் மொழி"),
    ("Telugu", "Telugu", "తెలుగు భాష"),
    ("Punjabi", "Gurmukhi", "ਪੰਜਾਬੀ ਬੋਲੀ"),
    ("Gujarati", "Gujarati", "ગુજરાતી ભાષા"),
    ("Odia", "Odia", "ଓଡ଼ିଆ ଭାଷା"),
    ("Kannada", "Kannada", "ಕನ್ನಡ ಭಾಷೆ"),
    ("Malayalam", "Malayalam", "മലയാളം"),
    ("English", "Latin", "Hello, world!"),
    ("Python", "Latin", "def f(x): return x**2"),
    ("Santali", "Ol Chiki", "ᱚᱞ ᱪᱤᱠᱤ"),
    ("Manipuri", "Meetei Mayek", "ꯃꯤꯇꯩ ꯃꯌꯦꯛ"),
    ("Sinhala", "Sinhala", "අකුරු"),
    ("Japanese", "Kanji and kana", "日本語のテキスト"),
    ("Emoji", "Emoji", "😀🙂"),
]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def read_csv(path: Path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def inline_svg(name: str) -> str:
    """One figure as inline SVG: scalable, with its own marker id, coloured by theme tokens."""
    s = (FIGS / f"{name}.svg").read_text(encoding="utf-8").strip()
    s, n = re.subn(r'<svg xmlns="http://www.w3.org/2000/svg" width="\d+" height="\d+"',
                   '<svg xmlns="http://www.w3.org/2000/svg" class="pfig"', s, count=1)
    if n != 1:
        raise SystemExit(f"{name}.svg: unexpected root element")
    s = s.replace('id="ah"', f'id="ah-{name}"').replace("url(#ah)", f"url(#ah-{name})")

    def themed(m):
        attrs, style = m.group(2), []
        for prop in ("fill", "stroke"):
            a = re.search(rf' {prop}="(#[0-9a-f]{{6}})"', attrs)
            if a:
                if a.group(1) not in THEME:
                    raise SystemExit(f"{name}.svg: no theme token for {a.group(1)}")
                style.append(f"{prop}:var({THEME[a.group(1)]})")
                attrs = attrs.replace(a.group(0), "", 1)
        return f"<{m.group(1)}{attrs}" + (f' style="{";".join(style)}"' if style else "") + m.group(3)

    return TAG.sub(themed, s)


def collect_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "teams/04-ke2/tests", "--collect-only", "-q",
                          "-p", "no:cacheprovider"], cwd=LAB, capture_output=True, text=True, encoding="utf-8")
    return [line.strip() for line in out.stdout.splitlines() if "::" in line]


def main() -> int:
    tok = C.load_tokenizer()
    specials = sorted(C.special_ids(tok))
    raw = C.raw_byte_table(tok)
    dec = C.decode_full_bytes(tok)
    det = C.read_json(C.RESULTS / "determinism.json")
    if C.sha256_bytes_list(raw) != det["hashes"]["raw"]["sha256"]:
        raise SystemExit("raw bytes differ from results/determinism.json")
    if C.sha256_bytes_list(dec) != det["hashes"]["decode"]["sha256"]:
        raise SystemExit("decode bytes differ from results/determinism.json")

    special_set = set(specials)
    classes = [C.script_class(b, i in special_set) for i, b in enumerate(raw)]
    names = C.ordered_classes(classes)
    index = {n: i for i, n in enumerate(names)}

    groups = {}
    for r in read_csv(C.RESULTS / "collisions.csv"):
        groups.setdefault(r["table"], []).append({
            "group": int(r["group"]), "size": int(r["size"]), "cause": r["cause"], "row": r["row_hex"],
            "text": r["row_text"], "ids": list(map(int, r["member_ids"].split())),
            "classes": r["script_classes"], "reason": r["reason"]})
    by_script = {}
    for r in read_csv(C.RESULTS / "collisions_by_script.csv"):
        by_script.setdefault(r["table"], []).append({
            "cls": r["script_class"], "tokens": int(r["tokens"]), "colliding": int(r["colliding_tokens"]),
            "groups": int(r["groups"])})

    hash_lines = []
    for line in (C.RESULTS / "decode_cpu_sha256.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            sha, run, *settings = line.split()
            hash_lines.append({"sha256": sha, "run": run, "settings": " ".join(settings)})

    proof = C.read_json(C.RESULTS / "proof_checks.json")
    summary = C.read_json(C.RESULTS / "collisions_summary.json")
    manifest = C.read_json(C.RESULTS / "tables_manifest.json")
    data = {
        "provenance": manifest["provenance"],
        "V": len(raw),
        "tokens": {
            "lengths": b64(np.array([len(r) for r in raw], dtype=np.uint8)),
            "blob": base64.b64encode(b"".join(raw)).decode("ascii"),
            "decode_diff": {str(i): d.hex() for i, (r, d) in enumerate(zip(raw, dec)) if r != d},
            "special": specials,
            "class_names": names,
            "classes": b64(np.array([index[c] for c in classes], dtype=np.uint8)),
        },
        "examples": [{"label": lab, "script": scr, "text": t, "ids": tok.encode(t, add_special_tokens=False)}
                     for lab, scr, t in EXAMPLES],
        "collisions": {"tables": summary["tables"], "refines": summary["p64_refines_p32"],
                       "groups": groups, "by_script": by_script},
        "proof": {"passed": proof["passed"], "total": proof["total"], "elapsed_s": proof["elapsed_s"],
                  "checks": proof["checks"]},
        "determinism": {"runs": det["runs"], "all_match": det["all_three_match"],
                        "cross": det["tables_match_manifest_and_roundtrip"],
                        "hashes": det["hashes"], "file_lines": hash_lines},
        "manifest": {k: {kk: v[kk] for kk in ("sha256", "distinct_rows", "ids_sharing_a_row", "n_truncated",
                                              "max_full_length", "rows_with_u_fffd")}
                     for k, v in manifest["tables"].items()},
        "tests": collect_tests(),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/").replace("<!--", "<\\!--").replace("�", "\\ufffd")
    html = PAGE.read_text(encoding="utf-8")
    if not BLOCK.search(html):
        raise SystemExit(f"data block not found in {PAGE}")
    html = BLOCK.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    slots = [m.group(2) for m in FIG_SLOT.finditer(html)]
    missing = sorted({p.stem for p in FIGS.glob("*.svg")} - set(slots))
    if missing:
        raise SystemExit(f"figures with no slot on the page: {missing}")
    html = FIG_SLOT.sub(lambda m: m.group(1) + inline_svg(m.group(2)) + m.group(4), html)
    PAGE.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote docs/{PAGE.name}: {len(html) / 1e6:.2f} MB, {len(slots)} figures, "
          f"{len(data['tests'])} tests listed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
