# eval-mini v1: result schema

**Status: draft, 1 Nov 2026.** It becomes final when its PR merges (due Wed 4 Nov). The `checkpoint` fields stay
provisional until Team 7 publishes its checkpoint path layout (due Wed 4 Nov).

Used by:
- Team 8, which ingests eval-mini results;
- Team 12, which scores quantized formats with eval-mini;
- `harness_lab/run_lab_ckpt.py`, which scores lab checkpoints.

Comments are welcome in G9 until **Tue 3 Nov, 12:00 IST**.

## 1. What changes from v0

| | v0 | v1 |
|---|---|---|
| Shape | two shapes: a flat `records` list (`eval_mini_v0.json`, PR #21) and a file nested by model (`eval_mini_v0_code_indic.json`, PR #88) | one flat `records` list for every source |
| Units | fractions 0–1 (PR #21), points (PR #88) | **points 0–100** everywhere |
| Model revision | per record (PR #21), per model (PR #88) | per record |
| Harness version | per record (PR #21), in `env` (PR #88) | per record, as `{name, version}` |
| Seed | 42 per record (PR #21), not recorded (PR #88) | per record, all four seeds |
| Lab checkpoints, quantized formats | no field for them | `checkpoint`, `format`, `backend` |

## 2. File layout

```json
{
  "schema_version": "eval_mini_v1",
  "generated": "2026-11-03",
  "sources": [
    {"path": "teams/09-evaluation/results/eval_mini_v0.json", "sha256": "<sha256 of the file>", "schema_version": "eval_mini_v0"}
  ],
  "notes": ["<file-level notes, see section 8>"],
  "records": [ { "...": "one object per score, section 3" } ]
}
```

UTF-8, LF line endings, records sorted by the key in section 4. Writing the same inputs twice gives a byte-identical file.

## 3. Record fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model` | str | yes | Hugging Face id as evaluated (`Qwen/Qwen2.5-1.5B`), or `lab:<run_id>` for a lab checkpoint |
| `model_revision` | str | yes | Hugging Face commit sha (40 hex characters); for a lab checkpoint, the sha256 of its `MANIFEST.json` |
| `checkpoint` | object or null | yes | null for Hugging Face models; for a lab checkpoint `{run, branch, step, epoch, tokens_seen, path}` (section 6) |
| `format` | str | yes | weights format as evaluated: `bf16` for the Hugging Face runs; Team 12's format name for quantized runs |
| `backend` | str | yes | `hf` (transformers through lm-eval) or the backend Team 12 reports (`llama.cpp`, …) |
| `dtype` | str or null | yes | torch dtype as reported (`bfloat16`) |
| `task` | str | yes | lm-eval task name (`arc_challenge`, `leaderboard_mmlu_pro`, `milu_Hindi`, …); code: `humaneval`, `humaneval_plus`, `mbpp`, `mbpp_plus`; derived: `arc_average` |
| `metric` | str | yes | `acc`, `acc_norm`, `exact_match`; code: `pass@1` |
| `filter` | str | yes | lm-eval answer filter: `none`, `strict-match`, `flexible-extract`; code: `none` |
| `primary` | bool | yes | true for the one metric and filter per task that reports use |
| `value` | number | yes | **points 0–100**, 2 decimals |
| `stderr` | number or null | yes | points, 2 decimals; null where it is not computed (code) |
| `n_items` | int | yes | items scored |
| `n_items_total` | int or null | no | items in the full split |
| `item_set` | str | yes | `full`; `limit:<N>` (lm-eval's first N items); or `spec:<spec_sha256>` (the items listed in an eval-mini spec file) |
| `profile` | str or null | no | v0 profile name (`mini`, `check`) or eval-mini part (`code_indic`). For reading only: not part of the key |
| `num_fewshot` | int | yes | few-shot examples (0 for code) |
| `decoding` | str | yes | `loglikelihood` (scored choices) or `greedy` (generation) |
| `seed` | object or null | yes | `{random, numpy, torch, fewshot}`; null only for 0-shot greedy runs, where no seed changes the result (section 5) |
| `harness` | object | yes | `{name, version}`, for example `{"name": "lm_eval", "version": "0.4.13"}` or `{"name": "evalplus", "version": "0.3.1"}`; optional `script_sha256` (the G9 script that ran it, as Team 12 records it) |
| `hardware` | str | yes | GPU or device as reported |
| `run_date` | str | yes | `YYYY-MM-DD` |
| `derived_from` | list of str or null | no | the tasks a derived score is computed from (`arc_average`: `arc_easy`, `arc_challenge`) |
| `notes` | list of str | no | notes that apply to this record only |

## 4. Record key

A record is identified by:

**`model`, `model_revision`, `format`, `item_set`, `task`, `num_fewshot`, `metric`, `filter`**

No two records in a file share a key. Every part is needed:
- `model_revision`: Qwen2.5-1.5B is scored at two revisions, `8faed761` (PR #21) and `e5dfabbc` (PR #88).
- `format`: Team 12 scores one model in several formats.
- `item_set`: GSM8K is scored on different items, the first 200 (PR #21, `mini`) and 100 spec items (PR #88).
- `num_fewshot`: PR #21 scores ARC-Challenge at 0 and 25 shots, and GSM8K at 4 and 5 shots, on the same full sets.

The model name alone never identifies a score.

## 5. Seeds

| Source | Seeds recorded in v1 | Where it comes from |
|---|---|---|
| `harness/run_eval_mini.py` (PR #21, Team 12) | `{42, 42, 42, 42}` | one `--seed`, passed to all four (`run_task`) |
| `harness_code_indic/run_lm_eval.py` (PR #88) | `{0, 1234, 1234, 1234}` | it passes no seed, so lm-eval 0.4.13's defaults apply (`lm_eval/defaults.py`) |
| evalplus code runs (PR #88) | null | 0-shot, greedy |

The eval-mini spec's `seed` (20261028) chooses which items are in the item set. It is not a run seed, and `item_set` already records it through `spec_sha256`.

## 6. Lab checkpoints (provisional)

Team 7's draft layout (PR #150, not merged) stores a checkpoint at
`<remote_root>/<run_id>/<branch>/step_<N>-e<E>/`: sharded files plus a `MANIFEST.json` that lists every file with its
sha256 and is written last. Each checkpoint records its run id, branch, step and tokens seen.

A sharded checkpoint has no single weights file, so v1 uses the sha256 of `MANIFEST.json` as `model_revision`.

```json
{
  "model": "lab:<run_id>",
  "model_revision": "<sha256 of MANIFEST.json>",
  "checkpoint": {"run": "<run_id>", "branch": "<branch>", "step": 2000, "epoch": 0, "tokens_seen": 0, "path": "<remote_root>/<run_id>/<branch>/step_2000-e0/"},
  "format": "bf16",
  "backend": "hf",
  "...": "the other fields as in section 3"
}
```

The exact form follows Team 7's layout on Wed 4 Nov. Team 8 plots scores over training by `checkpoint.step`.

## 7. Converting the v0 files

The converter is `results/convert_v0_to_v1.py`. It reads both v0 files and `harness_code_indic/eval_mini_v0_spec.json`, and writes
`results/eval_mini_v1.json`. Team 12 can run it on its own v0 files. `harness/run_eval_mini.py` does not change:
Team 12 checks its sha256 when combining runs.

| v1 field | From `eval_mini_v0.json` (PR #21; also Team 12's `summary.json`) | From `eval_mini_v0_code_indic.json` (PR #88) |
|---|---|---|
| `model` | `model` | `hf_id` |
| `model_revision` | `model_revision` | `revision` |
| `format`, `backend` | `bf16`, `hf` (Team 12: `run_meta.format`, `run_meta.backend`) | `bf16`, `hf` |
| `task`, `metric`, `filter` | as is | `scores` key; `metric` string split at the comma (`acc,none` → `acc` + `none`). Code: base → `humaneval` / `mbpp`, plus → `humaneval_plus` / `mbpp_plus`, metric `pass@1`, filter `none` |
| `value`, `stderr` | × 100, rounded to 2 decimals | as is; code `stderr` null |
| `n_items`, `n_items_total` | as is | `n`; total from the spec's `n_test_docs` (code: 164 and 378) |
| `item_set` | `full` if `limit` is null, else `limit:<limit>` | `spec:<spec_sha256>`; code `full` |
| `num_fewshot` | as is | from the spec (5); code 0 |
| `decoding` | `greedy` for `gsm8k`, else `loglikelihood` | the same; code `greedy` |
| `seed` | `{42, 42, 42, 42}` | `{0, 1234, 1234, 1234}`; code null |
| `harness` | `{lm_eval, harness_version}` | `{lm_eval, env.lm_eval}`; code `{evalplus, 0.3.1}` (the spec's `code.harness`) |
| `hardware` | `hardware` | `gpu.before.gpu` |
| `run_date` | `run_date` | the file's `generated` date (2026-10-27) |
| `primary` | `primary` | true (the file holds one metric per task) |
| `profile` | `profile` | `code_indic` |
| `derived_from` | `derived_from` (only `arc_average`) | null |

Expected result: **98 records** (59 from PR #21, 27 lm-eval scores and 12 code scores from PR #88).

The converter prints these checks:

| Check | Expected |
|---|---|
| record count | 98 |
| records missing a required field | 0 |
| values outside 0–100 | 0 |
| PR #21 records with `value == round(100 × v0 value, 2)` | 59 of 59 |
| duplicate keys (section 4) | 0 |
| a second run | byte-identical file |
| Team 12's `eval_mini/results/raw/runs_bf16/bf16/summary.json` | converts without errors |

## 8. File-level notes

- Subset scores (`limit:` and `spec:` item sets) are for comparing checkpoints and formats. They are not comparable to
  published full-set scores.
- Noise: on Qwen2.5-1.5B, changing only the attention kernel moved 100-item tasks by up to 2 items. Treat a difference of
  about 2 points on a 100-item task as noise.
- GSM8K: our 4-shot full-set scores are 6–7 points under Qwen's own report. The gap is prompt format: lm-eval and the
  SmolLM2 card's lighteval agree with each other and not with Qwen's figure (`results/compare_v0.md`).

## 9. Open points

1. **Team 8:** what does ingestion need? JSON or JSON Lines, nested objects (`checkpoint`, `seed`, `harness`) or flat
   columns, and how should nulls look?
2. **Team 7:** the final checkpoint layout (Wed 4 Nov) sets `checkpoint` and the lab `model_revision`.
3. **HellaSwag:** its items are internal only (licence unclear). Scores of public models are not items, so the proposal is
   to keep HellaSwag scores in v1 with a note.
