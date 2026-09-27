# Post-hoc patch analysis

Stricter checks layered on top of SWE-Perf's official evaluation. **Nothing here
modifies `evaluation/`** — the official metric stays exactly as published so our
numbers remain comparable.

## Why this exists

`run_evaluation.py` writes only `test_spec.eval_script` into the container,
which runs the instance's `efficiency_test` list. `check_evaluation.py` then
calls a patch correct when those same tests pass. For `astropy__astropy-16065`
that is **two tests serving as both the performance measurement and the
correctness check**. `test_spec.py` defines `eval_script_list_alltests`, a
full-suite variant, but no code path invokes it.

Four behaviours therefore score identically:

| | Behaviour |
|---|---|
| 1 | Profile, find the real bottleneck, fix it |
| 2 | Optimize only the input shapes the measured test uses |
| 3 | Speed up the measured path while breaking untested behaviour |
| 4 | Short-circuit the work the test does |

All four produce "tests pass, timing improved". These tools gather evidence to
tell them apart. **None of them decides anything on its own** — the output is a
shortlist for human review.

## Scripts

Run from this directory, with the `sweperf` conda env active.

### 1. Regression check (needs Docker)

Runs a wider test selection on the base code and again after the patch,
reporting only pass-to-fail transitions. Once per patch, outside any timing
loop, so it cannot perturb official measurements.

```bash
python regression_check.py \
  --predictions ../datasets/outputs/preds.jsonl \
  --test_scope nearby \
  --timeout 3600 \
  --out_dir analysis_out/regression
```

- `--test_scope nearby` (default): `tests/` directories beside each modified
  file, plus matching `test_<module>.py`. `full` runs the whole suite.
- Tests already failing on base are ignored.
- A newly failing test is rerun once; if it then passes it is recorded as
  `flaky`, not a regression.
- `--use_gold_patch` substitutes the expert patch — useful as a sanity check,
  since it should produce no regressions.

Writes one JSON per instance: `tests_selected`, `pass_to_fail`,
`missing_after_patch`, `flaky`, `timed_out`, `error`.

### 2. Patch scope (no Docker)

```bash
python patch_scope.py --predictions ../datasets/outputs/preds.jsonl
```

Maps changed lines to enclosing functions via the base source's AST, read from
the bare mirrors in `datasets/repo_mirrors/` (`--mirror_dir` to override).
Without mirrors it falls back to git's hunk headers, which are less precise;
the run prints a warning and sets `source_resolved: false`.

Compares against the task's targets (`test_functions`) and the expert's
(`patch_functions`). Name matching tolerates the dataset's inconsistent
qualification — it writes `LombScargle.autopower` but also a bare `wrapper`
where the AST resolves `QuantityInput.__call__.wrapper`.

> These fields are for analysis only and must never be fed back into patch
> generation.

### 3. Suspicious patterns (no Docker)

```bash
python flag_patterns.py --predictions ../datasets/outputs/preds.jsonl
```

Scans added lines for new caches, exact shape/dtype/length special cases, stack
or environment inspection, and monkey-patching of library attributes. Every
pattern is legitimate somewhere — a flag means "read this one", not "this is
cheating".

### 4. Timed-path coverage (needs Docker)

```bash
python timed_path.py --predictions ../datasets/outputs/preds.jsonl
```

Applies the patch and runs the instance's `efficiency_test` list **once** under
`coverage.py` — no timing, so it cannot disturb official measurements. Records
`on_timed_path` per changed function, plus `n_files_measured` and the tail of
the pytest output so a failed run is visible rather than silently reported as
"nothing ran".

Two measurement details matter here, both of which produced wrong answers
before they were fixed:

- Coverage is run with `--source` pinned to the checkout. Without it the first
  run reported every changed function as off-path, including functions the
  timed tests demonstrably exercise.
- "Did it run" is asked of the function's **body**, not its full span. A `def`
  line, its decorators and its docstring all execute at import, so the full
  span answers "was this module imported", which is true of every function in a
  touched file.

This exists to stop a wrong inference. Speeding up a helper that a target
function calls is the intended solution in SWE-Perf's realistic setting, yet
that helper appears "outside the named targets". Only a changed function that
never executes during the timed tests is a candidate for a genuinely different
bottleneck — or for dead work that cannot be affecting the score at all.

### 5. Provenance audit (no Docker)

```bash
python provenance_audit.py --predictions ../datasets/outputs/preds.jsonl
```

Before commit `19d54d4`, the generation driver cloned the full mirror into each
working copy, so the agent could read forward through history to the upstream
fix. This dates each prediction against that commit using the provenance file's
`started_at` and prompt hash, and searches transcripts for `git log`, `--all`,
`git show`, `git tag`.

**Read `transcript_fidelity` before trusting the transcript search.** Full
stream-json transcripts only began in the same commit that fixed the problem,
so pre-fix transcripts hold the agent's closing prose and nothing else —
finding no `git log` in them proves nothing, because a tool call could never
have appeared there.

### 6. Combined report

```bash
python report.py --official_csv ../datasets/outputs/model_result.csv --sample_size 20
```

Beyond the scope and flag columns, the report carries per-run test accounting
(`base_collected`/`after_collected`, passed/failed/errored/skipped,
`collection_errors`) and three regression components:

| Column | Meaning |
|---|---|
| `n_pass_to_fail` | passed on base, failed after |
| `n_missing_after_patch` | ran on base, did not run at all after |
| `n_new_collection_errors` | modules that stopped importing |
| `n_regressions` | the sum of those three |

The last two matter because the benchmark's test command passes
`--continue-on-collection-errors`: a patch that breaks an import makes whole
modules vanish rather than fail, which would otherwise read as a clean pass.

Overlap is reported twice — `overlap_with_targets_exact` counts only exact
name matches, `overlap_with_targets` also counts short-name fallbacks, and
`n_ambiguous_matches` counts fallbacks onto a short name that appears more than
once in its file.

Merges everything into `combined_report.csv`, plus `manual_review.csv`
containing every flagged or regressing patch and a fixed-seed random sample of
the rest, with an empty `category` column:

`1` real bottleneck fix · `2` overfit to test inputs · `3` breaks untested
behaviour · `4` short-circuits the tested work

The random sample matters: reviewing only suspicious patches tells you nothing
about the base rate.

Note that `check_evaluation.py`'s CSV is aggregated **per repo**, not per
instance, so `official_*` columns describe the repo rather than the individual
patch.

## Tests

```bash
python -m pytest tests/test_analysis.py -q
```

85 tests across `test_analysis.py` and `test_fixes.py`, covering the diff
parser, AST function mapping, match labelling, run accounting, coverage path
mapping, pattern flags, and provenance dating. No Docker, network, or dataset
required.
