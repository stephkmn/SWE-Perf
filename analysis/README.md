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

### 4. Combined report

```bash
python report.py --official_csv ../datasets/outputs/model_result.csv --sample_size 20
```

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

43 tests covering the diff parser, AST function mapping, name matching, and
pattern flags. No Docker, network, or dataset required.
