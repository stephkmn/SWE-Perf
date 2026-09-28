# Claude Code Inference Guide

`run_claude_code.py` generates SWE-Perf patches by running headless Claude Code
**inside each instance's own evaluation image**, so the agent works in the same
environment the benchmark later measures it in.

```
python generation/realistic/run_claude_code.py --model claude-opus-5 --effort xhigh
```

## What a run does, per instance

1. Pulls (never builds, never deletes) `docker.io/betty1202/sweb.eval.x86_64.<instance_id>`
   and starts one fresh container from it — the same image, platform, CPU and
   memory settings `evaluation/run_evaluation.py` uses, imported from
   `evaluation/docker_build.py` rather than restated. One container at a time.
2. Sets up the non-root `nonroot` user the base image already ships: `/testbed`
   is chowned to it, and its shell and `PATH` have the instance's conda
   environment active, so `python` and `pytest` work with no mention in the prompt.
   Claude *must* run non-root — `--dangerously-skip-permissions` refuses root.
3. Rewrites `/testbed` into a repository with exactly one commit and verifies it:
   `git log --all` = 1, no tags, no remotes, no reflog, clean tree.
4. Runs Claude with the dataset's `problem_statement_realistic`, verbatim.
5. Filters the working tree and takes `git diff` against that one commit.
6. Removes the container. Images are never removed.

## The baseline is the image's post-setup tree, not `base_commit`

Several images ship `/testbed` already modified by their own install step —
astropy rewrites `pyproject.toml`, for one. That post-setup state is committed
as the baseline, so the extracted patch contains only Claude's changes *and*
still applies on a fresh evaluation container. Checking out `base_commit`
instead would fold the image's own edits into the model's patch and make it
fail to apply at evaluation time.

## Authentication

Claude runs in a container and cannot reach the macOS keychain, so the
subscription credential has to be passed in:

```bash
claude setup-token                     # prints a long-lived token
export CLAUDE_CODE_OAUTH_TOKEN='...'   # the driver reads it from here
```

The token is handed to the container as an **exec-time environment variable
only**. It is never written to disk, an image, a log, a transcript or the
provenance file, and it is redacted out of captured stdout/stderr before
anything is saved. `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` are unset
inside the container (`env -u`) and the run aborts if they are not, so a
subscription run can never silently fall through to per-token API billing.

The driver fails immediately, with instructions, if the variable is missing.

## The CLI cache (`datasets/claude_cli_linux_x64/`)

The `@anthropic-ai/claude-code` npm package is a thin wrapper whose postinstall
drops a large native binary into place; after that, `claude` *is* that binary
and Node is never invoked again. Installing it per instance would re-download
it 140 times, expose the sweep to a registry hiccup, and — worst for a
benchmark — let the CLI version drift between instances.

So it is built **once**, inside a linux/amd64 container (an npm install on this
arm64 macOS host would fetch the `darwin-arm64` binary, which cannot run in the
container at all), into a host directory that every instance container
bind-mounts **read-only** at `/opt/claude-cli`. One version for the whole
sweep, no per-instance cost, and the agent cannot modify its own binary.

`MANIFEST.json` in that directory records the Node, npm-package and CLI
versions; they are copied into every provenance record. Rebuild with
`--bootstrap_cli`, or build it alone with `--bootstrap_only`.

At bootstrap the driver checks the *installed* CLI rather than assuming the
host's contract: `--effort` (with the requested level), `--model`,
`--output-format`, `--max-turns`, `--disallowedTools`,
`--dangerously-skip-permissions` and `--print` must all appear in that build's
`claude --help`, `setup-token` must still be a command, and
`CLAUDE_CODE_OAUTH_TOKEN` must still appear in the binary.

### Rosetta and AVX on Apple Silicon

The `linux-x64` Claude Code binary uses AVX. Docker Desktop on Apple Silicon
must therefore be running x86-64 containers under **Rosetta** (Settings →
General → "Use Rosetta for x86_64/amd64 emulation"), which supports AVX2 for
Linux binaries on recent macOS. Without it the bootstrap fails at
`claude --version` with an illegal-instruction crash rather than silently
recording garbage as the version.

## What is filtered out of the patch

| dropped | rule |
| --- | --- |
| edits to `tests/`, `test_*.py`, `*_test.py`, `conftest.py` | benchmark rule 1 forbids touching tests (`testing/` is library code and is *not* matched) |
| new files at the repo root, `.DS_Store` anywhere | agent scratch; a stray file at the root can break the package build |
| `__pycache__`, `*.pyc`, `*.so`/`*.pyd`/`*.o`, `build/`, `*.egg-info`, caches | `/testbed` arrives already built, so merely importing the package dirties these |
| any file `git diff --numstat` reports as binary | `git apply` refuses a "Binary files … differ" hunk |

Tracked files are restored to the baseline; untracked ones are deleted. Every
dropped path is recorded in the run's `_meta.jsonl`, with the reason.

## Output

Beside `--output` (default `datasets/outputs/claude_code_preds.jsonl`):

- `..._preds.jsonl` — `{instance_id, model_name_or_path, model_patch}`, ready for `evaluation/run_evaluation.py`
- `..._meta.jsonl` — per instance: status, image, conda env, baseline report, filtered paths, prompt hash, CLI version, `claude_result`, web-access audit hits, plus effort spent — `duration_s`, `turns`, `turns_source`, `tool_calls`
- `..._partial_preds.jsonl` — same format as the predictions file, but only the diffs from runs cut off by `timeout` or `max_turns`. Never fed to the evaluator: a mid-edit tree is an unfinished attempt, not a result. Kept so a three-hour run is readable instead of lost.
- `..._provenance.jsonl` — per run: `execution_env: "docker_with_deps"`, image names, model, effort, CLI/Node versions, tool allow/deny lists, prompt style and hashes
- `claude_code_logs/run<N>/<id>.jsonl` — the full stream-json transcript
- `claude_code_logs/run<N>/<id>.stderr.txt`

Instances that error, time out, or hit a usage limit get a meta line but **no**
prediction line, which is what makes `--resume` pick them up again. A cut-off
run's diff still lands in `..._partial_preds.jsonl`; that file is invisible to
`--resume`, so the instance is retried regardless.

`turns` comes from the CLI's own `num_turns` when the run printed a result
line. A killed run never prints one, so the turns are counted off the saved
transcript instead and `turns_source` says `transcript` — the two are not
measured the same way and should not be compared as if they were.

## Useful flags

| flag | |
| --- | --- |
| `--model` | **required**; full model ID, e.g. `claude-opus-5` |
| `--effort` | `low`/`medium`/`high`/`xhigh`/`max` (default `xhigh`) |
| `--instance_ids` | run only these |
| `--num_runs` | repeat the sweep; each run gets its own predictions file |
| `--resume` | skip instances already written for that run |
| `--timeout` | per-instance wall clock, default 1800s, enforced inside the container |
| `--keep_containers` | leave containers running for debugging |
| `--bootstrap_only` | build the CLI cache and exit |

The driver refuses to start a container while `analysis/timed_path.py` is
running and waits for it instead: two of these containers do not fit on a 16 GB
Mac under emulation. `--no-wait-for-timed-path` overrides that.

## Verification

```bash
# unit tests: patch filtering and the single-commit baseline logic (no Docker, no Claude)
python -m pytest generation/realistic/tests/test_run_claude_code.py

# container test: real container, real setup, dummy edits, NO Claude call.
# Ends by confirming the extracted patch passes `git apply --check` in a
# second, untouched container of the same image.
python generation/realistic/tests/check_container_setup.py --instance_id astropy__astropy-13496
```
