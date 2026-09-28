#!/usr/bin/env python3
"""
Generate SWE-Perf patches with Claude Code, inside the benchmark's own images.

For each dataset instance this script starts a fresh container from the same
image the evaluation harness runs (docker.io/betty1202/sweb.eval.x86_64.<id>),
runs headless Claude Code inside it as a non-root user with the instance's
conda environment already active, and captures the resulting `git diff` as the
instance's `model_patch`.

Running inside the image is the whole point: /testbed there is the repo *after*
the instance's install step, with every dependency present, so the agent can
import the package, run pytest, and measure what it changed. The previous
host-checkout mode could do none of that and has been removed.

Isolation: before Claude starts, /testbed is reduced to a single commit -- the
post-setup tree exactly as the image ships it, re-committed into a brand-new
repository. No earlier or later history, no tags, no remotes, no reflog. The
agent therefore cannot read the upstream fix, and the extracted diff contains
only its own changes while still applying cleanly on a fresh eval container.

Output is a JSONL file in the format expected by evaluation/run_evaluation.py:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

Authentication is CLAUDE_CODE_OAUTH_TOKEN from the host environment, handed to
the container as an exec-time environment variable only. It is never written to
disk, an image, a log, a transcript, or the provenance record, and it is
redacted out of captured output before anything is saved.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

import docker  # noqa: E402
import docker.errors  # noqa: E402

from constants import MAP_REPO_VERSION_TO_SPECS  # noqa: E402
from docker_build import LOCAL_MAX_MEM_LIMIT, LOCAL_MAX_NANO_CPUS, USE_HOST_NETWORK  # noqa: E402
from docker_utils import cleanup_container  # noqa: E402
from test_spec import make_test_spec  # noqa: E402

PROMPT_HASH_LEN = 16

# run_evaluation.py rewrites every instance image key to the published x86_64
# images under this prefix rather than building locally. Mirror it exactly so
# generation and evaluation see the same /testbed.
DOCKER_IMAGE_PREFIX = "docker.io/betty1202/"
IMAGE_TEMPLATE = DOCKER_IMAGE_PREFIX + "sweb.eval.x86_64.<instance_id>"

REPO_DIR = "/testbed"
# Created by the benchmark's own base image (`adduser ... nonroot`). Claude
# needs a non-root user: --dangerously-skip-permissions refuses to run as root.
AGENT_USER = "nonroot"
AGENT_HOME = "/home/nonroot"
CONDA_ROOT = "/opt/miniconda3"

# Read-only bind mount holding a Linux x86-64 Claude Code install, built once
# by --bootstrap_cli and reused by every instance. See bootstrap_cli().
CLI_MOUNT = "/opt/claude-cli"
CLI_BIN = f"{CLI_MOUNT}/npm/bin/claude"
CLI_NODE_BIN = f"{CLI_MOUNT}/node/bin"

# Node is a *bootstrap-time* dependency only: the npm package's postinstall
# swaps the platform's native binary over bin/claude.exe, after which `claude`
# execs that binary directly and no Node process is involved. Pinned so the
# cache is reproducible; the resolved versions land in the manifest.
NODE_VERSION = "22.23.3"
NODE_TARBALL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.gz"
CLI_NPM_PACKAGE = "@anthropic-ai/claude-code"
MANIFEST_NAME = "MANIFEST.json"

# Anchored to the repo, not the working directory: these used to be written
# "../../datasets/...", which only lands inside the repo when the script is run
# from generation/realistic and silently writes to the parent of the checkout
# otherwise. An explicit --output or --cli_cache is still taken as given.
DEFAULT_OUTPUT = str(REPO_ROOT / "datasets" / "outputs" / "claude_code_preds.jsonl")
DEFAULT_CLI_CACHE = str(REPO_ROOT / "datasets" / "claude_cli_linux_x64")

MARKER = "SWEPERF::"

ALLOWED_TOOLS = ["Edit", "Read", "Write", "Glob", "Grep", "Bash"]

# Network egress is denied outright: the experiment must measure what Claude can
# do from the source tree alone, not what it can look up about the upstream fix.
DISALLOWED_TOOLS = ["WebFetch", "WebSearch", "Bash(curl:*)", "Bash(wget:*)"]

# Substrings that, if they show up in a tool_use input, suggest the sandbox was
# probed for network access or repo history. Matched case-insensitively.
AUDIT_PATTERNS = ["http://", "https://", "github.com", "git log --all", "pip download"]

# problem_statement_realistic already carries the benchmark's own four rules, so
# it is sent untouched. This is the apples-to-apples comparison against
# published SWE-Perf numbers, and it is now the only supported prompt.
VERBATIM_TEMPLATE = "{problem_statement}"
PROMPT_STYLE = "verbatim"

# Intentionally does NOT match `testing/`: that is library code in several of the
# benchmark repos (e.g. xarray/testing/assertions.py), not a test suite.
TEST_PATH_RE = re.compile(
    r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$"
)

# Build output. Unlike the host-checkout setting, /testbed arrives already
# built, so these files exist in the baseline and Claude re-running the build
# (or merely importing the package) dirties them. None of it belongs in a patch.
BUILD_ARTIFACT_RE = re.compile(
    r"(^|/)__pycache__/|\.py[co]$|"
    r"\.(so|pyd|o|a|dylib)(\.[\w.]+)?$|"
    r"(^|/)build/|(^|/)\.eggs/|\.egg-info(/|$)|"
    r"(^|/)\.pytest_cache/|(^|/)\.hypothesis/|(^|/)\.mypy_cache/|(^|/)\.coverage$"
)

USAGE_LIMIT_RE = re.compile(
    r"usage limit|rate limit|rate[_-]limit|limit reached|limit_reached|"
    r"\b429\b|too many requests|overloaded",
    re.IGNORECASE,
)

# The ONLY statuses that earn a prediction line. Everything else is treated as
# unfinished work, left out of the predictions file, and picked up again by
# --resume. An allow-list rather than a deny-list on purpose: the first pilot
# run hit a 401, which exits 1 with an empty diff, and a deny-list recorded
# that as three legitimate "the model changed nothing" results. Any outcome we
# have not explicitly decided is trustworthy must not become a data point.
#
# "empty_patch" is kept because it IS a real result: the agent ran to
# completion and chose to change nothing. "timeout" is not, because a killed
# run leaves a mid-edit tree, and that truncated diff is a wrong data point
# rather than a slow one.
KEEP_STATUSES = ("ok", "empty_patch")

# Statuses where the agent was cut off mid-thought rather than finishing. The
# diff it left behind is a genuine partial attempt -- worth reading, and worth
# keeping so a run is not simply lost -- but it is not a result, so it goes to
# its own file and never reaches the predictions the evaluator scores.
PARTIAL_STATUSES = ("timeout", "max_turns")


def earns_a_prediction(status: str) -> bool:
    return status in KEEP_STATUSES


def earns_a_partial_prediction(status: str) -> bool:
    return status in PARTIAL_STATUSES


# Matched against the agent's own error text, never against the whole result
# JSON: that blob carries token counts, and a bare \b401\b would happily match
# "input_tokens": 401.
AUTH_ERROR_RE = re.compile(
    r"authentication_failed|oauth access token is invalid|"
    r"oauth token is invalid|invalid[_ ]api[_ ]key|unauthorized",
    re.IGNORECASE,
)


def classify_claude_failure(rc, result, stderr) -> str:
    """Name the failure, so an infrastructure fault is never a model result.

    An expired token, a 429 and a genuine agent crash all exit non-zero with
    an empty diff; only the last of those says anything about the model.
    """
    result = result or {}
    code = result.get("api_error_status")
    text = " ".join(x for x in (result.get("result_text", ""), stderr or "") if x)
    # Exhausting the turn budget is a real experimental outcome, not a crash,
    # and it exits 1 like everything else -- the first pilot reported it as a
    # bare "claude_exit_1", which hides the one fact that explains the run.
    # It still earns no prediction: like a timeout, it leaves a mid-edit tree.
    if result.get("terminal_reason") == "max_turns" or result.get("subtype") == "error_max_turns":
        return "max_turns"
    if code in (401, 403) or AUTH_ERROR_RE.search(text):
        return "auth_failed"
    if code == 429 or looks_like_usage_limit(text):
        return "usage_limit"
    if code or result.get("terminal_reason") == "api_error":
        return "api_error"
    return f"claude_exit_{rc}"


# --------------------------------------------------------------------------
# pure helpers (no Docker, no Claude -- these are what the unit tests cover)
# --------------------------------------------------------------------------

def remote_image_key(instance_id: str) -> str:
    """The published evaluation image for an instance."""
    return (DOCKER_IMAGE_PREFIX + "sweb.eval.x86_64."
            + instance_id.replace("__", "_s_")).lower()


def parse_markers(text: str, marker: str = MARKER) -> dict:
    """Collect `MARKERkey=value` lines out of a container's output."""
    found = {}
    for line in text.splitlines():
        line = line.strip()
        idx = line.find(marker)
        if idx < 0:
            continue
        payload = line[idx + len(marker):].strip()
        if "=" in payload:
            key, value = payload.split("=", 1)
            found[key.strip()] = value.strip()
    return found


def parse_porcelain_z(data) -> list:
    """Parse `git status --porcelain -z -uall` into (code, path) pairs."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    fields = data.split("\0")
    entries, i = [], 0
    while i < len(fields):
        item = fields[i]
        i += 1
        if not item:
            continue
        code, path = item[:2], item[3:]
        if code[0] in ("R", "C"):
            i += 1  # -z emits the rename/copy source as the next field
        entries.append((code, path))
    return entries


def is_untracked(code: str) -> bool:
    return code.strip() == "??"


def is_scratch(path: str) -> bool:
    """`.DS_Store` anywhere, or a brand-new file dropped at the repo root.

    Agents like to leave benchmark_foo.py / notes.md next to setup.py. Nothing
    at the top level is part of an optimization, and a stray file there can
    break the package build on the evaluation image.
    """
    return Path(path).name == ".DS_Store" or "/" not in path


def classify_worktree(entries) -> dict:
    """Decide what to do with every path `git status` reports.

    Order matters: a change to `tests/` is reverted as a test edit even if it
    also looks like build output, because rule 1 is the one being enforced.

    Returns lists of paths under four keys -- `revert` (restore the baseline
    copy), `remove` (delete an untracked file), `new_files` (untracked, kept in
    the patch, reported for review) -- plus the reason each path was filtered,
    so meta can say *why* something was dropped.
    """
    plan = {"revert": [], "remove": [], "new_files": [], "reasons": {}}

    def drop(path, reason):
        plan["reasons"][path] = reason

    for code, path in entries:
        untracked = is_untracked(code)
        if TEST_PATH_RE.search(path):
            reason = "test_file"
        elif BUILD_ARTIFACT_RE.search(path):
            reason = "build_artifact"
        elif untracked and is_scratch(path):
            reason = "scratch"
        else:
            if untracked:
                plan["new_files"].append(path)
            continue
        drop(path, reason)
        (plan["remove"] if untracked else plan["revert"]).append(path)
    return plan


def filtered_by(plan: dict, reason: str) -> list:
    """Paths the plan dropped for one reason, in the order they were seen."""
    return [p for p in plan["revert"] + plan["remove"]
            if plan["reasons"].get(p) == reason]


def parse_numstat_binaries(numstat: str) -> list:
    """Paths `git diff --cached --numstat` reports as binary (`-  -  path`)."""
    binaries = []
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "-" and parts[1] == "-":
            binaries.append(parts[-1])
    return binaries


def build_baseline_script(repo_dir: str = REPO_DIR) -> str:
    """Bash that turns /testbed into a repository with exactly one commit.

    The tree is committed *as the image ships it*, after its setup step has
    run. Several images leave /testbed dirty that way (astropy rewrites
    pyproject.toml during install), and that state is what the evaluation
    container will also have, so it is the only correct baseline: anything
    else would put the image's own edits into the model's patch and make the
    patch fail to apply at evaluation time.

    `git add -A -f` is deliberate. Some repos track files their own .gitignore
    matches (generated version modules, for one); without -f those land on
    disk but never enter the baseline, so a later edit to them is silently
    dropped from the diff. __pycache__ and .pyc are deleted first instead of
    being committed, because they are pure churn that Python regenerates --
    compiled extensions are NOT deleted, since the package needs them to
    import, and they are filtered at extraction instead.
    """
    return "\n".join([
        "set -e",
        f"cd {repo_dir}",
        "find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true",
        "find . -name '*.pyc' -delete 2>/dev/null || true",
        "find . -name '*.pyo' -delete 2>/dev/null || true",
        "rm -rf .git",
        "git -c init.defaultBranch=main init -q .",
        "git config user.name base",
        "git config user.email base@local",
        "git config commit.gpgsign false",
        "git add -A -f",
        # The message stays a bare "base": writing the upstream SHA into the
        # working copy would hand the agent an identifier for the real commit.
        "git commit -q --no-verify -m base",
        # A fresh init has no prior history in its reflog by construction;
        # dropping the log as well leaves literally nothing to walk back to.
        "rm -rf .git/logs",
    ])


def build_verify_script(repo_dir: str = REPO_DIR) -> str:
    """Bash that reports the isolation invariants for parse_baseline_report."""
    return "\n".join([
        f"cd {repo_dir}",
        f"echo '{MARKER}commits='$(git log --all --oneline | wc -l | tr -d ' ')",
        f"echo '{MARKER}tags='$(git tag | wc -l | tr -d ' ')",
        f"echo '{MARKER}remotes='$(git remote | wc -l | tr -d ' ')",
        f"echo '{MARKER}reflog='$(git reflog 2>/dev/null | wc -l | tr -d ' ')",
        f"echo '{MARKER}dirty='$(git status --porcelain | wc -l | tr -d ' ')",
        f"echo '{MARKER}head='$(git rev-parse HEAD)",
        "git status --porcelain | head -20",
    ])


def parse_baseline_report(text: str) -> dict:
    """Turn the verify script's marker lines into ints (and the HEAD sha)."""
    raw = parse_markers(text)
    report = {}
    for key in ("commits", "tags", "remotes", "reflog", "dirty"):
        try:
            report[key] = int(raw[key])
        except (KeyError, ValueError):
            report[key] = None
    report["head"] = raw.get("head")
    return report


def baseline_problems(report: dict) -> list:
    """Everything wrong with the isolated checkout; empty means it is clean."""
    problems = []
    checks = [
        ("commits", 1, "git log --all shows {got} commit(s), expected exactly 1"),
        ("tags", 0, "{got} tag(s) present, expected none"),
        ("remotes", 0, "{got} remote(s) present, expected none"),
        ("reflog", 0, "{got} reflog entr(ies) present, expected none"),
        ("dirty", 0, "{got} uncommitted path(s) in the baseline, expected none"),
    ]
    for key, expected, template in checks:
        got = report.get(key)
        if got is None:
            problems.append(f"could not read `{key}` from the container")
        elif got != expected:
            problems.append(template.format(got=got))
    if not report.get("head"):
        problems.append("baseline commit has no HEAD")
    return problems


def redact(text, secret):
    """Strip the OAuth token out of anything on its way to disk.

    Nothing should print it, but a single `env` in the transcript would put a
    live credential in the run's artifacts. Cheap insurance, applied to every
    byte the script writes.
    """
    if not text:
        return text or ""
    if secret:
        text = text.replace(secret, "<REDACTED_OAUTH_TOKEN>")
    return text


def problem_statement_for(inst):
    for key in ("problem_statement_realistic", "problem_statement"):
        if inst.get(key):
            return inst[key]
    raise KeyError(f"no problem statement on {inst['instance_id']}")


def build_prompt(inst):
    """The dataset's own text, verbatim, with nothing added."""
    return VERBATIM_TEMPLATE.format(problem_statement=problem_statement_for(inst))


def sha256_16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:PROMPT_HASH_LEN]


def parse_result_line(stdout: str) -> Optional[dict]:
    """Pull the fields we care about out of the stream's final `result` message."""
    keep = ("num_turns", "total_cost_usd", "usage", "duration_ms", "is_error",
            "subtype", "api_error_status", "terminal_reason")
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            found = {k: obj[k] for k in keep if k in obj}
            if "result" in obj and isinstance(obj["result"], str):
                found["result_text"] = obj["result"][:2000]
            return found
    return None


def _walk_tool_uses(node, out):
    if isinstance(node, dict):
        if node.get("type") == "tool_use":
            out.append(node)
        for value in node.values():
            _walk_tool_uses(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_tool_uses(value, out)


def stream_objects(stdout: str):
    """Yield the JSON objects out of a stream-json transcript, skipping noise.

    A killed run's transcript ends mid-line, and the CLI interleaves plain text
    on the same stream, so anything unparseable is simply passed over.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def count_assistant_turns(stdout: str) -> int:
    """Assistant messages in the transcript.

    This is the fallback count: when `timeout` kills the CLI there is no final
    `result` line and therefore no `num_turns`, and the transcript is the only
    surviving record of how far the agent actually got.
    """
    return sum(1 for obj in stream_objects(stdout) if obj.get("type") == "assistant")


def count_tool_calls(stdout: str) -> int:
    """Every tool_use block the agent emitted, over the whole transcript."""
    total = 0
    for obj in stream_objects(stdout):
        uses = []
        _walk_tool_uses(obj, uses)
        total += len(uses)
    return total


def turn_count(result, stdout: str):
    """(turns, source) -- the CLI's own count when the run reported one.

    A run that was killed never printed a result line, so the number is counted
    off the transcript instead and says so: the two are not measured the same
    way and must not be compared as if they were.
    """
    turns = (result or {}).get("num_turns")
    if isinstance(turns, int):
        return turns, "result_line"
    return count_assistant_turns(stdout), "transcript"


def audit_transcript(stdout: str):
    """Flag tool calls that look like network access or history spelunking."""
    hits = []
    for obj in stream_objects(stdout):
        uses = []
        _walk_tool_uses(obj, uses)
        for use in uses:
            try:
                blob = json.dumps(use.get("input", {}))
            except (TypeError, ValueError):
                blob = str(use.get("input", ""))
            low = blob.lower()
            for pattern in AUDIT_PATTERNS:
                idx = low.find(pattern)
                if idx < 0:
                    continue
                hits.append({
                    "pattern": pattern,
                    "tool": use.get("name", "?"),
                    "snippet": blob[max(0, idx - 40): idx + 120],
                })
                if len(hits) >= 25:
                    return hits
    return hits


def looks_like_usage_limit(*texts):
    return any(USAGE_LIMIT_RE.search(t) for t in texts if t)


def env_name_for(test_spec) -> str:
    """The conda env the benchmark's own eval script activates.

    Same rule regression_check.py uses; duplicated rather than imported so the
    generation driver depends only on evaluation/, never on analysis/.
    """
    match = re.search(r"conda activate (\S+)", test_spec.eval_script)
    return match.group(1) if match else "testbed"


def agent_environment(env_name: str, token: str = "") -> dict:
    """Environment for every exec run as the agent user.

    PATH puts the instance's conda env first so `python` and `pytest` resolve
    to the packages the benchmark installed -- that is the whole reason for
    running in the image, and it means the prompt never has to mention it.

    HOME has to be stated explicitly rather than left to `docker exec -u`: git
    reads its global config from $HOME, and the image's own environment points
    at /root, which the agent user cannot read.

    The token is added only when one is passed, so the setup and extraction
    execs never carry a credential they have no use for.
    """
    conda_bin = f"{CONDA_ROOT}/envs/{env_name}/bin"
    path = ":".join([
        conda_bin,
        f"{CONDA_ROOT}/condabin",
        f"{CLI_MOUNT}/npm/bin",
        CLI_NODE_BIN,
        "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin",
    ])
    env = {
        "HOME": AGENT_HOME,
        "USER": AGENT_USER,
        "LOGNAME": AGENT_USER,
        "SHELL": "/bin/bash",
        "PATH": path,
        "CONDA_PREFIX": f"{CONDA_ROOT}/envs/{env_name}",
        "CONDA_DEFAULT_ENV": env_name,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        # Pins the run to the version recorded in provenance; the mount is
        # read-only, so an update attempt would only produce noise anyway.
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CONFIG_DIR": f"{AGENT_HOME}/.claude",
    }
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def claude_command(prompt: str, model: str, effort: str, max_turns: int, timeout: int) -> list:
    """argv for the in-container Claude run.

    A list, not a shell string: the prompt is dataset text and must reach the
    CLI byte-for-byte, with no quoting layer in between.

    `env -u` rather than an empty value, so ANTHROPIC_API_KEY and
    ANTHROPIC_AUTH_TOKEN are genuinely absent and a subscription run can never
    silently fall through to per-token API billing.

    `timeout` enforces the wall clock inside the container, where it can
    actually reach the process group; the host-side join is only a backstop.
    """
    return [
        "/usr/bin/timeout", "--kill-after=30", "--signal=TERM", str(timeout),
        "/usr/bin/env", "-u", "ANTHROPIC_API_KEY", "-u", "ANTHROPIC_AUTH_TOKEN",
        CLI_BIN,
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--allowedTools", *ALLOWED_TOOLS,
        "--disallowedTools", *DISALLOWED_TOOLS,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        "--model", model,
        "--effort", effort,
    ]


# --------------------------------------------------------------------------
# docker plumbing
# --------------------------------------------------------------------------

def quiet_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def container_exec(container, cmd, *, user=None, environment=None, workdir=None,
                   timeout=None):
    """Run a command in a container; return (exit_code, stdout, stderr, timed_out).

    docker_utils.exec_run_with_timeout cannot be used here: it offers no way to
    pick the user, the working directory, or the environment, and it merges
    stderr into stdout. All three matter -- Claude must run as a non-root user
    with the token in its environment, and its stream-json stdout has to stay
    clean for the transcript.
    """
    api = container.client.api
    exec_id = api.exec_create(
        container.id, cmd,
        user=user or "",
        environment=environment or None,
        workdir=workdir,
        stdout=True, stderr=True,
    )["Id"]

    out_chunks, err_chunks = [], []
    failure = {}

    def pump():
        try:
            for stdout_chunk, stderr_chunk in api.exec_start(exec_id, stream=True, demux=True):
                if stdout_chunk:
                    out_chunks.append(stdout_chunk)
                if stderr_chunk:
                    err_chunks.append(stderr_chunk)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            failure["error"] = exc

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    thread.join(timeout)
    timed_out = thread.is_alive()
    if timed_out:
        _kill_exec(container, exec_id)
        thread.join(60)
    if failure.get("error") and not timed_out:
        raise failure["error"]

    try:
        code = api.exec_inspect(exec_id).get("ExitCode")
    except Exception:  # noqa: BLE001 - the exec may already be gone
        code = None
    stdout = b"".join(out_chunks).decode("utf-8", "replace")
    stderr = b"".join(err_chunks).decode("utf-8", "replace")
    return code, stdout, stderr, timed_out


def _kill_exec(container, exec_id):
    """Best effort stop of a runaway exec and anything it spawned."""
    try:
        pid = container.client.api.exec_inspect(exec_id).get("Pid") or 0
        if pid:
            container.exec_run(f"kill -TERM {pid}", user="root")
    except Exception:  # noqa: BLE001
        pass
    # The exec PID lives in the daemon's namespace, not the container's, so the
    # kill above often misses. Matching on the binary path inside this
    # single-purpose container is the reliable route.
    for signal_name in ("TERM", "KILL"):
        try:
            container.exec_run(f"pkill -{signal_name} -f {CLI_BIN}", user="root")
        except Exception:  # noqa: BLE001
            pass


def sh(container, script, *, user=None, environment=None, timeout=300):
    """Run a bash snippet (login-free) and return (code, stdout, stderr, timed_out)."""
    return container_exec(
        container, ["/bin/bash", "-c", script],
        user=user, environment=environment, timeout=timeout,
    )


def read_container_bytes(container, path):
    """Exact bytes of a file in the container, or None if it is not there."""
    try:
        stream, _ = container.get_archive(path)
    except docker.errors.NotFound:
        return None
    buf = io.BytesIO()
    for chunk in stream:
        buf.write(chunk)
    buf.seek(0)
    with tarfile.open(fileobj=buf) as tar:
        member = next((m for m in tar.getmembers() if m.isfile()), None)
        if member is None:
            return None
        extracted = tar.extractfile(member)
        return extracted.read() if extracted else None


def write_container_bytes(container, path, data: bytes):
    """Place exact bytes at `path` in the container.

    docker_utils.copy_to_container names the tar member after the *source*
    file, so it cannot write a chosen destination name, and it needs a real
    file on the host. This takes the bytes directly.
    """
    directory, _, name = path.rpartition("/")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    container.exec_run(f"mkdir -p {directory or '/'}", user="root")
    container.put_archive(directory or "/", buf.getvalue())


def ensure_image(client, image, quiet=False):
    """Make sure the published evaluation image is present locally.

    Images are only ever pulled here, never removed: they are several GB each
    and re-pulling one costs far more than the disk it occupies.
    """
    try:
        client.images.get(image)
        return False
    except docker.errors.ImageNotFound:
        pass
    if not quiet:
        print(f"[docker] pulling {image} (first use; several GB)")
    client.images.pull(image)
    return True


def remove_stale_container(client, name):
    try:
        existing = client.containers.get(name)
    except docker.errors.NotFound:
        return
    cleanup_container(client, existing, "quiet")


def create_instance_container(client, spec, run_id, cli_cache: Path):
    """A fresh container on the evaluation image, with the CLI cache mounted.

    docker_build.build_container() is the harness's own helper and would be the
    natural call here, but it takes no `volumes` argument and evaluation/ is
    off limits for this change. Every other setting is imported from it rather
    than restated, so the Mac clamps and the platform stay in lockstep with
    what the evaluation run uses:

      * platform  -- spec.platform, which test_spec.py pins to linux/x86_64
      * nano_cpus -- the repo's own value, clamped by LOCAL_MAX_NANO_CPUS
      * mem_limit -- LOCAL_MAX_MEM_LIMIT
      * network   -- USE_HOST_NETWORK

    The container itself runs as root so setup can chown /testbed and write the
    agent's shell profile; Claude is exec'd into it as AGENT_USER.
    """
    config = MAP_REPO_VERSION_TO_SPECS[spec.repo][spec.version]
    nano_cpus = min(config.get("nano_cpus", int(2e9)), LOCAL_MAX_NANO_CPUS)
    name = spec.get_instance_container_name(run_id)
    remove_stale_container(client, name)
    return client.containers.create(
        image=spec.instance_image_key,
        name=name,
        user="root",
        detach=True,
        command="tail -f /dev/null",
        nano_cpus=nano_cpus,
        platform=spec.platform,
        network_mode="host" if USE_HOST_NETWORK else None,
        mem_limit=LOCAL_MAX_MEM_LIMIT,
        oom_kill_disable=False,
        oom_score_adj=1000,
        volumes={str(cli_cache): {"bind": CLI_MOUNT, "mode": "ro"}},
    )


# --------------------------------------------------------------------------
# the Linux x86-64 Claude Code cache
# --------------------------------------------------------------------------

BOOTSTRAP_SCRIPT = """
set -e
CACHE={mount}
rm -rf "$CACHE"/node "$CACHE"/npm "$CACHE"/home "$CACHE"/{manifest}
mkdir -p "$CACHE"/node "$CACHE"/npm "$CACHE"/home
cd /tmp
echo "downloading node v{node_version} (linux-x64)"
curl -fsSL -o node.tar.gz "{node_tarball}"
tar xzf node.tar.gz -C "$CACHE"/node --strip-components=1
rm -f node.tar.gz
export PATH="$CACHE/node/bin:$PATH"
export HOME="$CACHE/home"
export npm_config_prefix="$CACHE/npm"
export npm_config_cache="$CACHE/home/.npm"
echo "installing {package}@{cli_version}"
npm install -g --no-fund --no-audit --loglevel=error "{package}@{cli_version}"
rm -rf "$CACHE/home/.npm"
echo '{marker}node_version='$(node --version)
echo '{marker}npm_version='$(npm --version)
echo '{marker}package_version='$(node -p \
  "require('$CACHE/npm/lib/node_modules/{package}/package.json').version")
echo '{marker}uname='$(uname -m)
# Run the binary for real, outside a command substitution, so `set -e` catches a
# crash here rather than letting the error text through as a "version".
# On Apple Silicon this is where an emulation problem surfaces: the linux-x64
# build uses AVX, and a Docker Desktop configured without Rosetta cannot run it.
"$CACHE"/npm/bin/claude --version > /tmp/cli_version.txt 2>&1
"$CACHE"/npm/bin/claude --help > "$CACHE"/claude-help.txt 2>&1
test -s "$CACHE"/claude-help.txt
echo '{marker}cli_version='$(head -1 /tmp/cli_version.txt)
# The auth variable is read straight out of the binary that will actually run,
# rather than assumed from whatever version is installed on the host. The same
# probe covers options that are real but undocumented -- --max-turns is in the
# binary and works, yet `claude --help` never lists it, so asserting on the
# help text would reject a perfectly good build. The control string must score
# 0: a grep that silently matches everything cannot pass this off as success.
# `|| true` rather than `|| echo 0`: grep -c prints the count and *then* exits 1
# when it is zero, so "|| echo 0" would make the marker read "0 0".
BIN="$CACHE"/npm/lib/node_modules/{package}/bin/claude.exe
echo '{marker}oauth_var='$(grep -ac CLAUDE_CODE_OAUTH_TOKEN "$BIN" 2>/dev/null || true)
echo '{marker}flag_max_turns='$(grep -ac -- 'max-turns' "$BIN" 2>/dev/null || true)
echo '{marker}grep_control='$(grep -ac -- 'definitely-not-a-claude-flag' "$BIN" 2>/dev/null || true)
"""


HELP_NAME = "claude-help.txt"


def verify_cli_capabilities(help_text: str, effort: str) -> list:
    """Check the flags this driver depends on against the CLI's own --help.

    The CLI in the container is not the one on the host and moves fast, so the
    contract is confirmed against the installed build rather than assumed:
    `--effort` with the level we ask for, the flags the run is built around,
    and `setup-token`, which is how the subscription credential this driver
    passes in CLAUDE_CODE_OAUTH_TOKEN is minted in the first place.
    """
    problems = []
    if not help_text.strip():
        return ["`claude --help` produced no output in the container"]
    # --max-turns is deliberately absent from this list: it is a working but
    # undocumented option, checked against the binary by verify_hidden_flags.
    for flag in ("--effort", "--model", "--output-format",
                 "--disallowedTools", "--dangerously-skip-permissions", "--print"):
        if flag not in help_text:
            problems.append(f"the installed CLI does not list {flag}")
    if "--effort" in help_text and effort not in help_text:
        problems.append(f"the installed CLI does not list an effort level named "
                        f"{effort!r}; `claude --help` describes --effort as: "
                        + _help_line(help_text, "--effort"))
    if "setup-token" not in help_text:
        problems.append("the installed CLI has no `setup-token` command, so "
                        "CLAUDE_CODE_OAUTH_TOKEN may no longer be the right "
                        "way to authenticate")
    return problems


def _marker_count(markers: dict, key: str) -> int:
    """A marker's value as a count; anything unparseable reads as zero."""
    try:
        return int(str(markers.get(key, "")).strip())
    except (TypeError, ValueError):
        return 0


def verify_hidden_flags(markers: dict) -> list:
    """Check options that work but are absent from `claude --help`.

    `--max-turns` is the case that matters: the CLI accepts it in print mode,
    but it is not in the visible option list, so asserting on the help text
    would reject a good build. Its presence in the binary is the signal
    instead -- guarded by a control string that must NOT be found, so a grep
    matching everything cannot masquerade as a pass.
    """
    control = markers.get("grep_control")
    if control is None or control == "":
        return ["the binary probe did not run: no control count came back, so a "
                "positive result for any flag would mean nothing"]
    if str(control) != "0":
        return [f"the binary probe is unreliable: a control string that should "
                f"appear 0 times was reported as {control!r}"]
    problems = []
    for key, flag in (("flag_max_turns", "--max-turns"),):
        if _marker_count(markers, key) <= 0:
            problems.append(f"{flag} does not appear in the installed CLI binary")
    return problems


def _help_line(help_text: str, flag: str) -> str:
    """The flag's own help entry, flattened, for a useful error message."""
    lines = help_text.splitlines()
    for i, line in enumerate(lines):
        if flag in line:
            return " ".join(x.strip() for x in lines[i:i + 3])
    return "(not found)"


def cli_cache_manifest(cli_cache: Path) -> Optional[dict]:
    path = cli_cache / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def bootstrap_cli(client, cli_cache: Path, image: str, cli_version: str,
                  effort: str = "xhigh", force=False):
    """Install Node + the Claude Code CLI into the host-side cache, once.

    Why a mounted, pre-installed copy rather than installing per instance:
    the npm package is a thin wrapper whose postinstall drops a ~hundreds-of-MB
    native binary into place. Doing that inside all 140 containers would
    re-download it 140 times, put the run at the mercy of a registry hiccup
    mid-sweep, and -- worst for a benchmark -- let the CLI version drift
    between instances. Building it once into a directory that every container
    bind-mounts read-only fixes the version for the whole sweep, costs nothing
    per instance, and keeps the agent from being able to modify its own binary.

    The build runs inside a linux/amd64 container so the installer resolves
    `linux-x64` and links against the same glibc the evaluation images use --
    an npm install on this arm64 macOS host would fetch the darwin-arm64
    binary, which cannot run in the container at all.

    Node ends up in the cache because npm needs it to run that postinstall.
    After it, `claude` is the native binary and Node is never invoked again.
    """
    existing = cli_cache_manifest(cli_cache)
    if existing and not force:
        help_path = cli_cache / HELP_NAME
        problems = verify_cli_capabilities(
            help_path.read_text(errors="replace") if help_path.exists() else "", effort)
        if "--max-turns" not in (existing.get("hidden_flags_verified") or []):
            problems.append("this cache predates the --max-turns binary check")
        if problems:
            raise RuntimeError(
                "the cached Claude Code CLI does not support what this run asks "
                "of it:\n  - " + "\n  - ".join(problems)
                + "\n\nRebuild the cache with --bootstrap_cli."
            )
        return existing, False

    cli_cache.mkdir(parents=True, exist_ok=True)
    ensure_image(client, image)
    print(f"[cli] building the Linux x86-64 Claude Code cache in {cli_cache}")
    container = client.containers.create(
        image=image,
        user="root",
        detach=True,
        command="tail -f /dev/null",
        platform="linux/x86_64",
        mem_limit=LOCAL_MAX_MEM_LIMIT,
        volumes={str(cli_cache): {"bind": CLI_MOUNT, "mode": "rw"}},
    )
    try:
        container.start()
        script = BOOTSTRAP_SCRIPT.format(
            mount=CLI_MOUNT, manifest=MANIFEST_NAME, node_version=NODE_VERSION,
            node_tarball=NODE_TARBALL, package=CLI_NPM_PACKAGE,
            cli_version=cli_version, marker=MARKER,
        )
        code, out, err, timed_out = sh(container, script, timeout=1800)
        if timed_out or code != 0:
            raise RuntimeError(
                "Claude Code CLI bootstrap failed "
                f"({'timed out' if timed_out else f'exit {code}'}).\n"
                + (out + err)[-4000:]
            )
        markers = parse_markers(out)
        if not markers.get("cli_version"):
            raise RuntimeError(
                "the CLI installed but `claude --version` produced nothing.\n"
                "On Apple Silicon this usually means the linux-x64 build hit an\n"
                "unsupported instruction under emulation -- see the Rosetta/AVX\n"
                "note in README_ClaudeCode.md.\n" + (out + err)[-4000:]
            )
        help_text = (cli_cache / HELP_NAME).read_text(errors="replace") \
            if (cli_cache / HELP_NAME).exists() else ""
        capability_problems = verify_cli_capabilities(help_text, effort)
        if capability_problems:
            raise RuntimeError(
                "the Claude Code CLI installed in the container does not "
                "support what this driver asks of it:\n  - "
                + "\n  - ".join(capability_problems)
            )
        hidden_problems = verify_hidden_flags(markers)
        if hidden_problems:
            raise RuntimeError(
                "the Claude Code CLI installed in the container is missing an "
                "option this driver passes:\n  - " + "\n  - ".join(hidden_problems)
            )
        if _marker_count(markers, "oauth_var") <= 0:
            raise RuntimeError(
                "CLAUDE_CODE_OAUTH_TOKEN does not appear in the installed CLI "
                "binary; its authentication contract has changed and this "
                "driver's token handling needs to be revisited."
            )
        manifest = {
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "built_in_image": image,
            "node_version": markers.get("node_version"),
            "npm_version": markers.get("npm_version"),
            "npm_package": CLI_NPM_PACKAGE,
            "npm_package_version": markers.get("package_version"),
            "claude_version": markers.get("cli_version"),
            "container_arch": markers.get("uname"),
            "effort_verified": effort,
            "hidden_flags_verified": ["--max-turns"],
            "help_file": HELP_NAME,
            "oauth_env_var": "CLAUDE_CODE_OAUTH_TOKEN",
        }
        (cli_cache / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[cli] {manifest['claude_version']} "
              f"(node {manifest['node_version']}, {CLI_NPM_PACKAGE}@"
              f"{manifest['npm_package_version']})")
        return manifest, True
    finally:
        cleanup_container(client, container, "quiet")


# --------------------------------------------------------------------------
# per-instance work
# --------------------------------------------------------------------------

def setup_agent_user(container, env_name, timeout=600):
    """Non-root user, writable /testbed, conda active in its shell.

    /testbed is chowned to the agent before git touches it so the repository is
    created by, and owned by, the same user Claude runs as -- otherwise git
    refuses to work in it ("dubious ownership") and the index is unwritable.
    """
    script = "\n".join([
        "set -e",
        f"id -u {AGENT_USER} >/dev/null 2>&1 || "
        f"adduser --disabled-password --gecos 'agent' {AGENT_USER}",
        f"mkdir -p {AGENT_HOME}",
        f"chown -R {AGENT_USER}:{AGENT_USER} {AGENT_HOME}",
        f"chown -R {AGENT_USER}:{AGENT_USER} {REPO_DIR}",
        # An interactive or login shell Claude opens lands in the same env its
        # own process already has, so `python` and `pytest` work everywhere.
        f"cat > {AGENT_HOME}/.bashrc <<'EOF_BASHRC'",
        f"source {CONDA_ROOT}/etc/profile.d/conda.sh",
        f"conda activate {env_name}",
        f"export PATH={CLI_MOUNT}/npm/bin:$PATH",
        "EOF_BASHRC",
        f"cat > {AGENT_HOME}/.profile <<'EOF_PROFILE'",
        f"[ -f {AGENT_HOME}/.bashrc ] && . {AGENT_HOME}/.bashrc",
        "EOF_PROFILE",
        f"chown {AGENT_USER}:{AGENT_USER} {AGENT_HOME}/.bashrc {AGENT_HOME}/.profile",
        f"su {AGENT_USER} -c 'git config --global --add safe.directory {REPO_DIR}'",
        f"su {AGENT_USER} -c 'git config --global init.defaultBranch main'",
    ])
    return sh(container, script, user="root", timeout=timeout)


def check_auth_env_clean(container, env_name, token, timeout=60):
    """Confirm the API-key variables really are absent inside the container."""
    cmd = ["/usr/bin/env", "-u", "ANTHROPIC_API_KEY", "-u", "ANTHROPIC_AUTH_TOKEN",
           "/bin/bash", "-c",
           # Reports presence only. Echoing a credential's *value* to find out
           # whether it is set would be the very leak this check exists to rule out.
           f'echo "{MARKER}api_key=$([ -n "$ANTHROPIC_API_KEY" ] && echo set || echo unset)"; '
           f'echo "{MARKER}auth_token=$([ -n "$ANTHROPIC_AUTH_TOKEN" ] && echo set || echo unset)"; '
           f'echo "{MARKER}oauth_present=$([ -n "$CLAUDE_CODE_OAUTH_TOKEN" ] && echo yes || echo no)"']
    _, out, _, _ = container_exec(
        container, cmd, user=AGENT_USER,
        environment=agent_environment(env_name, token), timeout=timeout,
    )
    markers = parse_markers(out)
    problems = []
    if markers.get("api_key") != "unset":
        problems.append("ANTHROPIC_API_KEY is set inside the container")
    if markers.get("auth_token") != "unset":
        problems.append("ANTHROPIC_AUTH_TOKEN is set inside the container")
    if markers.get("oauth_present") != "yes":
        problems.append("CLAUDE_CODE_OAUTH_TOKEN did not reach the container")
    return problems


def establish_baseline(container, environment, timeout=1800):
    """Single-commit /testbed, then verify it. Returns (report, problems, log)."""
    code, out, err, timed_out = sh(container, build_baseline_script(),
                                   user=AGENT_USER, environment=environment,
                                   timeout=timeout)
    if timed_out or code != 0:
        reason = "timed out" if timed_out else f"exit {code}"
        return {}, [f"baseline commit failed ({reason})"], (out + err)[-4000:]
    vcode, vout, verr, vtimed = sh(container, build_verify_script(),
                                   user=AGENT_USER, environment=environment,
                                   timeout=300)
    if vtimed or vcode != 0:
        reason = "timed out" if vtimed else f"exit {vcode}"
        return {}, [f"baseline verification failed ({reason})"], (vout + verr)[-4000:]
    report = parse_baseline_report(vout)
    return report, baseline_problems(report), vout[-4000:]


def container_cli_version(container, env_name, timeout=120):
    """The CLI version reported by the binary that will actually run.

    Asked without the token: printing a version needs no credential, and the
    fewer execs carry one the smaller the surface for it to escape through.
    """
    _, out, err, _ = container_exec(
        container, [CLI_BIN, "--version"], user=AGENT_USER,
        environment=agent_environment(env_name), timeout=timeout,
    )
    text = (out or err).strip().splitlines()
    return text[0].strip() if text else "unknown"


def run_claude(container, prompt, env_name, token, args, log_dir, iid):
    """Run headless Claude Code in the container, saving the full transcript.

    Returns (exit_code, timeout_error, claude_result, stdout, stderr). stdout
    and stderr are redacted before they are written or returned.
    """
    cmd = claude_command(prompt, args.model, args.effort, args.max_turns, args.timeout)
    code, stdout, stderr, host_timeout = container_exec(
        container, cmd,
        user=AGENT_USER,
        environment=agent_environment(env_name, token),
        workdir=REPO_DIR,
        # `timeout` inside the container owns the wall clock; the host join is
        # only there in case the exec stream itself wedges.
        timeout=args.timeout + 180,
    )
    stdout, stderr = redact(stdout, token), redact(stderr, token)

    # 124 is coreutils `timeout` giving up; 137 is the follow-up SIGKILL.
    err = None
    if host_timeout:
        err = f"exec stream stalled past {args.timeout + 180}s"
    elif code in (124, 137):
        err = f"timeout after {args.timeout}s"
    if err:
        stderr += f"\n--- TIMEOUT: {err} ---\n"

    (log_dir / f"{iid}.jsonl").write_text(stdout)
    (log_dir / f"{iid}.stderr.txt").write_text(stderr)
    return code, err, parse_result_line(stdout), stdout, stderr


def extract_patch(container, meta, environment, timeout=900):
    """Filter the working tree, then diff it against the baseline commit.

    Everything runs as the agent user, which owns the repository. The diff is
    written to a file and read back as raw bytes rather than scraped off an
    exec stream, so a patch is never corrupted by interleaving.
    """
    def git(script):
        return sh(container, f"cd {REPO_DIR}\n{script}", user=AGENT_USER,
                  environment=environment, timeout=timeout)

    status_path = "/tmp/sweperf_status.z"
    git(f"git status --porcelain -z -uall > {status_path}")
    entries = parse_porcelain_z(read_container_bytes(container, status_path) or b"")
    plan = classify_worktree(entries)

    apply_to_paths(container, plan["revert"], "git checkout --", environment, timeout)
    apply_to_paths(container, plan["remove"], "rm -rf --", environment, timeout)

    meta["reverted_tests"] = filtered_by(plan, "test_file")
    meta["removed_scratch_files"] = filtered_by(plan, "scratch")
    meta["build_artifacts_filtered"] = filtered_by(plan, "build_artifact")
    meta["new_files"] = plan["new_files"]

    # `git add -A` without -f on purpose: .gitignore no longer applies to the
    # paths the baseline tracks, so real edits are staged, while anything new
    # the build dropped in an ignored directory stays out of the diff.
    git("git add -A")

    # A binary file in the diff has no usable hunk -- `git diff` emits only
    # "Binary files ... differ", which `git apply` then refuses. Drop those
    # paths rather than ship a patch that cannot apply.
    _, numstat, _, _ = git("git diff --cached --numstat")
    binaries = parse_numstat_binaries(numstat)
    if binaries:
        apply_to_paths(container, binaries, "git reset -q HEAD --", environment, timeout)
        apply_to_paths(container, binaries, "git checkout --", environment, timeout)
        git("git add -A")
    meta["binary_files_dropped"] = binaries

    patch_path = "/tmp/sweperf_model.patch"
    git(f"git diff --cached > {patch_path}")
    raw = read_container_bytes(container, patch_path) or b""
    return raw.decode("utf-8", "replace")


def apply_to_paths(container, paths, command, environment, timeout=900):
    """Run `command` over `paths` inside the container, NUL-separated.

    Paths come out of the agent's working tree and can contain anything at all
    -- spaces, quotes, newlines. The list is written to a file as raw bytes and
    fed to `xargs -0`, so nothing is ever parsed by a shell.
    """
    if not paths:
        return
    list_path = f"{AGENT_HOME}/.sweperf_paths.z"
    write_container_bytes(container, list_path,
                          b"".join(p.encode("utf-8") + b"\0" for p in paths))
    sh(container,
       f"cd {REPO_DIR} && xargs -0 -r -a {list_path} {command} 2>/dev/null; "
       f"rm -f {list_path}",
       user=AGENT_USER, environment=environment, timeout=timeout)


def process(inst, args, client, log_dir, run_idx, token, cli_cache):
    iid = inst["instance_id"]
    started = time.time()
    meta = {
        "instance_id": iid, "run": run_idx, "status": "ok",
        "execution_env": "docker_with_deps",
        "reverted_tests": [], "removed_scratch_files": [], "new_files": [],
        "build_artifacts_filtered": [], "binary_files_dropped": [],
    }
    container = None
    patch = ""
    logger = quiet_logger(f"claudegen.{iid}")
    try:
        spec = make_test_spec(inst, is_eval=True)
        spec.instance_image_key = remote_image_key(iid)
        env_name = env_name_for(spec)
        meta["image"] = spec.instance_image_key
        meta["conda_env"] = env_name

        prompt = build_prompt(inst)
        meta["prompt_style"] = PROMPT_STYLE
        meta["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()

        ensure_image(client, spec.instance_image_key)
        container = create_instance_container(client, spec, args.run_id, cli_cache)
        container.start()

        code, out, err, timed_out = setup_agent_user(container, env_name)
        if timed_out or code != 0:
            meta["status"] = "setup_failed"
            meta["error"] = f"agent user setup failed: {(out + err)[-1500:]}"
            return None, meta, iid

        auth_problems = check_auth_env_clean(container, env_name, token)
        if auth_problems:
            meta["status"] = "setup_failed"
            meta["error"] = "; ".join(auth_problems)
            return None, meta, iid

        agent_env = agent_environment(env_name)
        report, problems, log = establish_baseline(container, agent_env)
        meta["baseline"] = report
        if problems:
            meta["status"] = "setup_failed"
            meta["error"] = "baseline not isolated: " + "; ".join(problems)
            meta["baseline_log"] = log
            return None, meta, iid

        meta["claude_version"] = container_cli_version(container, env_name)

        rc, err_text, result, stdout, stderr = run_claude(
            container, prompt, env_name, token, args, log_dir, iid
        )
        if result is not None:
            meta["claude_result"] = result
        meta["turns"], meta["turns_source"] = turn_count(result, stdout)
        meta["tool_calls"] = count_tool_calls(stdout)
        suspects = audit_transcript(stdout)
        if suspects:
            meta["web_access_suspect"] = suspects

        if err_text:
            meta["status"] = "timeout"
        elif rc != 0 or (result or {}).get("is_error"):
            meta["status"] = classify_claude_failure(rc, result, stderr)

        patch = extract_patch(container, meta, agent_env)
        if not patch.strip() and meta["status"] == "ok":
            meta["status"] = "empty_patch"
        meta["patch_bytes"] = len(patch)
        meta["files_changed"] = patch.count("diff --git ")
        if earns_a_partial_prediction(meta["status"]):
            meta["partial_patch"] = bool(patch.strip())
            meta["partial_patch_reason"] = (
                f"cut off by {meta['status']}; the diff was filtered the normal "
                f"way but is an unfinished attempt, so it is kept in the partial "
                f"file and left out of the predictions"
            )
    except Exception as exc:  # keep one bad instance from killing the sweep
        meta["status"] = "error"
        meta["error"] = redact(f"{type(exc).__name__}: {exc}", token)
        patch = ""
    finally:
        if container is not None and not args.keep_containers:
            cleanup_container(client, container, logger)
        elif container is not None:
            print(f"  [keep] container left running: {container.name}")
        # Set here, not after the block: the setup paths above return early
        # and the caller prints this field for every outcome.
        meta["duration_s"] = round(time.time() - started, 1)

    record = {
        "instance_id": iid,
        "model_name_or_path": args.model_name,
        "model_patch": patch,
    }
    return record, meta, iid


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def already_done(path: Path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def partial_stem(stem: str) -> str:
    """Name the partial file after the predictions file it shadows.

    `claude_code_pilot2_preds` -> `claude_code_pilot2_partial_preds`, so the
    two sort together and it is obvious at a glance which is which.
    """
    suffix = "_preds"
    if stem.endswith(suffix):
        return stem[: -len(suffix)] + "_partial" + suffix
    return stem + "_partial"


def run_paths(out_path: Path, run_idx: int, num_runs: int):
    """One predictions file per run: run_evaluation.py keys predictions by
    instance_id, so duplicate ids in a single file silently overwrite."""
    if num_runs == 1:
        preds = out_path
    else:
        preds = out_path.with_name(f"{out_path.stem}_run{run_idx}{out_path.suffix}")
    meta = preds.with_name(preds.stem + "_meta.jsonl")
    prov = preds.with_name(preds.stem + "_provenance.jsonl")
    partial = preds.with_name(partial_stem(preds.stem) + preds.suffix)
    return preds, meta, prov, partial


def append_jsonl(path: Path, record: dict):
    """Append one record, opening the file only when there is one to write.

    Deliberately not held open for the sweep: a run with nothing partial should
    not leave an empty file behind suggesting it had something.
    """
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_instances(args):
    """Either a SWE-Perf-style HF dataset or a JSONL of the same records.

    The JSONL form still has to be SWE-Perf shaped -- make_test_spec needs
    repo/version/base_commit/test_patch, and an evaluation image must exist for
    the instance_id, since that image *is* the environment now.
    """
    if args.instances_file:
        path = Path(args.instances_file).resolve()
        if not path.exists():
            sys.exit(f"error: no such instances file: {path}")
        instances = []
        required = ("instance_id", "repo", "base_commit", "version", "test_patch")
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                inst = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"error: {path}:{n} is not valid JSON: {exc}")
            missing = [k for k in required if inst.get(k) in (None, "")]
            if missing:
                sys.exit(f"error: {path}:{n} missing field(s): {', '.join(missing)}")
            if not (inst.get("problem_statement_realistic") or inst.get("problem_statement")):
                sys.exit(f"error: {path}:{n} has no problem_statement")
            instances.append(inst)
        print(f"[input] {len(instances)} instance(s) from {path}")
        return instances
    try:
        from datasets import load_dataset  # imported lazily: --instances_file
    except ImportError:                     # mode needs no HuggingFace deps
        sys.exit("error: the `datasets` package is required to load an HF dataset.\n"
                 "       Activate the project env first:  conda activate sweperf\n"
                 "       (or pass --instances_file to skip HuggingFace entirely)")
    dataset = load_dataset(args.dataset_name, split=args.split)
    print(f"[input] {len(dataset)} instance(s) from {args.dataset_name}:{args.split}")
    return list(dataset)


def execute_run(run_idx, instances, args, out_path, log_root, client, token,
                cli_cache, manifest):
    """Run one sweep, one container at a time. Returns (counts, hit_usage_limit)."""
    preds_path, meta_path, prov_path, partial_path = run_paths(
        out_path, run_idx, args.num_runs)
    log_dir = log_root / f"run{run_idx}"
    log_dir.mkdir(parents=True, exist_ok=True)

    todo = list(instances)
    if args.resume:
        done = already_done(preds_path)
        todo = [i for i in todo if i["instance_id"] not in done]
        print(f"[run {run_idx}] resume: skipping {len(done)} already-written instance(s)")
    if not todo:
        print(f"[run {run_idx}] nothing to do")
        return {}, False

    prompts = {i["instance_id"]: build_prompt(i) for i in todo}

    # Provenance is appended, not overwritten, so a CLI upgrade partway through
    # a resumed run stays visible in the record.
    with open(prov_path, "a") as f:
        f.write(json.dumps({
            "run": run_idx,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "execution_env": "docker_with_deps",
            "image_template": IMAGE_TEMPLATE,
            "images": {i["instance_id"]: remote_image_key(i["instance_id"]) for i in todo},
            "claude_version": manifest.get("claude_version"),
            "claude_cli_source": {
                "npm_package": manifest.get("npm_package"),
                "npm_package_version": manifest.get("npm_package_version"),
                "node_version": manifest.get("node_version"),
                "built_at": manifest.get("built_at"),
                "built_in_image": manifest.get("built_in_image"),
            },
            "model_flag": args.model,
            "model_id": args.model,
            "effort": args.effort,
            "model_name_or_path": args.model_name,
            "allowed_tools": ALLOWED_TOOLS,
            "disallowed_tools": DISALLOWED_TOOLS,
            "max_turns": args.max_turns,
            "prompt_style": PROMPT_STYLE,
            "prompt_source_field": "problem_statement_realistic",
            # Kept a plain string for analysis/provenance_audit.py, which
            # compares this field against a single hash.
            "prompt_sha256": sha256_16(VERBATIM_TEMPLATE),
            "prompt_sha256_by_instance": {
                iid: hashlib.sha256(p.encode()).hexdigest() for iid, p in prompts.items()
            },
            "timeout_s": args.timeout,
            "instances": len(todo),
        }) + "\n")

    print(f"[run {run_idx}/{args.num_runs}] {len(todo)} instance(s), "
          f"one container at a time -> {preds_path}")

    counts = {}
    stop_reason = None
    with open(preds_path, "a") as out_f, open(meta_path, "a") as meta_f:
        for n, inst in enumerate(todo, 1):
            record, meta, iid = process(inst, args, client, log_dir, run_idx,
                                        token, cli_cache)
            status = meta["status"]
            counts[status] = counts.get(status, 0) + 1
            # Anything that did not run cleanly gets a meta line but no
            # prediction, so a later --resume treats it as unfinished.
            wrote_partial = False
            if earns_a_prediction(status) and record is not None:
                out_f.write(json.dumps(record) + "\n"); out_f.flush()
            elif meta.get("partial_patch") and record is not None:
                # A cut-off run still produced a diff. It is kept out of the
                # predictions file -- --resume must still see the instance as
                # unfinished -- but it is not thrown away either.
                append_jsonl(partial_path, record)
                wrote_partial = True
            meta_f.write(json.dumps(meta) + "\n"); meta_f.flush()

            note = (f" (reverted {len(meta['reverted_tests'])} test file(s))"
                    if meta["reverted_tests"] else "")
            if wrote_partial:
                note += f" (partial patch -> {partial_path.name})"
            if meta.get("build_artifacts_filtered"):
                note += f" ({len(meta['build_artifacts_filtered'])} build artifact(s) filtered)"
            if meta.get("web_access_suspect"):
                note += f" (!! {len(meta['web_access_suspect'])} audit hit(s))"
            if meta.get("error"):
                note += f" -- {meta['error'][:200]}"
            # Turns are absent only on the setup paths, which return before
            # Claude is ever started.
            effort_note = ""
            if meta.get("turns") is not None:
                counted = (" counted from transcript"
                           if meta.get("turns_source") == "transcript" else "")
                effort_note = (f", {meta['turns']} turn(s){counted}, "
                               f"{meta.get('tool_calls', 0)} tool call(s)")
            print(f"  [run {run_idx}] [{n}/{len(todo)}] {iid}: "
                  f"{status}, {meta.get('files_changed', 0)} file(s), "
                  f"{meta['duration_s']}s{effort_note}{note}")

            # A bad credential fails every instance identically, so there is
            # nothing to learn from spending a container on each of the rest.
            # This one is not optional: unlike a usage limit, waiting does not
            # fix it.
            if status == "auth_failed":
                stop_reason = "auth_failed"
            elif status == "usage_limit" and args.stop_on_limit:
                stop_reason = "usage_limit"
            if stop_reason:
                remaining = len(todo) - n
                counts["cancelled"] = counts.get("cancelled", 0) + remaining
                label = ("authentication failed" if stop_reason == "auth_failed"
                         else "usage limit hit")
                print(f"  [run {run_idx}] {label} -- stopping with "
                      f"{remaining} instance(s) not started")
                break
    return counts, stop_reason


TIMED_PATH_PATTERN = r"[t]imed_path\.py"


def wait_for_timed_path(poll=60):
    """Never run beside analysis/timed_path.py: two containers do not fit.

    The Mac has 16 GB and runs these x86-64 images under emulation; a second
    container of the same family either fails to start or makes both runs so
    slow that the timing analysis it is doing becomes meaningless.
    """
    first = True
    while True:
        # "[t]imed_path\.py" rather than "timed_path.py": the bracket makes the
        # pattern not match its own text, so a wrapper whose command line
        # mentions this script (a watchdog, a `bash -c` loop) is not mistaken
        # for a running analysis and waited on forever.
        proc = subprocess.run(["pgrep", "-f", TIMED_PATH_PATTERN],
                              capture_output=True, text=True)
        pids = [p for p in proc.stdout.split() if p]
        if not pids:
            if not first:
                print("[wait] timed_path.py finished; continuing")
            return
        if first:
            print(f"[wait] analysis/timed_path.py is running (pid {', '.join(pids)}); "
                  f"waiting for it to finish before starting a container")
            first = False
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_name", default="SWE-Perf/SWE-Perf")
    p.add_argument("--split", default="test")
    p.add_argument("--instances_file", default=None,
                   help="JSONL of SWE-Perf-shaped records instead of an HF dataset; "
                        "each instance still needs a published evaluation image")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="predictions JSONL; meta, provenance and transcripts "
                        "are written beside it")
    p.add_argument("--model_name", default="claude-code",
                   help="value written to model_name_or_path (names the log dir at eval time)")
    p.add_argument("--model", required=True,
                   help="REQUIRED full model ID passed to the claude CLI "
                        "(e.g. claude-opus-5); required so the model can never "
                        "change silently between runs")
    p.add_argument("--effort", default="xhigh",
                   choices=("low", "medium", "high", "xhigh", "max"),
                   help="--effort passed to the claude CLI (default: xhigh)")
    p.add_argument("--max_turns", type=int, default=100,
                   help="--max-turns passed to the claude CLI (agentic turn cap)")
    p.add_argument("--num_runs", type=int, default=1,
                   help="repeat the whole sweep N times; each instance gets a fresh "
                        "container and a fresh claude process, and its own predictions file")
    p.add_argument("--instance_ids", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N instances")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-instance wall clock for the claude run (s)")
    p.add_argument("--run_id", default="claudegen",
                   help="suffix for container names, so a stale container is easy to spot")
    p.add_argument("--cli_cache", default=DEFAULT_CLI_CACHE,
                   help="host directory holding the Linux x86-64 Claude Code install "
                        "that every container mounts read-only")
    p.add_argument("--cli_version", default="latest",
                   help="npm version spec for @anthropic-ai/claude-code in the cache; "
                        "the resolved version is recorded in provenance")
    p.add_argument("--bootstrap_cli", action="store_true",
                   help="rebuild the CLI cache even if it already exists")
    p.add_argument("--bootstrap_image", default=None,
                   help="image to build the CLI cache in (default: the first "
                        "selected instance's evaluation image)")
    p.add_argument("--bootstrap_only", action="store_true",
                   help="build the CLI cache and exit without running any instance")
    p.add_argument("--keep_containers", action="store_true",
                   help="leave each instance container running for debugging")
    p.add_argument("--resume", action="store_true",
                   help="skip instance_ids already written for that run")
    p.add_argument("--stop_on_limit", dest="stop_on_limit", action="store_true",
                   default=True, help="stop the sweep after the first usage limit (default)")
    p.add_argument("--no-stop_on_limit", dest="stop_on_limit", action="store_false",
                   help="keep going after a usage limit instead of stopping")
    p.add_argument("--no-wait-for-timed-path", dest="wait_for_timed_path",
                   action="store_false", default=True,
                   help="start even if analysis/timed_path.py is running (not advised)")
    args = p.parse_args()

    if args.num_runs < 1:
        sys.exit("error: --num_runs must be at least 1")
    if args.max_turns < 1:
        sys.exit("error: --max_turns must be at least 1")

    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if not token:
        sys.exit(
            "error: CLAUDE_CODE_OAUTH_TOKEN is not set.\n"
            "       Claude runs inside the container, which has no access to the\n"
            "       macOS keychain, so the subscription token has to be passed in\n"
            "       explicitly. Create one and export it:\n\n"
            "           claude setup-token\n"
            "           export CLAUDE_CODE_OAUTH_TOKEN='<the token it prints>'\n\n"
            "       It is handed to the container as an environment variable only:\n"
            "       never written to disk, an image, a log, or the provenance file."
        )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cli_cache = Path(args.cli_cache).resolve()
    log_root = out_path.parent / "claude_code_logs"; log_root.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            sys.exit(f"error: unknown instance_ids: {sorted(missing)}")
    if args.limit:
        instances = instances[: args.limit]
    if not instances:
        sys.exit("error: no instances selected")

    if args.wait_for_timed_path:
        wait_for_timed_path()

    client = docker.from_env()
    bootstrap_image = args.bootstrap_image or remote_image_key(instances[0]["instance_id"])
    manifest, built = bootstrap_cli(client, cli_cache, bootstrap_image,
                                    args.cli_version, effort=args.effort,
                                    force=args.bootstrap_cli)
    if not built:
        print(f"[cli] reusing {manifest.get('claude_version')} from {cli_cache}")
    if args.bootstrap_only:
        print(json.dumps(manifest, indent=2))
        return

    print(f"[env] claude (in container): {manifest.get('claude_version')}, "
          f"model: {args.model}, effort: {args.effort}")

    totals, stopped = {}, None
    for run_idx in range(1, args.num_runs + 1):
        counts, stop_reason = execute_run(run_idx, instances, args, out_path,
                                          log_root, client, token, cli_cache, manifest)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        if stop_reason:
            stopped = stop_reason
            break

    print("\nsummary across all runs: " +
          (", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "nothing run"))
    for run_idx in range(1, args.num_runs + 1):
        preds, meta, _, partial = run_paths(out_path, run_idx, args.num_runs)
        if preds.exists():
            print(f"  run {run_idx}: {preds}  (metadata: {meta.name})")
        if partial.exists():
            print(f"  run {run_idx}: partial patches from cut-off runs "
                  f"(not predictions): {partial}")
    print(f"logs: {log_root}")
    if stopped == "auth_failed":
        print("\nSTOPPED: the Claude API rejected the credential (401/403). "
              "CLAUDE_CODE_OAUTH_TOKEN is set but not valid -- it has expired, "
              "been revoked, or was copied incompletely.\n"
              "Mint a fresh one with `claude setup-token`, export it, and rerun "
              "this same command with --resume.\n"
              "No prediction was written for any affected instance, so nothing "
              "has to be cleaned up first.")
    elif stopped == "usage_limit":
        print("\nSTOPPED: hit a Claude usage/rate limit. No prediction was written "
              "for the affected or unstarted instances, so rerun this same command "
              "later with --resume to pick them up.")
    if args.num_runs > 1:
        print("\nnote: evaluate each run separately, giving each its own --run_id "
              "so the evaluation logs don't overwrite each other.")


if __name__ == "__main__":
    main()
