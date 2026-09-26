#!/usr/bin/env python3
"""
Generate SWE-Perf patches with Claude Code (realistic setting).

For each dataset instance this script materializes the repo at its base commit
onto the host, runs headless Claude Code inside that checkout, and captures the
resulting `git diff` as the instance's `model_patch`.

The checkout contains exactly one commit (the base tree, re-committed into a
fresh repository): no history, no tags, no remotes. Claude therefore cannot
read the future of the repo, and the diff is always taken against the base.

Output is a JSONL file in the format expected by evaluation/run_evaluation.py:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

Authentication uses the Claude Code CLI's own login (subscription). Any
ANTHROPIC_API_KEY in the environment is deliberately stripped so runs never
fall through to per-token API billing.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

PROMPT_HASH_LEN = 16

ALLOWED_TOOLS = ["Edit", "Read", "Write", "Glob", "Grep", "Bash"]

# Network egress is denied outright: the experiment must measure what Claude can
# do from the source tree alone, not what it can look up about the upstream fix.
DISALLOWED_TOOLS = ["WebFetch", "WebSearch", "Bash(curl:*)", "Bash(wget:*)"]

# Substrings that, if they show up in a tool_use input, suggest the sandbox was
# probed for network access or repo history. Matched case-insensitively.
AUDIT_PATTERNS = ["http://", "https://", "github.com", "git log --all", "pip download"]

# NOTE: "custom" deliberately deviates from the benchmark's own wording. SWE-Perf
# builds its prompt in generation/oracle/make_datasets/create_text_dataset.py and
# states its four optimization rules there; this template restates those rules in
# our own words and adds the "no dependencies installed" caveat, which is a
# property of our host-checkout setting rather than of the benchmark. Use
# --prompt_style verbatim to send problem_statement_realistic untouched instead,
# which is the apples-to-apples comparison against published SWE-Perf numbers.
PROMPT_TEMPLATE = """You are optimizing this repository for runtime performance.

Rules:
1. Do NOT modify, add, or delete any unit tests. Existing tests must remain unaltered.
2. Preserve the exact behavior and public API of every function you touch.
3. Prioritize maximal efficiency gains where feasible.
4. Make your changes directly in the working tree by editing files.

Note: this checkout does NOT have the project's dependencies installed, so you
generally cannot import the package or run its test suite. Read the code and
reason about complexity and allocation instead.

Task:
{problem_statement}
"""

# --prompt_style verbatim: the problem statement is passed through untouched,
# because problem_statement_realistic already carries the benchmark's own rules.
VERBATIM_TEMPLATE = "{problem_statement}"

# Intentionally does NOT match `testing/`: that is library code in several of the
# benchmark repos (e.g. xarray/testing/assertions.py), not a test suite.
TEST_PATH_RE = re.compile(
    r"(^|/)tests?/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$"
)

USAGE_LIMIT_RE = re.compile(
    r"usage limit|rate limit|rate[_-]limit|limit reached|limit_reached|"
    r"\b429\b|too many requests|overloaded",
    re.IGNORECASE,
)

_mirror_locks: dict = {}
_mirror_locks_guard = threading.Lock()
_write_lock = threading.Lock()


def git(args, cwd=None, check=True):
    """Run a git command and return stdout."""
    proc = subprocess.run(
        ["git"] + args, cwd=cwd, check=False,
        capture_output=True, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def mirror_lock(repo):
    with _mirror_locks_guard:
        return _mirror_locks.setdefault(repo, threading.Lock())


def ensure_mirror(repo, cache_dir, base_commit):
    """Maintain one bare mirror per repo so each instance is built locally."""
    mirror = cache_dir / (repo.replace("/", "__") + ".git")
    with mirror_lock(repo):
        if not mirror.exists():
            print(f"[cache] cloning mirror for {repo} (one time, may be large)")
            git(["clone", "--bare", f"https://github.com/{repo}.git", str(mirror)])
        have = subprocess.run(
            ["git", "-C", str(mirror), "cat-file", "-e", f"{base_commit}^{{commit}}"],
            capture_output=True,
        ).returncode == 0
        if not have:
            print(f"[cache] fetching {base_commit[:8]} into {repo} mirror")
            git(["-C", str(mirror), "fetch", "--tags", "origin",
                 "+refs/heads/*:refs/heads/*"])
    return mirror


def prepare_checkout(repo, base_commit, mirror, dest):
    """Materialize ONLY the base commit's tree in a brand-new repository.

    `git clone` would drag along every commit, tag and remote of the mirror, so
    a capable agent could simply read the upstream optimization out of the future
    history. Instead the base tree is exported with `git archive`, unpacked into
    an empty directory, and committed once into a fresh `git init`. The result
    has one commit, no tags, no remotes, and no path back to the mirror.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # git archive <commit> | tar -x -C <dest>
    archive = subprocess.Popen(
        ["git", "-C", str(mirror), "archive", base_commit],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    untar = subprocess.Popen(
        ["tar", "-x", "-C", str(dest)],
        stdin=archive.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    archive.stdout.close()  # let git see EPIPE if tar dies first
    untar_err = untar.communicate()[1]
    archive_err = archive.stderr.read()
    archive.stderr.close()
    archive.wait()
    if archive.returncode != 0:
        raise RuntimeError(
            f"git archive {base_commit[:8]} failed: "
            f"{archive_err.decode('utf-8', 'replace').strip()}"
        )
    if untar.returncode != 0:
        raise RuntimeError(
            f"tar -x failed for {base_commit[:8]}: "
            f"{untar_err.decode('utf-8', 'replace').strip()}"
        )

    ident = ["-c", "user.name=base", "-c", "user.email=base@local",
             "-c", "commit.gpgsign=false"]
    git(["-c", "init.defaultBranch=main", "init", "-q", str(dest)])
    # -f is required: `git archive` exports files that upstream tracks even when
    # the repo's own .gitignore matches them. Without -f they land on disk but
    # never enter the base commit, so any edit to them is silently dropped from
    # the patch. extract_patch needs no -f: gitignore stops applying once a
    # file is tracked.
    git(["-C", str(dest)] + ident + ["add", "-A", "-f"])
    git(["-C", str(dest)] + ident +
        # the message stays a bare "base": writing the upstream SHA into the
        # working copy would hand the agent an identifier for the real commit.
        ["commit", "-q", "--no-verify", "-m", "base"])


def scrub_artifacts(root: Path):
    """Drop build noise Claude may have produced so it never reaches the diff."""
    for pyc in root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def status_entries(dest: Path):
    """Parse `git status --porcelain -z -uall` into (code, path) pairs."""
    out = git(["-C", str(dest), "status", "--porcelain", "-z", "-uall"], check=False)
    fields = out.split("\0")
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


def revert_test_edits(dest: Path):
    """Undo any change to a test file; rule 1 forbids touching them."""
    reverted = []
    for code, path in status_entries(dest):
        if not TEST_PATH_RE.search(path):
            continue
        reverted.append(path)
        if code.strip() == "??":
            target = dest / path
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        else:
            git(["-C", str(dest), "checkout", "--", path], check=False)
    return reverted


def clean_scratch_files(dest: Path):
    """Remove Claude's scratch output before staging.

    `.DS_Store` anywhere and brand-new files at the repo root (where agents tend
    to drop benchmark_foo.py / notes.md) are deleted. Every other new file is
    left in the patch but reported so it can be reviewed.
    """
    removed, new_files = [], []
    for code, path in status_entries(dest):
        if code.strip() != "??":
            continue
        target = dest / path
        is_scratch = Path(path).name == ".DS_Store" or "/" not in path
        if is_scratch:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            removed.append(path)
        else:
            new_files.append(path)
    return removed, new_files


def extract_patch(dest: Path):
    """Stage everything and return the unified diff against the base commit."""
    git(["-C", str(dest), "add", "-A"])
    return git(["-C", str(dest), "diff", "--cached"], check=False)


def build_env():
    """Inherit the user's shell env but force CLI (subscription) auth."""
    env = os.environ.copy()
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


def as_text(blob):
    """TimeoutExpired.stdout is bytes or None even when text=True was asked for."""
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", "replace")
    return blob


def parse_result_line(stdout: str) -> Optional[dict]:
    """Pull the fields we care about out of the stream's final `result` message."""
    keep = ("num_turns", "total_cost_usd", "usage", "duration_ms", "is_error", "subtype")
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


def audit_transcript(stdout: str):
    """Flag tool calls that look like network access or history spelunking."""
    hits = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
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


def run_claude(prompt, cwd, timeout, model, max_turns, log_dir, iid):
    """Run headless Claude Code, saving the full stream-json transcript.

    Returns (returncode, timeout_error, claude_result, stdout, stderr). A
    returncode of None means the process was killed by the timeout.
    """
    cmd = [
        "claude", "-p", prompt,
        "--allowedTools", *ALLOWED_TOOLS,
        "--disallowedTools", *DISALLOWED_TOOLS,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        "--model", model,
    ]
    stdout_path = log_dir / f"{iid}.jsonl"
    stderr_path = log_dir / f"{iid}.stderr.txt"
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=build_env(), capture_output=True,
            text=True, timeout=timeout,
        )
        rc, err = proc.returncode, None
        stdout, stderr = as_text(proc.stdout), as_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        rc, err = None, f"timeout after {timeout}s"
        stdout, stderr = as_text(exc.stdout), as_text(exc.stderr)
        stderr += f"\n--- TIMEOUT after {timeout}s ---\n"

    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return rc, err, parse_result_line(stdout), stdout, stderr


def problem_statement_for(inst):
    for key in ("problem_statement_realistic", "problem_statement"):
        if inst.get(key):
            return inst[key]
    raise KeyError(f"no problem statement on {inst['instance_id']}")


def prompt_template_for(style):
    return VERBATIM_TEMPLATE if style == "verbatim" else PROMPT_TEMPLATE


def build_prompt(inst, style):
    return prompt_template_for(style).format(problem_statement=problem_statement_for(inst))


def process(inst, args, cache_dir, work_dir, log_dir, run_idx, stop_event=None):
    iid = inst["instance_id"]
    if stop_event is not None and stop_event.is_set():
        return None, {"instance_id": iid, "run": run_idx, "status": "cancelled",
                      "reverted_tests": [], "duration_s": 0.0}
    dest = work_dir / f"{iid}__run{run_idx}"
    started = time.time()
    meta = {"instance_id": iid, "run": run_idx, "status": "ok", "reverted_tests": []}
    try:
        mirror = ensure_mirror(inst["repo"], cache_dir, inst["base_commit"])
        prepare_checkout(inst["repo"], inst["base_commit"], mirror, dest)

        prompt = build_prompt(inst, args.prompt_style)
        rc, err, result, stdout, stderr = run_claude(
            prompt, dest, args.timeout, args.model, args.max_turns, log_dir, iid
        )
        if result is not None:
            meta["claude_result"] = result
        suspects = audit_transcript(stdout)
        if suspects:
            meta["web_access_suspect"] = suspects

        if err:
            meta["status"] = "timeout"
        elif rc != 0:
            result_text = json.dumps(result) if result else ""
            if looks_like_usage_limit(stderr, result_text):
                meta["status"] = "usage_limit"
            else:
                meta["status"] = f"claude_exit_{rc}"

        scrub_artifacts(dest)
        meta["reverted_tests"] = revert_test_edits(dest)
        removed, new_files = clean_scratch_files(dest)
        meta["removed_scratch_files"] = removed
        meta["new_files"] = new_files
        patch = extract_patch(dest)
        if not patch.strip() and meta["status"] == "ok":
            meta["status"] = "empty_patch"
        meta["patch_bytes"] = len(patch)
        meta["files_changed"] = patch.count("diff --git ")
    except Exception as exc:  # keep one bad instance from killing the run
        meta["status"] = "error"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        patch = ""
    finally:
        if not args.keep_clones:
            shutil.rmtree(dest, ignore_errors=True)

    meta["duration_s"] = round(time.time() - started, 1)
    record = {
        "instance_id": iid,
        "model_name_or_path": args.model_name,
        "model_patch": patch,
    }
    return record, meta


# Statuses that must NOT get a prediction line: leaving them out of the
# predictions file is what makes --resume pick them up again.
RETRY_STATUSES = ("error", "usage_limit", "cancelled", "timeout")
# "timeout" is here deliberately: a killed run leaves a mid-edit working tree,
# and that truncated diff is a wrong data point, not a slow one.


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


def claude_version():
    try:
        proc = subprocess.run(["claude", "--version"], capture_output=True,
                              text=True, timeout=30)
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_paths(out_path: Path, run_idx: int, num_runs: int):
    """One predictions file per run: run_evaluation.py keys predictions by
    instance_id, so duplicate ids in a single file silently overwrite."""
    if num_runs == 1:
        preds = out_path
    else:
        preds = out_path.with_name(f"{out_path.stem}_run{run_idx}{out_path.suffix}")
    meta = preds.with_name(preds.stem + "_meta.jsonl")
    prov = preds.with_name(preds.stem + "_provenance.jsonl")
    return preds, meta, prov


def load_instances(args):
    """Either a SWE-Perf-style HF dataset or a JSONL of arbitrary instances.

    The JSONL form is what you want for SWE-fficiency and for real-life issues:
    one object per line with instance_id, repo, base_commit, problem_statement.
    """
    if args.instances_file:
        path = Path(args.instances_file).resolve()
        if not path.exists():
            sys.exit(f"error: no such instances file: {path}")
        instances, required = [], ("instance_id", "repo", "base_commit")
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                inst = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"error: {path}:{n} is not valid JSON: {exc}")
            missing = [k for k in required if not inst.get(k)]
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


def execute_run(run_idx, instances, args, out_path, cache_dir, work_dir, log_root, version):
    """Run one sweep. Returns (status counts, hit_usage_limit)."""
    preds_path, meta_path, prov_path = run_paths(out_path, run_idx, args.num_runs)
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

    template = prompt_template_for(args.prompt_style)
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()[:PROMPT_HASH_LEN]
    template_path = preds_path.with_name(f"prompt_template_{prompt_sha}.txt")
    if not template_path.exists():
        template_path.write_text(template)

    # Provenance is appended, not overwritten, so a CLI upgrade partway through
    # a resumed run stays visible in the record.
    with open(prov_path, "a") as f:
        f.write(json.dumps({
            "run": run_idx,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "claude_version": version,
            "model_flag": args.model,
            "model_id": args.model,
            "model_name_or_path": args.model_name,
            "allowed_tools": ALLOWED_TOOLS,
            "disallowed_tools": DISALLOWED_TOOLS,
            "max_turns": args.max_turns,
            "execution_env": "host_checkout_no_dependencies",
            "prompt_style": args.prompt_style,
            "prompt_template_file": template_path.name,
            "prompt_sha256": prompt_sha,
            "timeout_s": args.timeout,
            "instances": len(todo),
        }) + "\n")

    print(f"[run {run_idx}/{args.num_runs}] {len(todo)} instance(s), "
          f"{args.workers} worker(s) -> {preds_path}")

    counts = {}
    stop_event = threading.Event()
    hit_limit = False
    with open(preds_path, "a") as out_f, open(meta_path, "a") as meta_f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process, inst, args, cache_dir, work_dir, log_dir,
                        run_idx, stop_event)
            for inst in todo
        ]
        for n, fut in enumerate(as_completed(futures), 1):
            if fut.cancelled():
                continue
            record, meta = fut.result()
            status = meta["status"]
            counts[status] = counts.get(status, 0) + 1
            if status == "cancelled":
                continue
            with _write_lock:
                # error/usage_limit get a meta line but no prediction, so that a
                # later --resume treats them as unfinished and retries them.
                if status not in RETRY_STATUSES and record is not None:
                    out_f.write(json.dumps(record) + "\n"); out_f.flush()
                meta_f.write(json.dumps(meta) + "\n"); meta_f.flush()
            note = (f" (reverted {len(meta['reverted_tests'])} test file(s))"
                    if meta["reverted_tests"] else "")
            if meta.get("web_access_suspect"):
                note += f" (!! {len(meta['web_access_suspect'])} audit hit(s))"
            print(f"  [run {run_idx}] [{n}/{len(todo)}] {meta['instance_id']}: "
                  f"{status}, {meta.get('files_changed', 0)} file(s), "
                  f"{meta['duration_s']}s{note}")
            if status == "usage_limit" and args.stop_on_limit and not hit_limit:
                hit_limit = True
                stop_event.set()
                cancelled = sum(1 for f in futures if f.cancel())
                counts["cancelled"] = counts.get("cancelled", 0) + cancelled
                print(f"  [run {run_idx}] usage limit hit -- cancelled {cancelled} "
                      f"queued instance(s); waiting for in-flight work to finish")
    return counts, hit_limit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_name", default="SWE-Perf/SWE-Perf")
    p.add_argument("--split", default="test")
    p.add_argument("--instances_file", default=None,
                   help="JSONL of {instance_id, repo, base_commit, problem_statement} "
                        "instead of an HF dataset (use for SWE-fficiency / real-life issues)")
    p.add_argument("--output", default="../../datasets/outputs/claude_code_preds.jsonl")
    p.add_argument("--model_name", default="claude-code",
                   help="value written to model_name_or_path (names the log dir at eval time)")
    p.add_argument("--model", required=True,
                   help="REQUIRED full model ID passed to the claude CLI "
                        "(e.g. claude-opus-5); required so the model can never "
                        "change silently between runs")
    p.add_argument("--max_turns", type=int, default=100,
                   help="--max-turns passed to the claude CLI (agentic turn cap)")
    p.add_argument("--prompt_style", choices=("custom", "verbatim"), default="custom",
                   help="custom: our PROMPT_TEMPLATE wrapper; verbatim: send "
                        "problem_statement_realistic exactly as the benchmark writes it")
    p.add_argument("--num_runs", type=int, default=1,
                   help="repeat the whole sweep N times; each instance gets a fresh "
                        "claude process (fresh context) and its own predictions file")
    p.add_argument("--instance_ids", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N instances")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800, help="per-instance wall clock (s)")
    p.add_argument("--work_dir", default="../../datasets/claude_code_workspaces")
    p.add_argument("--cache_dir", default="../../datasets/repo_mirrors")
    p.add_argument("--keep_clones", action="store_true", help="keep checkouts for debugging")
    p.add_argument("--resume", action="store_true",
                   help="skip instance_ids already written for that run")
    p.add_argument("--stop_on_limit", dest="stop_on_limit", action="store_true",
                   default=True, help="stop the sweep after the first usage limit (default)")
    p.add_argument("--no-stop_on_limit", dest="stop_on_limit", action="store_false",
                   help="keep going after a usage limit instead of stopping")
    args = p.parse_args()

    if shutil.which("claude") is None:
        sys.exit("error: the `claude` CLI is not on PATH")
    if args.num_runs < 1:
        sys.exit("error: --num_runs must be at least 1")
    if args.max_turns < 1:
        sys.exit("error: --max_turns must be at least 1")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).resolve(); work_dir.mkdir(parents=True, exist_ok=True)
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

    version = claude_version()
    print(f"[env] claude CLI: {version}, model: {args.model}")

    totals, stopped = {}, False
    for run_idx in range(1, args.num_runs + 1):
        counts, hit_limit = execute_run(run_idx, instances, args, out_path,
                                        cache_dir, work_dir, log_root, version)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        if hit_limit:
            stopped = True
            break

    print("\nsummary across all runs: " +
          (", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "nothing run"))
    for run_idx in range(1, args.num_runs + 1):
        preds, meta, _ = run_paths(out_path, run_idx, args.num_runs)
        if preds.exists():
            print(f"  run {run_idx}: {preds}  (metadata: {meta.name})")
    print(f"logs: {log_root}")
    if stopped:
        print("\nSTOPPED: hit a Claude usage/rate limit. No prediction was written "
              "for the affected or cancelled instances, so rerun this same command "
              "later with --resume to pick them up.")
    if args.num_runs > 1:
        print("\nnote: evaluate each run separately, giving each its own --run_id "
              "so the evaluation logs don't overwrite each other.")


if __name__ == "__main__":
    main()
