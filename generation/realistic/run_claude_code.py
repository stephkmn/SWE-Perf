#!/usr/bin/env python3
"""
Generate SWE-Perf patches with Claude Code (realistic setting).

For each dataset instance this script clones the repo at its base commit onto
the host, runs headless Claude Code inside that checkout, and captures the
resulting `git diff` as the instance's `model_patch`.

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

PROMPT_HASH_LEN = 16

ALLOWED_TOOLS = ["Edit", "Read", "Write", "Glob", "Grep", "Bash"]

# Mirrors the four optimization rules the benchmark itself states in
# generation/oracle/make_datasets/create_text_dataset.py
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

TEST_PATH_RE = re.compile(
    r"(^|/)tests?/|(^|/)testing/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$"
)

_mirror_locks: dict[str, threading.Lock] = {}
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
    """Maintain one bare mirror per repo so each instance clones locally."""
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
    """Clone from the local mirror and detach at the base commit."""
    if dest.exists():
        shutil.rmtree(dest)
    git(["clone", "--no-checkout", str(mirror), str(dest)])
    git(["-C", str(dest), "checkout", "--detach", base_commit])
    git(["-C", str(dest), "clean", "-xfd"])


def scrub_artifacts(root: Path):
    """Drop build noise Claude may have produced so it never reaches the diff."""
    for pyc in root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for cache in list(root.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)


def revert_test_edits(dest: Path):
    """Undo any change to a test file; rule 1 forbids touching them."""
    reverted = []
    status = git(["-C", str(dest), "status", "--porcelain"], check=False)
    for line in status.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        if '"' in path:
            path = path.strip('"')
        if " -> " in path:  # rename
            path = path.split(" -> ", 1)[1]
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


def extract_patch(dest: Path):
    """Stage everything and return the unified diff."""
    git(["-C", str(dest), "add", "-A"])
    return git(["-C", str(dest), "diff", "--cached"], check=False)


def build_env():
    """Inherit the user's shell env but force CLI (subscription) auth."""
    env = os.environ.copy()
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(key, None)
    return env


def run_claude(prompt, cwd, timeout, model, log_path):
    cmd = ["claude", "-p", prompt, "--allowedTools", *ALLOWED_TOOLS]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=build_env(), capture_output=True,
            text=True, timeout=timeout,
        )
        log_path.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""))
        return proc.returncode, None
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") + f"\n--- TIMEOUT after {timeout}s ---\n")
        return None, f"timeout after {timeout}s"


def problem_statement_for(inst):
    for key in ("problem_statement_realistic", "problem_statement"):
        if inst.get(key):
            return inst[key]
    raise KeyError(f"no problem statement on {inst['instance_id']}")


def process(inst, args, cache_dir, work_dir, log_dir, run_idx):
    iid = inst["instance_id"]
    dest = work_dir / f"{iid}__run{run_idx}"
    started = time.time()
    meta = {"instance_id": iid, "run": run_idx, "status": "ok", "reverted_tests": []}
    try:
        mirror = ensure_mirror(inst["repo"], cache_dir, inst["base_commit"])
        prepare_checkout(inst["repo"], inst["base_commit"], mirror, dest)

        prompt = PROMPT_TEMPLATE.format(problem_statement=problem_statement_for(inst))
        rc, err = run_claude(prompt, dest, args.timeout, args.model, log_dir / f"{iid}.log")
        if err:
            meta["status"] = "timeout"
        elif rc != 0:
            meta["status"] = f"claude_exit_{rc}"

        scrub_artifacts(dest)
        meta["reverted_tests"] = revert_test_edits(dest)
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
        return {}

    # Provenance is appended, not overwritten, so a CLI upgrade partway through
    # a resumed run stays visible in the record.
    with open(prov_path, "a") as f:
        f.write(json.dumps({
            "run": run_idx,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "claude_version": version,
            "model_flag": args.model,
            "model_name_or_path": args.model_name,
            "allowed_tools": ALLOWED_TOOLS,
            "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:PROMPT_HASH_LEN],
            "timeout_s": args.timeout,
            "instances": len(todo),
        }) + "\n")

    print(f"[run {run_idx}/{args.num_runs}] {len(todo)} instance(s), "
          f"{args.workers} worker(s) -> {preds_path}")

    counts = {}
    with open(preds_path, "a") as out_f, open(meta_path, "a") as meta_f, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process, inst, args, cache_dir, work_dir, log_dir, run_idx)
            for inst in todo
        ]
        for n, fut in enumerate(as_completed(futures), 1):
            record, meta = fut.result()
            with _write_lock:
                out_f.write(json.dumps(record) + "\n"); out_f.flush()
                meta_f.write(json.dumps(meta) + "\n"); meta_f.flush()
            counts[meta["status"]] = counts.get(meta["status"], 0) + 1
            note = (f" (reverted {len(meta['reverted_tests'])} test file(s))"
                    if meta["reverted_tests"] else "")
            print(f"  [run {run_idx}] [{n}/{len(todo)}] {meta['instance_id']}: "
                  f"{meta['status']}, {meta.get('files_changed', 0)} file(s), "
                  f"{meta['duration_s']}s{note}")
    return counts


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
    p.add_argument("--model", default=None, help="optional --model passed to the claude CLI")
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
    args = p.parse_args()

    if shutil.which("claude") is None:
        sys.exit("error: the `claude` CLI is not on PATH")
    if args.num_runs < 1:
        sys.exit("error: --num_runs must be at least 1")

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
    print(f"[env] claude CLI: {version}")

    totals = {}
    for run_idx in range(1, args.num_runs + 1):
        counts = execute_run(run_idx, instances, args, out_path,
                             cache_dir, work_dir, log_root, version)
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v

    print("\nsummary across all runs: " +
          (", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "nothing run"))
    for run_idx in range(1, args.num_runs + 1):
        preds, meta, _ = run_paths(out_path, run_idx, args.num_runs)
        if preds.exists():
            print(f"  run {run_idx}: {preds}  (metadata: {meta.name})")
    print(f"logs: {log_root}")
    if args.num_runs > 1:
        print("\nnote: evaluate each run separately, giving each its own --run_id "
              "so the evaluation logs don't overwrite each other.")


if __name__ == "__main__":
    main()
