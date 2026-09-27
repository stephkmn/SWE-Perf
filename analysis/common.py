"""Shared helpers for the post-hoc patch analysis tools.

These tools sit alongside SWE-Perf's official evaluation and never alter it.
The official metric stays exactly as published so our numbers remain
comparable; everything here is a stricter check layered on top.

Why it exists: run_evaluation.py runs only the instance's `efficiency_test`
list, and check_evaluation.py calls a patch correct when those same tests pass.
The tests that measure the speedup are therefore the tests that validate it, so
a genuine optimization and one that games those tests score identically. These
modules look for the difference after the fact.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

DATASET_NAME = "SWE-Perf/SWE-Perf"
DATASET_SPLIT = "test"

# Where run_claude_code.py keeps its bare mirrors. Used to read base-commit
# sources for AST mapping without touching a container.
DEFAULT_MIRROR_DIR = Path(__file__).resolve().parents[1] / "datasets" / "repo_mirrors"


def load_instances(dataset_name=DATASET_NAME, split=DATASET_SPLIT):
    """Return {instance_id: row} for the benchmark split."""
    from datasets import load_dataset

    return {row["instance_id"]: row for row in load_dataset(dataset_name, split=split)}


def load_predictions(path):
    """Read a predictions JSONL into a list of dicts, skipping blank lines."""
    preds = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            preds.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n} is not valid JSON: {exc}") from exc
    return preds


def parse_function_map(raw):
    """Parse a {file: [functions]} mapping out of a dataset field.

    `test_functions` and `patch_functions` are JSON strings. As a fallback,
    `problem_statement_realistic` embeds the same mapping as a Python repr of a
    defaultdict, which json cannot read -- hence the literal_eval path.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {k: list(v) for k, v in raw.items()}
    text = str(raw)
    try:
        return {k: list(v) for k, v in json.loads(text).items()}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = ast.literal_eval(match.group(0))
    except (ValueError, SyntaxError):
        return {}
    return {k: list(v) for k, v in value.items()} if isinstance(value, dict) else {}


def target_functions(instance):
    """Functions the realistic task points the agent at."""
    fns = parse_function_map(instance.get("test_functions"))
    if fns:
        return fns
    return parse_function_map(instance.get("problem_statement_realistic"))


def expert_functions(instance):
    """Functions the human expert's patch actually changed."""
    return parse_function_map(instance.get("patch_functions"))


def flatten_functions(mapping):
    """{file: [f, g]} -> {"file::f", "file::g"} for set comparison."""
    return {f"{path}::{fn}" for path, fns in mapping.items() for fn in fns}


def mirror_path(repo, mirror_dir=DEFAULT_MIRROR_DIR):
    return Path(mirror_dir) / (repo.replace("/", "__") + ".git")


def base_source(repo, base_commit, file_path, mirror_dir=DEFAULT_MIRROR_DIR):
    """Read a file as of the base commit from the local bare mirror.

    Returns None when the mirror or the file is absent; callers fall back to
    the diff's own hunk headers, which are less precise but need no clone.
    """
    mirror = mirror_path(repo, mirror_dir)
    if not mirror.exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(mirror), "show", f"{base_commit}:{file_path}"],
        capture_output=True, text=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
