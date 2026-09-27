"""Decide whether existing predictions predate the git-history isolation fix.

Before that fix, run_claude_code.py cloned the full mirror into each working
copy, so the agent could read forward through history and find the upstream
performance fix. Patches generated then are not safely attributable to the
model. This reports which predictions are affected; it regenerates nothing.

Three independent signals, because any one of them can be missing:
  1. the provenance file's started_at, against the fix commit's timestamp
  2. the prompt hash, which changed in the same commit
  3. the transcripts, searched for history-reading commands
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_predictions  # noqa: E402

DRIVER_PATH = "generation/realistic/run_claude_code.py"
# The commit that replaced `git clone` with `git archive` into a fresh repo.
FIX_MARKER = "git archive"

HISTORY_COMMANDS = [
    ("git_log", re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?log\b")),
    ("git_all", re.compile(r"--all\b")),
    ("git_show", re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?show\b")),
    ("git_tag", re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?tag\b")),
]


def fix_commit(repo_root):
    """(sha, iso timestamp) of the commit that introduced history isolation."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "log", "-S", FIX_MARKER,
         "--format=%H|%cI", "--reverse", "--", DRIVER_PATH],
        capture_output=True, text=True,
    )
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        return None, None
    sha, when = lines[0].split("|", 1)
    return sha, when


def parse_iso(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    # provenance writes %z without a colon, e.g. 2026-09-26T16:04:08-0700
    match = re.match(r"(.*)([+-]\d{2})(\d{2})$", text)
    if match:
        try:
            return datetime.fromisoformat(f"{match.group(1)}{match.group(2)}:{match.group(3)}")
        except ValueError:
            return None
    return None


def prompt_hash_after_fix(repo_root, sha):
    """The prompt hash the driver produced at the fixed commit, if derivable."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{sha}:{DRIVER_PATH}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    match = re.search(r'PROMPT_TEMPLATE\s*=\s*"""(.*?)"""', proc.stdout, re.DOTALL)
    if not match:
        return None
    import hashlib
    return hashlib.sha256((match.group(1)).encode()).hexdigest()[:16]


def read_provenance(preds_path):
    """Provenance records sitting beside a predictions file."""
    path = Path(preds_path)
    candidates = [path.with_name(path.stem + "_provenance.jsonl")]
    candidates += sorted(path.parent.glob("*_provenance.jsonl"))
    records = []
    for candidate in candidates:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            break
    return records


def scan_transcripts(log_dir):
    """Search agent transcripts for commands that read repository history.

    Returns (hits, fidelity). Fidelity matters more than the hits: the driver
    only began saving full stream-json transcripts in the same commit that
    fixed history isolation, so transcripts from before it contain the agent's
    closing prose and nothing else. Searching those for `git log` proves
    nothing -- a tool call would never have appeared in them. Absence of hits
    is only meaningful when tool calls were actually recorded.
    """
    hits = {}
    tool_calls_visible = False
    scanned = 0
    if not log_dir or not Path(log_dir).exists():
        return hits, "no_transcripts", 0
    for path in sorted(Path(log_dir).rglob("*")):
        if not path.is_file() or path.suffix not in (".jsonl", ".log", ".txt"):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        scanned += 1
        if '"type": "tool_use"' in text or '"type":"tool_use"' in text:
            tool_calls_visible = True
        found = sorted({name for name, pattern in HISTORY_COMMANDS if pattern.search(text)})
        if found:
            hits[path.name] = found
    if scanned == 0:
        return hits, "no_transcripts", 0
    return hits, ("tool_calls_visible" if tool_calls_visible else "final_text_only"), scanned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--transcript_dir", default=None,
                   help="agent transcripts (default: claude_code_logs beside the predictions)")
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--output", default="analysis_out/provenance_audit.json")
    args = p.parse_args()

    sha, when = fix_commit(args.repo_root)
    fixed_at = parse_iso(when)
    preds = load_predictions(args.predictions)
    provenance = read_provenance(args.predictions)

    transcript_dir = args.transcript_dir or str(Path(args.predictions).parent / "claude_code_logs")
    transcript_hits, fidelity, n_scanned = scan_transcripts(transcript_dir)

    expected_hash = prompt_hash_after_fix(args.repo_root, sha) if sha else None
    runs = []
    for record in provenance:
        started = parse_iso(record.get("started_at"))
        before = bool(started and fixed_at and started < fixed_at)
        runs.append({
            "run": record.get("run"),
            "started_at": record.get("started_at"),
            "prompt_sha256": record.get("prompt_sha256"),
            "generated_before_fix": before,
            "prompt_hash_matches_fixed_driver": (
                None if expected_hash is None else record.get("prompt_sha256") == expected_hash
            ),
        })

    verdict = "unknown"
    if runs:
        verdict = "before_fix" if any(r["generated_before_fix"] for r in runs) else "after_fix"

    report = {
        "fix_commit": sha,
        "fix_committed_at": when,
        "driver_path": DRIVER_PATH,
        "predictions_path": str(args.predictions),
        "n_predictions": len(preds),
        "provenance_records": len(provenance),
        "runs": runs,
        "verdict": verdict,
        "transcript_dir": transcript_dir,
        "transcripts_scanned": n_scanned,
        "transcript_fidelity": fidelity,
        "transcripts_with_history_commands": transcript_hits,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"history-isolation fix : {sha[:8] if sha else 'not found'} ({when})")
    print(f"predictions           : {len(preds)} in {args.predictions}")
    for r in runs:
        state = "BEFORE fix (history was visible)" if r["generated_before_fix"] else "after fix"
        print(f"  run {r['run']}: started {r['started_at']} -> {state}")
    if not runs:
        print("  no provenance records found -- cannot date these predictions")
    if transcript_hits:
        print(f"transcripts referencing history commands: {len(transcript_hits)}")
        for name, found in list(transcript_hits.items())[:10]:
            print(f"  {name}: {', '.join(found)}")
    elif fidelity == "final_text_only":
        print(f"transcripts: {n_scanned} scanned, no history commands found -- but these "
              f"record only the agent's closing text,\n"
              f"             not its tool calls, so this is NOT evidence that history "
              f"went unread.")
    elif fidelity == "no_transcripts":
        print("transcripts: none found")
    else:
        print(f"transcripts: {n_scanned} scanned with tool calls visible, "
              f"no history commands found")
    print(f"\nverdict: {verdict}\nwrote {out}")


if __name__ == "__main__":
    main()
