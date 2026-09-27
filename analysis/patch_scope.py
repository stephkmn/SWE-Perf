"""Map each patch to the functions it changes, and compare against the task.

A patch confined entirely to the functions the task names is a candidate for
overfitting: it may be tuned to the measured tests rather than fixing a real
bottleneck. One that also touches shared machinery is likelier to be genuine.
Neither is proof -- this produces evidence for human review, and the fields it
emits must never be fed back into patch generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATASET_NAME, DATASET_SPLIT, DEFAULT_MIRROR_DIR, base_source, expert_functions, flatten_functions,
    load_instances, load_predictions, target_functions,
)

SECTION_NAME_RE = re.compile(r"\b(?:def|class)\s+(\w+)")


def parse_diff(diff_text):
    """Split a unified diff into per-file changed-line information.

    Returns {path: {"old_lines": set[int], "added": [(new_lineno, text)],
                    "sections": [str]}}.

    `old_lines` holds base-file line numbers, which is what the AST of the base
    source can be queried with. A pure insertion has no base line of its own, so
    it anchors to the last base line seen -- the function it was inserted into.
    """
    from unidiff import PatchSet

    files = {}
    try:
        patch = PatchSet(diff_text)
    except Exception:
        return files

    for pfile in patch:
        path = pfile.path
        entry = files.setdefault(path, {"old_lines": set(), "added": [], "sections": []})
        for hunk in pfile:
            if hunk.section_header:
                entry["sections"].append(hunk.section_header)
            anchor = hunk.source_start
            for line in hunk:
                if line.source_line_no is not None:
                    anchor = line.source_line_no
                if line.is_removed:
                    entry["old_lines"].add(line.source_line_no)
                elif line.is_added:
                    entry["old_lines"].add(anchor)
                    entry["added"].append((line.target_line_no, line.value.rstrip("\n")))
    return files


def function_ranges(source):
    """List (qualified_name, start_line, end_line) for every def and class.

    Names are qualified the way the dataset writes them -- "LombScargle.autopower"
    -- so they can be compared directly against test_functions/patch_functions.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    ranges = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                starts = [child.lineno] + [d.lineno for d in getattr(child, "decorator_list", [])]
                ranges.append((name, min(starts), getattr(child, "end_lineno", child.lineno)))
                walk(child, name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return ranges


def functions_for_lines(ranges, lines):
    """Innermost def/class enclosing each changed line.

    Returns (names, module_level_count). Lines outside any def or class -- module
    scope, imports, constants -- are counted rather than named.
    """
    names, module_level = set(), 0
    for line in sorted(x for x in lines if x):
        enclosing = [r for r in ranges if r[1] <= line <= r[2]]
        if not enclosing:
            module_level += 1
            continue
        names.add(min(enclosing, key=lambda r: r[2] - r[1])[0])
    return names, module_level


def functions_from_sections(sections):
    """Fallback when base source is unavailable: read git's hunk headers.

    Less precise than the AST -- it names the nearest preceding def, which for a
    change inside a nested function is the outer one -- but needs no checkout.
    """
    names = set()
    for section in sections:
        match = SECTION_NAME_RE.search(section or "")
        if match:
            names.add(match.group(1))
    return names


def _tail(name):
    """"pkg/f.py::A.b.c" -> ("pkg/f.py", "c")."""
    path, _, qual = name.partition("::")
    return path, qual.rsplit(".", 1)[-1]


def match_functions(changed, reference):
    """Match changed functions against a reference set, tolerating the dataset's
    inconsistent qualification.

    The dataset writes some names fully qualified ("LombScargle.autopower") and
    others bare ("wrapper", which the AST resolves as
    "QuantityInput.__call__.wrapper"). Exact matching alone therefore misses
    real overlaps and would overstate how much novel ground a patch covered.
    Falling back to the final name component is scoped to the same file, so the
    residual risk is two same-named functions in one module.

    Returns (matched_reference_entries, unmatched_changed_entries).
    """
    by_tail = {}
    for ref in reference:
        by_tail.setdefault(_tail(ref), set()).add(ref)

    matched, unmatched = set(), []
    for name in changed:
        if name in reference:
            matched.add(name)
        elif _tail(name) in by_tail:
            matched |= by_tail[_tail(name)]
        else:
            unmatched.append(name)
    return matched, unmatched


def analyze_patch(instance, patch_text, mirror_dir=None):
    """Compare one patch's footprint against the task targets and the expert's."""
    per_file = parse_diff(patch_text)
    targets = target_functions(instance)
    expert = expert_functions(instance)

    mirrors = Path(mirror_dir) if mirror_dir else DEFAULT_MIRROR_DIR
    changed, module_level, used_ast = set(), 0, False
    for path, info in per_file.items():
        source = base_source(instance["repo"], instance["base_commit"], path, mirrors)
        if source:
            used_ast = True
            names, mod = functions_for_lines(function_ranges(source), info["old_lines"])
            module_level += mod
        else:
            names = functions_from_sections(info["sections"])
        changed |= {f"{path}::{n}" for n in names}

    target_set, expert_set = flatten_functions(targets), flatten_functions(expert)
    matched_targets, outside_targets = match_functions(changed, target_set)
    matched_expert, _ = match_functions(changed, expert_set)
    overlap_targets = sorted(matched_targets)
    overlap_expert = sorted(matched_expert)

    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "files_changed": sorted(per_file),
        "functions_changed": sorted(changed),
        "module_level_changes": module_level,
        "target_functions": sorted(target_set),
        "expert_functions": sorted(expert_set),
        "overlap_with_targets": overlap_targets,
        "overlap_with_expert": overlap_expert,
        "functions_outside_targets": sorted(outside_targets),
        # True only when the patch changed something and stayed entirely inside
        # the named targets -- the shape most worth reviewing by hand.
        "touches_only_targets": bool(changed) and not outside_targets,
        "n_functions_changed": len(changed),
        "source_resolved": used_ast,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", default="analysis_out/patch_scope.jsonl")
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--mirror_dir", default=None,
                   help="bare mirrors for base sources (default: datasets/repo_mirrors); "
                        "without them, hunk headers are used instead")
    args = p.parse_args()

    instances = load_instances(args.dataset_name, args.split)
    preds = load_predictions(args.predictions)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    unresolved = 0
    with open(out, "w") as fh:
        for pred in preds:
            inst = instances.get(pred["instance_id"])
            if inst is None:
                print(f"warning: {pred['instance_id']} not in dataset, skipping")
                continue
            row = analyze_patch(inst, pred.get("model_patch") or "", args.mirror_dir)
            row["model_name_or_path"] = pred.get("model_name_or_path")
            unresolved += 0 if row["source_resolved"] else 1
            fh.write(json.dumps(row) + "\n")

    print(f"wrote {out}")
    if unresolved:
        print(f"note: {unresolved} patch(es) fell back to hunk headers "
              f"(no base source in mirrors) -- function names are approximate")


if __name__ == "__main__":
    main()
