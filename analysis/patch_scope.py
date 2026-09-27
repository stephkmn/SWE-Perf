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

    Returns {path: {"old_lines": set[int],
                    "added": [(new_lineno, text, base_anchor)],
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
                    entry["added"].append((line.target_line_no, line.value.rstrip("\n"), anchor))
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


def function_body_ranges(source):
    """(qualified_name, first_body_line, end_line) for every def and class.

    Distinct from function_ranges: a `def` line, its decorators and its
    docstring all execute when the module is imported, so asking "did any line
    of this function run" against the full span answers "was it imported",
    which is true of every function in a touched module. Only the executable
    body separates a function that actually ran from one that was merely
    defined.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    ranges = []

    def body_start(node):
        body = [st for st in node.body if not (
            isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
            and isinstance(st.value.value, str))]
        return (body[0].lineno if body else getattr(node, "end_lineno", node.lineno))

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                ranges.append((name, body_start(child),
                               getattr(child, "end_lineno", child.lineno)))
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


DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
CLASS_RE = re.compile(r"^\s*class\s+(\w+)")
DECORATOR_RE = re.compile(r"^\s*@")


def added_definitions(added):
    """Functions and classes the patch introduces.

    A brand-new definition cannot be found by mapping lines onto the base
    AST -- it does not exist there -- so without this a patch that adds a
    helper looks like an untargeted module-level edit.
    """
    names = set()
    for _, text, _ in added:
        body = text.lstrip("+")
        match = DEF_RE.match(body) or CLASS_RE.match(body)
        if match:
            names.add(match.group(1))
    return names


def decorated_functions(ranges, added):
    """Existing functions that the patch decorates.

    A decorator inserted above an existing `def` anchors to the blank line
    before it, so line mapping calls it a module-level edit even though it
    changes that function's behaviour -- which is how SWE-Perf's own
    patch_functions labels it.
    """
    starts = {}
    for name, start, _ in ranges:
        starts.setdefault(start, name)
    names = set()
    for _, text, anchor in added:
        if DECORATOR_RE.match(text.lstrip("+")) and anchor is not None:
            following = starts.get(anchor + 1)
            if following:
                names.add(following)
    return names


def _tail(name):
    """"pkg/f.py::A.b.c" -> ("pkg/f.py", "c")."""
    path, _, qual = name.partition("::")
    return path, qual.rsplit(".", 1)[-1]


def short_name_counts(ranges):
    """How many defs/classes in one file share each final name component.

    Files routinely contain several `wrapper`s or `__init__`s, so a fallback
    match on the short name can attach a change to the wrong function. Counting
    them lets each match say whether it was ambiguous.
    """
    counts = {}
    for name, _, _ in ranges:
        tail = name.rsplit(".", 1)[-1]
        counts[tail] = counts.get(tail, 0) + 1
    return counts


def match_functions(changed, reference, ambiguity=None):
    """Match changed functions against a reference set, labelling each match.

    The dataset writes some names fully qualified ("LombScargle.autopower") and
    others bare ("wrapper", which the AST resolves as
    "QuantityInput.__call__.wrapper"), so exact matching alone misses real
    overlaps and overstates how much novel ground a patch covered. The fallback
    compares the final name component within the same file.

    `ambiguity` maps a file path to {short_name: count} from short_name_counts;
    a fallback match on a name appearing more than once in that file is marked
    ambiguous rather than silently trusted.

    Returns (matches, unmatched_changed) where each match is
    {reference, changed, match_type, ambiguous}.
    """
    ambiguity = ambiguity or {}
    by_tail = {}
    for ref in reference:
        by_tail.setdefault(_tail(ref), set()).add(ref)

    matches, unmatched = [], []
    for name in sorted(changed):
        if name in reference:
            matches.append({"reference": name, "changed": name,
                            "match_type": "exact", "ambiguous": False})
            continue
        key = _tail(name)
        if key in by_tail:
            path, short = key
            ambiguous = ambiguity.get(path, {}).get(short, 0) > 1
            for ref in sorted(by_tail[key]):
                matches.append({"reference": ref, "changed": name,
                                "match_type": "fallback", "ambiguous": ambiguous})
            continue
        unmatched.append(name)
    return matches, unmatched


def _refs(matches, match_type=None):
    return sorted({m["reference"] for m in matches
                   if match_type is None or m["match_type"] == match_type})


def analyze_patch(instance, patch_text, mirror_dir=None):
    """Compare one patch's footprint against the task targets and the expert's."""
    per_file = parse_diff(patch_text)
    targets = target_functions(instance)
    expert = expert_functions(instance)

    mirrors = Path(mirror_dir) if mirror_dir else DEFAULT_MIRROR_DIR
    changed, module_level, used_ast = set(), 0, False
    new_functions = set()
    ambiguity = {}
    for path, info in per_file.items():
        source = base_source(instance["repo"], instance["base_commit"], path, mirrors)
        if source:
            used_ast = True
            ranges = function_ranges(source)
            names, mod = functions_for_lines(ranges, info["old_lines"])
            names |= decorated_functions(ranges, info["added"])
            module_level += mod
            ambiguity[path] = short_name_counts(ranges)
        else:
            names = functions_from_sections(info["sections"])
        introduced = added_definitions(info["added"])
        new_functions |= {f"{path}::{n}" for n in introduced}
        changed |= {f"{path}::{n}" for n in names | introduced}

    target_set, expert_set = flatten_functions(targets), flatten_functions(expert)
    target_matches, outside_targets = match_functions(changed, target_set, ambiguity)
    expert_matches, _ = match_functions(changed, expert_set, ambiguity)

    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "files_changed": sorted(per_file),
        "functions_changed": sorted(changed),
        "module_level_changes": module_level,
        "target_functions": sorted(target_set),
        "expert_functions": sorted(expert_set),
        "overlap_with_targets": _refs(target_matches),
        "overlap_with_targets_exact": _refs(target_matches, "exact"),
        "overlap_with_expert": _refs(expert_matches),
        "overlap_with_expert_exact": _refs(expert_matches, "exact"),
        "target_matches": target_matches,
        "expert_matches": expert_matches,
        "n_ambiguous_matches": sum(1 for m in target_matches if m["ambiguous"]),
        "new_functions": sorted(new_functions),
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
