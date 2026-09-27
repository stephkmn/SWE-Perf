"""Unit tests for the four post-review fixes.

All run without Docker, network, or the dataset.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from patch_scope import (analyze_patch, added_definitions, decorated_functions,
                         function_ranges, match_functions, parse_diff, short_name_counts)
from provenance_audit import HISTORY_COMMANDS, parse_iso, scan_transcripts
from regression_check import collection_errors, outcomes_from_report, summarise_run
from timed_path import executed_lines


def report(tests=(), collectors=()):
    return {
        "tests": [{"nodeid": n, "outcome": o} for n, o in tests],
        "collectors": [{"nodeid": n, "outcome": o} for n, o in collectors],
    }


# --- Fix 1: every test must be accounted for --------------------------------

class TestRunAccounting:
    def test_counts_each_outcome(self):
        counts, outcomes, errors = summarise_run(report(tests=[
            ("t.py::a", "passed"), ("t.py::b", "failed"),
            ("t.py::c", "error"), ("t.py::d", "skipped"),
        ]))
        assert counts["collected"] == 4
        assert (counts["passed"], counts["failed"], counts["error"], counts["skipped"]) == (1, 1, 1, 1)
        assert counts["collection_errors"] == 0
        assert len(outcomes) == 4 and errors == []

    def test_unknown_outcome_lands_in_other(self):
        counts, _, _ = summarise_run(report(tests=[("t.py::a", "xfailed")]))
        assert counts["other"] == 1

    def test_collection_errors_are_extracted(self):
        assert collection_errors(report(collectors=[
            ("pkg/test_broken.py", "failed"), ("pkg/test_fine.py", "passed"),
        ])) == ["pkg/test_broken.py"]

    def test_empty_report_is_safe(self):
        counts, outcomes, errors = summarise_run(None)
        assert counts["collected"] == 0 and outcomes == {} and errors == []
        assert outcomes_from_report(None) == {}

    def test_vanished_test_is_detectable(self):
        """A broken import removes tests rather than failing them."""
        _, base, _ = summarise_run(report(tests=[("t.py::a", "passed"), ("t.py::b", "passed")]))
        _, after, _ = summarise_run(report(tests=[("t.py::a", "passed")]))
        passed_on_base = {n for n, o in base.items() if o == "passed"}
        missing = sorted(n for n in passed_on_base if n not in after)
        assert missing == ["t.py::b"], "a test that stops running must not read as a pass"

    def test_new_collection_error_is_detectable(self):
        _, _, base_errors = summarise_run(report(collectors=[]))
        _, _, after_errors = summarise_run(report(collectors=[("pkg/test_x.py", "failed")]))
        assert sorted(set(after_errors) - set(base_errors)) == ["pkg/test_x.py"]


# --- Fix 2: match provenance ------------------------------------------------

SAME_NAME_SOURCE = '''\
def outer_a():
    def wrapper(x):
        return x

class B:
    def wrapper(self, x):
        return x
'''

UNIQUE_NAME_SOURCE = '''\
class C:
    def only_one(self):
        return 1
'''


class TestMatchLabelling:
    def test_exact_match_is_labelled(self):
        matches, unmatched = match_functions({"a.py::f"}, {"a.py::f"})
        assert matches == [{"reference": "a.py::f", "changed": "a.py::f",
                            "match_type": "exact", "ambiguous": False}]
        assert unmatched == []

    def test_fallback_match_is_labelled(self):
        matches, _ = match_functions({"d.py::Q.__call__.wrapper"}, {"d.py::wrapper"})
        assert matches[0]["match_type"] == "fallback"
        assert matches[0]["reference"] == "d.py::wrapper"

    def test_ambiguous_when_short_name_repeats_in_file(self):
        ambiguity = {"d.py": short_name_counts(function_ranges(SAME_NAME_SOURCE))}
        assert ambiguity["d.py"]["wrapper"] == 2
        matches, _ = match_functions({"d.py::B.wrapper"}, {"d.py::wrapper"}, ambiguity)
        assert matches[0]["match_type"] == "fallback" and matches[0]["ambiguous"] is True

    def test_not_ambiguous_when_short_name_is_unique(self):
        ambiguity = {"c.py": short_name_counts(function_ranges(UNIQUE_NAME_SOURCE))}
        matches, _ = match_functions({"c.py::C.only_one"}, {"c.py::only_one"}, ambiguity)
        assert matches[0]["ambiguous"] is False

    def test_exact_match_never_ambiguous(self):
        ambiguity = {"d.py": {"wrapper": 5}}
        matches, _ = match_functions({"d.py::wrapper"}, {"d.py::wrapper"}, ambiguity)
        assert matches[0]["match_type"] == "exact" and matches[0]["ambiguous"] is False

    def test_analyze_patch_reports_both_overlap_counts(self, tmp_path):
        instance = {
            "instance_id": "demo__demo-1", "repo": "demo/demo", "base_commit": "0" * 40,
            "test_functions": '{"pkg/mod.py": ["top_level"]}', "patch_functions": "{}",
        }
        diff = ("diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
                "@@ -7,2 +7,2 @@ def top_level(n):\n def top_level(n):\n-    return 1\n+    return 2\n")
        row = analyze_patch(instance, diff, mirror_dir=str(tmp_path))
        assert "overlap_with_targets" in row and "overlap_with_targets_exact" in row
        assert row["n_ambiguous_matches"] == 0


# --- Fix 3: coverage mapping ------------------------------------------------

class TestExecutedLines:
    payload = {"files": {
        "astropy/units/core.py": {"executed_lines": [10, 11, 12]},
        "./pkg/mod.py": {"executed_lines": [5]},
        "/testbed/deep/nested/file.py": {"executed_lines": [1, 2]},
    }}

    def test_exact_path(self):
        assert executed_lines(self.payload, "astropy/units/core.py") == {10, 11, 12}

    def test_leading_dot_slash_is_normalised(self):
        assert executed_lines(self.payload, "pkg/mod.py") == {5}

    def test_suffix_match_for_absolute_keys(self):
        assert executed_lines(self.payload, "deep/nested/file.py") == {1, 2}

    def test_unknown_file_returns_empty(self):
        assert executed_lines(self.payload, "not/here.py") == set()

    def test_empty_payload_is_safe(self):
        assert executed_lines(None, "a.py") == set()
        assert executed_lines({}, "a.py") == set()

    def test_on_path_decision_uses_function_span(self):
        """A function is on the timed path when any of its lines executed."""
        ranges = {n: (s, e) for n, s, e in function_ranges(UNIQUE_NAME_SOURCE)}
        span = ranges["C.only_one"]
        ran = {span[0]}
        assert any(span[0] <= line <= span[1] for line in ran)
        assert not any(span[0] <= line <= span[1] for line in {span[1] + 50})


# --- Fix 4: provenance audit ------------------------------------------------

class TestProvenanceAudit:
    def test_parses_offset_without_colon(self):
        """The driver writes %z, e.g. -0700, which fromisoformat rejects on 3.9."""
        assert parse_iso("2026-09-26T16:04:08-0700") is not None

    def test_parses_standard_iso(self):
        assert parse_iso("2026-09-26T16:42:57-07:00") is not None

    def test_rejects_garbage(self):
        assert parse_iso("not a date") is None and parse_iso(None) is None

    def test_orders_before_and_after(self):
        before = parse_iso("2026-09-26T16:04:08-0700")
        fixed = parse_iso("2026-09-26T16:42:57-07:00")
        assert before < fixed

    @pytest.mark.parametrize("text,expected", [
        ("git log --oneline", ["git_log"]),
        ("git log --all --oneline", ["git_all", "git_log"]),
        ("git show HEAD~3:file.py", ["git_show"]),
        ("git tag -l", ["git_tag"]),
        ("git -C /repo log", ["git_log"]),
    ])
    def test_history_commands_detected(self, tmp_path, text, expected):
        (tmp_path / "t.jsonl").write_text(json.dumps({"x": text}))
        hits, _, _ = scan_transcripts(str(tmp_path))
        assert sorted(hits["t.jsonl"]) == sorted(expected)

    def test_plain_code_is_not_flagged(self, tmp_path):
        (tmp_path / "t.log").write_text("I optimized the loop using a closed form.")
        hits, fidelity, scanned = scan_transcripts(str(tmp_path))
        assert hits == {} and scanned == 1

    def test_fidelity_final_text_only(self, tmp_path):
        """Transcripts without tool calls cannot prove history went unread."""
        (tmp_path / "t.log").write_text("Summary of what I changed.")
        _, fidelity, _ = scan_transcripts(str(tmp_path))
        assert fidelity == "final_text_only"

    def test_fidelity_tool_calls_visible(self, tmp_path):
        (tmp_path / "t.jsonl").write_text(json.dumps({"type": "tool_use", "name": "Bash"}))
        _, fidelity, _ = scan_transcripts(str(tmp_path))
        assert fidelity == "tool_calls_visible"

    def test_no_transcripts(self, tmp_path):
        hits, fidelity, scanned = scan_transcripts(str(tmp_path / "missing"))
        assert hits == {} and fidelity == "no_transcripts" and scanned == 0


# --- Fix 2b: changes invisible to the base AST ------------------------------

DECORATED_BASE = """\
def first():
    return 1


def target():
    return 2
"""

ADD_HELPER_AND_DECORATOR = """\
diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -2,4 +2,11 @@ def first():
     return 1
 
 
+def helper(func):
+    def inner(*a):
+        return func(*a)
+    return inner
+
+
+@helper
 def target():
"""


class TestInvisibleChanges:
    """A new def is absent from the base AST; a decorator anchors above its def."""

    def test_added_definitions_found(self):
        added = parse_diff(ADD_HELPER_AND_DECORATOR)["pkg/mod.py"]["added"]
        assert added_definitions(added) == {"helper", "inner"}

    def test_added_definitions_ignores_ordinary_lines(self):
        assert added_definitions([(1, "+    return n * 2", 5)]) == set()

    def test_decorator_attributed_to_following_function(self):
        ranges = function_ranges(DECORATED_BASE)
        target_start = {n: s for n, s, _ in ranges}["target"]
        added = [(9, "+@helper", target_start - 1)]
        assert decorated_functions(ranges, added) == {"target"}

    def test_decorator_without_following_def_is_ignored(self):
        ranges = function_ranges(DECORATED_BASE)
        assert decorated_functions(ranges, [(9, "+@helper", 999)]) == set()

    def test_non_decorator_added_line_ignored(self):
        ranges = function_ranges(DECORATED_BASE)
        target_start = {n: s for n, s, _ in ranges}["target"]
        assert decorated_functions(ranges, [(9, "+x = 1", target_start - 1)]) == set()

    def test_parse_diff_carries_anchor(self):
        added = parse_diff(ADD_HELPER_AND_DECORATOR)["pkg/mod.py"]["added"]
        assert all(len(entry) == 3 for entry in added), "added entries carry (lineno, text, anchor)"


# --- Fix 3b: import-time execution must not count as "ran" ------------------

DOCSTRING_FN = '''\
def outer():
    """Explain."""
    value = 1
    return value
'''


class TestBodyRanges:
    """A def line executes at import; only the body shows the function ran."""

    def test_body_skips_def_and_docstring(self):
        from patch_scope import function_body_ranges
        (name, start, end), = function_body_ranges(DOCSTRING_FN)
        assert name == "outer"
        assert DOCSTRING_FN.splitlines()[start - 1].strip() == "value = 1"

    def test_full_span_starts_at_def(self):
        from patch_scope import function_ranges
        (_, start, _), = function_ranges(DOCSTRING_FN)
        assert DOCSTRING_FN.splitlines()[start - 1].startswith("def ")

    def test_import_only_execution_reads_as_not_run(self):
        from patch_scope import function_body_ranges, function_ranges
        full = {n: (s, e) for n, s, e in function_ranges(DOCSTRING_FN)}["outer"]
        body = {n: (s, e) for n, s, e in function_body_ranges(DOCSTRING_FN)}["outer"]
        executed_at_import = {full[0]}          # just the def line
        assert any(full[0] <= l <= full[1] for l in executed_at_import), "full span is fooled"
        assert not any(body[0] <= l <= body[1] for l in executed_at_import), "body span is not"

    def test_body_only_function_is_safe(self):
        from patch_scope import function_body_ranges
        name, start, _ = function_body_ranges("def f():\n    pass\n")[0]
        assert (name, start) == ("f", 2)

    def test_syntax_error_yields_nothing(self):
        from patch_scope import function_body_ranges
        assert function_body_ranges("def broken(:") == []


# --- Fix 5: the patch reaches the container, and "applied" means applied -----

class TestPatchDelivery:
    """The tar member has to be named for the destination, not the source.

    docker_utils.copy_to_container names it after the source file, so the
    archive unpacks beside the requested path under a different name and the
    path the apply command reads never exists.
    """

    def test_tar_member_is_named_for_the_destination(self, tmp_path):
        import io
        import tarfile

        from timed_path import copy_file_to_container

        src = tmp_path / "sympy__sympy-26358.patch"
        src.write_text("diff --git a/x.py b/x.py\n")
        sent = {}

        class FakeContainer:
            def exec_run(self, cmd):
                sent["mkdir"] = cmd

            def put_archive(self, path, data):
                sent["path"] = path
                sent["names"] = tarfile.open(fileobj=io.BytesIO(data)).getnames()

        copy_file_to_container(FakeContainer(), src, "/tmp/timed_path.diff")
        assert sent["path"] == "/tmp"
        assert sent["names"] == ["timed_path.diff"]


class TestApplyVerification:
    """`git diff --stat` alone answers the wrong question.

    Several eval images ship /testbed already dirty from their install step's
    sed, so the old check read as success no matter what the apply did.
    """

    def test_reads_modified_and_added_paths(self):
        from timed_path import porcelain_paths

        assert porcelain_paths(
            " M sympy/integrals/heurisch.py\n?? new_file.py\nA  staged.py\n"
        ) == {"sympy/integrals/heurisch.py", "new_file.py", "staged.py"}

    def test_rename_keeps_the_new_name(self):
        from timed_path import porcelain_paths

        assert porcelain_paths('R  old.py -> new.py\n') == {"new.py"}

    def test_blank_lines_are_ignored(self):
        from timed_path import porcelain_paths

        assert porcelain_paths("\n\n M a.py\n") == {"a.py"}

    def test_dirty_baseline_does_not_look_like_an_applied_patch(self):
        from timed_path import porcelain_paths

        baseline = porcelain_paths(" M pyproject.toml\n")          # astropy images
        patched = {"astropy/units/core.py"}
        assert [p for p in patched if p not in baseline] == ["astropy/units/core.py"]
