"""Unit tests for the diff parser, AST function mapping, and pattern flags.

These use small synthetic diffs rather than benchmark data so they run without
Docker, a network, or the dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from common import flatten_functions, parse_function_map
from flag_patterns import scan_patch
from patch_scope import (
    analyze_patch, function_ranges, functions_for_lines, functions_from_sections,
    match_functions, parse_diff,
)

BASE_SOURCE = '''\
import functools


CONSTANT = 1


def top_level(n):
    total = 0
    for i in range(n):
        total += i
    return total


class Widget:
    """A widget."""

    def method_a(self, x):
        return x * 2

    @property
    def size(self):
        return 3

    def method_b(self, y):
        def inner(z):
            return z + 1
        return inner(y)
'''

DIFF_MODIFY_FUNCTION = '''\
diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -7,5 +7,2 @@ def top_level(n):
 def top_level(n):
-    total = 0
-    for i in range(n):
-        total += i
-    return total
+    return n * (n - 1) // 2
'''

DIFF_MODIFY_METHOD = '''\
diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -17,2 +17,2 @@ class Widget:
     def method_a(self, x):
-        return x * 2
+        return x + x
'''


class TestDiffParser:
    def test_collects_file_and_added_lines(self):
        parsed = parse_diff(DIFF_MODIFY_FUNCTION)
        assert list(parsed) == ["pkg/mod.py"]
        added = [t for _, t in parsed["pkg/mod.py"]["added"]]
        assert any("n * (n - 1)" in t for t in added)

    def test_records_base_line_numbers(self):
        parsed = parse_diff(DIFF_MODIFY_FUNCTION)
        assert parsed["pkg/mod.py"]["old_lines"], "removed lines must map to base line numbers"

    def test_captures_section_header(self):
        parsed = parse_diff(DIFF_MODIFY_FUNCTION)
        assert any("top_level" in s for s in parsed["pkg/mod.py"]["sections"])

    def test_malformed_diff_returns_empty(self):
        assert parse_diff("this is not a diff") == {}

    def test_empty_patch_returns_empty(self):
        assert parse_diff("") == {}


class TestFunctionRanges:
    def test_finds_functions_classes_and_methods(self):
        names = {n for n, _, _ in function_ranges(BASE_SOURCE)}
        assert "top_level" in names
        assert "Widget" in names
        assert "Widget.method_a" in names
        assert "Widget.method_b.inner" in names, "nested functions should be qualified"

    def test_decorated_method_starts_at_decorator(self):
        ranges = {n: (s, e) for n, s, e in function_ranges(BASE_SOURCE)}
        start, _ = ranges["Widget.size"]
        assert BASE_SOURCE.splitlines()[start - 1].strip() == "@property"

    def test_syntax_error_yields_no_ranges(self):
        assert function_ranges("def broken(:\n  pass") == []


class TestLineMapping:
    def test_maps_line_to_enclosing_function(self):
        ranges = function_ranges(BASE_SOURCE)
        names, module_level = functions_for_lines(ranges, {8})
        assert names == {"top_level"} and module_level == 0

    def test_picks_innermost_scope(self):
        ranges = function_ranges(BASE_SOURCE)
        inner_line = BASE_SOURCE.splitlines().index("            return z + 1") + 1
        names, _ = functions_for_lines(ranges, {inner_line})
        assert names == {"Widget.method_b.inner"}

    def test_module_level_change_is_counted_not_named(self):
        ranges = function_ranges(BASE_SOURCE)
        names, module_level = functions_for_lines(ranges, {4})
        assert names == set() and module_level == 1

    def test_section_header_fallback(self):
        assert functions_from_sections(["def top_level(n):"]) == {"top_level"}
        assert functions_from_sections(["class Widget:"]) == {"Widget"}
        assert functions_from_sections([""]) == set()


class TestAnalyzePatch:
    """analyze_patch with no mirrors available, so it uses hunk headers."""

    instance = {
        "instance_id": "demo__demo-1",
        "repo": "demo/demo",
        "base_commit": "0" * 40,
        "test_functions": '{"pkg/mod.py": ["top_level"]}',
        "patch_functions": '{"pkg/other.py": ["helper"]}',
    }

    def test_detects_target_only_patch(self, tmp_path):
        row = analyze_patch(self.instance, DIFF_MODIFY_FUNCTION, mirror_dir=str(tmp_path))
        assert row["functions_changed"] == ["pkg/mod.py::top_level"]
        assert row["overlap_with_targets"] == ["pkg/mod.py::top_level"]
        assert row["touches_only_targets"] is True
        assert row["overlap_with_expert"] == []
        assert row["source_resolved"] is False

    def test_non_target_patch_is_not_flagged_as_target_only(self, tmp_path):
        row = analyze_patch(self.instance, DIFF_MODIFY_METHOD, mirror_dir=str(tmp_path))
        assert row["touches_only_targets"] is False

    def test_empty_patch_is_not_target_only(self, tmp_path):
        row = analyze_patch(self.instance, "", mirror_dir=str(tmp_path))
        assert row["touches_only_targets"] is False
        assert row["functions_changed"] == []


def added(*lines):
    """Wrap lines in a minimal one-hunk diff so scan_patch can read them."""
    body = "".join(f"+{l}\n" for l in lines)
    return ("diff --git a/pkg/mod.py b/pkg/mod.py\n"
            "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
            f"@@ -1,0 +1,{len(lines)} @@\n{body}")


class TestPatternFlags:
    @pytest.mark.parametrize("line,category", [
        ("@functools.lru_cache(maxsize=None)", "cache_decorator"),
        ("@cache", "cache_decorator"),
        ("_result_cache = {}", "cache_container"),
        ("_MEMO = dict()", "cache_container"),
        ("if arr.dtype == np.float64:", "shape_special_case"),
        ("if x.shape == (3, 3):", "shape_special_case"),
        ("if len(items) == 7:", "length_special_case"),
        ("frame = sys._getframe(1)", "stack_inspection"),
        ("if inspect.stack()[1]:", "stack_inspection"),
        ("if 'pytest' in sys.modules:", "test_env_detection"),
        ("flag = os.environ.get('FAST')", "test_env_detection"),
        ("np.dot = faster_dot", "monkey_patch"),
    ])
    def test_flags_expected_patterns(self, line, category):
        assert category in {f["category"] for f in scan_patch(added(line))}

    @pytest.mark.parametrize("line", [
        "return n * (n - 1) // 2",
        "total = sum(range(n))",
        "if value is None:",
        "result = np.dot(a, b)",              # a call, not an assignment
        "# lru_cache would help here",        # comment prose
        "if arr.dtype is not None:",
    ])
    def test_ignores_ordinary_code(self, line):
        assert scan_patch(added(line)) == []

    def test_flag_records_location_and_text(self):
        flags = scan_patch(added("_cache = {}"))
        assert flags[0]["file"] == "pkg/mod.py"
        assert flags[0]["line"] == 1
        assert "_cache" in flags[0]["text"]
        assert flags[0]["rationale"]


class TestFunctionMapParsing:
    def test_parses_json_field(self):
        assert parse_function_map('{"a.py": ["f", "g"]}') == {"a.py": ["f", "g"]}

    def test_parses_defaultdict_repr_from_problem_statement(self):
        raw = ("Optimize these:\n"
               "defaultdict(<class 'list'>, {'a.py': ['f'], 'b.py': ['C.m']})\n"
               "Conditions apply.")
        assert parse_function_map(raw) == {"a.py": ["f"], "b.py": ["C.m"]}

    def test_handles_missing_and_garbage(self):
        assert parse_function_map(None) == {}
        assert parse_function_map("") == {}
        assert parse_function_map("no mapping here") == {}

    def test_flatten(self):
        assert flatten_functions({"a.py": ["f", "g"]}) == {"a.py::f", "a.py::g"}


class TestFunctionMatching:
    """The dataset qualifies some names and not others; matching must cope."""

    def test_exact_match(self):
        matched, unmatched = match_functions({"a.py::f"}, {"a.py::f"})
        assert matched == {"a.py::f"} and unmatched == []

    def test_bare_reference_matches_qualified_change(self):
        # dataset says "wrapper"; the AST resolves "QuantityInput.__call__.wrapper"
        matched, unmatched = match_functions(
            {"d.py::QuantityInput.__call__.wrapper"}, {"d.py::wrapper"})
        assert matched == {"d.py::wrapper"} and unmatched == []

    def test_tail_match_does_not_cross_files(self):
        matched, unmatched = match_functions({"a.py::C.wrapper"}, {"b.py::wrapper"})
        assert matched == set() and unmatched == ["a.py::C.wrapper"]

    def test_reports_unmatched_changes(self):
        matched, unmatched = match_functions({"a.py::f", "a.py::g"}, {"a.py::f"})
        assert matched == {"a.py::f"} and unmatched == ["a.py::g"]

    def test_empty_inputs(self):
        assert match_functions(set(), {"a.py::f"}) == (set(), [])
        assert match_functions({"a.py::f"}, set()) == (set(), ["a.py::f"])
