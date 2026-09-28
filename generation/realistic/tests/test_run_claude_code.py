"""Unit tests for the generation driver's patch filtering and baseline logic.

These are deliberately Docker-free and Claude-free: every function under test
is a pure transformation of text that a container produced, so the interesting
cases (a test edit, a recompiled extension, a scratch file at the repo root, a
checkout that still has history) can be exercised as strings.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from run_claude_code import (
    BUILD_ARTIFACT_RE, MARKER, TEST_PATH_RE,
    agent_environment, baseline_problems, build_baseline_script, build_prompt,
    build_verify_script, claude_command, classify_worktree, filtered_by,
    is_scratch, looks_like_usage_limit, parse_baseline_report, parse_markers,
    parse_numstat_binaries, parse_porcelain_z, redact, remote_image_key,
    verify_cli_capabilities, verify_hidden_flags,
    classify_claude_failure, earns_a_prediction, parse_result_line,
    KEEP_STATUSES, PARTIAL_STATUSES, count_assistant_turns, count_tool_calls,
    earns_a_partial_prediction, partial_stem, run_paths, turn_count,
    ARMS, CONSTRAINT_BLOCK, RUN_LOG_FIELDS, append_jsonl, hint_sentence,
    patch_touches_tests, prompt_style_for, run_log_record, termination_for,
)


def porcelain(*entries):
    """Build a `git status --porcelain -z -uall` payload from (code, path)."""
    return "".join(f"{code} {path}\0" for code, path in entries)


class TestPorcelainParsing:
    def test_reads_code_and_path(self):
        data = porcelain((" M", "astropy/units/core.py"), ("??", "notes.md"))
        assert parse_porcelain_z(data) == [
            (" M", "astropy/units/core.py"), ("??", "notes.md"),
        ]

    def test_accepts_bytes(self):
        assert parse_porcelain_z(porcelain((" M", "a.py")).encode()) == [(" M", "a.py")]

    def test_rename_consumes_the_source_field(self):
        # -z emits the rename source as its own field, with no status code.
        data = "R  new.py\0old.py\0 M other.py\0"
        assert parse_porcelain_z(data) == [("R ", "new.py"), (" M", "other.py")]

    def test_path_with_spaces_survives(self):
        data = porcelain(("??", "src/a file with spaces.py"))
        assert parse_porcelain_z(data) == [("??", "src/a file with spaces.py")]

    def test_empty_input(self):
        assert parse_porcelain_z("") == []


class TestTestPathMatching:
    @pytest.mark.parametrize("path", [
        "tests/test_units.py",
        "astropy/units/tests/test_core.py",
        "test/helper.py",
        "sympy/core/tests/test_expr.py",
        "xarray/conftest.py",
        "some/dir/thing_test.py",
    ])
    def test_matches_test_files(self, path):
        assert TEST_PATH_RE.search(path)

    @pytest.mark.parametrize("path", [
        # `testing/` is library code in several benchmark repos, not a suite.
        "xarray/testing/assertions.py",
        "astropy/units/core.py",
        "sklearn/utils/latest.py",
    ])
    def test_leaves_library_code_alone(self, path):
        assert not TEST_PATH_RE.search(path)


class TestBuildArtifactMatching:
    @pytest.mark.parametrize("path", [
        "astropy/units/__pycache__/core.cpython-39.pyc",
        "sklearn/tree/_tree.cpython-39-x86_64-linux-gnu.so",
        "build/lib.linux-x86_64/astropy/units.py",
        "astropy.egg-info/PKG-INFO",
        "src/thing.o",
        ".pytest_cache/v/cache/lastfailed",
        "sympy/core/expr.pyc",
    ])
    def test_matches_build_output(self, path):
        assert BUILD_ARTIFACT_RE.search(path)

    @pytest.mark.parametrize("path", [
        "astropy/units/core.py",
        "docs/building.rst",
        "xarray/core/rebuild.py",
    ])
    def test_leaves_sources_alone(self, path):
        assert not BUILD_ARTIFACT_RE.search(path)


class TestScratchDetection:
    def test_top_level_file_is_scratch(self):
        assert is_scratch("benchmark_foo.py")

    def test_ds_store_anywhere_is_scratch(self):
        assert is_scratch("astropy/units/.DS_Store")

    def test_nested_file_is_not_scratch(self):
        assert not is_scratch("astropy/units/core.py")


class TestClassifyWorktree:
    def test_test_edits_are_reverted(self):
        plan = classify_worktree([(" M", "astropy/units/tests/test_core.py")])
        assert plan["revert"] == ["astropy/units/tests/test_core.py"]
        assert filtered_by(plan, "test_file") == ["astropy/units/tests/test_core.py"]

    def test_new_test_files_are_removed_not_reverted(self):
        plan = classify_worktree([("??", "astropy/units/tests/test_new.py")])
        assert plan["remove"] == ["astropy/units/tests/test_new.py"]
        assert plan["revert"] == []

    def test_modified_extension_is_reverted(self):
        # /testbed ships built, so importing the package alone can touch these.
        plan = classify_worktree([(" M", "sklearn/tree/_tree.cpython-39-x86_64-linux-gnu.so")])
        assert filtered_by(plan, "build_artifact") == [
            "sklearn/tree/_tree.cpython-39-x86_64-linux-gnu.so"]
        assert plan["revert"] == ["sklearn/tree/_tree.cpython-39-x86_64-linux-gnu.so"]

    def test_new_pycache_is_removed(self):
        plan = classify_worktree([("??", "astropy/units/__pycache__/core.cpython-39.pyc")])
        assert plan["remove"] == ["astropy/units/__pycache__/core.cpython-39.pyc"]

    def test_scratch_at_root_is_removed(self):
        plan = classify_worktree([("??", "notes.md"), ("??", "benchmark.py")])
        assert plan["remove"] == ["notes.md", "benchmark.py"]
        assert filtered_by(plan, "scratch") == ["notes.md", "benchmark.py"]
        assert plan["new_files"] == []

    def test_modified_root_file_is_kept(self):
        # setup.py is a legitimate thing to edit; only *new* root files are scratch.
        plan = classify_worktree([(" M", "setup.py")])
        assert plan["revert"] == [] and plan["remove"] == []

    def test_new_source_file_is_kept_and_reported(self):
        plan = classify_worktree([("??", "astropy/units/fastpath.py")])
        assert plan["new_files"] == ["astropy/units/fastpath.py"]
        assert plan["remove"] == []

    def test_real_edit_is_left_untouched(self):
        plan = classify_worktree([(" M", "astropy/units/core.py")])
        assert plan["revert"] == [] and plan["remove"] == []
        assert plan["new_files"] == []

    def test_test_rule_wins_over_build_rule(self):
        # A .pyc under tests/ is reported as a test edit, because rule 1 is the
        # one whose violation matters for the benchmark.
        plan = classify_worktree([(" M", "astropy/tests/__pycache__/x.cpython-39.pyc")])
        assert filtered_by(plan, "test_file") == ["astropy/tests/__pycache__/x.cpython-39.pyc"]
        assert filtered_by(plan, "build_artifact") == []

    def test_deleted_source_is_kept_in_the_patch(self):
        plan = classify_worktree([(" D", "astropy/units/old.py")])
        assert plan["revert"] == [] and plan["remove"] == []

    def test_mixed_tree(self):
        plan = classify_worktree([
            (" M", "astropy/units/core.py"),
            (" M", "astropy/units/tests/test_core.py"),
            ("??", "astropy/units/__pycache__/core.cpython-39.pyc"),
            ("??", "scratch.py"),
            ("??", "astropy/units/helper.py"),
        ])
        assert plan["new_files"] == ["astropy/units/helper.py"]
        assert filtered_by(plan, "test_file") == ["astropy/units/tests/test_core.py"]
        assert filtered_by(plan, "scratch") == ["scratch.py"]
        assert filtered_by(plan, "build_artifact") == [
            "astropy/units/__pycache__/core.cpython-39.pyc"]


class TestNumstatBinaries:
    def test_finds_binary_rows(self):
        numstat = "12\t3\tastropy/units/core.py\n-\t-\tastropy/_compiler.so\n"
        assert parse_numstat_binaries(numstat) == ["astropy/_compiler.so"]

    def test_text_only_diff_has_none(self):
        assert parse_numstat_binaries("1\t1\ta.py\n") == []

    def test_renames_keep_the_last_field(self):
        assert parse_numstat_binaries("-\t-\tsrc/a.bin\n") == ["src/a.bin"]

    def test_empty(self):
        assert parse_numstat_binaries("") == []


class TestBaselineScript:
    def test_replaces_history_with_a_single_commit(self):
        script = build_baseline_script("/testbed")
        assert "rm -rf .git" in script
        assert "git -c init.defaultBranch=main init -q ." in script
        assert "git commit -q --no-verify -m base" in script
        # -f is required: some repos track files their own .gitignore matches.
        assert "git add -A -f" in script
        assert "rm -rf .git/logs" in script

    def test_does_not_check_out_the_base_commit(self):
        # The point of the rewrite: the baseline is the image's post-setup
        # tree, not a fresh checkout of base_commit. A reset/checkout here
        # would throw away the setup step's own edits (astropy's pyproject).
        script = build_baseline_script("/testbed")
        assert "git reset --hard" not in script
        assert "git checkout" not in script
        assert "git clone" not in script

    def test_clears_pycache_but_keeps_compiled_extensions(self):
        script = build_baseline_script("/testbed")
        assert "__pycache__" in script and "*.pyc" in script
        # deleting these would break `import astropy` inside the container
        assert "*.so" not in script

    def test_honours_the_repo_dir(self):
        assert "cd /elsewhere" in build_baseline_script("/elsewhere")


class TestBaselineVerification:
    CLEAN = (
        f"{MARKER}commits=1\n{MARKER}tags=0\n{MARKER}remotes=0\n"
        f"{MARKER}reflog=0\n{MARKER}dirty=0\n{MARKER}head=abc123\n"
    )

    def test_verify_script_asks_for_every_invariant(self):
        script = build_verify_script("/testbed")
        for probe in ("git log --all", "git tag", "git remote", "git reflog",
                      "git status --porcelain", "git rev-parse HEAD"):
            assert probe in script

    def test_clean_baseline_has_no_problems(self):
        report = parse_baseline_report(self.CLEAN)
        assert report == {"commits": 1, "tags": 0, "remotes": 0, "reflog": 0,
                          "dirty": 0, "head": "abc123"}
        assert baseline_problems(report) == []

    def test_leftover_history_is_reported(self):
        report = parse_baseline_report(self.CLEAN.replace("commits=1", "commits=4212"))
        problems = baseline_problems(report)
        assert len(problems) == 1 and "4212 commit(s)" in problems[0]

    def test_tags_and_remotes_are_reported(self):
        text = self.CLEAN.replace("tags=0", "tags=97").replace("remotes=0", "remotes=1")
        problems = baseline_problems(parse_baseline_report(text))
        assert any("97 tag(s)" in p for p in problems)
        assert any("1 remote(s)" in p for p in problems)

    def test_dirty_baseline_is_reported(self):
        # Anything uncommitted at this point would land in the model's patch.
        problems = baseline_problems(parse_baseline_report(
            self.CLEAN.replace("dirty=0", "dirty=3")))
        assert any("3 uncommitted path(s)" in p for p in problems)

    def test_missing_output_is_reported_not_ignored(self):
        problems = baseline_problems(parse_baseline_report(""))
        assert len(problems) == 6  # five counters plus the missing HEAD

    def test_markers_survive_surrounding_noise(self):
        noisy = f"warning: something\n{MARKER}commits=1\nOn branch main\n"
        assert parse_markers(noisy) == {"commits": "1"}

    def test_marker_value_may_contain_equals(self):
        assert parse_markers(f"{MARKER}cli_version=2.1.283 (Claude Code)") == {
            "cli_version": "2.1.283 (Claude Code)"}


class TestPrompt:
    """The prompt is protocol, so these assert on the bytes, not the shape."""

    ISSUE = "Make `foo` faster.\n\nIt is slow on large inputs."

    EXPECTED_NO_HINT = (
        "Make `foo` faster.\n"
        "\n"
        "It is slow on large inputs.\n"
        "\n"
        "Constraints:\n"
        "- Do not modify, delete, or skip any existing test.\n"
        "- Preserve the program's behavior; only performance may change.\n"
        "- When you are done, the final state of the working tree is your patch.\n"
        "  It will be taken as a unified diff against the starting commit."
    )

    EXPECTED_HINTED = EXPECTED_NO_HINT + (
        "\nThe developer's fix for this issue changed src/foo.py, "
        "in Foo.compute, Foo.reduce."
    )

    def inst(self, iid="repo__repo-1"):
        return {"instance_id": iid, "problem_statement_realistic": self.ISSUE}

    def test_no_hint_prompt_is_byte_for_byte(self):
        assert build_prompt(self.inst(), "no-hint") == self.EXPECTED_NO_HINT

    def test_hinted_prompt_is_byte_for_byte(self):
        hints = {"repo__repo-1": {"file": "src/foo.py",
                                  "methods": ["Foo.compute", "Foo.reduce"]}}
        assert build_prompt(self.inst(), "hinted", hints) == self.EXPECTED_HINTED

    def test_the_arms_differ_by_exactly_one_line(self):
        hints = {"repo__repo-1": {"file": "src/foo.py",
                                  "methods": ["Foo.compute", "Foo.reduce"]}}
        plain = build_prompt(self.inst(), "no-hint")
        hinted = build_prompt(self.inst(), "hinted", hints)
        extra = hinted[len(plain):].splitlines()
        assert hinted.startswith(plain)
        assert extra == ["", "The developer's fix for this issue changed "
                             "src/foo.py, in Foo.compute, Foo.reduce."]

    def test_the_issue_text_is_verbatim_and_first(self):
        assert build_prompt(self.inst(), "no-hint").startswith(self.ISSUE)

    def test_a_blank_line_separates_issue_and_constraints(self):
        assert self.ISSUE + "\n\n" + CONSTRAINT_BLOCK == \
            build_prompt(self.inst(), "no-hint")

    def test_nothing_about_benchmarks_or_speedup_is_added(self):
        # The protocol forbids role text, examples, and any mention of the
        # benchmark: the model must not know it is being timed.
        added = build_prompt(self.inst(), "no-hint")[len(self.ISSUE):].lower()
        for word in ("benchmark", "speedup", "speed up", "swe-perf",
                     "you are a", "for example", "performance engineer",
                     "your task is", "expert"):
            assert word not in added, word

    def test_a_single_method_needs_no_list(self):
        assert hint_sentence({"file": "a/b.py", "methods": "solve"}) == (
            "The developer's fix for this issue changed a/b.py, in solve.")

    def test_falls_back_to_problem_statement(self):
        inst = {"instance_id": "x__y-1", "problem_statement": self.ISSUE}
        assert build_prompt(inst, "no-hint") == self.EXPECTED_NO_HINT

    def test_unknown_arm_is_refused(self):
        with pytest.raises(ValueError):
            build_prompt(self.inst(), "sideways")


class TestMissingHintFailsLoudly:
    ISSUE = "slow"

    def inst(self, iid="repo__repo-1"):
        return {"instance_id": iid, "problem_statement_realistic": self.ISSUE}

    def test_no_entry_at_all(self):
        with pytest.raises(KeyError, match="repo__repo-1"):
            build_prompt(self.inst(), "hinted", {})

    def test_hints_for_a_different_instance(self):
        with pytest.raises(KeyError, match="repo__repo-1"):
            build_prompt(self.inst(), "hinted",
                         {"other__other-2": {"file": "a.py", "methods": ["m"]}})

    def test_entry_missing_the_file(self):
        with pytest.raises(KeyError, match="file"):
            build_prompt(self.inst(), "hinted",
                         {"repo__repo-1": {"methods": ["m"]}})

    def test_entry_missing_the_methods(self):
        with pytest.raises(KeyError, match="methods"):
            build_prompt(self.inst(), "hinted",
                         {"repo__repo-1": {"file": "a.py"}})

    def test_empty_methods_list_is_not_a_hint(self):
        with pytest.raises(KeyError, match="methods"):
            build_prompt(self.inst(), "hinted",
                         {"repo__repo-1": {"file": "a.py", "methods": []}})

    def test_no_hint_arm_never_needs_hints(self):
        assert build_prompt(self.inst(), "no-hint", {})


class TestArmsKeepOutputsApart:
    def test_each_arm_gets_its_own_files(self, tmp_path):
        out = tmp_path / "claude_code_preds.jsonl"
        plain = run_paths(out, 1, 1, "no-hint")
        hinted = run_paths(out, 1, 1, "hinted")
        assert plain[0].name == "claude_code_preds_no-hint.jsonl"
        assert hinted[0].name == "claude_code_preds_hinted.jsonl"
        # No path of one arm may collide with any path of the other.
        assert not set(plain) & set(hinted)

    def test_prompt_style_names_the_arm(self):
        assert prompt_style_for("no-hint") == "verbatim+constraints"
        assert prompt_style_for("hinted") == "verbatim+constraints+hint"

    def test_both_arms_are_offered(self):
        assert ARMS == ("no-hint", "hinted")


class TestClaudeCommand:
    CMD = claude_command("do the thing", "claude-opus-5", "xhigh", 100, 1800)

    def test_prompt_is_its_own_argv_entry(self):
        # A list, never a shell string: the prompt is dataset text and must
        # reach the CLI byte for byte.
        assert self.CMD[self.CMD.index("-p") + 1] == "do the thing"

    def test_model_and_effort(self):
        assert self.CMD[self.CMD.index("--model") + 1] == "claude-opus-5"
        assert self.CMD[self.CMD.index("--effort") + 1] == "xhigh"

    def test_unsets_the_api_key_variables(self):
        joined = " ".join(self.CMD)
        assert "-u ANTHROPIC_API_KEY" in joined
        assert "-u ANTHROPIC_AUTH_TOKEN" in joined

    def test_denies_network_tools(self):
        for tool in ("WebFetch", "WebSearch", "Bash(curl:*)", "Bash(wget:*)"):
            assert tool in self.CMD
        assert self.CMD.index("--disallowedTools") < self.CMD.index("WebFetch")

    def test_wall_clock_is_enforced_in_the_container(self):
        assert self.CMD[0] == "/usr/bin/timeout"
        assert "1800" in self.CMD[:5]

    def test_runs_without_permission_prompts(self):
        assert "--dangerously-skip-permissions" in self.CMD

    def test_streams_json_transcript(self):
        assert self.CMD[self.CMD.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in self.CMD


class TestAgentEnvironment:
    ENV = agent_environment("testbed", "sk-ant-oat-SECRET")

    def test_conda_env_comes_first_on_path(self):
        assert self.ENV["PATH"].split(":")[0] == "/opt/miniconda3/envs/testbed/bin"

    def test_cli_is_on_path(self):
        assert "/opt/claude-cli/npm/bin" in self.ENV["PATH"].split(":")

    def test_home_is_the_non_root_user(self):
        assert self.ENV["HOME"] == "/home/nonroot"

    def test_carries_the_oauth_token_and_nothing_else(self):
        assert self.ENV["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-SECRET"
        assert "ANTHROPIC_API_KEY" not in self.ENV
        assert "ANTHROPIC_AUTH_TOKEN" not in self.ENV


class TestRedaction:
    def test_token_never_reaches_a_transcript(self):
        text = 'tool_use {"command": "env"} SECRETTOKEN trailing'
        assert "SECRETTOKEN" not in redact(text, "SECRETTOKEN")
        assert "<REDACTED_OAUTH_TOKEN>" in redact(text, "SECRETTOKEN")

    def test_handles_empty_input(self):
        assert redact("", "tok") == ""
        assert redact(None, "tok") == ""

    def test_no_token_is_a_no_op(self):
        assert redact("hello", "") == "hello"


class TestImageKey:
    def test_matches_the_evaluation_harness_naming(self):
        assert remote_image_key("astropy__astropy-12907") == (
            "docker.io/betty1202/sweb.eval.x86_64.astropy_s_astropy-12907")

    def test_lowercases(self):
        assert remote_image_key("Sympy__Sympy-1") == (
            "docker.io/betty1202/sweb.eval.x86_64.sympy_s_sympy-1")


class TestUsageLimitDetection:
    @pytest.mark.parametrize("text", [
        "Claude usage limit reached", "HTTP 429 Too Many Requests", "rate_limit_error",
    ])
    def test_detects(self, text):
        assert looks_like_usage_limit(text)

    def test_ignores_ordinary_output(self):
        assert not looks_like_usage_limit("finished editing core.py")


class TestCliCapabilityCheck:
    """The CLI in the container is not the host's and moves fast, so the flags
    this driver depends on are confirmed against that build's own --help."""

    HELP = (
        "Usage: claude [options] [command] [prompt]\n"
        "  --effort <level>   Effort level for the current session\n"
        "                     (low, medium, high, xhigh, max)\n"
        "  --model <model>    Model for the current session\n"
        "  --output-format <format>  Output format\n"
        "  --disallowedTools, --disallowed-tools <tools...>\n"
        "  --dangerously-skip-permissions  Bypass all permission checks.\n"
        "  -p, --print        Print response and exit\n"
        "Commands:\n"
        "  setup-token        Set up a long-lived authentication token\n"
    )

    def test_accepts_a_cli_that_supports_everything(self):
        assert verify_cli_capabilities(self.HELP, "xhigh") == []

    def test_rejects_an_unknown_effort_level(self):
        problems = verify_cli_capabilities(self.HELP, "ultra")
        assert len(problems) == 1
        assert "'ultra'" in problems[0]
        # the message quotes the real help line so the fix is obvious
        assert "low, medium, high, xhigh, max" in problems[0]

    def test_reports_a_missing_flag(self):
        stripped = self.HELP.replace("  --effort <level>   Effort level for the "
                                     "current session\n", "")
        problems = verify_cli_capabilities(stripped, "xhigh")
        assert any("--effort" in p for p in problems)

    def test_reports_a_missing_setup_token_command(self):
        stripped = self.HELP.replace("  setup-token        Set up a long-lived "
                                     "authentication token\n", "")
        problems = verify_cli_capabilities(stripped, "xhigh")
        assert any("setup-token" in p for p in problems)

    def test_empty_help_is_a_single_clear_failure(self):
        assert verify_cli_capabilities("", "xhigh") == [
            "`claude --help` produced no output in the container"]


class TestHiddenFlagCheck:
    """--max-turns works but is not in `claude --help`, so it is checked
    against the binary instead of the documentation."""

    OK = {"flag_max_turns": "10", "grep_control": "0"}

    def test_accepts_a_binary_that_contains_the_flag(self):
        assert verify_hidden_flags(self.OK) == []

    def test_reports_a_flag_the_binary_does_not_have(self):
        problems = verify_hidden_flags({"flag_max_turns": "0", "grep_control": "0"})
        assert len(problems) == 1 and "--max-turns" in problems[0]

    def test_a_control_string_that_matches_invalidates_the_probe(self):
        # If the control is found, grep is matching things it should not, and a
        # positive result for the real flag would mean nothing.
        problems = verify_hidden_flags({"flag_max_turns": "10", "grep_control": "3"})
        assert len(problems) == 1 and "unreliable" in problems[0]

    def test_missing_markers_are_a_failure_not_a_pass(self):
        assert verify_hidden_flags({}) != []

    def test_non_numeric_count_is_a_failure(self):
        assert verify_hidden_flags({"flag_max_turns": "?", "grep_control": "0"}) != []

    def test_the_grep_c_zero_zero_reading_is_rejected(self):
        # `grep -c` prints the count and then exits 1 when it is zero, so a
        # "|| echo 0" fallback yields "0 0". An equality test against "0" would
        # quietly accept that as a clean control.
        problems = verify_hidden_flags({"flag_max_turns": "10", "grep_control": "0 0"})
        assert len(problems) == 1 and "unreliable" in problems[0]

    def test_a_missing_control_is_reported_as_a_probe_that_did_not_run(self):
        problems = verify_hidden_flags({"flag_max_turns": "10", "grep_control": ""})
        assert len(problems) == 1 and "did not run" in problems[0]


class TestFailureClassification:
    """An expired token, a 429 and a genuine crash all exit non-zero with an
    empty diff. Only the last says anything about the model, so they must not
    share a status -- the first pilot run recorded three 401s as three
    legitimate "the model changed nothing" predictions."""

    AUTH_RESULT = {
        "is_error": True, "api_error_status": 401, "terminal_reason": "api_error",
        "result_text": "Failed to authenticate. API Error: 401 OAuth access token is invalid.",
    }

    def test_401_is_an_auth_failure(self):
        assert classify_claude_failure(1, self.AUTH_RESULT, "") == "auth_failed"

    def test_403_is_an_auth_failure(self):
        assert classify_claude_failure(1, {"api_error_status": 403}, "") == "auth_failed"

    def test_auth_failure_is_recognised_from_stderr_alone(self):
        assert classify_claude_failure(1, None, "authentication_failed") == "auth_failed"

    def test_429_is_a_usage_limit(self):
        assert classify_claude_failure(1, {"api_error_status": 429}, "") == "usage_limit"

    def test_other_api_errors_are_their_own_status(self):
        result = {"api_error_status": 500, "terminal_reason": "api_error"}
        assert classify_claude_failure(1, result, "") == "api_error"

    def test_exhausted_turn_budget_is_named(self):
        # Observed in the pilot: num_turns 101 against --max-turns 100 exits 1,
        # which a bare exit-code status reports as an indistinguishable crash.
        result = {"is_error": True, "subtype": "error_max_turns",
                  "terminal_reason": "max_turns", "num_turns": 101}
        assert classify_claude_failure(1, result, "") == "max_turns"

    def test_max_turns_earns_no_prediction(self):
        # Cut off mid-edit, exactly like a timeout.
        assert not earns_a_prediction("max_turns")

    def test_a_plain_crash_keeps_the_exit_code(self):
        assert classify_claude_failure(2, None, "segfault") == "claude_exit_2"

    def test_token_counts_are_not_mistaken_for_status_codes(self):
        # The result blob carries usage numbers; a bare \b401\b regex over the
        # whole JSON would match "input_tokens": 401 and call a good run an
        # auth failure.
        result = {"usage": {"input_tokens": 401, "output_tokens": 403},
                  "result_text": "done, made core.py faster"}
        assert classify_claude_failure(1, result, "") == "claude_exit_1"


class TestWhatEarnsAPrediction:
    def test_a_clean_run_is_recorded(self):
        assert earns_a_prediction("ok")

    def test_a_deliberate_no_change_is_recorded(self):
        # The agent ran to completion and chose to change nothing; that is a
        # real result, not a failure.
        assert earns_a_prediction("empty_patch")

    @pytest.mark.parametrize("status", [
        "auth_failed", "api_error", "usage_limit", "timeout", "error",
        "setup_failed", "cancelled", "claude_exit_1", "claude_exit_2",
    ])
    def test_nothing_else_is_recorded(self, status):
        assert not earns_a_prediction(status)

    def test_an_unknown_status_defaults_to_not_recorded(self):
        # An allow-list, so a status added later cannot silently become data.
        assert not earns_a_prediction("some_future_status")


class TestResultLineKeepsErrorFields:
    def test_api_error_fields_survive_parsing(self):
        line = json.dumps({"type": "result", "is_error": True,
                           "api_error_status": 401, "terminal_reason": "api_error",
                           "result": "Failed to authenticate.", "num_turns": 1})
        parsed = parse_result_line(line)
        assert parsed["api_error_status"] == 401
        assert parsed["terminal_reason"] == "api_error"


def stream(*objs):
    """A stream-json transcript: one JSON object per line."""
    return "\n".join(json.dumps(o) for o in objs) + "\n"


def assistant(*tool_names):
    """An assistant message that calls the named tools."""
    content = [{"type": "text", "text": "thinking"}]
    content += [{"type": "tool_use", "name": n, "input": {"x": 1}} for n in tool_names]
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


class TestTurnCounting:
    def test_the_result_line_is_believed_when_the_run_finished(self):
        turns, source = turn_count({"num_turns": 87}, stream(assistant("Read")))
        assert (turns, source) == (87, "result_line")

    def test_zero_turns_from_the_result_line_is_still_the_result_line(self):
        # `or` on the count would quietly fall through to the transcript here.
        turns, source = turn_count({"num_turns": 0}, stream(assistant("Read")))
        assert (turns, source) == (0, "result_line")

    def test_a_killed_run_is_counted_off_the_transcript(self):
        # No result line at all: `timeout` took the CLI down before it printed.
        text = stream({"type": "system", "subtype": "init"},
                      assistant("Read"),
                      {"type": "user", "message": {"content": "tool result"}},
                      assistant("Edit"))
        turns, source = turn_count(None, text)
        assert (turns, source) == (2, "transcript")

    def test_a_result_line_without_num_turns_falls_back(self):
        turns, source = turn_count({"is_error": True}, stream(assistant("Read")))
        assert (turns, source) == (1, "transcript")

    def test_a_truncated_last_line_does_not_break_the_count(self):
        text = stream(assistant("Read"), assistant("Bash")) + '{"type": "assist'
        assert count_assistant_turns(text) == 2

    def test_plain_text_on_the_stream_is_ignored(self):
        text = "Loading...\n" + stream(assistant("Read")) + "killed\n"
        assert count_assistant_turns(text) == 1


class TestToolCallCounting:
    def test_every_tool_use_block_counts(self):
        text = stream(assistant("Read", "Grep"), assistant(), assistant("Bash"))
        assert count_tool_calls(text) == 3

    def test_a_transcript_with_no_tools_counts_zero(self):
        assert count_tool_calls(stream(assistant())) == 0

    def test_an_empty_transcript_counts_zero(self):
        assert count_tool_calls("") == 0
        assert count_assistant_turns("") == 0


class TestPartialPredictions:
    @pytest.mark.parametrize("status", ["timeout", "max_turns"])
    def test_a_cut_off_run_yields_a_partial(self, status):
        assert earns_a_partial_prediction(status)

    @pytest.mark.parametrize("status", [
        "ok", "empty_patch", "auth_failed", "usage_limit", "api_error",
        "error", "setup_failed", "claude_exit_1",
    ])
    def test_nothing_else_yields_a_partial(self, status):
        assert not earns_a_partial_prediction(status)

    def test_a_partial_is_never_also_a_prediction(self):
        # The two files must not disagree about the same instance.
        assert not set(KEEP_STATUSES) & set(PARTIAL_STATUSES)
        for status in PARTIAL_STATUSES:
            assert not earns_a_prediction(status)


class TestPartialPath:
    def test_partial_sits_beside_the_predictions_file(self, tmp_path):
        out = tmp_path / "claude_code_pilot2_preds.jsonl"
        preds, meta, prov, partial = run_paths(out, 1, 1, "no-hint")
        assert preds.name == "claude_code_pilot2_preds_no-hint.jsonl"
        assert partial.name == "claude_code_pilot2_partial_preds_no-hint.jsonl"
        assert meta.name == "claude_code_pilot2_preds_no-hint_meta.jsonl"
        assert prov.name == "claude_code_pilot2_preds_no-hint_provenance.jsonl"

    def test_each_run_of_a_repeat_sweep_gets_its_own_partial(self, tmp_path):
        out = tmp_path / "claude_code_preds.jsonl"
        preds, _, _, partial = run_paths(out, 2, 3, "hinted")
        assert preds.name == "claude_code_preds_run2_hinted.jsonl"
        assert partial.name == "claude_code_partial_preds_run2_hinted.jsonl"

    def test_a_name_that_does_not_end_in_preds_is_suffixed(self):
        assert partial_stem("results") == "results_partial"

    def test_only_the_trailing_preds_is_renamed(self):
        assert partial_stem("preds_of_preds") == "preds_of_partial_preds"

    def test_the_partial_path_is_never_the_predictions_path(self, tmp_path):
        for name in ("a_preds.jsonl", "results.jsonl", "preds.jsonl"):
            for arm in ARMS:
                preds, _, _, partial = run_paths(tmp_path / name, 1, 1, arm)
                assert preds != partial


class Args:
    """The handful of argparse fields the run log reads."""
    def __init__(self, **kw):
        self.model = "claude-opus-5"
        self.effort = "high"
        self.max_turns = 2000
        self.timeout = 10800
        self.operator = "SN"
        self.notes = ""
        self.__dict__.update(kw)


def meta_for(status="ok", **kw):
    m = {
        "instance_id": "repo__repo-1", "arm": "no-hint", "status": status,
        "turns": 215, "turns_source": "result_line", "tool_calls": 214,
        "duration_s": 2209.0, "claude_version": "2.1.283 (Claude Code)",
        "claude_result": {"usage": {
            "input_tokens": 396, "output_tokens": 144656,
            "cache_creation_input_tokens": 309255,
            "cache_read_input_tokens": 36948640,
        }},
    }
    m.update(kw)
    return m


class TestTerminationMapping:
    def test_a_clean_run_finished(self):
        assert termination_for("ok") == "finished"

    def test_a_deliberate_no_change_finished(self):
        # The agent ran to completion; changing nothing is a finished run.
        assert termination_for("empty_patch") == "finished"

    def test_an_exhausted_turn_budget_is_the_turn_limit(self):
        assert termination_for("max_turns") == "turn_limit"

    def test_a_killed_run_is_the_time_limit(self):
        assert termination_for("timeout") == "time_limit"

    @pytest.mark.parametrize("status", [
        "error", "setup_failed", "auth_failed", "usage_limit", "api_error",
        "claude_exit_1", "cancelled",
    ])
    def test_everything_else_is_an_error(self, status):
        assert termination_for(status) == "error"

    def test_an_unknown_status_is_an_error_not_finished(self):
        # A status added later must never read as a completed run.
        assert termination_for("some_future_status") == "error"

    def test_the_mapping_agrees_with_the_status_groups(self):
        # Nothing is derived twice: every kept status finishes, and the two
        # partial statuses are exactly the two limit terminations.
        for status in KEEP_STATUSES:
            assert termination_for(status) == "finished"
        assert {termination_for(s) for s in PARTIAL_STATUSES} == \
            {"turn_limit", "time_limit"}


class TestRunLogLine:
    def test_it_has_exactly_the_listed_fields(self, tmp_path):
        rec = run_log_record(meta_for(), Args(), "", tmp_path / "p.jsonl",
                             tmp_path / "t.jsonl")
        assert tuple(rec) == RUN_LOG_FIELDS

    def test_no_extra_and_no_missing_keys(self, tmp_path):
        rec = run_log_record(meta_for(), Args(), "", None, None)
        assert set(rec) == set(RUN_LOG_FIELDS)
        assert len(rec) == len(RUN_LOG_FIELDS)

    def test_values_come_from_the_run(self, tmp_path):
        rec = run_log_record(meta_for(), Args(notes="pilot 2 rerun"),
                             "", tmp_path / "p.jsonl", tmp_path / "t.jsonl")
        assert rec["issue_id"] == "repo__repo-1"
        assert rec["arm"] == "no-hint"
        assert rec["model_id"] == "claude-opus-5"
        assert rec["harness"] == "claude-code"
        assert rec["harness_version"] == "2.1.283 (Claude Code)"
        assert rec["effort"] == "high"
        assert rec["max_turns"] == 2000
        assert rec["max_wall_clock_s"] == 10800
        assert rec["turns_used"] == 215
        assert rec["tokens_in"] == 396 + 309255 + 36948640
        assert rec["tokens_out"] == 144656
        assert rec["wall_clock_s"] == 2209.0
        assert rec["termination"] == "finished"
        assert rec["operator"] == "SN"
        assert rec["notes"] == "pilot 2 rerun"
        assert rec["patch_path"].endswith("p.jsonl")
        assert rec["trajectory_path"].endswith("t.jsonl")

    def test_the_model_id_is_the_exact_string(self):
        rec = run_log_record(meta_for(), Args(model="claude-opus-5[1m]"), "", None, None)
        assert rec["model_id"] == "claude-opus-5[1m]"

    def test_a_run_with_no_patch_records_no_patch_path(self):
        rec = run_log_record(meta_for("error"), Args(), "", None, None)
        assert rec["patch_path"] is None
        assert rec["termination"] == "error"

    def test_tokens_in_counts_cache_reads_and_creation(self):
        # `input_tokens` alone understated xarray's pilot by five orders of
        # magnitude, which would make the run log useless for cost analysis.
        rec = run_log_record(meta_for(), Args(), "", None, None)
        assert rec["tokens_in"] == 36_948_640 + 309_255 + 396

    def test_the_full_breakdown_stays_in_meta(self):
        meta = meta_for()
        run_log_record(meta, Args(), "", None, None)
        usage = meta["claude_result"]["usage"]
        assert usage["input_tokens"] == 396
        assert usage["cache_read_input_tokens"] == 36948640
        assert usage["cache_creation_input_tokens"] == 309255

    def test_a_partial_usage_block_still_sums(self):
        # A result line that reports no cache fields must not produce None.
        meta = meta_for(claude_result={"usage": {"input_tokens": 10,
                                                 "output_tokens": 20}})
        assert run_log_record(meta, Args(), "", None, None)["tokens_in"] == 10

    def test_a_missing_result_line_leaves_tokens_null(self):
        rec = run_log_record(meta_for("timeout", claude_result={}), Args(),
                             "", None, None)
        assert rec["tokens_in"] is None and rec["tokens_out"] is None
        assert rec["termination"] == "time_limit"

    def test_run_date_is_iso_and_json_serialisable(self):
        rec = run_log_record(meta_for(), Args(), "", None, None)
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", rec["run_date"])
        json.loads(json.dumps(rec))

    def test_it_is_appended_never_rewritten(self, tmp_path):
        path = tmp_path / "pilot_runs.jsonl"
        append_jsonl(path, run_log_record(meta_for(), Args(), "", None, None))
        append_jsonl(path, run_log_record(meta_for("timeout"), Args(), "", None, None))
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert [l["termination"] for l in lines] == ["finished", "time_limit"]


class TestModifiedTestsFlag:
    def test_a_library_only_patch_is_false(self):
        patch = ("diff --git a/sympy/core/basic.py b/sympy/core/basic.py\n"
                 "--- a/sympy/core/basic.py\n+++ b/sympy/core/basic.py\n")
        assert patch_touches_tests(patch) is False

    def test_a_patch_touching_a_test_is_true(self):
        patch = ("diff --git a/sympy/core/basic.py b/sympy/core/basic.py\n"
                 "diff --git a/sympy/core/tests/test_basic.py "
                 "b/sympy/core/tests/test_basic.py\n")
        assert patch_touches_tests(patch) is True

    def test_conftest_counts(self):
        assert patch_touches_tests("diff --git a/conftest.py b/conftest.py\n") is True

    def test_library_code_under_testing_is_not_a_test(self):
        # xarray/testing/assertions.py is shipped library code, not a suite.
        patch = ("diff --git a/xarray/testing/assertions.py "
                 "b/xarray/testing/assertions.py\n")
        assert patch_touches_tests(patch) is False

    def test_an_empty_patch_is_false(self):
        assert patch_touches_tests("") is False

    def test_it_reaches_the_run_log(self):
        patch = "diff --git a/pkg/tests/test_x.py b/pkg/tests/test_x.py\n"
        assert run_log_record(meta_for(), Args(), patch, None, None)["modified_tests"]
