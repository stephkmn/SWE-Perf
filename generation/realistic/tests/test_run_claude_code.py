"""Unit tests for the generation driver's patch filtering and baseline logic.

These are deliberately Docker-free and Claude-free: every function under test
is a pure transformation of text that a container produced, so the interesting
cases (a test edit, a recompiled extension, a scratch file at the repo root, a
checkout that still has history) can be exercised as strings.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from run_claude_code import (
    BUILD_ARTIFACT_RE, MARKER, TEST_PATH_RE, VERBATIM_TEMPLATE,
    agent_environment, baseline_problems, build_baseline_script, build_prompt,
    build_verify_script, claude_command, classify_worktree, filtered_by,
    is_scratch, looks_like_usage_limit, parse_baseline_report, parse_markers,
    parse_numstat_binaries, parse_porcelain_z, redact, remote_image_key,
    verify_cli_capabilities, verify_hidden_flags,
    classify_claude_failure, earns_a_prediction, parse_result_line,
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
    def test_is_the_dataset_text_verbatim(self):
        statement = "Optimize `foo`.\n\nRules:\n1. {not a format field}\n100% faster"
        inst = {"instance_id": "x__y-1", "problem_statement_realistic": statement}
        assert build_prompt(inst) == statement

    def test_template_adds_nothing(self):
        assert VERBATIM_TEMPLATE == "{problem_statement}"

    def test_falls_back_to_problem_statement(self):
        inst = {"instance_id": "x__y-1", "problem_statement": "text"}
        assert build_prompt(inst) == "text"

    def test_missing_statement_raises(self):
        with pytest.raises(KeyError):
            build_prompt({"instance_id": "x__y-1"})


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
