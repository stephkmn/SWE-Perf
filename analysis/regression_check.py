"""Run a broader test suite before and after each patch, and report new failures.

SWE-Perf's official run executes only the instance's `efficiency_test` list and
treats those same tests as the correctness check. A patch that speeds up the
measured path while breaking behavior no measured test covers scores as a clean
success. This runs a wider selection of tests on the base code and again after
the patch, and reports only pass-to-fail transitions.

Runs once per patch, outside any timing loop, so it cannot perturb the official
measurements. It reuses the benchmark's own images and container settings.
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docker  # noqa: E402

from common import DATASET_NAME, DATASET_SPLIT, load_instances, load_predictions, write_json  # noqa: E402
from constants import MAP_REPO_VERSION_TO_SPECS  # noqa: E402
from docker_build import build_container  # noqa: E402
from docker_utils import cleanup_container, copy_to_container  # noqa: E402
from docker_utils import exec_run_with_timeout  # noqa: E402
from patch_scope import parse_diff  # noqa: E402
from test_spec import make_test_spec  # noqa: E402

# run_evaluation.py rewrites the image key to the published x86_64 images
# rather than building locally; mirror that so we use the same containers.
DOCKER_IMAGE_PREFIX = "docker.io/betty1202/"
CONDA_ACTIVATE = "source /opt/miniconda3/bin/activate"
FAILING_OUTCOMES = {"failed", "error"}


def quiet_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def remote_image_key(instance_id):
    return (DOCKER_IMAGE_PREFIX + "sweb.eval.x86_64." + instance_id.replace("__", "_s_")).lower()


def env_name_for(test_spec):
    """The conda env the benchmark's own eval script activates."""
    match = re.search(r"conda activate (\S+)", test_spec.eval_script)
    return match.group(1) if match else "testbed"


def run_in_container(container, script, timeout):
    """Execute a bash snippet; return (output_text, timed_out)."""
    out, timed_out, _ = exec_run_with_timeout(container, f"/bin/bash -c {json.dumps(script)}", timeout)
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    return out or "", timed_out


def read_container_file(container, path):
    result = container.exec_run(f"cat {path}", demux=False)
    if result.exit_code != 0:
        return None
    try:
        return json.loads(result.output.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def candidate_test_paths(patch_text):
    """Test files and directories sitting next to each modified source file."""
    candidates = []
    for path in parse_diff(patch_text):
        if not path.endswith(".py"):
            continue
        directory = posixpath.dirname(path)
        stem = posixpath.basename(path)[:-3]
        prefix = f"{directory}/" if directory else ""
        candidates += [
            f"{prefix}tests",
            f"{prefix}tests/test_{stem}.py",
            f"{prefix}test_{stem}.py",
        ]
    return sorted(set(candidates))


def select_tests(container, repo_dir, patch_text, scope, timeout):
    """Resolve which test paths actually exist in the container.

    "full" returns [] so pytest collects the whole suite. "nearby" keeps only
    paths that exist, then drops files already covered by a selected directory
    so pytest does not run them twice.
    """
    if scope == "full":
        return []
    candidates = candidate_test_paths(patch_text)
    if not candidates:
        return []
    checks = " ; ".join(f'[ -e "{c}" ] && echo "{c}"' for c in candidates)
    out, _ = run_in_container(container, f"cd {repo_dir} && ({checks}) 2>/dev/null || true", timeout)
    existing = [line.strip() for line in out.splitlines() if line.strip() in candidates]
    dirs = {p for p in existing if not p.endswith(".py")}
    return sorted(d for d in dirs) + sorted(
        f for f in existing if f.endswith(".py")
        and not any(f.startswith(d + "/") for d in dirs)
    )


def pytest_command(repo, version, report_file, paths, extra=""):
    cmd = MAP_REPO_VERSION_TO_SPECS[repo.lower()][version]["test_all_cmd"]
    target = " ".join(f"'{p}'" for p in paths)
    return f"{cmd} --json-report --json-report-file={report_file} {extra} {target}".strip()


def outcomes_from_report(report):
    """{nodeid: outcome} from a pytest-json-report payload."""
    if not report or "tests" not in report:
        return {}
    return {t["nodeid"]: t.get("outcome") for t in report["tests"]}


def collection_errors(report):
    """Node ids that failed to import or collect.

    pytest-json-report records these under "collectors" with outcome "failed".
    They matter because the benchmark's test command passes
    --continue-on-collection-errors: a patch that breaks an import makes whole
    modules silently disappear from the run instead of failing it.
    """
    if not report:
        return []
    errors = []
    for collector in report.get("collectors", []) or []:
        if collector.get("outcome") == "failed":
            errors.append(collector.get("nodeid") or "<unknown>")
    return sorted(set(errors))


def summarise_run(report):
    """Counts for one pytest run, so a vanished test cannot look like a pass."""
    outcomes = outcomes_from_report(report)
    tally = {"collected": len(outcomes), "passed": 0, "failed": 0,
             "error": 0, "skipped": 0, "other": 0}
    for outcome in outcomes.values():
        key = outcome if outcome in ("passed", "failed", "error", "skipped") else "other"
        tally[key] += 1
    errors = collection_errors(report)
    tally["collection_errors"] = len(errors)
    return tally, outcomes, errors


def check_instance(instance, prediction, client, args):
    """Run the selected tests before and after the patch for one instance."""
    iid = instance["instance_id"]
    logger = quiet_logger(f"regression.{iid}")
    result = {
        "instance_id": iid,
        "repo": instance["repo"],
        "model_name_or_path": prediction.get("model_name_or_path"),
        "test_scope": args.test_scope,
        "tests_selected": [],
        "n_tests_run_base": 0,
        "base_counts": {},
        "after_counts": {},
        "base_collection_errors": [],
        "after_collection_errors": [],
        "new_collection_errors": [],
        "pass_to_fail": [],
        "missing_after_patch": [],
        "flaky": [],
        "regression_count": 0,
        "timed_out": False,
        "error": None,
    }
    patch_text = prediction.get("model_patch") or ""
    if not patch_text.strip():
        result["error"] = "empty patch"
        return result

    spec = make_test_spec(instance, is_eval=True)
    spec.instance_image_key = remote_image_key(iid)
    repo_dir = f"/{env_name_for(spec)}"
    activate = f"{CONDA_ACTIVATE} && conda activate {env_name_for(spec)} && cd {repo_dir}"

    deadline = time.time() + args.timeout

    def remaining():
        return max(1, int(deadline - time.time()))

    container = None
    try:
        container = build_container(spec, client, args.run_id, logger, nocache=False, from_remote=True)
        container.start()

        run_in_container(container, f"{activate} && pip install -q pytest-json-report", remaining())

        paths = select_tests(container, repo_dir, patch_text, args.test_scope, remaining())
        result["tests_selected"] = paths
        if args.test_scope == "nearby" and not paths:
            result["error"] = "no neighbouring tests found for the modified files"
            return result

        # Baseline, before the patch is applied.
        base_cmd = pytest_command(instance["repo"], instance["version"], "/tmp/reg_base.json", paths)
        _, timed_out = run_in_container(container, f"{activate} && {base_cmd}", remaining())
        if timed_out:
            result["timed_out"] = True
            return result
        base_counts, base_outcomes, base_errors = summarise_run(
            read_container_file(container, "/tmp/reg_base.json"))
        result["base_counts"] = base_counts
        result["base_collection_errors"] = base_errors
        result["n_tests_run_base"] = len(base_outcomes)
        if not base_outcomes:
            result["error"] = "no tests collected on base"
            return result

        # Apply the patch exactly as the official harness does.
        patch_path = Path(args.out_dir) / f"{iid}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch_text)
        copy_to_container(container, patch_path, Path("/tmp/analysis_patch.diff"))
        applied, _ = run_in_container(
            container,
            f"cd {repo_dir} && (git apply --allow-empty -v /tmp/analysis_patch.diff || "
            f"patch --batch --fuzz=5 -p1 -i /tmp/analysis_patch.diff) 2>&1",
            remaining(),
        )
        verify, _ = run_in_container(container, f"cd {repo_dir} && git diff --stat", remaining())
        if not verify.strip():
            result["error"] = f"patch did not apply: {applied.strip()[:300]}"
            return result

        after_cmd = pytest_command(instance["repo"], instance["version"], "/tmp/reg_after.json", paths)
        _, timed_out = run_in_container(container, f"{activate} && {after_cmd}", remaining())
        if timed_out:
            result["timed_out"] = True
            return result
        after_counts, after_outcomes, after_errors = summarise_run(
            read_container_file(container, "/tmp/reg_after.json"))
        result["after_counts"] = after_counts
        result["after_collection_errors"] = after_errors
        result["new_collection_errors"] = sorted(set(after_errors) - set(base_errors))

        passed_on_base = {n for n, o in base_outcomes.items() if o == "passed"}
        broke = sorted(n for n in passed_on_base if after_outcomes.get(n) in FAILING_OUTCOMES)
        # A test that ran on base and did not run at all afterwards is a
        # regression, not a pass: --continue-on-collection-errors lets a broken
        # import remove whole modules from the run without failing anything.
        result["missing_after_patch"] = sorted(n for n in passed_on_base if n not in after_outcomes)

        # A single rerun separates a real regression from an order- or
        # timing-dependent flake, which these suites have plenty of.
        if broke:
            node_args = " ".join(f"'{n}'" for n in broke)
            rerun_cmd = (
                MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]["test_all_cmd"]
                + f" --json-report --json-report-file=/tmp/reg_rerun.json {node_args}"
            )
            _, timed_out = run_in_container(container, f"{activate} && {rerun_cmd}", remaining())
            if not timed_out:
                rerun = outcomes_from_report(read_container_file(container, "/tmp/reg_rerun.json"))
                result["flaky"] = sorted(n for n in broke if rerun.get(n) == "passed")
                broke = [n for n in broke if n not in set(result["flaky"])]
        result["pass_to_fail"] = broke
        result["regression_count"] = (
            len(broke) + len(result["missing_after_patch"]) + len(result["new_collection_errors"])
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if container is not None:
            cleanup_container(client, container, logger)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out_dir", default="analysis_out/regression")
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--test_scope", choices=["nearby", "full"], default="nearby")
    p.add_argument("--timeout", type=int, default=3600, help="per-instance wall clock (s)")
    p.add_argument("--run_id", default="analysis_regression")
    p.add_argument("--instance_ids", nargs="*", default=None)
    p.add_argument("--use_gold_patch", action="store_true",
                   help="ignore model_patch and check the expert patch instead (smoke test)")
    args = p.parse_args()

    instances = load_instances(args.dataset_name, args.split)
    preds = load_predictions(args.predictions)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        preds = [p_ for p_ in preds if p_["instance_id"] in wanted]
    if not preds:
        sys.exit("error: no predictions selected")

    client = docker.from_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n, pred in enumerate(preds, 1):
        inst = instances.get(pred["instance_id"])
        if inst is None:
            print(f"[{n}/{len(preds)}] {pred['instance_id']}: not in dataset, skipping")
            continue
        if args.use_gold_patch:
            pred = dict(pred, model_patch=inst["patch"], model_name_or_path="gold")
        started = time.time()
        result = check_instance(inst, pred, client, args)
        result["duration_s"] = round(time.time() - started, 1)
        write_json(out_dir / f"{result['instance_id']}.json", result)
        status = (result["error"] or ("timed out" if result["timed_out"] else
                  f"{result['regression_count']} regression(s)"))
        detail = []
        if result["missing_after_patch"]:
            detail.append(f"{len(result['missing_after_patch'])} vanished")
        if result["new_collection_errors"]:
            detail.append(f"{len(result['new_collection_errors'])} new collection error(s)")
        if result["flaky"]:
            detail.append(f"{len(result['flaky'])} flaky")
        print(f"[{n}/{len(preds)}] {result['instance_id']}: {status}, "
              f"{result['n_tests_run_base']} test(s), {result['duration_s']}s"
              + (", " + ", ".join(detail) if detail else ""))

    print(f"\nresults: {out_dir}")


if __name__ == "__main__":
    main()
