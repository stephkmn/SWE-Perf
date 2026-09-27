"""Determine which changed functions actually execute during the timed tests.

Motivation: "outside the named targets" is not the same as "a new performance
issue." SWE-Perf's realistic setting names a handful of target functions, but
speeding up a helper those targets call is exactly the intended solution, and
such a helper shows up as outside the targets. Only a change that never runs
during the timed tests is a candidate for a genuinely different bottleneck --
or for dead work.

Runs the instance's efficiency_test list once under coverage.py, with no
timing involved, so it cannot disturb the official measurements.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docker  # noqa: E402

from common import DATASET_NAME, DATASET_SPLIT, base_source, load_instances  # noqa: E402
from common import load_predictions, write_json  # noqa: E402
from docker_build import build_container  # noqa: E402
from docker_utils import cleanup_container, copy_to_container  # noqa: E402
from patch_scope import analyze_patch, function_body_ranges  # noqa: E402
from regression_check import CONDA_ACTIVATE, env_name_for, quiet_logger  # noqa: E402
from regression_check import read_container_file, remote_image_key, run_in_container  # noqa: E402
from test_spec import make_test_spec  # noqa: E402

COVERAGE_JSON = "/tmp/analysis_coverage.json"


def executed_lines(coverage_payload, file_path):
    """Set of executed line numbers for one file in a coverage.py JSON report.

    Keys are paths relative to the run's working directory, but coverage may
    record them with or without a leading directory, so match on suffix.
    """
    files = (coverage_payload or {}).get("files", {})
    for key, entry in files.items():
        normalised = key.lstrip("./")
        if normalised == file_path or normalised.endswith("/" + file_path):
            return set(entry.get("executed_lines", []))
    return set()


def check_instance(instance, prediction, client, args):
    iid = instance["instance_id"]
    logger = quiet_logger(f"timedpath.{iid}")
    result = {
        "instance_id": iid,
        "repo": instance["repo"],
        "model_name_or_path": prediction.get("model_name_or_path"),
        "on_timed_path": {},
        "n_on_path": 0,
        "n_off_path": 0,
        "off_path_functions": [],
        "coverage_available": False,
        "pytest_tail": "",
        "n_files_measured": 0,
        "error": None,
    }
    patch_text = prediction.get("model_patch") or ""
    if not patch_text.strip():
        result["error"] = "empty patch"
        return result

    scope = analyze_patch(instance, patch_text, args.mirror_dir)
    changed = set(scope["functions_changed"])
    if not changed:
        result["error"] = "no changed functions resolved"
        return result

    spec = make_test_spec(instance, is_eval=True)
    spec.instance_image_key = remote_image_key(iid)
    env = env_name_for(spec)
    repo_dir = f"/{env}"
    activate = f"{CONDA_ACTIVATE} && conda activate {env} && cd {repo_dir}"
    deadline = time.time() + args.timeout

    def remaining():
        return max(1, int(deadline - time.time()))

    container = None
    try:
        container = build_container(spec, client, args.run_id, logger, nocache=False, from_remote=True)
        container.start()
        run_in_container(container, f"{activate} && python -m pip install -q coverage", remaining())

        patch_path = Path(args.out_dir) / f"{iid}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch_text)
        copy_to_container(container, patch_path, Path("/tmp/timed_path.diff"))
        run_in_container(
            container,
            f"cd {repo_dir} && (git apply --allow-empty -v /tmp/timed_path.diff || "
            f"patch --batch --fuzz=5 -p1 -i /tmp/timed_path.diff) 2>&1",
            remaining(),
        )
        verify, _ = run_in_container(container, f"cd {repo_dir} && git diff --stat", remaining())
        if not verify.strip():
            result["error"] = "patch did not apply"
            return result

        tests = " ".join(f"'{t}'" for t in instance["efficiency_test"])
        # One pass, no repeats: this measures which lines run, never how long.
        # --source pins measurement to the checkout: without it coverage skips
        # modules already imported when it starts, which is most of the package.
        run_in_container(
            container,
            f"{activate} && coverage run --source={repo_dir} -m pytest "
            f"-p no:cacheprovider -q {tests} > /tmp/cov_run.log 2>&1 || true",
            remaining(),
        )
        cov_log, _ = run_in_container(container, "tail -5 /tmp/cov_run.log", remaining())
        result["pytest_tail"] = cov_log.strip()[-500:]
        run_in_container(
            container,
            f"{activate} && coverage json -o {COVERAGE_JSON} --pretty-print > /dev/null 2>&1 || true",
            remaining(),
        )
        payload = read_container_file(container, COVERAGE_JSON)
        if not payload:
            result["error"] = "coverage produced no report"
            return result
        result["coverage_available"] = True
        result["n_files_measured"] = len((payload or {}).get("files", {}))

        # Read the patched sources back so function ranges match what ran.
        on_path = {}
        for path in sorted({q.split("::", 1)[0] for q in changed}):
            if not path.endswith(".py"):
                continue
            source, _ = run_in_container(container, f"cat {repo_dir}/{path}", remaining())
            ran = executed_lines(payload, path)
            # body ranges, not full spans: a def line runs at import time
            ranges = {n: (s, e) for n, s, e in function_body_ranges(source)}
            for qualified in sorted(q for q in changed if q.startswith(path + "::")):
                name = qualified.split("::", 1)[1]
                span = ranges.get(name)
                on_path[qualified] = bool(span) and any(span[0] <= l <= span[1] for l in ran)

        result["on_timed_path"] = on_path
        result["n_on_path"] = sum(1 for v in on_path.values() if v)
        result["off_path_functions"] = sorted(k for k, v in on_path.items() if not v)
        result["n_off_path"] = len(result["off_path_functions"])

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if container is not None:
            cleanup_container(client, container, logger)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out_dir", default="analysis_out/timed_path")
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--run_id", default="analysis_timedpath")
    p.add_argument("--instance_ids", nargs="*", default=None)
    p.add_argument("--mirror_dir", default=None)
    p.add_argument("--use_gold_patch", action="store_true")
    args = p.parse_args()

    instances = load_instances(args.dataset_name, args.split)
    preds = load_predictions(args.predictions)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        preds = [x for x in preds if x["instance_id"] in wanted]
    if not preds:
        sys.exit("error: no predictions selected")

    client = docker.from_env()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for n, pred in enumerate(preds, 1):
        inst = instances.get(pred["instance_id"])
        if inst is None:
            continue
        if args.use_gold_patch:
            pred = dict(pred, model_patch=inst["patch"], model_name_or_path="gold")
        started = time.time()
        res = check_instance(inst, pred, client, args)
        res["duration_s"] = round(time.time() - started, 1)
        write_json(Path(args.out_dir) / f"{res['instance_id']}.json", res)
        print(f"[{n}/{len(preds)}] {res['instance_id']}: "
              + (res["error"] or f"{res['n_on_path']} on path, {res['n_off_path']} off path")
              + f", {res['duration_s']}s")
    print(f"\nresults: {args.out_dir}")


if __name__ == "__main__":
    main()
