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
import io
import json
import logging
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import docker  # noqa: E402

from common import DATASET_NAME, DATASET_SPLIT, base_source, load_instances  # noqa: E402
from common import load_predictions, write_json  # noqa: E402
from docker_build import build_container  # noqa: E402
from docker_utils import cleanup_container  # noqa: E402
from patch_scope import analyze_patch, function_body_ranges, parse_diff  # noqa: E402
from regression_check import CONDA_ACTIVATE, env_name_for, quiet_logger  # noqa: E402
from regression_check import read_container_file, remote_image_key, run_in_container  # noqa: E402
from test_spec import make_test_spec  # noqa: E402

REMOTE_PATCH = "/tmp/timed_path.diff"
COVERAGE_JSON = "/tmp/analysis_coverage.json"
COVERAGE_DATA = "/tmp/analysis_coverage.data"
COVERAGE_RC = "/tmp/analysis_coveragerc"
# pytest's own exit codes: 0 passed, 1 tests failed. Anything above that means
# the run never happened (collection error, bad usage, nothing collected), so
# an empty coverage report there is a broken run, not an off-path result.
PYTEST_DID_NOT_RUN = {2, 3, 4, 5}
TAIL = 2000


def copy_file_to_container(container, src, dst):
    """Place a local file at exactly `dst` inside the container.

    docker_utils.copy_to_container names the tar member after the *source*
    file, so the archive unpacks beside `dst` under the source's own name and
    `dst` itself is never created. Name the member after the destination.
    """
    dst = PurePosixPath(dst)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(str(src), arcname=dst.name)
    container.exec_run(f"mkdir -p {dst.parent}")
    container.put_archive(str(dst.parent), buf.getvalue())


def run_with_status(container, script, timeout):
    """run_in_container plus the script's exit status.

    exec_run_with_timeout returns no exit code, so ask the shell for it and
    strip the marker back off. Returns (output, exit_code, timed_out); the
    code is None only if the marker never arrived (killed, or timed out).
    """
    # Chained with ";" rather than a newline: docker-py shlex-splits the
    # command, and a "\n" survives that as two literal characters.
    marker = "__timed_path_status__"
    out, timed_out = run_in_container(container, f"{script}; {marker}=$?; echo {marker}=${marker}", timeout)
    match = re.search(rf"{marker}=(\d+)\s*\Z", out)
    if not match:
        return out, None, timed_out
    return out[: match.start()], int(match.group(1)), timed_out


def porcelain_paths(porcelain):
    """Paths git reports as modified/added by `git status --porcelain`."""
    paths = set()
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:                      # a rename: keep the new name
            path = path.split(" -> ", 1)[1]
        paths.add(path.strip('"'))
    return paths


def executed_lines(coverage_payload, file_path):
    """Set of executed line numbers for one file in a coverage.py JSON report.

    Keys are paths relative to the run's working directory, but coverage may
    record them with or without a leading directory, so match on suffix.
    """
    return measured_lines(coverage_payload, file_path)[1]


def measured_lines(coverage_payload, file_path):
    """(was the file in the report, executed line numbers) for one file.

    Absent and measured-but-never-executed both leave the line set empty, and
    only the first of those means the measurement failed rather than the code
    genuinely sitting off the timed path. Keep them apart.
    """
    files = (coverage_payload or {}).get("files", {})
    for key, entry in files.items():
        normalised = key.lstrip("./")
        if normalised == file_path or normalised.endswith("/" + file_path):
            return True, set(entry.get("executed_lines", []))
    return False, set()


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
        "pytest_exit_code": None,
        "n_files_measured": 0,
        "baseline_dirty": [],
        "files_not_measured": [],
        "apply_output": "",
        "verify_output": "",
        "coverage_output": "",
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
        copy_file_to_container(container, patch_path, REMOTE_PATCH)

        # These images do not all ship a git new enough for --allow-empty
        # (2.35+); an empty patch is rejected earlier anyway. Keep patch(1) as
        # the fallback for diffs git is stricter about than patch is.
        before, _ = run_in_container(container, f"cd {repo_dir} && git status --porcelain", remaining())
        result["baseline_dirty"] = sorted(porcelain_paths(before))
        apply_out, apply_code, apply_timeout = run_with_status(
            container,
            f"cd {repo_dir} && echo '--- git apply ---' && "
            f"git apply -v --whitespace=nowarn {REMOTE_PATCH} 2>&1; "
            f"if [ $? -ne 0 ]; then echo '--- patch(1) fallback ---'; "
            f"patch --batch --fuzz=5 -p1 -i {REMOTE_PATCH} 2>&1; fi",
            remaining(),
        )
        # Judge by the apply command's own exit status and by the files the
        # diff names actually turning up as changed. The old check -- "does
        # git diff --stat print anything" -- reports success whenever the image
        # ships a dirty checkout (several do, from their install step's sed)
        # and failure whenever anything upstream of it silently went wrong.
        verify, _ = run_in_container(container, f"cd {repo_dir} && git status --porcelain", remaining())
        touched = porcelain_paths(verify)
        missing = [path for path in sorted(set(parse_diff(patch_text))) if path not in touched]
        if apply_code != 0 or missing or apply_timeout:
            result["apply_output"] = apply_out.strip()[-TAIL:]
            result["verify_output"] = verify.strip()[-TAIL:]
            reason = "timed out" if apply_timeout else f"exit {apply_code}"
            if missing:
                reason += "; unchanged after apply: " + ", ".join(missing[:5])
            result["error"] = f"patch did not apply ({reason})"
            return result

        tests = " ".join(f"'{t}'" for t in instance["efficiency_test"])
        # Measure under our own config, not the repo's. coverage.py picks up
        # setup.cfg/tox.ini/pyproject.toml from the working directory, and
        # sphinx (among others) sets parallel = true there, which writes
        # .coverage.<host>.<pid>.<rand> instead of the file `coverage json`
        # then looks for -- the whole run reads as "no report". Pinning
        # data_file also survives a suite that chdirs away mid-run.
        run_in_container(
            container,
            f"printf '[run]\\nparallel = False\\nbranch = False\\ndata_file = {COVERAGE_DATA}\\n'"
            f" > {COVERAGE_RC}",
            remaining(),
        )
        # One pass, no repeats: this measures which lines run, never how long.
        # --source pins measurement to the checkout: without it coverage skips
        # modules already imported when it starts, which is most of the package.
        _, pytest_code, pytest_timeout = run_with_status(
            container,
            f"{activate} && coverage run --rcfile={COVERAGE_RC} --source={repo_dir} -m pytest "
            f"-p no:cacheprovider -q {tests} > /tmp/cov_run.log 2>&1",
            remaining(),
        )
        result["pytest_exit_code"] = pytest_code
        cov_log, _ = run_in_container(container, "tail -5 /tmp/cov_run.log", remaining())
        result["pytest_tail"] = cov_log.strip()[-500:]
        if pytest_timeout:
            result["error"] = "timed out running the efficiency tests"
            return result
        if pytest_code in PYTEST_DID_NOT_RUN:
            result["error"] = f"efficiency tests did not run (pytest exit {pytest_code})"
            return result
        cov_json_out, _ = run_in_container(
            container,
            f"{activate} && coverage json --rcfile={COVERAGE_RC} -o {COVERAGE_JSON} --pretty-print 2>&1",
            remaining(),
        )
        payload = read_container_file(container, COVERAGE_JSON)
        if not payload:
            result["coverage_output"] = cov_json_out.strip()[-TAIL:]
            result["error"] = "coverage produced no report"
            return result
        result["coverage_available"] = True
        result["n_files_measured"] = len((payload or {}).get("files", {}))

        # Read the patched sources back so function ranges match what ran.
        on_path = {}
        unmeasured = []
        for path in sorted({q.split("::", 1)[0] for q in changed}):
            if not path.endswith(".py"):
                continue
            source, _ = run_in_container(container, f"cat {repo_dir}/{path}", remaining())
            measured, ran = measured_lines(payload, path)
            if not measured:
                unmeasured.append(path)
            # body ranges, not full spans: a def line runs at import time
            ranges = {n: (s, e) for n, s, e in function_body_ranges(source)}
            for qualified in sorted(q for q in changed if q.startswith(path + "::")):
                name = qualified.split("::", 1)[1]
                span = ranges.get(name)
                on_path[qualified] = bool(span) and any(span[0] <= l <= span[1] for l in ran)

        # A file coverage never saw cannot testify that its functions did not
        # run, so say so rather than filing them all as off-path.
        result["files_not_measured"] = unmeasured
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
        unmeasured = res.get("files_not_measured") or []
        print(f"[{n}/{len(preds)}] {res['instance_id']}: "
              + (res["error"] or f"{res['n_on_path']} on path, {res['n_off_path']} off path")
              + (f" [{len(unmeasured)} file(s) not measured]" if unmeasured else "")
              + f", {res['duration_s']}s")
    print(f"\nresults: {args.out_dir}")


if __name__ == "__main__":
    main()
