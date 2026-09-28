#!/usr/bin/env python3
"""End-to-end check of the container setup, WITHOUT calling Claude.

Exercises the same functions run_claude_code.py uses for every instance --
agent user, conda environment, single-commit baseline, patch extraction -- then
stands up a second, untouched container of the same image and confirms the
extracted patch applies there with `git apply --check`. That last step is the
one that matters: it proves the baseline really is the image's post-setup tree,
so a patch taken against it is still valid at evaluation time.

No Claude process is started and no API call is made; the "model's work" is a
handful of dummy edits made with sed, chosen to hit every filtering rule.

    python generation/realistic/tests/check_container_setup.py \
        --instance_id astropy__astropy-13496
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import docker  # noqa: E402

from run_claude_code import (  # noqa: E402
    AGENT_USER, CLI_BIN, REPO_DIR, agent_environment, bootstrap_cli,
    build_verify_script, check_auth_env_clean, cleanup_container,
    container_exec, create_instance_container, ensure_image, establish_baseline,
    extract_patch, make_test_spec, env_name_for, quiet_logger, remote_image_key,
    setup_agent_user, sh, wait_for_timed_path,
)

# The container check never talks to the API, so no real credential is needed.
# check_auth_env_clean only asks whether the variable arrived and whether the
# API-key variables are absent, which a placeholder answers just as well.
DUMMY_TOKEN = "not-a-real-token-container-check-only"

DUMMY_EDITS = r"""
set -e
cd {repo}
src=$(git ls-files '*.py' | grep -v -E '(^|/)tests?/' | head -1)
tst=$(git ls-files '*.py' | grep -E '(^|/)tests?/' | head -1)
so=$(git ls-files '*.so' | head -1)
echo "SRC=$src"
echo "TST=$tst"
echo "SO=$so"
# a real source edit: the only thing that should survive into the patch
printf '\n# dummy edit from check_container_setup.py\n' >> "$src"
# a test edit: rule 1 says this must be reverted
printf '\n# dummy test edit\n' >> "$tst"
# scratch at the repo root: must be removed
printf 'notes\n' > benchmark_notes.md
# a compiled extension the image shipped, dirtied the way importing or
# rebuilding the package would dirty it: must be reverted, and must never
# reach the diff as an unappliable binary hunk
if [ -n "$so" ]; then printf 'x' >> "$so"; fi
# fresh build output: must be filtered out
mkdir -p "$(dirname "$src")/__pycache__"
printf 'x' > "$(dirname "$src")/__pycache__/dummy.cpython-39.pyc"
printf 'x' > "$(dirname "$src")/dummy_artifact.so"
"""

def show(title, text):
    print(f"\n--- {title} ---")
    print(text.strip() or "(no output)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance_id", default="astropy__astropy-13496")
    p.add_argument("--dataset_name", default="SWE-Perf/SWE-Perf")
    p.add_argument("--split", default="test")
    p.add_argument("--run_id", default="containercheck")
    p.add_argument("--cli_cache",
                   default=str(Path(__file__).resolve().parents[3]
                               / "datasets" / "claude_cli_linux_x64"))
    p.add_argument("--keep", action="store_true", help="leave the containers running")
    args = p.parse_args()

    from datasets import load_dataset
    rows = {r["instance_id"]: r for r in load_dataset(args.dataset_name, split=args.split)}
    inst = rows.get(args.instance_id)
    if inst is None:
        sys.exit(f"error: {args.instance_id} is not in {args.dataset_name}:{args.split}")

    # Never two containers at once on this machine.
    wait_for_timed_path()

    client = docker.from_env()
    spec = make_test_spec(inst, is_eval=True)
    spec.instance_image_key = remote_image_key(args.instance_id)
    env_name = env_name_for(spec)
    cli_cache = Path(args.cli_cache).resolve()
    logger = quiet_logger("containercheck")
    failures = []

    print(f"instance : {args.instance_id}")
    print(f"image    : {spec.instance_image_key}")
    print(f"platform : {spec.platform}")
    print(f"conda env: {env_name}")

    manifest, built = bootstrap_cli(client, cli_cache, spec.instance_image_key,
                                    "latest", effort="xhigh")
    print(f"cli      : {manifest.get('claude_version')} "
          f"({'built now' if built else 'from cache'}), "
          f"node {manifest.get('node_version')}, arch {manifest.get('container_arch')}")

    ensure_image(client, spec.instance_image_key)
    container = create_instance_container(client, spec, args.run_id, cli_cache)
    verifier = None
    try:
        container.start()

        # ---- step 4: non-root user, writable /testbed, conda env on PATH ----
        code, out, err, _ = setup_agent_user(container, env_name)
        if code != 0:
            failures.append(f"agent setup exited {code}")
            show("agent setup output", out + err)

        env = agent_environment(env_name, DUMMY_TOKEN)
        _, who, _, _ = container_exec(
            container,
            ["/bin/bash", "-lc",
             "id; echo cwd=$(pwd); echo python=$(command -v python); "
             "echo pytest=$(command -v pytest); python -c 'import sys; print(sys.version)'; "
             f"touch {REPO_DIR}/.write_probe && echo write_ok && rm {REPO_DIR}/.write_probe"],
            user=AGENT_USER, environment=env, workdir=REPO_DIR, timeout=300)
        show("step 4: agent user, cwd, conda env, write access", who)
        for needle in (f"uid=", "/envs/", "write_ok"):
            if needle not in who:
                failures.append(f"step 4: expected {needle!r} in the shell probe")
        if "uid=0(root)" in who:
            failures.append("step 4: Claude would run as root "
                            "(--dangerously-skip-permissions refuses that)")

        _, ver, verr, _ = container_exec(container, [CLI_BIN, "--version"],
                                         user=AGENT_USER, environment=env, timeout=180)
        show("step 4: claude --version from inside the container", ver + verr)
        if not ver.strip():
            failures.append("step 4: the in-container CLI printed no version")

        auth_problems = check_auth_env_clean(container, env_name, DUMMY_TOKEN)
        show("step 4: auth environment",
             "clean" if not auth_problems else "\n".join(auth_problems))
        failures += [f"auth: {p}" for p in auth_problems]

        # ---- step 5: single-commit baseline with no leakage ----
        report, problems, log = establish_baseline(container, env)
        _, checks, _, _ = sh(container, build_verify_script(), user=AGENT_USER,
                             environment=env, timeout=300)
        show("step 5: git isolation checks", checks)
        print(f"parsed   : {report}")
        if problems:
            failures += [f"baseline: {p}" for p in problems]
            show("baseline log", log)
        else:
            print("baseline : 1 commit, no tags, no remotes, no reflog, clean tree")

        _, hist, _, _ = sh(container,
                           f"cd {REPO_DIR} && git log --all --oneline && "
                           f"echo '-- tags --' && git tag && "
                           f"echo '-- remotes --' && git remote -v && "
                           f"echo '-- reflog --' && git reflog",
                           user=AGENT_USER, environment=env, timeout=120)
        show("step 5: raw git log --all / tag / remote / reflog", hist)

        # ---- a dummy edit standing in for the model's work ----
        _, edits, eerr, _ = sh(container, DUMMY_EDITS.format(repo=REPO_DIR),
                               user=AGENT_USER, environment=env, timeout=600)
        show("dummy edits", edits + eerr)

        _, dirty, _, _ = sh(container,
                            f"cd {REPO_DIR} && git status --porcelain -uall | head -20",
                            user=AGENT_USER, environment=env, timeout=300)
        show("working tree before extraction", dirty)

        meta = {}
        patch = extract_patch(container, meta, env)
        show("extraction meta", "\n".join(
            f"{k}: {v}" for k, v in meta.items() if k != "new_files") +
            f"\nnew_files: {meta.get('new_files')}")
        show("extracted patch", patch[:2000] or "(empty)")

        if not patch.strip():
            failures.append("extraction: the patch is empty")
        if "dummy edit from check_container_setup" not in patch:
            failures.append("extraction: the real source edit is missing from the patch")
        for needle, what in (
            ("dummy test edit", "a test-file edit"),
            ("benchmark_notes.md", "a root scratch file"),
            ("dummy_artifact.so", "a build artifact"),
            ("dummy.cpython-39.pyc", "a __pycache__ file"),
            ("Binary files", "a binary diff"),
            ("GIT binary patch", "a binary patch"),
        ):
            if needle in patch:
                failures.append(f"extraction: {what} reached the patch ({needle!r})")
        if "SO=" in edits and edits.split("SO=", 1)[1].split("\n", 1)[0].strip():
            if not meta.get("build_artifacts_filtered"):
                failures.append("extraction: the dirtied .so was not filtered, so the "
                                "build-artifact rule never fired")

        # ---- the patch must apply on a fresh container of the same image ----
        verifier = create_instance_container(
            client, spec, args.run_id + "verify", cli_cache)
        verifier.start()
        from run_claude_code import write_container_bytes
        write_container_bytes(verifier, "/tmp/model.patch", patch.encode())
        code, chk, cerr, _ = sh(
            verifier,
            f"git config --global --add safe.directory {REPO_DIR}; "
            f"cd {REPO_DIR} && git apply --check -v /tmp/model.patch && echo APPLY_CHECK_OK",
            user="root", timeout=300)
        show("git apply --check in a fresh container", chk + cerr)
        if "APPLY_CHECK_OK" not in chk:
            failures.append(f"git apply --check failed (exit {code})")
    finally:
        if not args.keep:
            cleanup_container(client, container, logger)
            if verifier is not None:
                cleanup_container(client, verifier, logger)
        else:
            print(f"\nkept: {container.name}"
                  + (f", {verifier.name}" if verifier is not None else ""))

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASSED: container setup, isolation, extraction, and re-apply all check out")


if __name__ == "__main__":
    main()
