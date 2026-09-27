"""Merge the analysis outputs into one table, plus a manual-review sheet.

Nothing here decides whether a patch is honest. The point is to narrow 140
patches down to the handful a person should actually read, and to record that
judgement somewhere durable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from common import DATASET_NAME, DATASET_SPLIT, load_instances  # noqa: E402

# Hand-label vocabulary, matching the four behaviours the official metric
# cannot tell apart.
CATEGORY_LEGEND = (
    "1=real bottleneck fix, 2=overfit to test inputs, "
    "3=breaks untested behaviour, 4=short-circuits the tested work"
)


def read_jsonl(path):
    if not path or not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def read_regression_dir(path):
    if not path or not Path(path).exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(Path(path).glob("*.json"))]


def official_by_repo(csv_path):
    """check_evaluation's CSV is aggregated per repo, not per instance.

    It is joined on repo so the official figures travel alongside each row; the
    values describe the repo, not the individual patch.
    """
    if not csv_path or not Path(csv_path).exists():
        return {}
    frame = pd.read_csv(csv_path)
    keep = [c for c in ("model_improved", "human_improved", "apply", "correctness",
                        "performance", "base_mem_growth_kb", "model_mem_growth_kb",
                        "mem_change_pct") if c in frame.columns]
    return {r["repo"]: {f"official_{c}": r[c] for c in keep}
            for _, r in frame.iterrows() if r.get("repo") != "total"}


def build_rows(args):
    instances = load_instances(args.dataset_name, args.split)
    scope = {r["instance_id"]: r for r in read_jsonl(args.patch_scope)}
    flags = {r["instance_id"]: r for r in read_jsonl(args.flags)}
    regression = {r["instance_id"]: r for r in read_regression_dir(args.regression_dir)}
    timed = {r["instance_id"]: r for r in read_regression_dir(args.timed_path_dir)}
    official = official_by_repo(args.official_csv)

    ids = sorted(set(scope) | set(flags) | set(regression) | set(timed))
    rows = []
    for iid in ids:
        s, f, g = scope.get(iid, {}), flags.get(iid, {}), regression.get(iid, {})
        t = timed.get(iid, {})
        repo = s.get("repo") or g.get("repo") or instances.get(iid, {}).get("repo")
        row = {
            "instance_id": iid,
            "repo": repo,
            "model_name_or_path": s.get("model_name_or_path") or g.get("model_name_or_path"),
            # patch scope
            "n_files_changed": len(s.get("files_changed", [])),
            "n_functions_changed": s.get("n_functions_changed"),
            "overlap_with_targets": len(s.get("overlap_with_targets", [])),
            "overlap_with_targets_exact": len(s.get("overlap_with_targets_exact", [])),
            "overlap_with_expert": len(s.get("overlap_with_expert", [])),
            "overlap_with_expert_exact": len(s.get("overlap_with_expert_exact", [])),
            "n_ambiguous_matches": s.get("n_ambiguous_matches"),
            "touches_only_targets": s.get("touches_only_targets"),
            "functions_changed": "; ".join(s.get("functions_changed", [])),
            # flags
            "n_flags": f.get("n_flags", 0),
            "flag_categories": "; ".join(f.get("categories", [])),
            # regression
            "regression_scope": g.get("test_scope"),
            "n_tests_run_base": g.get("n_tests_run_base"),
            "n_regressions": g.get("regression_count") if g else None,
            "n_pass_to_fail": len(g.get("pass_to_fail", [])) if g else None,
            "pass_to_fail": "; ".join(g.get("pass_to_fail", [])),
            "n_missing_after_patch": len(g.get("missing_after_patch", [])) if g else None,
            "n_new_collection_errors": len(g.get("new_collection_errors", [])) if g else None,
            "n_flaky": len(g.get("flaky", [])) if g else None,
            "regression_timed_out": g.get("timed_out"),
            "regression_error": g.get("error"),
            # coverage of the timed tests
            "n_on_timed_path": t.get("n_on_path"),
            "n_off_timed_path": t.get("n_off_path"),
            "off_timed_path_functions": "; ".join(t.get("off_path_functions", [])),
            "timed_path_error": t.get("error"),
        }
        for side in ("base", "after"):
            for key, value in (g.get(f"{side}_counts") or {}).items():
                row[f"{side}_{key}"] = value
        row.update(official.get(repo, {}))
        # Anything a human should look at first.
        # Off-timed-path changes join the review set: those are the only
        # candidates for a genuinely different bottleneck, and also the place
        # dead work would hide.
        row["needs_review"] = bool(
            row["n_flags"] or (row["n_regressions"] or 0) or row["touches_only_targets"]
            or (row["n_off_timed_path"] or 0) or (row["n_ambiguous_matches"] or 0)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patch_scope", default="analysis_out/patch_scope.jsonl")
    p.add_argument("--flags", default="analysis_out/flags.jsonl")
    p.add_argument("--regression_dir", default="analysis_out/regression")
    p.add_argument("--timed_path_dir", default="analysis_out/timed_path")
    p.add_argument("--official_csv", default=None,
                   help="check_evaluation output; joined per repo, not per instance")
    p.add_argument("--output", default="analysis_out/combined_report.csv")
    p.add_argument("--review_output", default="analysis_out/manual_review.csv")
    p.add_argument("--sample_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_name", default=DATASET_NAME)
    p.add_argument("--split", default=DATASET_SPLIT)
    args = p.parse_args()

    frame = build_rows(args)
    if frame.empty:
        sys.exit("error: no analysis inputs found -- run the other scripts first")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    # Every flagged or regressing patch, plus a fixed-seed random sample so the
    # review set is not composed only of suspicious patches.
    must_review = frame[frame["needs_review"]]
    remainder = frame[~frame["needs_review"]]
    n_sample = min(args.sample_size, len(remainder))
    sample = (remainder.sample(n=n_sample, random_state=args.seed)
              if n_sample else remainder.head(0))

    review = pd.concat([must_review, sample]).sort_values("instance_id").copy()
    review["review_reason"] = review["needs_review"].map(
        {True: "flagged/regression/target-only", False: "random sample"})
    review["category"] = ""
    review["notes"] = ""
    columns = ["instance_id", "repo", "review_reason", "n_flags", "flag_categories",
               "n_regressions", "n_pass_to_fail", "n_missing_after_patch",
               "n_new_collection_errors", "touches_only_targets", "n_ambiguous_matches",
               "n_on_timed_path", "n_off_timed_path", "off_timed_path_functions",
               "n_functions_changed", "functions_changed", "category", "notes"]
    review[[c for c in columns if c in review.columns]].to_csv(args.review_output, index=False)

    print(f"combined report : {out}  ({len(frame)} instance(s))")
    print(f"manual review   : {args.review_output}  "
          f"({len(must_review)} flagged + {n_sample} sampled)")
    print(f"category legend : {CATEGORY_LEGEND}")


if __name__ == "__main__":
    main()
