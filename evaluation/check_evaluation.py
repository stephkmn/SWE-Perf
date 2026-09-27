import json
import os
import statistics
import pandas as pd
from argparse import ArgumentParser
from utils import filter_outliers, find_max_significant_improvement, load_sweperf_dataset


# Memory readings carried through by run_evaluation.mem_fields(). Reports
# produced before the memory plugin existed simply lack these keys, so every
# accessor below degrades to None rather than failing.
MEM_FIELD = "rss_growth_kb"


def median_mem(entries):
    """Median memory growth (kB) across a test's repeats, or None if unrecorded."""
    vals = [e[MEM_FIELD] for e in entries.values() if e.get(MEM_FIELD) is not None]
    return statistics.median(vals) if vals else None


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def pct_change(new, old):
    """Percent change from old to new; None when either side is unmeasured."""
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def calculate_performance_result(sweperf_data, log_root):
    without_prediction = 0
    without_run = 0
    run_failed = 0
    human_improved = []
    model_improved = []
    human_total = []
    model_mem = []   # per-instance median memory growth, patched run
    base_mem = []    # per-instance median memory growth, base run
    for _, data in sweperf_data.iterrows():
        id = data['instance_id']
        duration_changes = data["duration_changes"]
        efficiency_test = data["efficiency_test"]
        # calculate human total
        ht = []
        for test in efficiency_test:
            duration_change_base = filter_outliers([duration_change[test]["base"] for duration_change in duration_changes])
            duration_change_head = filter_outliers([duration_change[test]["head"] for duration_change in duration_changes])
            ht.append(find_max_significant_improvement(duration_change_head, duration_change_base))
        human_total.append(sum(ht)/len(ht))

        # calculate model total
        if not os.path.exists(os.path.join(log_root, id, "run_instance.log")):
            without_prediction += 1
            continue
        if not os.path.exists(os.path.join(log_root, id, "report.json")):
            without_run += 1
            continue

        with open(os.path.join(log_root, id, "report.json"), "r") as file:
            report = json.load(file)
        
        durations = {}
        resutls = {}
        durations_base = {}
        results_base = {}
        mem_base_inst = {}
        mem_head_inst = {}
        mem_base, mem_head = mem_base_inst, mem_head_inst
        for test in report.keys():
            if "base" in report[test]:
                durations_base_ = [rep["duration"] for rep in report[test]["base"].values()]
                durations_base_ = filter_outliers(durations_base_)
                durations_base[test] = durations_base_

                results_base_ = [rep["outcome"] for rep in report[test]["base"].values()]
                results_base[test] = set(results_base_) == {"passed"}

                mem_base[test] = median_mem(report[test]["base"])
            if "human" in report[test]:
                durations_head_ = [rep["duration"] for rep in report[test]["human"].values()]
                durations_head_ = filter_outliers(durations_head_)
                durations[test] = durations_head_
                resutls[test] = set([rep["outcome"] for rep in report[test]["human"].values()]) == {"passed"}
                mem_head[test] = median_mem(report[test]["human"])
        if durations==None:
            run_failed+=1
            continue

        hi = []
        mi = []
        for test in efficiency_test:
            if test not in durations or test not in durations_base:
                # print(f"{log_root} {id} {test} not in durations or durations_base")
                run_failed += 1
                break
            if not results_base[test] or not resutls[test]:
                # print(f"{log_root} {id} {test} not in results or results_base")
                run_failed += 1
                break
            duration_change_base = filter_outliers([duration_change[test]["base"] for duration_change in duration_changes])
            duration_change_head = filter_outliers([duration_change[test]["head"] for duration_change in duration_changes])
            hi_sig = find_max_significant_improvement(duration_change_head, duration_change_base)
            hi.append(hi_sig)
            mi_sig = find_max_significant_improvement(durations[test], durations_base[test])
            mi.append(mi_sig)
        else:
            human_improved.append(sum(hi)/len(hi))
            model_improved.append(sum(mi)/len(mi))
            model_mem.append(mean_or_none([mem_head_inst.get(t) for t in efficiency_test]))
            base_mem.append(mean_or_none([mem_base_inst.get(t) for t in efficiency_test]))

    with_prediction = len(sweperf_data) - without_prediction
    total = len(sweperf_data)
    print(f"There are {len(sweperf_data)} data, {without_prediction} without prediction and {with_prediction} with prediction. ")
    print(f"There are {without_run/total} ({without_run}/{total}) failed patch, {(with_prediction - without_run)/total} ({with_prediction - without_run}/{total}) success patch")
    print(f"There are {run_failed/total} ({run_failed}/{total}) failed run, {(with_prediction - without_run - run_failed)/total} ({with_prediction - without_run - run_failed}/{total}) success run")
    model_mem_kb = mean_or_none(model_mem)
    base_mem_kb = mean_or_none(base_mem)
    mem_delta_pct = pct_change(model_mem_kb, base_mem_kb)
    print(f"Model efficiency improved: {sum(model_improved)/total}")
    print(f"Human efficiency improved: {sum(human_improved)/total}")
    print(f"Human total efficiency improved: {sum(human_total)/total}")
    if model_mem_kb is None:
        print("Memory: not recorded in these logs")
    else:
        print(f"Memory (median growth per test): base {base_mem_kb:.0f} kB, "
              f"model {model_mem_kb:.0f} kB"
              + (f", {mem_delta_pct:+.1f}%" if mem_delta_pct is not None else ""))
    return {
        "model": log_root,
        "total": total,
        "with_prediction": with_prediction,
        "with_run": with_prediction - without_run,
        "success": with_prediction - without_run - run_failed,
        "model_improved": sum(model_improved),
        "human_improved": sum(human_improved),
        "human_total_improved": sum(human_total),
        "apply": (with_prediction - without_run)/total,
        "correctness": (with_prediction - without_run - run_failed)/total,
        "performance": sum(model_improved)/total,
        "human_performance": sum(human_improved)/total,
        "human_total_performance": sum(human_total)/total,
        # Memory is reported as a median, not a significance-tested improvement:
        # unlike duration it is near-deterministic across repeats, so the
        # Mann-Whitney ratchet used for timing would be misleading here.
        "base_mem_growth_kb": base_mem_kb,
        "model_mem_growth_kb": model_mem_kb,
        "mem_change_pct": mem_delta_pct,
    }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="SWE-Perf/SWE-Perf", required=True, type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--log_root", required=True, type=str, help="log path")
    parser.add_argument("--output_path", required=True, type=str, help="performence output path")
    args = parser.parse_args()

    output_path = args.output_path

    log_root = args.log_root
    sweperf_data = load_sweperf_dataset(args.dataset_dir)
    sweperf_data = pd.DataFrame(sweperf_data)
    results = []
    # performence_paths = []
    # for log_root in log_roots:
    print("================================")
    print(log_root)
    print("[total]")
    result = calculate_performance_result(sweperf_data, log_root)
    result["repo"] = "total"
    results.append(result)
    for repo, group in sweperf_data.groupby('repo'):
        print("================================")
        print(log_root)
        print(f"[{repo}]")
        result = calculate_performance_result(group, log_root)
        if result:
            result["repo"] = repo
            results.append(result)
    # save results to csv
    df = pd.DataFrame(results)
    log_instance = log_root.split('/')[-1]

    df.to_csv(output_path, index=False)

