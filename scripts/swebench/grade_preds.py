"""
Grades a mini-swe-agent preds.json file using the pyxis adapter
(swebench_pyxis.py), running instances concurrently via a thread pool.

Usage: python grade_preds.py <preds.json> [--workers N]
"""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from swebench_pyxis import build_test_specs, grade_instance, load_verified_dataset, run_instance_container, stage_instance

STAGING_ROOT_DEFAULT = "/scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl/swebench_runs/grading"


def grade_one(instance_id, model_patch, model_name, test_spec, staging_root, timeout_s):
    staging_dir = stage_instance(test_spec, model_patch, staging_root)
    log_path, returncode = run_instance_container(test_spec, staging_dir, timeout_s=timeout_s)
    report = grade_instance(test_spec, model_name, model_patch, log_path)
    resolved = report.get(instance_id, {}).get("resolved")
    return {"instance_id": instance_id, "resolved": resolved, "returncode": returncode, "log_path": str(log_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--staging-root", default=STAGING_ROOT_DEFAULT)
    args = ap.parse_args()

    preds = json.loads(Path(args.preds_file).read_text())
    preds = {k: v for k, v in preds.items() if v.get("model_patch")}
    print(f"Grading {len(preds)} instances with non-empty patches (skipping empty ones)")

    instance_ids = list(preds.keys())
    dataset = load_verified_dataset(instance_ids=instance_ids)
    test_specs = build_test_specs(dataset)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                grade_one,
                iid,
                preds[iid]["model_patch"],
                preds[iid].get("model_name_or_path", "unknown"),
                test_specs[iid],
                args.staging_root,
                args.timeout,
            ): iid
            for iid in instance_ids
        }
        for fut in concurrent.futures.as_completed(futures):
            iid = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"instance_id": iid, "resolved": None, "error": str(e)}
            results.append(r)
            print(f"{iid}: resolved={r.get('resolved')}", flush=True)

    n_resolved = sum(1 for r in results if r.get("resolved"))
    print(f"\n=== {n_resolved}/{len(results)} resolved ===")
    Path(args.staging_root, "grading_results.json").parent.mkdir(parents=True, exist_ok=True)
    Path(args.staging_root, "grading_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
