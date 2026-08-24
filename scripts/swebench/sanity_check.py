"""
Validates the pyxis adapter end-to-end on a small number of instances before
trusting it at scale (plan item: "Validate correctness before trusting
results"). For each selected instance, runs BOTH the gold patch (expect
resolved=True) and an empty patch (expect resolved=False) through the same
container/grading path, and reports whether each matched expectation.

Usage: python sanity_check.py [--n N] [--timeout SECONDS]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from swebench_pyxis import (
    build_test_specs,
    grade_instance,
    load_verified_dataset,
    run_instance_container,
    stage_instance,
)

STAGING_ROOT = "/scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl/swebench_runs/sanity_check"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="number of instances to test")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    dataset = load_verified_dataset()
    instances = dataset[: args.n]
    test_specs = build_test_specs(instances)

    results = []
    for inst in instances:
        iid = inst["instance_id"]
        ts = test_specs[iid]
        for label, patch in [("gold", inst["patch"]), ("empty", "")]:
            staging_dir = Path(STAGING_ROOT) / label / iid
            staging_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"=== {iid} [{label}] === staging at {staging_dir}", flush=True)
            sd = stage_instance(ts, patch, staging_dir.parent)
            log_path, returncode = run_instance_container(
                ts, sd, timeout_s=args.timeout
            )
            report = grade_instance(ts, f"sanity-{label}", patch, log_path)
            resolved = report.get(iid, {}).get("resolved")
            expected = True if label == "gold" else False
            ok = resolved == expected
            print(
                f"    srun_returncode={returncode} resolved={resolved} "
                f"expected={expected} {'OK' if ok else 'MISMATCH'}",
                flush=True,
            )
            results.append(
                {
                    "instance_id": iid,
                    "label": label,
                    "resolved": resolved,
                    "expected": expected,
                    "ok": ok,
                    "log_path": str(log_path),
                }
            )

    print("\n=== SUMMARY ===")
    n_ok = sum(r["ok"] for r in results)
    for r in results:
        print(
            f"{r['instance_id']:45s} {r['label']:6s} "
            f"resolved={r['resolved']!s:6s} expected={r['expected']!s:6s} "
            f"{'OK' if r['ok'] else 'MISMATCH -- see ' + r['log_path']}"
        )
    print(f"\n{n_ok}/{len(results)} matched expectations")
    Path(STAGING_ROOT, "sanity_results.json").write_text(json.dumps(results, indent=2))
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
