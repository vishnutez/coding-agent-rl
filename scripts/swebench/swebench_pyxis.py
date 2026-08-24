"""
Shared library for running SWE-bench Verified instances via Pyxis/Enroot
(srun --container-image=docker://...) instead of the docker SDK the official
swebench harness uses. See plans/2026-08-24-swe-verified-implementation-path.md
for why this exists.

Reuses swebench's own TestSpec construction and get_eval_report() grading
unmodified — only the container-execution glue is replaced.
"""
import json
import subprocess
from pathlib import Path

from swebench.harness.grading import get_eval_report
from swebench.harness.utils import load_swebench_dataset, make_test_spec

from image_cache import resolve_image_ref

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]


def load_verified_dataset(instance_ids=None):
    return load_swebench_dataset(
        name="SWE-bench/SWE-bench_Verified", split="test", instance_ids=instance_ids
    )


def build_test_specs(instances):
    return {inst["instance_id"]: make_test_spec(inst) for inst in instances}


def stage_instance(test_spec, model_patch, staging_root):
    """Write patch.diff + eval.sh into staging_root/<instance_id>/."""
    staging_dir = Path(staging_root) / test_spec.instance_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "patch.diff").write_text(model_patch or "")
    (staging_dir / "eval.sh").write_text(test_spec.eval_script)
    return staging_dir


def _container_command() -> str:
    """
    Bash run inside the container: apply the model patch at /testbed using
    swebench's own fallback chain, then run the dataset-provided eval script.
    Mirrors swebench/harness/run_evaluation.py's EvaluationContainerRunner
    patch-application logic exactly (same commands, same order, same
    already-applied fallback check).
    """
    apply_attempts = "\n".join(
        f'  if [ "$APPLIED" -eq 0 ] && {cmd} /staging/patch.diff; then APPLIED=1; fi'
        for cmd in GIT_APPLY_CMDS
    )
    return f"""
set -uo pipefail
cd /testbed
APPLIED=0
{apply_attempts}
if [ "$APPLIED" -eq 0 ] && git apply --check --reverse /staging/patch.diff 2>/dev/null; then
  APPLIED=1
fi
if [ "$APPLIED" -eq 1 ]; then
  echo ">>>>> Applied Patch"
  bash /staging/eval.sh
else
  echo ">>>>> Patch Apply Failed"
fi
"""


def run_instance_container(
    test_spec, staging_dir, cpus=2, mem="8G", timeout_s=1800, partition="def"
):
    """
    Run one instance's container via srun/pyxis. Returns the path to the
    captured combined stdout/stderr log (test_output.txt), written on the
    host side via shell redirection of the srun call itself — no need to
    mount an output path, only the input staging dir (patch.diff, eval.sh).
    """
    staging_dir = Path(staging_dir)
    log_path = staging_dir / "test_output.txt"
    cmd = [
        "srun",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={max(1, timeout_s // 60)}",
        f"--container-image={resolve_image_ref(test_spec.image)}",
        f"--container-mounts={staging_dir}:/staging",
        "--container-workdir=/testbed",
        # Enroot mounts the container rootfs read-only by default (unlike
        # docker's writable-by-default overlay); the eval needs to git-apply
        # patches and write test artifacts under /testbed.
        "--container-writable",
        "bash",
        "-c",
        _container_command(),
    ]
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s + 120)
    return log_path, proc.returncode


def grade_instance(test_spec, model_name_or_path, model_patch, log_path):
    prediction = {
        "instance_id": test_spec.instance_id,
        "model_patch": model_patch or "",
        "model_name_or_path": model_name_or_path,
    }
    return get_eval_report(
        test_spec, prediction, str(log_path), include_tests_status=True
    )
