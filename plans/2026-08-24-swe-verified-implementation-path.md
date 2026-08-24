# SWE-bench Verified on this cluster — implementation path

Deferred while we run EvalPlus first. Come back to this once the vLLM endpoint
is validated and we want deeper coding-agent signal.

## Why this needs a plan

SWE-bench Verified's official harness drives evaluation entirely through the
`docker` Python SDK / CLI (build per-instance image → run container → apply
patch → run tests → parse log). This cluster has **no docker, apptainer,
singularity, or podman** anywhere (checked login node and compute node
`dgx012` directly — no binaries, no daemon, no systemd unit).

What it *does* have: **Pyxis + Enroot**, NVIDIA's SLURM-native container
stack. Verified working on 2026-08-24 — `srun --container-image=docker://...`
successfully pulled and ran `ubuntu:22.04` on a compute node. No docker
daemon required, pulls straight from a registry via enroot.

The eval containers themselves don't need GPU (they just apply a patch and
run the repo's test suite on CPU) — GPU is only needed for the vLLM server,
which runs separately. So this is a harness-adapter problem, not an
infra-capability problem.

## Findings so far

- [x] Confirm no docker/apptainer/singularity/podman on login node
- [x] Confirm no container runtime on GPU compute node (`dgx012`, via `srun`)
- [x] Confirm no relevant module (`module spider` checked docker, apptainer,
      singularity, podman, enroot, pyxis, charliecloud, shifter, sarus)
- [x] Confirm Pyxis SPANK plugin is active (`srun --help` shows
      `--container-image` etc. tagged `[pyxis]`)
- [x] Confirm enroot config/prolog/epilog present under
      `/cm/shared/apps/slurm/etc`
- [x] Smoke-test: `srun --container-image=docker://hello-world` — image
      pulled fine, run failed only because that image has no `/bin/sh`
      (expected, not a real failure)
- [x] Smoke-test: `srun --container-image=docker://ubuntu:22.04` — pulled and
      ran successfully, correct OS inside container

## Remaining work

### 1. Understand the harness internals
- [ ] `pip install swebench`, read `swebench/harness/run_evaluation.py` and
      `docker_build.py`/`docker_utils.py` to see exactly what docker calls
      are made per instance (build vs. pull, what gets mounted, where the
      patch is injected, where the test script lives inside the image)
- [ ] Confirm the registry SWE-bench Verified eval images are actually
      published to (expect Docker Hub under `swebench/sweb.eval.x86_64.*` —
      verify exact naming/tag scheme used by the installed `swebench`
      version) and that enroot can pull from it directly
- [ ] Identify the per-instance eval entrypoint/script path baked into each
      image (e.g. a `run_tests.sh` equivalent) and the exact file the patch
      needs to be applied against (`/testbed` or similar)

### 2. Build the pyxis adapter
- [ ] Write a script that, given `(instance_id, model_patch)`, does the
      equivalent of the harness's per-instance docker run using
      `srun --container-image=docker://<image> --container-mounts=<patch>:<path>`
      instead of `docker run`
- [ ] Reuse SWE-bench's own log-parsing/grading utilities (these just parse
      text output, no docker dependency) to turn the raw container log into
      a pass/fail verdict — don't reimplement this part
- [ ] Decide how to batch across 500 Verified instances: SLURM job array vs.
      a driver script issuing many `srun` calls, respecting partition
      concurrency (queue was fairly loaded during setup — check `squeue -p def`
      before running the full set)

### 3. Validate correctness before trusting results
- [ ] Sanity check: run a handful of instances with the **gold patch**
      (ground truth) through the adapter and confirm they score PASS —
      proves the adapter's execution/grading path is correct independent of
      the model
- [ ] Run a handful of instances with an obviously-wrong patch (e.g. empty
      diff) and confirm they score FAIL — proves it's not a rubber stamp
- [ ] Spot-check a few real model-generated patches against the raw log
      output, not just the parsed verdict

### 4. Wire in the model
- [ ] Decide on an agent scaffold to generate patches from the vLLM endpoint
      (e.g. mini-swe-agent, or something bespoke) — SWE-bench Verified needs
      a trajectory that produces a diff, not just a single completion
- [ ] Point the scaffold's LLM client at the vLLM OpenAI-compatible endpoint
- [ ] Generate predictions for all 500 instances, feed into the adapter from
      step 2

### 5. Scale + report
- [ ] Run the full Verified set, aggregate resolve rate
- [ ] Decide whether the pyxis adapter is worth generalizing into reusable
      infra for future evals/RL rollouts on this cluster, given verl is the
      eventual training stack

## Open questions to revisit

- Does the RL rollout loop (verl) itself need sandboxed code execution during
  training, not just eval? If so, this same pyxis adapter is probably the
  right long-term building block — worth designing it with that reuse in
  mind rather than as a one-off eval script.
- Is there a lighter-weight path we're missing (e.g. SWE-bench's Modal
  backend, if we get a Modal account) that avoids building/maintaining this
  adapter at all? Worth a cost/effort comparison before committing to option
  above.
