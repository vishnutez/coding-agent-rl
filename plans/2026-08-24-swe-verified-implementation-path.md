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

### 1. Understand the harness internals — done 2026-08-24
- [x] `pip install swebench`, read the harness source to see exactly what
      docker calls are made per instance and what a pyxis replacement needs
      to replicate. Findings:
  - **Image**: each dataset row already carries its own full image string
    (`instance["image"]`, e.g.
    `swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest` on Docker
    Hub) via `TestSpec.image` from `make_test_spec()` — no naming scheme to
    reconstruct, pull it straight from the dataset. Pyxis pulls it the same
    way it already pulled `ubuntu`/`hello-world` in our smoke tests.
  - **Workdir**: `/testbed`, root user. No baked-in eval script in the
    image — the *entire* eval script (conda activate, apply the dataset's
    test-patch, run pytest/etc.) is a **dataset field**
    (`TestSpec.eval_script`), already fully assembled per-instance including
    exit-code capture. Nothing to author ourselves; just write it out and run
    `bash` on it.
  - **Patch application**: write the model's patch to a file, apply at
    `/testbed` trying in order `git apply --verbose` →
    `git apply --verbose --3way` → `git apply --verbose --reject` →
    `patch --batch --forward --fuzz=5 -p1 -i` (first exit-0 wins); if all
    fail, `git apply --check --reverse` against the same file — exit 0 there
    means it was already applied, still a pass. Only then is it a real
    failure (log `>>>>> Patch Apply Failed`; success logs
    `>>>>> Applied Patch`). *Then*, separately, run `TestSpec.eval_script`
    (which does its own test-patch apply + test run).
  - **Grading is docker-free already**: `grading.py`'s `get_eval_report()`
    only imports `re` and swebench's own parser/type modules — call it
    directly with `(test_spec, prediction, test_output_path)` exactly like
    the harness does. Don't reimplement any parsing.
  - **Predictions schema**: JSON/JSONL, one object per instance:
    `{"instance_id", "model_patch", "model_name_or_path"}`. Empty/`None`
    patches are filtered out (auto-fail) before running.
  - **No hidden docker dependency elsewhere**: container cleanup shells out
    to `docker stop/kill/rm` directly, and there's ghost-container-name
    collision handling — both irrelevant, since a `srun` step's container is
    inherently isolated per-invocation and torn down when the step exits.
    Nothing to replicate there.
  - Confirmed via smoke test: bind-mounting a `/scratch`-backed directory
    into a pyxis container and writing a file from inside it persists back
    to the host after the container exits — this is the mechanism for
    getting the patch in and the test log out. **Mount sources must live on
    `/scratch`** — a login-node-local `/tmp` path is invisible to compute
    nodes and fails with "No such file or directory".
  - No explicit per-instance CPU/mem/disk limits in the harness (default
    `--timeout 1800s`, default `--max_workers 4`, both artifacts of a
    single-docker-daemon setup, not real constraints) — start with a
    conservative 2 CPU / 8GB / 1800s per instance and adjust if some repos
    need more.

### 2. Build the pyxis adapter — core mechanics done 2026-08-24
- [x] `scripts/swebench/swebench_pyxis.py`: given `(instance_id, model_patch)`,
      stages `patch.diff`/`eval.sh` into a `/scratch`-backed per-instance dir,
      then runs
      `srun --container-image=docker://<image> --container-mounts=<staging_dir>:/staging --container-workdir=/testbed --container-writable`
      instead of `docker run`. The `--container-writable` flag was the one
      real surprise: enroot mounts the container rootfs **read-only by
      default** (docker's default is a writable overlay), which broke `git
      apply`/`.git/index.lock` until added.
- [x] Reuse SWE-bench's own `get_eval_report()` for grading — confirmed
      docker-free, call it directly (see findings above)
- [ ] Batch across instances via a SLURM job array (`--array=0-N%K`), one
      instance per array task — natural fit since the harness already treats
      instances as fully independent with no shared-daemon coordination to
      replicate. Check `squeue -p def` load before sizing `%K` concurrency.
      **Not built yet** — `sanity_check.py` runs instances sequentially from
      a plain background process, fine for validation (~1-2 min/instance)
      but far too slow for the full 500-instance set.

### 3. Validate correctness before trusting results — done 2026-08-24
- [x] Sanity check: ran gold patches for 2 instances (`astropy-12907`,
      `astropy-13033`) through the adapter — both scored `resolved=True`.
      First attempt failed on both (read-only filesystem blocked `git
      apply`), fixed by adding `--container-writable`; reran clean.
- [x] Ran the same 2 instances with an empty patch — both scored
      `resolved=False` as expected, confirming the eval isn't a rubber stamp.
      4/4 gold/empty × 2-instance matrix matched expectations.
- [ ] Spot-check a few real model-generated patches against the raw log
      output, not just the parsed verdict — deferred until the agent
      scaffold (section 4) is producing real patches to check.

### 3.5. Serving config decision for this phase — done 2026-08-24
Unlike EvalPlus (hardcoded 768-token budget forced thinking off), our own
agent scaffold controls its own per-turn token budget, so:
- [x] **Thinking mode: ON** — reverted `serve_vllm.sbatch` to the model's
      default chat template (dropped the `chat_template_no_think.jinja`
      override). SWE-bench needs real multi-file reasoning to locate and fix
      bugs, unlike HumanEval/MBPP's function-level synthesis — worth the
      extra tokens/turn.
- [x] **Max context: 200000** — initially set to 131072, then raised to
      match Qwen's own published SWE-bench Verified eval config for this
      model exactly (HF model card footnote: *"Internal agent scaffold
      (bash + file-edit tools); temp=1.0, top_p=0.95, 200K context window"*)
      so our numbers are comparable to their reported 73.4% resolve rate.
- [x] Restarted the vLLM server (job 461714 on `dgx010:8000`) with this
      config; verified thinking re-enabled via a direct completion test.

### 4. Wire in the model — scaffold built, smoke-testing 2026-08-24
- [x] **Scaffold: mini-swe-agent**, not Qwen Code. Confirmed via the HF model
      card footnote above that Qwen's own reported number came from an
      undisclosed *internal* scaffold, not Qwen Code — described only as
      "bash + file-edit tools", architecturally identical to mini-swe-agent's
      minimal ReAct loop. Qwen Code (their interactive CLI product) would add
      uncontrolled tooling/prompting on top, working against reproducibility.
- [x] **Prompt format: backticks, not tool-calling.** mini-swe-agent ships
      two variants: `swebench.yaml` (native OpenAI tool-calling) and
      `swebench_backticks.yaml` (plain-text fenced-code-block commands,
      regex-parsed). Went with backticks since our vLLM server isn't
      configured with a tool-call parser for this model's architecture
      (`qwen3_engine_tool_parser.py` exists in vLLM but untested here) —
      avoids a whole class of "does structured tool-calling actually work
      for this specific model" risk for no real benefit.
- [x] **Custom environment: `scripts/swebench/pyxis_environment.py`**
      (`PyxisEnvironment`, plugged in as `environment_class:
      pyxis_environment.PyxisEnvironment`). mini-swe-agent's built-in
      `DockerEnvironment`/`SingularityEnvironment` start one persistent
      container and `exec` each agent step into it — pyxis has no
      "exec into an already-running container" primitive. Design (confirmed
      by direct testing): reserve one node via a placeholder
      `sbatch --wrap="sleep N"` allocation, then run every agent step as a
      `srun --jobid=<placeholder> --container-name=X` job step — confirmed
      this persists container filesystem state across separate `srun`
      invocations as long as they land on the same node (guaranteed here
      since they're steps of one allocation), at ~2-3s overhead per step.
      `cleanup()` just `scancel`s the placeholder job.
- [x] **Wiring `get_sb_environment()`**: the installed package only
      auto-resolves `instance["image"]` into the environment config for a
      hardcoded list of class names (`docker`, `singularity`, `swerex_modal`,
      `contree`) — a custom class needs a small monkeypatch to get the image
      threaded through at all. `scripts/swebench/run_mini_swebench.py` does
      this, plus dynamically injects the vLLM server's current host:port
      (read from `server_endpoint.txt`) as `model.model_kwargs.api_base`,
      since the server's node changes across restarts.
- [x] **No SLURM array needed** — `mini-extra swebench --workers N` already
      parallelizes across instances via a thread pool, and each thread's
      `PyxisEnvironment` reserves its own placeholder allocation, so SLURM's
      own scheduler handles the concurrency. This supersedes the earlier
      "batch via SLURM job array" plan in section 2 — that's still the right
      approach for the *grading* pass (running the swebench eval containers
      against generated patches), but not needed for *generation*.
- [x] **Smoke test on 1 instance (`astropy-12907`) — pipeline validated,
      2 real bugs found and fixed en route:**
  - `_query()` needs the paired `litellm_textbased` model class, not the
    default `litellm` — the backticks *prompt* alone isn't enough, since the
    default `LitellmModel` unconditionally sends `tools=[BASH_TOOL]` +
    `tool_choice=auto` regardless of prompt wording, which our vLLM server
    (no `--tool-call-parser` configured) rejects outright. Fixed by adding
    `model.model_class: litellm_textbased` to `mini_config_overrides.yaml`.
  - **`--container-writable` is required on *every* `execute()` call, not
    just the initial container-creation step** — contradicts what our
    earlier persistence smoke test suggested, because that test only did a
    `cat` (read); pyxis re-mounts a named container read-only by default on
    each new invocation regardless of how it was created. Without this, the
    agent could explore/read the repo fine but every actual patch/build
    attempt failed with "Read-only file system", derailing the trajectory.
    Fixed in `pyxis_environment.py`'s `execute()`.
  - After both fixes: the agent correctly diagnosed the real bug (nested
    `CompoundModel` handling in `_cstack`), read the right files, and
    applied a source fix ("Fix applied successfully"). It didn't reach
    submission on this run — got stuck late in a `RepeatedFormatError` loop
    while fighting `astropy`/`erfa`'s C-extension build (a real
    environment-complexity issue on this instance, not an infra bug;
    mini-swe-agent correctly gives up rather than hanging). `preds.json`
    correctly recorded an empty patch for the unsubmitted instance, and the
    placeholder SLURM allocation cleaned up properly.
- [ ] Run a slightly larger batch (~5 instances, a few workers) to see the
      typical submission rate/timing before committing to the full 500 —
      one stuck instance isn't enough to judge normal behavior.
- [ ] Generate predictions for all 500 instances once that looks healthy
- [ ] Feed the resulting `preds.json` into the adapter from section 2 for
      grading

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
