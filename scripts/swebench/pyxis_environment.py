"""
mini-swe-agent Environment implementation backed by Pyxis/Enroot instead of
docker/singularity (neither is installed on this cluster).

Docker's model (used by minisweagent.environments.docker.DockerEnvironment):
start one long-lived container, `docker exec` each agent step into it.
Pyxis has no "exec into an already-running container" primitive — each
`srun --container-image=...` invocation is a fresh container instantiation.

Confirmed by direct testing (2026-08-24) that `srun --container-name=X`
reuses/persists a named container's filesystem state across separate `srun`
invocations *as long as they land on the same node*. So the design here is:
  1. Reserve one node up front via a placeholder `sbatch --wrap="sleep N"`
     allocation (cheap, CPU-only — no GPU needed for eval containers).
  2. Every agent step becomes a `srun --jobid=<placeholder> --container-name=X`
     *job step* of that allocation (fast, ~2-3s, no scheduling queue since
     it's a step of an existing allocation, and guaranteed same node).
  3. `cleanup()` cancels the placeholder job, tearing down the container.
"""
import logging
import os
import shlex
import subprocess
import time
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class PyxisEnvironmentConfig(BaseModel):
    image: str
    """docker://... image reference to run (e.g. docker://swebench/sweb.eval...)."""
    cwd: str = "/"
    env: dict[str, str] = {}
    forward_env: list[str] = []
    timeout: int = 60
    """Timeout (seconds) for each individual command execution."""
    interpreter: list[str] = ["bash", "-c"]
    partition: str = "def"
    cpus: int = 2
    mem: str = "8G"
    placeholder_time_s: int = 7200
    """How long to reserve the placeholder node allocation for (whole trajectory)."""
    start_timeout: int = 300
    """Timeout for the placeholder allocation to start running + first container pull."""


class PyxisEnvironment:
    def __init__(
        self,
        *,
        config_class: type = PyxisEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.container_name = f"msweap-{uuid.uuid4().hex[:10]}"
        self.job_id: str | None = None
        self._start_allocation()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        import platform

        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_allocation(self):
        hh = self.config.placeholder_time_s // 3600
        mm = (self.config.placeholder_time_s % 3600) // 60
        ss = self.config.placeholder_time_s % 60
        time_str = f"{hh:02d}:{mm:02d}:{ss:02d}"
        sbatch_cmd = [
            "sbatch",
            f"--partition={self.config.partition}",
            f"--cpus-per-task={self.config.cpus}",
            f"--mem={self.config.mem}",
            f"--time={time_str}",
            "--parsable",
            "--output=/dev/null",
            "--error=/dev/null",
            f"--wrap=sleep {self.config.placeholder_time_s}",
        ]
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, timeout=60, check=True)
        self.job_id = result.stdout.strip().split(";")[0]
        self.logger.info(f"Reserved placeholder allocation {self.job_id} for container {self.container_name}")

        deadline = time.time() + self.config.start_timeout
        while time.time() < deadline:
            st = subprocess.run(
                ["squeue", "-j", self.job_id, "-h", "-o", "%T"], capture_output=True, text=True
            ).stdout.strip()
            if st == "RUNNING":
                break
            if st == "":
                raise RuntimeError(f"Placeholder job {self.job_id} disappeared from queue before starting")
            time.sleep(1)
        else:
            raise RuntimeError(f"Placeholder job {self.job_id} did not start within {self.config.start_timeout}s")

        # Eagerly pull the image and create the named container so the first
        # real agent step doesn't pay the pull latency (and errors here
        # surface clearly, matching DockerEnvironment's check=True at start).
        init_cmd = [
            "srun",
            f"--jobid={self.job_id}",
            f"--container-image={self.config.image}",
            f"--container-name={self.container_name}",
            "--container-writable",
            f"--container-workdir={self.config.cwd}",
            "true",
        ]
        subprocess.run(init_cmd, capture_output=True, text=True, timeout=self.config.start_timeout, check=True)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.job_id, "Placeholder allocation not started"

        env_prefix = ""
        env_assignments = {**{k: os.environ[k] for k in self.config.forward_env if k in os.environ}, **self.config.env}
        if env_assignments:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_assignments.items()) + " "

        cmd = [
            "srun",
            f"--jobid={self.job_id}",
            f"--container-name={self.container_name}",
            # Confirmed necessary on every invocation, not just container
            # creation: pyxis re-mounts a named container read-only by
            # default on each srun call unless this is passed again.
            "--container-writable",
            f"--container-workdir={cwd}",
            *self.config.interpreter,
            env_prefix + command,
        ]

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        if getattr(self, "job_id", None) is not None:
            subprocess.Popen(["scancel", self.job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __del__(self):
        self.cleanup()
