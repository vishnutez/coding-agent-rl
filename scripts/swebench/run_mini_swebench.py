#!/usr/bin/env python3
"""
Thin wrapper around `mini-extra swebench` (mini-swe-agent's batch SWE-bench
runner) that:
  1. Makes our custom PyxisEnvironment importable and wires it into
     get_sb_environment()'s per-instance image-name resolution — the
     installed package only special-cases "docker"/"singularity"/
     "swerex_modal"/"contree" for that, so a custom environment_class
     string needs this one small monkeypatch to get instance["image"]
     threaded through at all.
  2. Reads the vLLM server's current host:port from server_endpoint.txt
     (written by serve_vllm.sbatch) and injects it as model.model_kwargs.api_base,
     since the server's node changes across restarts.

All other CLI args (--subset, --split, --slice, --workers, -o, etc.) pass
straight through to the real `mini-extra swebench` CLI.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))  # for `import pyxis_environment`

import minisweagent.run.benchmarks.swebench as sb  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.environments import get_environment  # noqa: E402
from image_cache import resolve_image_ref  # noqa: E402

_original_get_sb_environment = sb.get_sb_environment


def _patched_get_sb_environment(config: dict, instance: dict):
    env_config = {**config.get("environment", {})}
    if env_config.get("environment_class") == "pyxis_environment.PyxisEnvironment":
        # mini-swe-agent loads princeton-nlp/SWE-Bench_Verified, an older
        # dataset schema with no "image" column, so swebench's own
        # make_test_spec(instance)["image"] (which requires that field)
        # KeyErrors here. Fall back to mini-swe-agent's own
        # get_swebench_docker_image_name() naming convention instead — but
        # strip its baked-in "docker.io/" registry prefix first, since that
        # combined with our "docker://" scheme prefix produces a malformed
        # "docker://docker.io/..." reference pyxis rejects.
        image_name = sb.get_swebench_docker_image_name(instance).removeprefix("docker.io/")
        # Prefer a locally-cached squashfs (see image_cache.py) over a fresh
        # docker:// pull -- Docker Hub's anonymous rate limit (100/hour,
        # shared cluster-wide) can't sustain ~500 distinct image pulls in
        # one run.
        env_config["image"] = resolve_image_ref(image_name)
        env = get_environment(env_config)
        if startup_command := config.get("run", {}).get("env_startup_command"):
            from jinja2 import StrictUndefined, Template

            rendered = Template(startup_command, undefined=StrictUndefined).render(**instance)
            out = env.execute({"command": rendered})
            if out["returncode"] != 0:
                raise RuntimeError(f"Error executing startup command: {out}")
        return env
    return _original_get_sb_environment(config, instance)


sb.get_sb_environment = _patched_get_sb_environment


def _read_endpoint() -> str:
    endpoint_file = PROJECT_DIR / "scripts" / "server_endpoint.txt"
    if not endpoint_file.exists():
        raise SystemExit(f"No vLLM server endpoint found at {endpoint_file} — is serve_vllm.sbatch running?")
    return endpoint_file.read_text().strip()


if __name__ == "__main__":
    endpoint = _read_endpoint()
    base_url = f"http://{endpoint}/v1"
    backticks_config = str(builtin_config_dir / "benchmarks" / "swebench_backticks.yaml")
    overrides_config = str(SCRIPT_DIR / "mini_config_overrides.yaml")

    sys.argv = [
        sys.argv[0],
        "-c",
        backticks_config,
        "-c",
        overrides_config,
        "-c",
        f"model.model_kwargs.api_base={base_url}",
        *sys.argv[1:],
    ]
    sb.app()
