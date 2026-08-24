"""
Local enroot squashfs cache for SWE-bench eval images, to avoid re-hitting
Docker Hub's anonymous pull rate limit (100/hour, shared across the whole
cluster's egress IP) on every run. See plans/2026-08-24-swe-verified-
implementation-path.md for the incident that made this necessary: a
500-instance run needing ~500 distinct images blew through the limit in
~30 minutes, and even Docker Official Images (ubuntu) share the same quota.

Once an image is cached here, both the agent-generation path
(pyxis_environment.py, via run_mini_swebench.py) and the grading path
(swebench_pyxis.py) resolve to the local file instead of docker://, so a
fully-cached run never touches Docker Hub again.
"""
from pathlib import Path

CACHE_DIR = Path("/scratch/project/prj-02-llm-reasoning-kalathil/vishnukunde/coding-agent-rl/image_cache")


def safe_name(image: str) -> str:
    return image.replace("/", "_").replace(":", "_") + ".sqsh"


def cached_path(image: str) -> Path:
    return CACHE_DIR / safe_name(image)


def resolve_image_ref(image: str) -> str:
    """Local squashfs path if cached (nonzero size), else a docker:// pull."""
    p = cached_path(image)
    if p.is_file() and p.stat().st_size > 0:
        return str(p)
    return f"docker://{image}"
