"""
Pre-pulls every distinct SWE-bench Verified eval image once and saves it as
a local enroot squashfs under image_cache/, self-paced to stay under Docker
Hub's anonymous rate limit (100 pulls/hour, shared cluster-wide -- even
Docker Official Images share this quota, confirmed 2026-08-24). Resumable:
skips images that already have a nonzero-size .sqsh file.

Usage: python cache_images.py [--sleep SECONDS] [--max-attempts N]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from image_cache import CACHE_DIR, cached_path
from swebench_pyxis import load_verified_dataset


def cache_one(image: str, timeout_s: int = 600) -> bool:
    dest = cached_path(image)
    tmp_dest = dest.with_suffix(".sqsh.tmp")
    tmp_dest.unlink(missing_ok=True)
    container_name = "cache_" + "".join(c if c.isalnum() else "_" for c in image)[:60]
    cmd = [
        "srun",
        "-p",
        "def",
        "--cpus-per-task=2",
        "--mem=8G",
        f"--time={max(1, timeout_s // 60)}",
        "--output=/dev/null",
        "--error=/dev/null",
        f"--container-image=docker://{image}",
        f"--container-name={container_name}",
        f"--container-save={tmp_dest}",
        "true",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        tmp_dest.unlink(missing_ok=True)
        return False
    if proc.returncode == 0 and tmp_dest.is_file() and tmp_dest.stat().st_size > 0:
        tmp_dest.rename(dest)
        return True
    tmp_dest.unlink(missing_ok=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=45.0, help="seconds between attempts, regardless of outcome")
    ap.add_argument("--max-attempts", type=int, default=3, help="retry attempts per image before giving up")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_verified_dataset()
    images = sorted({inst["image"] for inst in dataset})
    print(f"{len(images)} distinct images in SWE-bench Verified")

    already = [img for img in images if cached_path(img).is_file() and cached_path(img).stat().st_size > 0]
    todo = [img for img in images if img not in already]
    print(f"{len(already)} already cached, {len(todo)} remaining")

    failed = []
    for i, image in enumerate(todo, 1):
        ok = False
        for attempt in range(1, args.max_attempts + 1):
            ok = cache_one(image)
            if ok:
                break
            print(f"[{i}/{len(todo)}] {image}: attempt {attempt}/{args.max_attempts} failed", flush=True)
            time.sleep(args.sleep)
        status = "OK" if ok else "FAILED (giving up for now)"
        print(f"[{i}/{len(todo)}] {image}: {status}", flush=True)
        if not ok:
            failed.append(image)
        time.sleep(args.sleep)

    print(f"\n=== Done. {len(todo) - len(failed)}/{len(todo)} newly cached, {len(failed)} failed ===")
    if failed:
        failed_file = CACHE_DIR / "failed_images.txt"
        failed_file.write_text("\n".join(failed) + "\n")
        print(f"Failed images written to {failed_file} -- rerun this script to retry them")


if __name__ == "__main__":
    main()
