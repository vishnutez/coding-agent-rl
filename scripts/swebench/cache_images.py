"""
Pre-pulls SWE-bench Verified eval images and saves them as local enroot
squashfs files under image_cache/, self-paced to stay under Docker Hub's
anonymous rate limit (100 pulls/hour, shared cluster-wide -- even Docker
Official Images share this quota, confirmed 2026-08-24). Resumable: skips
images that already have a nonzero-size .sqsh file.

Images average ~2.7GB each; all ~500 distinct Verified images would need
~1.35TB, which doesn't fit in available scratch space. Worse, the
*entire shared cluster filesystem* (not just this project's quota) was
at 96% full with only ~347GB free total when this was discovered
(2026-08-24) -- so the cache budget default is kept modest (100GB, ~35
images) to avoid eating most of the remaining shared headroom. Whatever
isn't cached falls back to a paced live docker:// pull at eval time
(image_cache.py's resolve_image_ref()), which costs no persistent disk
since enroot extracts uncached pulls to the compute node's local /tmp,
not /scratch.

Usage: python cache_images.py [--sleep SECONDS] [--max-attempts N] [--max-cache-gb N]
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


def cache_dir_size_gb() -> float:
    return sum(f.stat().st_size for f in CACHE_DIR.glob("*.sqsh")) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=45.0, help="seconds between attempts, regardless of outcome")
    ap.add_argument("--max-attempts", type=int, default=1, help="retry attempts per image before giving up")
    ap.add_argument("--max-cache-gb", type=float, default=100.0, help="stop once cache reaches this size")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_verified_dataset()
    images = sorted({inst["image"] for inst in dataset})
    print(f"{len(images)} distinct images in SWE-bench Verified")

    already = [img for img in images if cached_path(img).is_file() and cached_path(img).stat().st_size > 0]
    todo = [img for img in images if img not in already]
    print(f"{len(already)} already cached, {len(todo)} remaining, budget={args.max_cache_gb}GB")

    failed = []
    for i, image in enumerate(todo, 1):
        size_gb = cache_dir_size_gb()
        if size_gb >= args.max_cache_gb:
            print(f"\nReached cache budget ({size_gb:.1f}GB >= {args.max_cache_gb}GB), stopping. "
                  f"{len(todo) - i + 1} images left uncached -- they'll fall back to a live "
                  f"docker:// pull at eval time.")
            break
        ok = False
        for attempt in range(1, args.max_attempts + 1):
            ok = cache_one(image)
            if ok:
                break
            print(f"[{i}/{len(todo)}] {image}: attempt {attempt}/{args.max_attempts} failed", flush=True)
            time.sleep(args.sleep)
        status = f"OK (cache now {cache_dir_size_gb():.1f}GB)" if ok else "FAILED (giving up for now)"
        print(f"[{i}/{len(todo)}] {image}: {status}", flush=True)
        if not ok:
            failed.append(image)
        time.sleep(args.sleep)

    print(f"\n=== Done. cache size: {cache_dir_size_gb():.1f}GB, {len(failed)} failed this pass ===")
    if failed:
        failed_file = CACHE_DIR / "failed_images.txt"
        failed_file.write_text("\n".join(failed) + "\n")
        print(f"Failed images written to {failed_file} -- rerun this script to retry them")


if __name__ == "__main__":
    main()
