from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import isolated_inference
import model_cache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one isolated rare-model task and print RSS before/after."
    )
    parser.add_argument(
        "task",
        choices=[
            "several_lines_parallel_furrow",
            "several_lines_reticular_asymmetric",
            "several_globules_asymmetrical_melanin",
            "several_globules_asymmetrical_other",
        ],
    )
    parser.add_argument("image_path")
    parser.add_argument(
        "--sleep-after",
        type=float,
        default=1.0,
        help="Seconds to wait before measuring final RSS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image_path}")

    mask = np.full(image.shape[:2], 255, dtype=np.uint8)

    rss_before = model_cache.current_rss_bytes()
    started_at = time.perf_counter()
    result = isolated_inference.run_isolated(args.task, image, mask)
    elapsed = time.perf_counter() - started_at
    rss_after_call = model_cache.current_rss_bytes()

    time.sleep(args.sleep_after)
    rss_after_sleep = model_cache.current_rss_bytes()

    payload = {
        "task": args.task,
        "result": result,
        "elapsed_sec": round(elapsed, 3),
        "rss_before_mib": round(rss_before / 1024 / 1024, 1),
        "rss_after_call_mib": round(rss_after_call / 1024 / 1024, 1),
        "rss_after_sleep_mib": round(rss_after_sleep / 1024 / 1024, 1),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
