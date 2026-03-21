#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from collections import Counter

import webdataset as wds


def decode_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Safe WebDataset smoke reader for SAIL workflow")
    parser.add_argument("--shards", type=str, required=True, help="Shard pattern, e.g. /blob/.../cc3m-train-{0000..0287}.tar")
    parser.add_argument("--max-samples", type=int, default=64, help="How many samples to inspect")
    parser.add_argument("--output-dir", type=str, default="/home/aiscuser/sail_runs/cc3m_wds_smoke", help="Local output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "report.json")
    preview_path = os.path.join(args.output_dir, "preview.jsonl")

    dataset = wds.WebDataset(args.shards, shardshuffle=False)

    key_counter = Counter()
    inspected = 0

    with open(preview_path, "w", encoding="utf-8") as fw:
        for sample in dataset:
            inspected += 1
            keys = sorted(sample.keys())
            key_counter.update(keys)

            image_bytes = sample.get("jpg")
            short_caption = decode_text(sample.get("short"))[:200]
            long_caption = decode_text(sample.get("long"))[:200]

            item = {
                "index": inspected,
                "sample_key": decode_text(sample.get("__key__")),
                "keys": keys,
                "jpg_bytes": len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else None,
                "jpg_sha1_prefix": hashlib.sha1(image_bytes).hexdigest()[:12] if isinstance(image_bytes, (bytes, bytearray)) else None,
                "short_preview": short_caption,
                "long_preview": long_caption,
            }
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

            if inspected >= args.max_samples:
                break

    report = {
        "shards": args.shards,
        "max_samples": args.max_samples,
        "inspected_samples": inspected,
        "observed_keys": key_counter,
        "preview_file": preview_path,
        "note": "Read-only inspection. No writes under /blob.",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] inspected_samples={inspected}")
    print(f"[OK] report={report_path}")
    print(f"[OK] preview={preview_path}")


if __name__ == "__main__":
    main()
