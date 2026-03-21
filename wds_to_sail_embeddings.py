#!/usr/bin/env python3
import argparse
import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import torch
import webdataset as wds


def slugify(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def decode_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def chunked(items: List, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def decode_rgb_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def build_paths(output_root: str, dataset_name: str, text_model: str, vision_model: str, text_key: str, extra_text_key: Optional[str]) -> Dict[str, str]:
    text_model_slug = slugify(text_model)
    vision_model_slug = slugify(vision_model)

    image_dir = os.path.join(output_root, "image_embedding", vision_model_slug, dataset_name)
    text_dir = os.path.join(output_root, "text_embedding", text_model_slug, f"{dataset_name}_{text_key}")
    extra_text_dir = None
    if extra_text_key:
        extra_text_dir = os.path.join(output_root, "text_embedding", text_model_slug, f"{dataset_name}_{extra_text_key}")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    if extra_text_dir:
        os.makedirs(extra_text_dir, exist_ok=True)

    return {
        "image_dir": image_dir,
        "text_dir": text_dir,
        "extra_text_dir": extra_text_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Convert WebDataset shards to SAIL .pt embedding files")
    parser.add_argument("--shards", type=str, required=True, help="WebDataset shard pattern, e.g. /blob/.../cc3m-train-{0000..0287}.tar")
    parser.add_argument("--dataset-name", type=str, default="cc3m_wds_wbf", help="Dataset name used in output folder names")
    parser.add_argument("--output-root", type=str, default="/home/aiscuser/sail_data", help="All outputs will be written under this directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for sample count")
    parser.add_argument("--image-key", type=str, default="jpg", help="Image field key in webdataset sample")
    parser.add_argument("--text-key", type=str, default="short", help="Primary text field key")
    parser.add_argument("--extra-text-key", type=str, default="long", help="Optional extra text key, set '' to disable")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-base-patch32", help="Vision model name for SAIL ImageEmbedding")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Text model name for SAIL SentenceEmbedding")
    parser.add_argument("--agg-mode", type=str, default="concat", choices=["concat", "cls", "patch"], help="Vision aggregation mode")
    parser.add_argument("--start-index", type=int, default=0, help="Skip first N matched samples")
    parser.add_argument("--read-workers", type=int, default=8, help="WebDataset loader worker processes")
    parser.add_argument("--decode-workers", type=int, default=8, help="Thread workers for image byte decoding")
    return parser.parse_args()


def main():
    args = parse_args()
    extra_text_key = args.extra_text_key.strip() or None

    # Import SAIL model classes lazily so --help works without repo import path.
    import sys
    sail_root = "/home/aiscuser/SAIL"
    if sail_root not in sys.path:
        sys.path.insert(0, sail_root)
    from model.vision_model import ImageEmbedding
    from model.language_model import SentenceEmbedding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = build_paths(
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        text_model=args.text_model,
        vision_model=args.vision_model,
        text_key=args.text_key,
        extra_text_key=extra_text_key,
    )

    print(f"[INFO] device={device}")
    print(f"[INFO] reading shards={args.shards}")
    print(f"[INFO] writing under={args.output_root}")

    image_encoder = ImageEmbedding(args.vision_model, device=device, agg_mode=args.agg_mode)
    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    text_encoder = SentenceEmbedding(args.text_model)
    text_encoder.eval()

    base_ds = wds.WebDataset(args.shards, shardshuffle=False, handler=wds.warn_and_continue)
    tuple_keys = [args.image_key, args.text_key]
    if extra_text_key:
        tuple_keys.append(extra_text_key)
    tuple_keys.append("__key__")
    ds = base_ds.to_tuple(*tuple_keys)
    if args.read_workers > 0:
        loader = wds.WebLoader(
            ds,
            batch_size=None,
            num_workers=args.read_workers,
            persistent_workers=True,
        )
    else:
        loader = ds

    kept = 0
    seen = 0
    written_batches = 0
    sample_keys: List[str] = []

    pending_image_bytes: List[bytes] = []
    pending_keys: List[str] = []
    texts: List[str] = []
    extra_texts: List[str] = []

    def flush_batch(batch_idx: int):
        nonlocal pending_image_bytes, pending_keys, texts, extra_texts
        if not pending_image_bytes:
            return

        if args.decode_workers > 1:
            with ThreadPoolExecutor(max_workers=args.decode_workers) as executor:
                decoded_images = list(executor.map(decode_rgb_from_bytes, pending_image_bytes))
        else:
            decoded_images = [decode_rgb_from_bytes(x) for x in pending_image_bytes]

        valid_indices = [i for i, img in enumerate(decoded_images) if img is not None]
        if not valid_indices:
            pending_image_bytes = []
            pending_keys = []
            texts = []
            extra_texts = []
            return

        images = [decoded_images[i] for i in valid_indices]
        valid_texts = [texts[i] for i in valid_indices]
        valid_keys = [pending_keys[i] for i in valid_indices]
        if extra_text_key:
            valid_extra_texts = [extra_texts[i] for i in valid_indices]
        else:
            valid_extra_texts = None

        with torch.no_grad():
            image_inputs = image_encoder.image_processor(images, return_tensors="pt")
            if isinstance(image_inputs, torch.Tensor):
                image_inputs = image_inputs.to(device=device, dtype=image_encoder.model.dtype)
            else:
                image_inputs = image_inputs.to(device=device, dtype=image_encoder.model.dtype)

            image_emb = image_encoder.forward(image_inputs).detach().cpu().to(torch.float16)
            text_emb = text_encoder.get_sentence_embeddings(valid_texts).detach().cpu().to(torch.float16)
            if extra_text_key:
                extra_text_emb = text_encoder.get_sentence_embeddings(valid_extra_texts).detach().cpu().to(torch.float16)
            else:
                extra_text_emb = None

        torch.save(image_emb, os.path.join(paths["image_dir"], f"{batch_idx}.pt"))
        torch.save(text_emb, os.path.join(paths["text_dir"], f"{batch_idx}.pt"))
        if extra_text_emb is not None and paths["extra_text_dir"]:
            torch.save(extra_text_emb, os.path.join(paths["extra_text_dir"], f"{batch_idx}.pt"))

        sample_keys.extend(valid_keys)
        pending_image_bytes = []
        pending_keys = []
        texts = []
        extra_texts = []

    for sample in loader:
        seen += 1

        if seen <= args.start_index:
            continue

        if extra_text_key:
            image_bytes, primary_text_raw, secondary_text_raw, sample_key_raw = sample
        else:
            image_bytes, primary_text_raw, sample_key_raw = sample
            secondary_text_raw = ""

        primary_text = decode_text(primary_text_raw).strip()
        secondary_text = decode_text(secondary_text_raw).strip() if extra_text_key else ""

        if image_bytes is None or not primary_text:
            continue
        if extra_text_key and not secondary_text:
            continue

        pending_image_bytes.append(image_bytes)
        pending_keys.append(decode_text(sample_key_raw))
        texts.append(primary_text)
        if extra_text_key:
            extra_texts.append(secondary_text)
        kept += 1

        if len(pending_image_bytes) >= args.batch_size:
            flush_batch(written_batches)
            written_batches += 1
            if written_batches % 10 == 0:
                print(f"[INFO] written_batches={written_batches}, kept_samples={kept}")

        if args.max_samples is not None and kept >= args.max_samples:
            break

    if pending_image_bytes:
        flush_batch(written_batches)
        written_batches += 1

    metadata = {
        "time": datetime.now().isoformat(),
        "input_shards": args.shards,
        "dataset_name": args.dataset_name,
        "image_key": args.image_key,
        "text_key": args.text_key,
        "extra_text_key": extra_text_key,
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "agg_mode": args.agg_mode,
        "seen_samples": seen,
        "kept_samples": kept,
        "written_batches": written_batches,
        "batch_size": args.batch_size,
        "output_paths": paths,
        "sample_keys_head": sample_keys[:20],
        "safety_note": "Read-only from /blob; all generated files are under output_root.",
    }

    metadata_path = os.path.join(args.output_root, f"{args.dataset_name}_conversion_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] kept_samples={kept}")
    print(f"[OK] written_batches={written_batches}")
    print(f"[OK] metadata={metadata_path}")
    print(f"[OK] image_embedding_dir={paths['image_dir']}")
    print(f"[OK] text_embedding_dir={paths['text_dir']}")
    if paths["extra_text_dir"]:
        print(f"[OK] extra_text_embedding_dir={paths['extra_text_dir']}")


if __name__ == "__main__":
    main()
