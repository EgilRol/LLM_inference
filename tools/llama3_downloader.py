# python tools/llama3_downloader.py --out ./assets/llama3/
# THE MODEL FILES "MUST" BE ASSUMED TO BE IN ./assets/llama3/ FOR OTHER SCRIPTS.
# DO NOT CHANGE THIS FILE 

import os
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

DEFAULT_ALLOW = [
    # model weights (Transformers / safetensors)
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    # configs
    "config.json",
    "generation_config.json",
    # tokenizer (some repos have tokenizer.json, some have tokenizer.model too)
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=False,
        help="Output directory, e.g. ./checkpoints/llama3_8b_instruct",
    )
    ap.add_argument(
        "--revision", default="main", help="Branch/tag/commit (default: main)"
    )
    ap.add_argument(
        "--full", action="store_true", help="Download full repo (no filtering)"
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing HF token.\n"
            "Do one of:\n"
            "  export HF_TOKEN=...   (or HUGGINGFACE_HUB_TOKEN)\n"
            "or\n"
            "  huggingface-cli login\n"
        )

    allow_patterns = None if args.full else DEFAULT_ALLOW

    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.out or "./assets/llama3/",
        local_dir_use_symlinks=False,  # make a real directory copy (handy for projects/containers)
        token=token,
        allow_patterns=allow_patterns,
        max_workers=8,
        resume_download=True,
    )

    print("Downloaded to:", path)
    if not args.full:
        print("Downloaded (filtered) patterns:", DEFAULT_ALLOW)


if __name__ == "__main__":
    main()


