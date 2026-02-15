"""
DO NOT CHANGE THIS FILE 

This is just a simple script to test if the tokenizer works and produces the same token IDs as the C++ implementation.
You have to download model first to see it working. You can run it with a sentence as an argument or it will prompt you for input.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from transformers import AutoTokenizer


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tokenize a sentence with Hugging Face and print token IDs",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Sentence to tokenize. If omitted, you will be prompted.",
    )

    parser.set_defaults(add_special_tokens=True)

    args = parser.parse_args(argv)

    # Prepare input text
    if args.text:
        text = " ".join(args.text)
    else:
        try:
            text = input("Enter a sentence: ").strip()
        except EOFError:
            print("No input provided.", file=sys.stderr)
            return 2

    if not text:
        print("Empty input.", file=sys.stderr)
        return 2

    # Load tokenizer from local assets; avoid network fetches
    tokenizer = AutoTokenizer.from_pretrained(
        'assets/llama3',
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    # Encode and print IDs
    token_ids = tokenizer.encode(text, add_special_tokens=args.add_special_tokens)
    print(token_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
