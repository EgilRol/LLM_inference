# Dump LLaMA weights to binary files with a header for C++ loading.
#
# Binary format. All tensors are float32.
#   magic[4]     = "LLW\x01"
#   num_tensors  = uint32
#   for each tensor:
#     name_len   = uint32
#     name       = uint8[name_len] (UTF-8)
#     ndim       = uint32
#     shape      = uint32[ndim]
#   then raw tensor data in same order, row-major, float32.
import json
import os
import struct
from collections import defaultdict
import torch
from safetensors import safe_open


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(TOOLS_DIR, "..", "assets", "llama3")
CONFIG_PATH = os.path.join(ASSET_DIR, "config.json")
INDEX_PATH = os.path.join(ASSET_DIR, "model.safetensors.index.json")
OUT_DIR = os.path.join(TOOLS_DIR, "..", "data")
MAGIC = b"LLW\x01"


def param_to_out_key(param_name: str, num_layers: int) -> str | None:
    """Return which output file this param belongs to, or None to skip."""
    if param_name.startswith("model.embed_tokens"):
        return "embed_tokens"
    if param_name.startswith("lm_head"):
        return "lm_head"
    if param_name.startswith("model.norm."):
        return "norm"
    if param_name.startswith("model.layers."):
        parts = param_name.split(".")
        if len(parts) >= 3:
            try:
                layer_num = int(parts[2])
                if 0 <= layer_num < num_layers:
                    return f"layer_{layer_num:02d}"
            except ValueError:
                pass
    return None


def _tensor_bytes(tensor) -> bytes:
    """Convert tensor to float32 and return raw bytes (4 bytes per element)."""
    t = tensor.detach().float().contiguous()
    return t.numpy().tobytes()


def write_bin(out_path: str, entries: list) -> None:
    """Write one .bin file: header (magic, num_tensors, per-tensor metadata) then raw float32 data."""
    with open(out_path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", len(entries)))

        for name, tensor in entries:
            name_bytes = name.encode("utf-8")
            out.write(struct.pack("<I", len(name_bytes)))
            out.write(name_bytes)
            shape = tensor.shape
            out.write(struct.pack("<I", len(shape)))
            for d in shape:
                out.write(struct.pack("<I", d))

        for _name, tensor in entries:
            out.write(_tensor_bytes(tensor))


def main() -> None:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]

    with open(INDEX_PATH, "r") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    by_safetensor = defaultdict(list)
    for param_name, file_name in weight_map.items():
        by_safetensor[file_name].append(param_name)

    buckets = defaultdict(list)

    for safetensor_file, param_list in by_safetensor.items():
        path = os.path.join(ASSET_DIR, safetensor_file)
        with safe_open(path, framework="pt") as f:
            for param_name in param_list:
                key = param_to_out_key(param_name, num_layers)
                if key is None:
                    continue
                tensor = f.get_tensor(param_name)
                buckets[key].append((param_name, tensor))

    # Deterministic order per file (sort by param name)
    for key in buckets:
        buckets[key].sort(key=lambda x: x[0])

    os.makedirs(OUT_DIR, exist_ok=True)
    for key, entries in sorted(buckets.items()):
        out_path = os.path.join(OUT_DIR, f"{key}.bin")
        write_bin(out_path, entries)
        print(f"Wrote {out_path} ({len(entries)} tensors)")


if __name__ == "__main__":
    main()
