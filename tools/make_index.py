import json
import os
import pickle
import struct
from pathlib import Path

import pandas as pd


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DATA_DIR = REPO_ROOT / "data"
SAFETENSORS_INDEX_PATH = REPO_ROOT / "assets" / "llama3" / "model.safetensors.index.json"
OUTPUT_PATH = TOOLS_DIR / "weight_index.pkl"
MAGIC = b"LLW\x01"
DTYPE = "float32"
ELEMENT_SIZE_BYTES = 4


def param_to_dump_key(param_name: str, num_layers: int) -> str | None:
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
            except ValueError:
                return None
            if 0 <= layer_num < num_layers:
                return f"layer_{layer_num:02d}"
    return None


def load_safetensors_metadata() -> tuple[dict[str, str], dict[str, str]]:
    with SAFETENSORS_INDEX_PATH.open("r", encoding="utf-8") as f:
        index_payload = json.load(f)

    with (REPO_ROOT / "assets" / "llama3" / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    num_layers = int(config["num_hidden_layers"])
    source_file_by_name = index_payload["weight_map"]
    dump_file_by_name = {}
    for tensor_name in source_file_by_name:
        dump_key = param_to_dump_key(tensor_name, num_layers)
        if dump_key is not None:
            dump_file_by_name[tensor_name] = f"{dump_key}.bin"
    return source_file_by_name, dump_file_by_name


def parse_dump_file(bin_path: Path) -> list[dict]:
    rows: list[dict] = []
    file_size = bin_path.stat().st_size

    with bin_path.open("rb") as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"{bin_path} has invalid magic {magic!r}")

        num_tensors = struct.unpack("<I", f.read(4))[0]
        header_entries = []
        for tensor_index in range(num_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = list(struct.unpack(f"<{ndim}I", f.read(4 * ndim)))
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            num_bytes = num_elements * ELEMENT_SIZE_BYTES
            header_entries.append(
                {
                    "tensor_index_in_file": tensor_index,
                    "tensor_name": name,
                    "ndim": ndim,
                    "shape": shape,
                    "num_elements": num_elements,
                    "num_bytes": num_bytes,
                }
            )

        data_offset = f.tell()
        for entry in header_entries:
            entry["data_offset"] = data_offset
            entry["data_end_offset"] = data_offset + entry["num_bytes"]
            if entry["data_end_offset"] > file_size:
                raise ValueError(
                    f"{bin_path} tensor {entry['tensor_name']} extends past end of file"
                )
            data_offset = entry["data_end_offset"]

        file_name = bin_path.name
        file_stem = bin_path.stem
        layer_index = (
            int(file_stem.split("_")[1]) if file_stem.startswith("layer_") else None
        )
        for entry in header_entries:
            rows.append(
                {
                    "dump_file": file_name,
                    "dump_path": os.fspath(bin_path.resolve()),
                    "dump_key": file_stem,
                    "layer_index": layer_index,
                    "file_num_tensors": num_tensors,
                    "dump_file_size_bytes": file_size,
                    "dtype": DTYPE,
                    **entry,
                }
            )

    return rows


def build_index() -> pd.DataFrame:
    source_file_by_name, dump_file_by_name = load_safetensors_metadata()

    rows = []
    for bin_path in sorted(DATA_DIR.glob("*.bin")):
        rows.extend(parse_dump_file(bin_path))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No dump files found in {DATA_DIR}")

    df["layer_index"] = pd.array(df["layer_index"], dtype="Int64")
    df["source_safetensors_file"] = df["tensor_name"].map(source_file_by_name)
    df["expected_dump_file"] = df["tensor_name"].map(dump_file_by_name)
    df["dump_file_matches_source_mapping"] = df["dump_file"] == df["expected_dump_file"]

    missing_source = df["source_safetensors_file"].isna()
    if missing_source.any():
        missing_names = df.loc[missing_source, "tensor_name"].tolist()
        raise ValueError(f"Missing safetensors mapping for tensors: {missing_names}")

    mismatched_dump_file = ~df["dump_file_matches_source_mapping"]
    if mismatched_dump_file.any():
        mismatched_names = df.loc[mismatched_dump_file, "tensor_name"].tolist()
        raise ValueError(
            "Dump file mismatch for tensors: "
            f"{mismatched_names}"
        )

    df = df[
        [
            "tensor_name",
            "dump_key",
            "dump_file",
            "dump_path",
            "layer_index",
            "tensor_index_in_file",
            "file_num_tensors",
            "dtype",
            "ndim",
            "shape",
            "num_elements",
            "num_bytes",
            "data_offset",
            "data_end_offset",
            "dump_file_size_bytes",
            "source_safetensors_file",
            "expected_dump_file",
            "dump_file_matches_source_mapping",
        ]
    ].sort_values(["dump_file", "tensor_index_in_file"], kind="stable")

    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    df = build_index()
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {OUTPUT_PATH} with {len(df)} rows")


if __name__ == "__main__":
    main()
