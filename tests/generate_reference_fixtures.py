import argparse
import os
import struct
from dataclasses import dataclass

import torch

import reference
from reference_fixture_format import FixtureTensor, write_fixture


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "reference")

WEIGHT_DTYPE_FP32 = 0
WEIGHT_DTYPE_BF16 = 1
FIXED_LAYERS = (0, 16, 31)
FIXED_SEQ_LENS = (1, 10, 100)
FIXED_SEED = 1337
LOGIT_CHUNK_ROWS = 2048


@dataclass
class TensorMeta:
    name: str
    dtype: int
    shape: tuple[int, ...]
    num_elements: int
    num_bytes: int
    data_offset: int


class WeightDump:
    def __init__(self, path: str):
        self.path = path
        self.entries = self._read_index(path)

    def _read_index(self, path: str) -> dict[str, TensorMeta]:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"LLW\x01":
                raise ValueError(f"Invalid weight dump magic in {path}")

            num_tensors = struct.unpack("<I", f.read(4))[0]
            ordered: list[TensorMeta] = []
            for _ in range(num_tensors):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                dtype = struct.unpack("<I", f.read(4))[0]
                ndim = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))

                num_elements = 1
                for dim in shape:
                    num_elements *= dim
                dtype_size = 4 if dtype == WEIGHT_DTYPE_FP32 else 2
                num_bytes = num_elements * dtype_size
                ordered.append(
                    TensorMeta(
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        num_elements=num_elements,
                        num_bytes=num_bytes,
                        data_offset=0,
                    )
                )

            data_offset = f.tell()
            entries: dict[str, TensorMeta] = {}
            for meta in ordered:
                meta.data_offset = data_offset
                data_offset += meta.num_bytes
                entries[meta.name] = meta
            return entries

    def meta(self, tensor_name: str) -> TensorMeta:
        return self.entries[tensor_name]

    def _decode_bytes(self, raw: bytes, dtype: int, shape: tuple[int, ...]) -> torch.Tensor:
        if dtype == WEIGHT_DTYPE_FP32:
            tensor = torch.frombuffer(memoryview(raw), dtype=torch.float32).clone()
            return tensor.reshape(shape)
        if dtype == WEIGHT_DTYPE_BF16:
            tensor = torch.frombuffer(memoryview(raw), dtype=torch.uint16).clone()
            return tensor.view(torch.bfloat16).reshape(shape).to(torch.float32)
        raise ValueError(f"Unsupported weight dtype {dtype}")

    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        meta = self.meta(tensor_name)
        with open(self.path, "rb") as f:
            f.seek(meta.data_offset)
            raw = f.read(meta.num_bytes)
        return self._decode_bytes(raw, meta.dtype, meta.shape)

    def load_row_block(self, tensor_name: str, start_row: int, row_count: int) -> torch.Tensor:
        meta = self.meta(tensor_name)
        if len(meta.shape) != 2:
            raise ValueError(f"{tensor_name} is not rank-2")
        rows, cols = meta.shape
        if start_row < 0 or row_count < 0 or start_row + row_count > rows:
            raise ValueError(f"Row block out of bounds for {tensor_name}")

        dtype_size = 4 if meta.dtype == WEIGHT_DTYPE_FP32 else 2
        row_bytes = cols * dtype_size
        with open(self.path, "rb") as f:
            f.seek(meta.data_offset + start_row * row_bytes)
            raw = f.read(row_count * row_bytes)
        return self._decode_bytes(raw, meta.dtype, (row_count, cols))

    def load_rows(self, tensor_name: str, row_indices: list[int]) -> torch.Tensor:
        meta = self.meta(tensor_name)
        if len(meta.shape) != 2:
            raise ValueError(f"{tensor_name} is not rank-2")
        rows, cols = meta.shape
        dtype_size = 4 if meta.dtype == WEIGHT_DTYPE_FP32 else 2
        row_bytes = cols * dtype_size

        raw = bytearray(len(row_indices) * row_bytes)
        with open(self.path, "rb") as f:
            for out_row, row_idx in enumerate(row_indices):
                if row_idx < 0 or row_idx >= rows:
                    raise ValueError(f"Row {row_idx} out of bounds for {tensor_name}")
                f.seek(meta.data_offset + row_idx * row_bytes)
                chunk = f.read(row_bytes)
                begin = out_row * row_bytes
                raw[begin : begin + row_bytes] = chunk
        return self._decode_bytes(bytes(raw), meta.dtype, (len(row_indices), cols))


def int_scalar(value: int) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.int32)


def float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float32).contiguous()


def deterministic_token_ids(seq_len: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(FIXED_SEED + seq_len)
    return torch.randint(0, reference.VOCAB, (seq_len,), generator=generator).tolist()


def layer_file_path(layer_idx: int) -> str:
    return os.path.join(DATA_DIR, f"layer_{layer_idx:02d}.bin")


def load_layer_weights(layer_idx: int) -> dict[str, torch.Tensor]:
    dump = WeightDump(layer_file_path(layer_idx))
    prefix = f"model.layers.{layer_idx}."
    tensor_names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    return {name: dump.load_tensor(prefix + name) for name in tensor_names}


def runtime_rope_tables(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair_dim = head_dim // 2
    pairs = torch.arange(pair_dim, dtype=torch.float32)
    theta = 1.0 / (reference.ROPE_BASE ** (pairs / pair_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, theta)
    return angles.cos(), angles.sin()


def flat_heads_to_reference_qkv(
    q_flat: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = q_flat.shape[0]
    q = q_flat.view(seq_len, reference.H, reference.H_D).transpose(0, 1)
    k = k_flat.view(seq_len, reference.H_K, reference.H_D).transpose(0, 1)
    v = v_flat.view(seq_len, reference.H_K, reference.H_D).transpose(0, 1)
    return q, k, v


def reference_qk_to_flat(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = q.shape[1]
    q_flat = q.transpose(0, 1).contiguous().view(seq_len, reference.H * reference.H_D)
    k_flat = k.transpose(0, 1).contiguous().view(seq_len, reference.H_K * reference.H_D)
    return q_flat, k_flat


def embed_tokens(token_ids: list[int], embed_dump: WeightDump) -> torch.Tensor:
    return embed_dump.load_rows("model.embed_tokens.weight", token_ids)


def output_logits_chunked(x: torch.Tensor, norm_weight: torch.Tensor, lm_head_dump: WeightDump) -> torch.Tensor:
    x_norm = reference.rmsnorm(x, norm_weight)
    x_last = x_norm[-1]
    lm_meta = lm_head_dump.meta("lm_head.weight")
    logits_parts = []
    for start in range(0, lm_meta.shape[0], LOGIT_CHUNK_ROWS):
        count = min(LOGIT_CHUNK_ROWS, lm_meta.shape[0] - start)
        weights = lm_head_dump.load_row_block("lm_head.weight", start, count)
        logits_parts.append(weights @ x_last)
    return torch.cat(logits_parts, dim=0)


def write_case(output_dir: str, file_name: str, tensors: list[FixtureTensor]) -> None:
    path = os.path.join(output_dir, file_name)
    write_fixture(path, tensors)
    print(f"Wrote {path}")


def generate_synthetic_cases(output_dir: str) -> None:
    matmul_cases = {
        "matmul_square.bin": (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        ),
        "matmul_rect.bin": (
            torch.tensor([[0.5, -1.0, 2.0], [3.0, 4.0, -2.0]], dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [3.0, -4.0], [0.5, 1.5]], dtype=torch.float32),
        ),
        "matmul_tall.bin": (
            torch.tensor([[1.0, -2.0], [0.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
            torch.tensor([[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]], dtype=torch.float32),
        ),
    }
    for file_name, (a, b) in matmul_cases.items():
        write_case(
            output_dir,
            file_name,
            [
                FixtureTensor("input_a", a),
                FixtureTensor("input_b", b),
                FixtureTensor("expected", a @ b),
            ],
        )

    residual_input = torch.tensor([[1.0, 2.0, 3.0], [-1.5, 0.0, 4.5]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -2.0, 1.0], [3.0, 1.5, -0.5]], dtype=torch.float32)
    write_case(
        output_dir,
        "residual_add_basic.bin",
        [
            FixtureTensor("input", residual_input),
            FixtureTensor("residual", residual),
            FixtureTensor("expected", reference.residual_add(residual_input, residual)),
        ],
    )

    argmax_input = torch.tensor(
        [[1.0, 3.5, -2.0, 0.0], [5.0, 4.9, 5.1, -1.0], [-7.0, -3.0, -4.0, -2.5]],
        dtype=torch.float32,
    )
    write_case(
        output_dir,
        "argmax_basic.bin",
        [
            FixtureTensor("input", argmax_input),
            FixtureTensor("expected", argmax_input.argmax(dim=-1).to(torch.int32)),
        ],
    )

    seq_len = 3
    head_dim = 4
    num_q_heads = 2
    num_kv_heads = 1
    q_flat = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            [0.0, 1.0, -2.0, 2.0, 1.5, -0.5, 0.25, 3.0],
            [-1.0, 4.0, 0.5, -2.5, 2.0, 1.0, -1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    k_flat = torch.tensor(
        [
            [2.0, -1.0, 0.5, 1.5],
            [0.0, 3.0, -2.0, 4.0],
            [1.25, -0.75, 2.5, -3.5],
        ],
        dtype=torch.float32,
    )
    cos_half, sin_half = runtime_rope_tables(seq_len, head_dim)
    cos_full = torch.cat([cos_half, cos_half], dim=-1)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)
    q_ref = q_flat.view(seq_len, num_q_heads, head_dim).transpose(0, 1)
    k_ref = k_flat.view(seq_len, num_kv_heads, head_dim).transpose(0, 1)
    q_rot, k_rot = reference.apply_rope(q_ref, k_ref, cos_full, sin_full)
    expected_q = q_rot.transpose(0, 1).contiguous().view(seq_len, num_q_heads * head_dim)
    expected_k = k_rot.transpose(0, 1).contiguous().view(seq_len, num_kv_heads * head_dim)
    write_case(
        output_dir,
        "rope_synthetic_small.bin",
        [
            FixtureTensor("q", q_flat),
            FixtureTensor("k", k_flat),
            FixtureTensor("cos", cos_half),
            FixtureTensor("sin", sin_half),
            FixtureTensor("num_q_heads", int_scalar(num_q_heads)),
            FixtureTensor("num_kv_heads", int_scalar(num_kv_heads)),
            FixtureTensor("head_dim", int_scalar(head_dim)),
            FixtureTensor("position_offset", int_scalar(0)),
            FixtureTensor("expected_q", expected_q),
            FixtureTensor("expected_k", expected_k),
        ],
    )


def generate_real_cases(output_dir: str) -> None:
    embed_dump = WeightDump(os.path.join(DATA_DIR, "embed_tokens.bin"))
    lm_head_dump = WeightDump(os.path.join(DATA_DIR, "lm_head.bin"))
    norm_dump = WeightDump(os.path.join(DATA_DIR, "norm.bin"))
    final_norm_weight = norm_dump.load_tensor("model.norm.weight")

    for seq_len in FIXED_SEQ_LENS:
        token_ids = deterministic_token_ids(seq_len)
        x = embed_tokens(token_ids, embed_dump)
        cos_full, sin_full = reference.precompute_rope_tables(reference.H_D, seq_len)
        cos_half, sin_half = runtime_rope_tables(seq_len, reference.H_D)

        for layer_idx in range(reference.N_LAYERS):
            layer_weights = load_layer_weights(layer_idx)
            layer_input = x.clone()

            x_norm = reference.rmsnorm(layer_input, layer_weights["input_layernorm.weight"])
            q_flat, k_flat, v_flat = reference.qkv_projections(
                x_norm,
                layer_weights["self_attn.q_proj.weight"],
                layer_weights["self_attn.k_proj.weight"],
                layer_weights["self_attn.v_proj.weight"],
            )

            q_heads, k_heads, v_heads = flat_heads_to_reference_qkv(q_flat, k_flat, v_flat)
            q_rot, k_rot = reference.apply_rope(q_heads, k_heads, cos_full, sin_full)
            q_rot_flat, k_rot_flat = reference_qk_to_flat(q_rot, k_rot)
            attention_output = reference.grouped_query_attention(q_rot, k_rot, v_heads)
            attn_projected = reference.attention_output_proj(
                attention_output, layer_weights["self_attn.o_proj.weight"]
            )
            attn_residual = reference.residual_add(layer_input, attn_projected)
            ffn_input = reference.rmsnorm(
                attn_residual, layer_weights["post_attention_layernorm.weight"]
            )
            ffn_output = reference.swiglu_ffn(
                ffn_input,
                layer_weights["mlp.gate_proj.weight"],
                layer_weights["mlp.up_proj.weight"],
                layer_weights["mlp.down_proj.weight"],
            )
            block_output = reference.residual_add(attn_residual, ffn_output)

            if layer_idx in FIXED_LAYERS:
                suffix = f"layer{layer_idx:02d}_seq{seq_len:03d}.bin"
                base_tensors = [
                    FixtureTensor("layer_index", int_scalar(layer_idx)),
                    FixtureTensor("seq_len", int_scalar(seq_len)),
                    FixtureTensor("position_offset", int_scalar(0)),
                ]

                write_case(
                    output_dir,
                    f"rmsnorm_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("input", float_tensor(layer_input)),
                        FixtureTensor("expected", float_tensor(x_norm)),
                    ],
                )
                write_case(
                    output_dir,
                    f"qkv_projection_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("input", float_tensor(x_norm)),
                        FixtureTensor("expected_q", float_tensor(q_flat)),
                        FixtureTensor("expected_k", float_tensor(k_flat)),
                        FixtureTensor("expected_v", float_tensor(v_flat)),
                    ],
                )
                write_case(
                    output_dir,
                    f"rope_real_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("q", float_tensor(q_flat)),
                        FixtureTensor("k", float_tensor(k_flat)),
                        FixtureTensor("cos", float_tensor(cos_half)),
                        FixtureTensor("sin", float_tensor(sin_half)),
                        FixtureTensor("num_q_heads", int_scalar(reference.H)),
                        FixtureTensor("num_kv_heads", int_scalar(reference.H_K)),
                        FixtureTensor("head_dim", int_scalar(reference.H_D)),
                        FixtureTensor("expected_q", float_tensor(q_rot_flat)),
                        FixtureTensor("expected_k", float_tensor(k_rot_flat)),
                    ],
                )
                write_case(
                    output_dir,
                    f"attention_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("q", float_tensor(q_rot_flat)),
                        FixtureTensor("k", float_tensor(k_rot_flat)),
                        FixtureTensor("v", float_tensor(v_flat)),
                        FixtureTensor("num_q_heads", int_scalar(reference.H)),
                        FixtureTensor("num_kv_heads", int_scalar(reference.H_K)),
                        FixtureTensor("head_dim", int_scalar(reference.H_D)),
                        FixtureTensor("expected", float_tensor(attention_output)),
                    ],
                )
                write_case(
                    output_dir,
                    f"swiglu_ffn_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("input", float_tensor(ffn_input)),
                        FixtureTensor("expected", float_tensor(ffn_output)),
                    ],
                )
                write_case(
                    output_dir,
                    f"decoder_block_{suffix}",
                    base_tensors
                    + [
                        FixtureTensor("input", float_tensor(layer_input)),
                        FixtureTensor("expected", float_tensor(block_output)),
                    ],
                )

            x = block_output

        logits = output_logits_chunked(x, final_norm_weight, lm_head_dump)
        next_token = int(torch.argmax(logits).item())
        seq_suffix = f"seq{seq_len:03d}.bin"
        write_case(
            output_dir,
            f"output_layer_{seq_suffix}",
            [
                FixtureTensor("seq_len", int_scalar(seq_len)),
                FixtureTensor("input", float_tensor(x)),
                FixtureTensor("expected_logits", float_tensor(logits)),
            ],
        )
        write_case(
            output_dir,
            f"forward_one_step_{seq_suffix}",
            [
                FixtureTensor("seq_len", int_scalar(seq_len)),
                FixtureTensor("token_ids", torch.tensor(token_ids, dtype=torch.int32)),
                FixtureTensor("expected_logits", float_tensor(logits)),
                FixtureTensor("expected_next_token", int_scalar(next_token)),
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reference fixtures for C++/CUDA tests.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reference fixture .bin files will be written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        generate_synthetic_cases(args.output_dir)
        generate_real_cases(args.output_dir)


if __name__ == "__main__":
    main()
