import array
import os
import struct
from dataclasses import dataclass

import torch


MAGIC = b"RFX\x01"
DTYPE_INT32 = 1
DTYPE_FLOAT32 = 2


@dataclass
class FixtureTensor:
    name: str
    tensor: torch.Tensor


def _tensor_payload(tensor: torch.Tensor) -> tuple[int, bytes]:
    if tensor.dtype == torch.int32:
        values = array.array("i", tensor.reshape(-1).tolist())
        return DTYPE_INT32, values.tobytes()
    if tensor.dtype == torch.float32:
        values = array.array("f", tensor.reshape(-1).tolist())
        return DTYPE_FLOAT32, values.tobytes()
    raise ValueError(f"Unsupported fixture tensor dtype: {tensor.dtype}")


def write_fixture(path: str, tensors: list[FixtureTensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    prepared: list[tuple[str, torch.Tensor, int, bytes]] = []
    for entry in tensors:
        tensor = entry.tensor.detach().cpu().contiguous()
        dtype, payload = _tensor_payload(tensor)
        prepared.append((entry.name, tensor, dtype, payload))

    with open(path, "wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", len(prepared)))

        for name, tensor, dtype, _payload in prepared:
            name_bytes = name.encode("utf-8")
            out.write(struct.pack("<I", len(name_bytes)))
            out.write(name_bytes)
            out.write(struct.pack("<I", dtype))
            out.write(struct.pack("<I", tensor.ndim))
            for dim in tensor.shape:
                out.write(struct.pack("<I", dim))

        for _name, _tensor, _dtype, payload in prepared:
            out.write(payload)
