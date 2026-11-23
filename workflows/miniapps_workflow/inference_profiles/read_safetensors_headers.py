"""Read tensor names, dtypes and shapes from safetensors weight-file headers.

Only each file's header is read: an 8-byte length followed by a JSON object. The
weights themselves are never loaded, so this runs in seconds on a login node and
needs nothing beyond the standard library. The output is the
``safetensors-headers.json`` input of ``inference_profile.build_profile``.

    python read_safetensors_headers.py MODEL_DIR > safetensors-headers.json
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


def read_headers(model_dir: str | Path) -> dict[str, Any]:
    """Return every tensor's dtype, shape and file, plus per-file header hashes."""
    result: dict[str, Any] = {"files": {}, "tensors": {}}
    for path in sorted(Path(model_dir).glob("model-*.safetensors")):
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            raw = handle.read(length)
        header = json.loads(raw)
        result["files"][path.name] = {
            "bytes": path.stat().st_size,
            "header_bytes": length,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata": header.get("__metadata__"),
        }
        for name, info in header.items():
            if name != "__metadata__":
                result["tensors"][name] = {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "file": path.name,
                }
    return result


if __name__ == "__main__":
    print(json.dumps(read_headers(sys.argv[1])))
