"""Temp-only synthetic encryption timing probe. No replay, bench suite or API calls.

Run with `.venv/bin/python tools/measure_encryption_overhead.py`. Keys exist only in
the fake Keychain's memory. Timings exclude native Keychain prompt/lookup latency.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from discrete import encryption as enc
from discrete.models import ClaimDraft
from discrete.store import Store


class MemoryKeychain:
    def __init__(self) -> None:
        self.keys: dict[str, bytes] = {}

    def get(self, key_id: str) -> bytes:
        return self.keys[key_id]

    def put(self, key_id: str, key: bytes) -> None:
        self.keys[key_id] = key


@dataclass
class Timings:
    insert_us: float
    fts_us: float
    dense_warm_us: float
    dense_cold_ms: float
    frame_read_us: float
    frame_decode_us: float


async def measure(root: Path, *, encrypted: bool, rows: int = 3000) -> Timings:
    os.environ["DISCRETE_ENCRYPT_AT_REST"] = str(encrypted).lower()
    path = root / "store.db"
    store = await Store.open(str(path))
    try:
        start = time.perf_counter()
        for i in range(rows):
            await store.create_claim(ClaimDraft(kind="fact", text=f"Synthetic violet notebook subject{i % 50} item {i}"))
        insert_us = (time.perf_counter() - start) * 1e6 / rows
        vec = np.random.default_rng(7).normal(size=(rows, 128)).astype(np.float32)
        await store.save_embeddings([("claim", i + 1, vector, f"hash{i}") for i, vector in enumerate(vec)])
        start = time.perf_counter()
        for i in range(200):
            await store.fts_ids("claim", f"subject{i % 50}", limit=20)
        fts_us = (time.perf_counter() - start) * 1e6 / 200
        start = time.perf_counter()
        await store.dense_search("claim", vec[0], limit=20)
        dense_cold_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        for _ in range(200):
            await store.dense_search("claim", vec[0], limit=20)
        dense_warm_us = (time.perf_counter() - start) * 1e6 / 200
        pixels = np.random.default_rng(3).integers(0, 256, (900, 1600, 3), dtype=np.uint8)
        data = io.BytesIO()
        Image.fromarray(pixels).save(data, "JPEG", quality=65)
        plain = data.getvalue()
        state = enc.read_state(path)
        frame = root / "frame.jpg"
        enc.atomic_write(frame, enc.encrypt_frame(plain, state.key_id) if state else plain)
        start = time.perf_counter()
        for _ in range(200):
            enc.read_frame(frame)
        frame_read_us = (time.perf_counter() - start) * 1e6 / 200
        start = time.perf_counter()
        for _ in range(100):
            with enc.open_frame(frame) as image:
                image.load()
        frame_decode_us = (time.perf_counter() - start) * 1e6 / 100
        return Timings(insert_us, fts_us, dense_warm_us, dense_cold_ms, frame_read_us, frame_decode_us)
    finally:
        await store.close()


async def main() -> None:
    enc._keychain = MemoryKeychain()
    os.environ["EMBED_LM_MODEL_NAME"] = "synthetic"
    samples: dict[str, list[Timings]] = {"plaintext": [], "encrypted": []}
    with tempfile.TemporaryDirectory(prefix="discrete-encryption-timing-") as folder:
        os.environ["DISCRETE_DATA_DIR"] = folder
        os.environ.pop("DISCRETE_DB", None)
        # Alternate order to reduce CPU/disk cache warmup bias.
        for repeat in range(5):
            for encrypted in ([False, True] if repeat % 2 == 0 else [True, False]):
                name = "encrypted" if encrypted else "plaintext"
                root = Path(folder) / f"{name}-{repeat}"
                root.mkdir()
                samples[name].append(await measure(root, encrypted=encrypted))
    result = {name: {metric: round(statistics.median(asdict(sample)[metric] for sample in timings), 3)
                     for metric in asdict(timings[0])} for name, timings in samples.items()}
    result["overhead_percent"] = {metric: round(100 * (result["encrypted"][metric] / value - 1), 1)
                                  for metric, value in result["plaintext"].items()}
    print(json.dumps({"rows": 3000, "repetitions": 5, "frame_pixels": "1600x900", "timings": result}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
