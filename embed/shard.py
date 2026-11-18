"""Embed one 100k-passage slice of the corpus. The embed workflow runs 27 of
these at once on separate runners; ids are saved next to the vectors so the
parts can be checked against the index's doc order when they're put together.

    python embed/shard.py corpus.parquet <part> <out dir>
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastembed import TextEmbedding

from embed import MODEL, PART


def main():
    path, part, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(path, columns=["_id", "title", "text"]).slice(part * PART, PART)
    ids = table.column("_id").to_pylist()
    texts = [f"{t or ''} {x}".strip() for t, x in zip(table.column("title").to_pylist(), table.column("text").to_pylist())]
    if not texts:
        return
    # sorting by length keeps padding down inside each batch
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    model = TextEmbedding(MODEL, threads=os.cpu_count())
    t = time.time()
    vecs = np.zeros((len(texts), 384), dtype=np.float16)
    for i, v in zip(order, model.embed([texts[i] for i in order], batch_size=64)):
        vecs[i] = v
    np.save(out / f"part-{part:04d}.npy", vecs)
    (out / f"part-{part:04d}.ids.txt").write_text("\n".join(ids) + "\n")
    print(f"part {part}: {len(texts)} passages in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
