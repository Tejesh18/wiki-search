"""Embed every passage with bge-small (384 dims) on CPU.

Writes data/emb/part-NNNN.npy (float16, 100k passages each). Resumable:
parts that already exist are skipped, since the whole corpus takes hours.

    python embed/embed.py data/nq/corpus.jsonl data/emb [--limit N]
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

PART = 100_000
MODEL = "BAAI/bge-small-en-v1.5"


def passages(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            yield f"{d.get('title', '')} {d['text']}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(MODEL, threads=os.cpu_count())

    docs = passages(args.corpus)
    if args.limit:
        docs = itertools.islice(docs, args.limit)
    for part in itertools.count():
        chunk = list(itertools.islice(docs, PART))
        if not chunk:
            break
        path = out / f"part-{part:04d}.npy"
        if path.exists():
            continue
        t = time.time()
        vecs = np.stack(list(model.embed(chunk, batch_size=128))).astype(np.float16)
        np.save(path.with_suffix(".tmp.npy"), vecs)
        os.replace(path.with_suffix(".tmp.npy"), path)
        print(f"{path.name}: {len(chunk)} in {time.time() - t:.0f}s ({len(chunk) / (time.time() - t):.0f}/s)", flush=True)


if __name__ == "__main__":
    main()
