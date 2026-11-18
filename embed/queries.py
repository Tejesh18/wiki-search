"""Embed the benchmark queries. bge expects this instruction in front of
queries (not passages).

    python embed/queries.py data/nq/queries.jsonl data/emb
"""

import json
import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from embed import MODEL

PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    qs = [json.loads(l) for l in open(sys.argv[1])]
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(MODEL)
    vecs = np.stack(list(model.embed([PREFIX + q["text"] for q in qs], batch_size=64))).astype(np.float32)
    vecs.tofile(out / "queries.f32")
    (out / "qids.txt").write_text("\n".join(q["_id"] for q in qs) + "\n")
    print(vecs.shape)


if __name__ == "__main__":
    main()
