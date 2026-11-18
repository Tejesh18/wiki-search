"""Put the 27 downloaded parts back together into one float32 file, checking
the ids line up with the index's doc order, and compute exact nearest
neighbours (the ground truth HNSW is measured against).

    gh run download <run id> -D data/parts
    python embed/assemble.py data/parts data/index/ids.txt data/emb
"""

import sys
from pathlib import Path

import numpy as np


def main():
    parts, ids_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    files = sorted(parts.glob("part-*/part-*.npy"))
    want = ids_path.read_text().split("\n")[:-1]
    got = []
    with open(out / "corpus.f32", "wb") as f:
        for p in files:
            got += p.with_suffix(".ids.txt").read_text().split("\n")[:-1]
            np.load(p).astype(np.float32).tofile(f)
    assert got == want, "parts don't match the index's doc order"
    print(f"{len(files)} parts, {len(got)} vectors")

    # exact top 100 per query by brute force, in chunks to bound memory
    docs = np.fromfile(out / "corpus.f32", dtype=np.float32).reshape(-1, 384)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.fromfile(out / "queries.f32", dtype=np.float32).reshape(-1, 384)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    best_s = np.full((len(q), 100), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), 100), dtype=np.int64)
    for start in range(0, len(docs), 200_000):
        s = q @ docs[start:start + 200_000].T
        top = np.argpartition(-s, 100, axis=1)[:, :100]
        cs = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1)
        ci = np.concatenate([best_i, top + start], 1)
        keep = np.argsort(-cs, axis=1)[:, :100]
        best_s, best_i = np.take_along_axis(cs, keep, 1), np.take_along_axis(ci, keep, 1)
    np.save(out / "truth.npy", best_i)
    qids = (out / "qids.txt").read_text().split("\n")[:-1]
    with open(out / "exact.tsv", "w") as f:
        for qi, qid in enumerate(qids):
            for r in range(100):
                f.write(f"{qid}\t{want[best_i[qi, r]]}\t{r + 1}\t{best_s[qi, r]:.4f}\n")
    print("exact neighbours written")


if __name__ == "__main__":
    main()
