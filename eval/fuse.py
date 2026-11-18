"""Reciprocal rank fusion of several runs (Cormack et al. 2009).

Each document gets sum(1 / (60 + rank)) over the runs it appears in. Only
ranks are used, so BM25 scores (unbounded) and cosine similarities (0-1)
don't need to be put on the same scale, and there's nothing to tune on the
test queries.

    python eval/fuse.py runs/bm25.tsv runs/dense.tsv > runs/hybrid.tsv
"""

import sys
from collections import defaultdict

K = 60


def main():
    fused = defaultdict(lambda: defaultdict(float))
    for path in sys.argv[1:]:
        for line in open(path):
            qid, doc, rank, _ = line.split()
            fused[qid][doc] += 1 / (K + int(rank))
    for qid, docs in fused.items():
        ranked = sorted(docs.items(), key=lambda x: -x[1])[:100]
        for r, (doc, s) in enumerate(ranked, 1):
            print(f"{qid}\t{doc}\t{r}\t{s:.6f}")


if __name__ == "__main__":
    main()
