"""recall@10 of an approximate run against the exact neighbours.

    python eval/recall.py data/emb/exact.tsv runs/hnsw-ef64.tsv
"""

import sys
from collections import defaultdict


def top10(path):
    out = defaultdict(list)
    for line in open(path):
        qid, doc, rank, _ = line.rstrip("\n").split("\t")
        if int(rank) <= 10:
            out[qid].append(doc)
    return out


exact, approx = top10(sys.argv[1]), top10(sys.argv[2])
print(f"recall@10 {sum(len(set(exact[q]) & set(approx[q])) / 10 for q in exact) / len(exact):.3f}")
