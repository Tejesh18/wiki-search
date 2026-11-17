"""nDCG@10 and recall@100 for a run file against BEIR qrels.

    python eval/score.py data/nq/qrels/test.tsv runs/bm25.tsv
"""

import csv
import math
import sys
from collections import defaultdict


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        for qid, doc, rel in r:
            if int(rel) > 0:
                qrels[qid][doc] = int(rel)
    return qrels


def load_run(path):
    run = defaultdict(list)
    with open(path) as f:
        for qid, doc, rank, score in csv.reader(f, delimiter="\t"):
            run[qid].append((int(rank), doc))
    return {q: [d for _, d in sorted(v)] for q, v in run.items()}


def ndcg(ranked, rels, k=10):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def main():
    qrels, run = load_qrels(sys.argv[1]), load_run(sys.argv[2])
    n = r100 = 0.0
    for qid, rels in qrels.items():
        ranked = run.get(qid, [])
        n += ndcg(ranked, rels)
        r100 += len(set(ranked[:100]) & set(rels)) / len(rels)
    print(f"{len(qrels)} queries  nDCG@10 {n / len(qrels):.3f}  recall@100 {r100 / len(qrels):.3f}")


if __name__ == "__main__":
    main()
