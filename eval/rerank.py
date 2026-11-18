"""Rerank the top of a run with a cross-encoder (ms-marco MiniLM-L6).

The first stage scores query and passage separately (BM25 terms, or two
embeddings). A cross-encoder reads them together, so it can tell "who
founded X" from "who did X found", but it costs a model call per pair, so
it only sees the first stage's top N. On a 4-core CPU that's ~10 pairs a
second, so the rerank workflow splits the queries across runners.

    python eval/rerank.py <run.tsv> <out.tsv> --corpus corpus.parquet --queries queries.parquet [--top 50] [--shard 0 --of 6]
"""

import argparse
import time
from collections import defaultdict

import pyarrow.parquet as pq
from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()

    run = defaultdict(list)
    for line in open(a.run):
        qid, doc, rank, _ = line.split()
        if int(rank) <= a.top:
            run[qid].append(doc)
    qids = sorted(run)[a.shard::a.of]
    need = {d for q in qids for d in run[q]}

    t = pq.read_table(a.corpus, columns=["_id", "title", "text"])
    text = {i: f"{ti or ''} {x}".strip()
            for i, ti, x in zip(t.column("_id").to_pylist(), t.column("title").to_pylist(), t.column("text").to_pylist())
            if i in need}
    t = pq.read_table(a.queries, columns=["_id", "text"])
    queries = dict(zip(t.column("_id").to_pylist(), t.column("text").to_pylist()))

    model = TextCrossEncoder(MODEL)
    start = time.time()
    with open(a.out, "w") as f:
        for i, qid in enumerate(qids):
            docs = run[qid]
            scores = list(model.rerank(queries[qid], [text[d] for d in docs], batch_size=64))
            for r, (doc, s) in enumerate(sorted(zip(docs, scores), key=lambda x: -x[1]), 1):
                f.write(f"{qid}\t{doc}\t{r}\t{s:.4f}\n")
            if i % 100 == 0:
                print(f"{i}/{len(qids)} queries, {(time.time() - start) / (i + 1):.1f} s each", flush=True)
    print(f"{len(qids)} queries reranked in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
