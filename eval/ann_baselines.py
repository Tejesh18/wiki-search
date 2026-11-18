"""My HNSW next to hnswlib and FAISS on the same vectors and settings.

Codespace CPUs are shared, so speed drifts by 30%+ over minutes. To keep the
comparison fair, the three run in rounds with the order rotated each round,
and each number is the median over rounds. Only one index is in memory at a
time (all three at once doesn't fit in 16 GB). Queries run one at a time on
one thread.

    python eval/ann_baselines.py [rounds]
"""

import subprocess
import sys
import time

import numpy as np

M, EF_BUILD, DIM = 16, 200, 384
EFS = (16, 32, 64, 128, 256)


def load():
    docs = np.fromfile("data/emb/corpus.f32", dtype=np.float32).reshape(-1, DIM)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = np.fromfile("data/emb/queries.f32", dtype=np.float32).reshape(-1, DIM)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return docs, q, np.load("data/emb/truth.npy")[:, :10]


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])


def timed(search, q, ef):
    times, found = [], []
    for v in q:
        t = time.perf_counter()
        found.append(search(v, ef))
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return found, times[len(times) // 2], times[int(len(times) * 0.99)]


def mine(ef):
    # the Rust binary prints "N queries  p50 X ms  p99 Y ms" on its last line
    out = subprocess.run(
        ["./target/release/wiki-search", "hnsw-run", "data/emb/m16.hnsw", "data/emb/corpus.f32", "data/index/ids.txt",
         "data/emb/queries.f32", "data/emb/qids.txt", "/tmp/ann.tsv", str(ef), "10"],
        capture_output=True, text=True, check=True).stderr.strip().splitlines()[-1].split()
    return float(out[3]), float(out[6])


HL_PATH, FI_PATH = "data/emb/hnswlib.bin", "data/emb/faiss.index"


def build(docs):
    """Build each library's index once and save it, so rounds can load one at a time."""
    import os

    import faiss
    import hnswlib

    if not os.path.exists(HL_PATH):
        hl = hnswlib.Index(space="ip", dim=DIM)
        hl.init_index(max_elements=len(docs), M=M, ef_construction=EF_BUILD)
        t = time.time()
        hl.add_items(docs, num_threads=4)
        print(f"hnswlib built in {time.time() - t:.0f}s", flush=True)
        hl.save_index(HL_PATH)
        del hl
    if not os.path.exists(FI_PATH):
        faiss.omp_set_num_threads(4)
        fi = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
        fi.hnsw.efConstruction = EF_BUILD
        t = time.time()
        fi.add(docs)
        print(f"faiss built in {time.time() - t:.0f}s", flush=True)
        faiss.write_index(fi, FI_PATH)
        del fi


def run_lib(name, q, truth):
    """Load one library's index, run every ef, free it."""
    out = {}
    if name == "hnswlib":
        import hnswlib

        idx = hnswlib.Index(space="ip", dim=DIM)
        idx.load_index(HL_PATH)
        idx.set_num_threads(1)

        def search(v, ef):
            idx.set_ef(ef)
            return idx.knn_query(v, k=10)[0][0]
    else:
        import faiss

        idx = faiss.read_index(FI_PATH)
        faiss.omp_set_num_threads(1)

        def search(v, ef):
            idx.hnsw.efSearch = ef
            return idx.search(v[None], 10)[1][0]
    for ef in EFS:
        found, p50, p99 = timed(search, q, ef)
        out[ef] = (recall(found, truth), p50, p99)
    return out


def run_mine(q, truth):
    ids = open("data/index/ids.txt").read().splitlines()
    pos = {d: i for i, d in enumerate(ids)}
    qids = open("data/emb/qids.txt").read().split()
    out = {}
    for ef in EFS:
        p50, p99 = mine(ef)
        got = {}
        for line in open("/tmp/ann.tsv"):
            qid, doc, _, _ = line.split()
            got.setdefault(qid, []).append(pos[doc])
        out[ef] = (recall([got[x] for x in qids], truth), p50, p99)
    return out


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    docs, q, truth = load()
    build(docs)
    del docs
    names = ["mine", "hnswlib", "faiss"]
    results = {}
    for r in range(rounds):
        order = names[r % 3:] + names[:r % 3]  # rotate who goes first
        for name in order:
            res = run_mine(q, truth) if name == "mine" else run_lib(name, q, truth)
            for ef, v in res.items():
                results.setdefault((name, ef), []).append(v)
        print(f"round {r + 1} done ({' -> '.join(order)})", flush=True)

    print()
    print(f"{'':8}{'ef':>5}{'recall@10':>11}{'p50 ms':>9}{'p99 ms':>9}   (median of {rounds} rounds)")
    for name in names:
        for ef in EFS:
            rs = np.array(results[(name, ef)])
            print(f"{name:8}{ef:5d}{rs[0, 0]:11.3f}{np.median(rs[:, 1]):9.2f}{np.median(rs[:, 2]):9.2f}")


if __name__ == "__main__":
    main()
