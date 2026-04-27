import numpy as np
import faiss
import time


def solve(base_vectors, query_vectors, k, K, time_budget):
    t0 = time.time()
    DEADLINE = t0 + time_budget * 0.92

    N, d = base_vectors.shape
    Q = query_vectors.shape[0]

    faiss.omp_set_num_threads(8)

    base_f32 = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)

    def time_left():
        return DEADLINE - time.time()

    def rank_counts(counts):
        # stable sort: ties broken by lower index (spec requirement)
        return np.argsort(-counts, kind="stable")[:K]

    counts = np.zeros(N, dtype=np.int32)

    # ── Calibrate actual machine BLAS throughput (~0.001s overhead) ───────────
    probe_q = min(50, Q)
    probe_n = min(10_000, N)
    mini = faiss.IndexFlatL2(d)
    mini.add(base_f32[:probe_n])
    t1 = time.time()
    mini.search(query_f32[:probe_q], k)
    blas_flops = 2.0 * probe_q * probe_n * d / max(time.time() - t1, 1e-6)

    # ── Exact search when affordable ──────────────────────────────────────────
    exact_cost_est = 2.0 * Q * N * d / blas_flops
    if exact_cost_est < time_budget * 0.65:
        index = faiss.IndexFlatL2(d)
        index.add(base_f32)
        batch = min(Q, 8192)
        for i in range(0, Q, batch):
            if time_left() <= 0:
                break
            end = min(i + batch, Q)
            _, inds = index.search(query_f32[i:end], k)
            flat = inds.ravel()
            counts += np.bincount(flat[flat >= 0], minlength=N).astype(np.int32)
        return rank_counts(counts)

    # ── IVF approximate search ─────────────────────────────────────────────────
    # index.add costs ≈ 2*N*nlist*d FLOPs; budget 15% of time for all building
    nlist_budget = int(0.15 * time_budget * blas_flops / (2.0 * N * d))
    nlist = max(64, min(8192, min(int(4 * np.sqrt(N)), nlist_budget)))

    train_n = min(max(40 * nlist, 50_000), N)
    rng = np.random.default_rng(42)
    if train_n < N:
        train_data = np.ascontiguousarray(
            base_f32[rng.choice(N, train_n, replace=False)]
        )
    else:
        train_data = base_f32

    # IVFPQ for large N: ~8x faster per probe → more nprobe coverage per second
    USE_PQ = N > 2_000_000
    if USE_PQ:
        M = max(4, d // 8)
        while d % M != 0 and M > 1:
            M -= 1
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, 8)
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    index.train(train_data)
    index.add(base_f32)

    if time_left() <= 0:
        return rank_counts(counts)

    # ── Two-point nprobe calibration ───────────────────────────────────────────
    bench_q = min(200, Q)
    np_lo, np_hi = 4, min(32, nlist)

    index.nprobe = np_lo
    t1 = time.time()
    index.search(query_f32[:bench_q], k)
    t_lo = (time.time() - t1) / bench_q

    if time_left() > 0 and np_hi > np_lo:
        index.nprobe = np_hi
        t1 = time.time()
        index.search(query_f32[:bench_q], k)
        t_hi = (time.time() - t1) / bench_q
        b = max((t_hi - t_lo) / (np_hi - np_lo), 1e-9)
        a = max(t_lo - np_lo * b, 0.0)
    else:
        a, b = 0.0, max(t_lo / max(np_lo, 1), 1e-9)

    remaining = time_left() * 0.85
    if remaining <= 0:
        return rank_counts(counts)

    nprobe = int((remaining / max(Q, 1) - a) / b)
    nprobe = max(1, min(nlist, nprobe))
    index.nprobe = nprobe

    # ── Main search ────────────────────────────────────────────────────────────
    batch = 8192
    for i in range(0, Q, batch):
        if time_left() <= 0:
            break
        end = min(i + batch, Q)
        _, inds = index.search(query_f32[i:end], k)
        flat = inds.ravel()
        counts += np.bincount(flat[flat >= 0], minlength=N).astype(np.int32)

    # ── Ensemble: second IVFFlat if ≥15% budget remains ───────────────────────
    if (not USE_PQ) and time_left() > time_budget * 0.15:
        q2 = faiss.IndexFlatL2(d)
        idx2 = faiss.IndexIVFFlat(q2, d, nlist, faiss.METRIC_L2)
        train2 = np.ascontiguousarray(base_f32[rng.choice(N, train_n, replace=False)])
        idx2.train(train2)
        idx2.add(base_f32)
        idx2.nprobe = nprobe
        for i in range(0, Q, batch):
            if time_left() <= 0:
                break
            end = min(i + batch, Q)
            _, inds2 = idx2.search(query_f32[i:end], k)
            flat2 = inds2.ravel()
            counts += np.bincount(flat2[flat2 >= 0], minlength=N).astype(np.int32)

    return rank_counts(counts)
