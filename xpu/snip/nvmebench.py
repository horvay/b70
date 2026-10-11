# NVMe read throughput of the tier's reader pool for different piece counts (random records, into the bounce ring).
import time, random, concurrent.futures as cf
random.seed(1)
N = tier.n_bounce
for pieces, threads in ((8, 16), (16, 16), (16, 32), (4, 32), (32, 32)):
    tier.n_pieces = pieces
    if hasattr(tier, "_piece_list"):
        del tier._piece_list
    old = tier.pool
    tier.pool = cf.ThreadPoolExecutor(threads)
    keys = random.sample(range(tier.L * tier.E), N)
    t0 = time.perf_counter()
    fs = [f for j, k in enumerate(keys) for f in tier._submit(k, tier.bounce_np[j])]
    for f in fs:
        f.result()
    dt = time.perf_counter() - t0
    # single-record latency (what a lone miss costs)
    t1 = time.perf_counter()
    for f in tier._submit(random.randrange(tier.L * tier.E), tier.bounce_np[0]):
        f.result()
    lat = time.perf_counter() - t1
    print(f"pieces {pieces:2d} threads {threads}: {N} records {N * tier.rec / dt / 1e9:.2f} GB/s; one record {lat * 1e3:.1f} ms")
    tier.pool.shutdown(); tier.pool = old
tier.n_pieces = 8; del tier._piece_list
