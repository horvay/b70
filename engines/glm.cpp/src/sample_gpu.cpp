#include "sample_gpu.h"
#include <cmath>
#include <cstring>

namespace glm {

using sycl::nd_item;
using sycl::nd_range;

GpuPool::GpuPool(int V, int qmax) : V_(V), qmax_(qmax) {
    auto& q = Dev::get().q;
    ids_d_ = Tensor::empty({qmax, CAP}, DType::I32);
    raw_d_ = Tensor::empty({qmax, CAP}, DType::F16);
    meta_d_ = Tensor::empty({qmax, 4}, DType::I32);
    rows_d_ = Tensor::empty({(int64_t)(qmax * sizeof(PoolRow))}, DType::U8);
    ids_h_ = sycl::malloc_host<int>((size_t)qmax * CAP, q);
    raw_h_ = sycl::malloc_host<half>((size_t)qmax * CAP, q);
    meta_h_ = sycl::malloc_host<int>((size_t)qmax * 4, q);
    rows_h_ = sycl::malloc_host<PoolRow>(qmax, q);
    GLM_CHECK(ids_h_ && raw_h_ && meta_h_ && rows_h_, "sampling pool: pinned buffers");
}

GpuPool::~GpuPool() {
    auto& q = Dev::get().q;
    q.wait();
    for (void* p : {(void*)ids_h_, (void*)raw_h_, (void*)meta_h_, (void*)rows_h_, (void*)pen_h_}) if (p) sycl::free(p, q);
}

float GpuPool::rest_sum(int r) const { return sycl::bit_cast<float>(meta_h_[r * 4 + 2]); }
float GpuPool::gmax(int r) const { return sycl::bit_cast<float>(meta_h_[r * 4 + 3]); }

void GpuPool::run(sycl::queue& q, const half* logits, int Q, const std::vector<PoolRow>& rows,
                  const std::vector<PenEntry>& pen) {
    GLM_CHECK(Q >= 1 && Q <= qmax_ && (int)rows.size() >= Q, "sampling pool: %d rows", Q);
    const int V = V_;
    std::memcpy(rows_h_, rows.data(), (size_t)Q * sizeof(PoolRow));
    q.memcpy(rows_d_.p, rows_h_, (size_t)Q * sizeof(PoolRow));
    const PoolRow* rd = static_cast<const PoolRow*>(rows_d_.p);
    const float* scp = nullptr;
    if (!pen.empty()) {
        // penalized rows: their logits as fp32 with the penalties applied (the selection's view; the host redoes the
        // penalties exactly on the pool)
        const int np = (int)pen.size();
        if (np > pen_cap_) {
            if (pen_h_) { q.wait(); sycl::free(pen_h_, q); }
            pen_cap_ = std::max(np, 2 * pen_cap_);
            pen_h_ = sycl::malloc_host<PenEntry>(pen_cap_, q);
            pen_d_ = Tensor::empty({(int64_t)(pen_cap_ * sizeof(PenEntry))}, DType::U8);
            GLM_CHECK(pen_h_, "sampling pool: pinned penalty list");
        }
        std::memcpy(pen_h_, pen.data(), (size_t)np * sizeof(PenEntry));
        q.memcpy(pen_d_.p, pen_h_, (size_t)np * sizeof(PenEntry));
        if (!scratch_.p) scratch_ = Tensor::empty({qmax_, V}, DType::F32);
        float* sc = scratch_.ptr<float>();
        const PenEntry* pe = static_cast<const PenEntry*>(pen_d_.p);
        int maxn = 0;
        for (int r = 0; r < Q; r++) maxn = std::max(maxn, rows[r].pen_n);
        q.parallel_for(sycl::range<2>(Q, V), [=](sycl::item<2> it) {
            const int r = it[0], v = it[1];
            if (rd[r].pen_n > 0) sc[(size_t)r * V + v] = (float)logits[(size_t)r * V + v];
        });
        q.parallel_for(sycl::range<2>(Q, maxn), [=](sycl::item<2> it) {
            const int r = it[0], i = it[1];
            const PoolRow pr = rd[r];
            if (i >= pr.pen_n) return;
            const PenEntry e = pe[pr.pen_off + i];
            float& x = sc[(size_t)r * V + e.id];
            float y = x;
            if (pr.rep != 1.0f) y = y > 0 ? y / pr.rep : y * pr.rep;
            x = y - e.sub;
        });
        scp = sc;
    }
    int* ids = ids_d_.ptr<int>();
    half* raw = raw_d_.ptr<half>();
    int* meta = meta_d_.ptr<int>();
    constexpr int WG = 1024, NBk = NB, CAPk = CAP;
    static_assert(NB == WG, "a bin per work-item");
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> hist(sycl::range<1>(NBk), h);
        sycl::local_accessor<int, 1> cnt(sycl::range<1>(1), h);
        h.parallel_for(nd_range<1>((size_t)Q * WG, WG), [=](nd_item<1> it) {
            const int r = it.get_group(0), l = it.get_local_id(0);
            auto grp = it.get_group();
            const PoolRow pr = rd[r];
            const half* lr = logits + (size_t)r * V;
            const float* sr = scp && pr.pen_n > 0 ? scp + (size_t)r * V : nullptr;
            auto val = [&](int v) { return sr ? sr[v] : (float)lr[v]; };
            float m = -INFINITY;
            for (int v = l; v < V; v += WG) m = sycl::fmax(m, val(v));
            const float M = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
            hist[l] = 0;
            if (l == 0) cnt[0] = 0;
            sycl::group_barrier(grp);
            const float W = pr.window, lo = M - W, scale = NBk / W;
            auto bin = [&](float x) { return sycl::min(NBk - 1, (int)((M - x) * scale)); };
            for (int v = l; v < V; v += WG) {
                const float x = val(v);
                if (x >= lo) sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                              sycl::access::address_space::local_space>(hist[bin(x)]) += 1;
            }
            sycl::group_barrier(grp);
            // the last bin whose count from the top stays within CAP (-1: the top bin alone is more)
            const int cum = sycl::inclusive_scan_over_group(grp, hist[l], sycl::plus<int>());
            const int bstar = sycl::reduce_over_group(grp, cum <= CAPk ? l : -1, sycl::maximum<int>());
            int rest_n = 0;
            float rs = 0, rc = 0;                         // Kahan sum of the left-out window's mass above the floor
            if (bstar >= 0) {
                for (int v = l; v < V; v += WG) {
                    const float x = val(v);
                    if (x < lo) continue;
                    if (bin(x) <= bstar) {
                        const int slot = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                                          sycl::access::address_space::local_space>(cnt[0]).fetch_add(1);
                        ids[(size_t)r * CAPk + slot] = v;
                        raw[(size_t)r * CAPk + slot] = lr[v];
                    } else {
                        rest_n++;
                        const float xs = (x - M) * pr.inv_t;
                        if (pr.inv_t > 0 && xs >= pr.log_floor) {
                            const float y = sycl::exp(xs) - rc, t = rs + y;
                            rc = (t - rs) - y;
                            rs = t;
                        }
                    }
                }
            }
            rest_n = sycl::reduce_over_group(grp, rest_n, sycl::plus<int>());
            rs = sycl::reduce_over_group(grp, rs, sycl::plus<float>());
            sycl::group_barrier(grp);
            if (l == 0) {
                meta[r * 4] = bstar < 0 ? -1 : cnt[0];
                meta[r * 4 + 1] = rest_n;
                meta[r * 4 + 2] = sycl::bit_cast<int>(rs);
                meta[r * 4 + 3] = sycl::bit_cast<int>(M);
            }
        });
    });
    q.memcpy(meta_h_, meta, (size_t)Q * 4 * sizeof(int));
    q.memcpy(ids_h_, ids, (size_t)Q * CAP * sizeof(int));
    q.memcpy(raw_h_, raw, (size_t)Q * CAP * sizeof(half));
    q.wait();
}

}  // namespace glm
