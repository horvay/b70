// MoE routing alignment: group the (token, pick) pairs by expert. An ordinary SYCL kernel, kept apart from the ESIMD
// files (ahead-of-time ESIMD code generation drops ordinary kernels' names). From llm/glm/xpu/csrc/exl3_moe_align.sycl.
#include "rt.h"

namespace glm {

void moe_align(sycl::queue& q, const int* ids, const float* w, int npairs, int topk, int E, int pad, int* act_expert,
               int* act_start, int* act_cnt, int* n_act, int* act_pstart, int* pair_tok, float* pair_w, int* pair_of,
               int* blk_a, int* n_blk) {
    constexpr int WG = 1024;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> cnt(sycl::range<1>(E), h), start(sycl::range<1>(E), h), fill(sycl::range<1>(E), h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            int l = it.get_local_id(0);
            auto g = it.get_group();
            using lref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                          sycl::access::address_space::local_space>;
            for (int e = l; e < E; e += WG) { cnt[e] = 0; fill[e] = 0; }
            sycl::group_barrier(g);
            for (int i = l; i < npairs; i += WG) lref(cnt[ids[i]]).fetch_add(1);
            sycl::group_barrier(g);
            int c = l < E ? cnt[l] : 0;
            int st = sycl::exclusive_scan_over_group(g, c, sycl::plus<int>());
            int act = c > 0 ? 1 : 0;
            int ai = sycl::exclusive_scan_over_group(g, act, sycl::plus<int>());
            int nact = sycl::reduce_over_group(g, act, sycl::plus<int>());
            int pc = (c + pad - 1) / pad * pad;
            int pst = sycl::exclusive_scan_over_group(g, pc, sycl::plus<int>());
            if (l < E) {
                start[l] = st;
                if (act) { act_expert[ai] = l; act_start[ai] = st; act_cnt[ai] = c; act_pstart[ai] = pst; }
                if (act && blk_a)
                    for (int j = 0; j < pc / pad; ++j) blk_a[pst / pad + j] = ai;
            }
            int ptot = sycl::reduce_over_group(g, pc, sycl::plus<int>());
            if (l == 0) { n_act[0] = nact; if (n_blk) n_blk[0] = ptot / pad; }
            sycl::group_barrier(g);
            for (int i = l; i < npairs; i += WG) {
                int e = ids[i];
                int pos = start[e] + lref(fill[e]).fetch_add(1);
                pair_tok[pos] = i / topk;
                pair_w[pos] = w[i];
                pair_of[i] = pos;
            }
        });
    });
}

}  // namespace glm
