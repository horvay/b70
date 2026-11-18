// Launchers of Ripple's attention kernels (kernels/g4_fa.h, kernels/rp_dec.h); 256-register ESIMD, so in the PF library.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../kernels/rp_dec.h"
#include "../kernels/g4_fa.h"
#include "rp_ops.h"

namespace glm::rpfa {

namespace {
template <class K, int WGS = 8>
void launch(sycl::queue& q, size_t n, const K& k) {
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>((n + WGS - 1) / WGS * WGS, WGS), props, k);
    });
}
}  // namespace

void prompt(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase, int win, half* o) {
    constexpr int D = rp::HD, QR = 8, Hk = rp::NKV, G = rp::NH / Hk, BT = 4;
    int nqt = (Lq + QR - 1) / QR;
    ::g4fa::FaKernel<D, QR, false, false, BT, ::rpdec::SCORE> kern{qb, k, v, o, nullptr, nullptr, Lq, Lk, rp::NH, Hk, G, qbase, kbase, win, nqt};
    launch(q, (size_t)nqt * Hk * G, kern);
}

void dec_sliding(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                 float* pm, float* pl, float* po, int ns, int blocks) {
    ::rpdec::Dec<true> kern{qb, K, V, pm, pl, po, pdev, sdev, p0, R, rp::RING, rp::WIN, ns, blocks};
    launch(q, (size_t)R * rp::NKV * ns, kern);
}

void dec_global(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                int TS, float* pm, float* pl, float* po, int ns) {
    ::rpdec::Dec<false> kern{qb, K, V, pm, pl, po, pdev, sdev, p0, R, TS, 0, ns, 0};
    launch(q, (size_t)R * rp::NKV * ns, kern);
}

void dec_draft(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int span, int win,
               float* pm, float* pl, float* po, int ns, int blocks) {
    ::rpdec::Dec<true> kern{qb, K, V, pm, pl, po, pdev, sdev, 0, R, span, win, ns, blocks, R - 1};
    launch(q, (size_t)R * rp::NKV * ns, kern);
}

}  // namespace glm::rpfa
