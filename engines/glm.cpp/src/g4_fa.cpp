// Launchers of the prompt attention kernels (kernels/g4_fa.h); ESIMD, so built into libglm_esimd.so.
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../kernels/g4_dec.h"
#include "../kernels/g4_fa.h"
#include "g4_ops.h"

namespace glm::g4fa {

namespace {
using ::g4fa::FaKernel;

template <class K, int WGS = 8>
void launch(sycl::queue& q, size_t n, const K& k) {
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>((n + WGS - 1) / WGS * WGS, WGS), props, k);
    });
}
}  // namespace

void sliding(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase,
             int win, half* o) {
    constexpr int D = g4::SD, QR = 8, Hk = g4::SKV, G = g4::NH / Hk;
    int nqt = (Lq + QR - 1) / QR;
    FaKernel<D, QR, false, false, 4> kern{qb, k, v, o, nullptr, nullptr, Lq, Lk, g4::NH, Hk, G, qbase, kbase, win, nqt};
    launch(q, (size_t)nqt * Hk * G, kern);
}

void full_block(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase,
                float* o32, float* lse) {
    constexpr int D = g4::FD, QR = 8, Hk = g4::FKV, G = g4::NH / Hk;
    int nqt = (Lq + QR - 1) / QR;
    FaKernel<D, QR, true, true, 2> kern{qb, k, v, nullptr, o32, lse, Lq, Lk, g4::NH, Hk, G, qbase, kbase, 0, nqt};
    launch(q, (size_t)nqt * Hk * G * 2, kern);       // 2 threads a head; a work-group of 8 holds 4 heads of one query tile
}

void sliding_dec(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                 float* pm, float* pl, float* po, int ns, int blocks) {
    ::g4dec::SlidingDec kern{qb, K, V, pm, pl, po, pdev, sdev, p0, R, g4::RING, g4::WIN, ns, blocks};
    launch(q, (size_t)R * g4::SKV * ns, kern);
}

void full_dec(sycl::queue& q, const half* qb, const half* Pc, const half* Kr, const float* Rc, const half* kn, int R,
              const int* pdev, const int* sdev, int TS, int p0, float* pm, float* pl, float* po, int ns) {
    ::g4dec::FullDec kern{qb, Pc, Kr, Rc, kn, pm, pl, po, pdev, sdev, p0, R, ns, TS};
    launch<::g4dec::FullDec, 2>(q, (size_t)R * g4::FKV * ns * 2, kern);    // a work-group = a pair of threads
}

namespace {
template <int QR, class TO, bool CAP>
void dense_q(sycl::queue& q, const half* x, int xs, const half* w, int K, int N, TO* y) {
    ::g4dec::DenseKernel<QR, TO, CAP> kern{x, w, y, K, N, xs};
    launch(q, (size_t)N / 16, kern);
}
template <class TO, bool CAP>
void dense_rows(sycl::queue& q, const half* x, int Q, int xs, const half* w, int K, int N, TO* y) {
    switch (Q) {
        case 1: dense_q<1, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 2: dense_q<2, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 3: dense_q<3, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 4: dense_q<4, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 5: dense_q<5, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 6: dense_q<6, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 7: dense_q<7, TO, CAP>(q, x, xs, w, K, N, y); break;
        case 8: dense_q<8, TO, CAP>(q, x, xs, w, K, N, y); break;
        default: GLM_CHECK(false, "dense: %d rows", Q);
    }
}
}  // namespace

void dense16(sycl::queue& q, const half* x, int Q, int xs, const half* w, int K, int N, half* y, bool softcap) {
    if (softcap) dense_rows<half, true>(q, x, Q, xs, w, K, N, y);
    else dense_rows<half, false>(q, x, Q, xs, w, K, N, y);
}
void dense32(sycl::queue& q, const half* x, int Q, int xs, const half* w, int K, int N, float* y) {
    dense_rows<float, false>(q, x, Q, xs, w, K, N, y);
}

}  // namespace glm::g4fa
