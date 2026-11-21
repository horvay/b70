// Qwen3.5 / Qwen3.8 prompt kernels in ESIMD (256 registers, libglm_moe_pf.so): flash attention over the selected
// sequence's cache (kernels/g4_fa.h at head dim 256, one thread per query head and 8 rows) and the chunked delta rule
// (kernels/q35_gdn.h).
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../kernels/g4_fa.h"
#include "../kernels/q35_dec.h"
#include "../kernels/q35_fa.h"
#include "../kernels/q35_gdn.h"
#include "q35_ops.h"

namespace glm::q35ops {

using namespace q35;

namespace {
template <class K, int WGS = 8>
void launch(sycl::queue& q, size_t n, const K& k) {
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>((n + WGS - 1) / WGS * WGS, WGS), props, k);
    });
}
}  // namespace

void attn_prompt(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, int p0, half* o, int variant,
                 const half* kt, int TS, const half* vt) {
    constexpr int QR = 8, G = AH / AKV;
    const int nqt = (rows + QR - 1) / QR;
    auto go = [&](auto kern) {
        kern.kt = kt;
        kern.vt = vt;
        kern.TS = TS;
        launch(q, (size_t)nqt * AKV * G, kern);
    };
    if (variant == 13 || variant == 15) {
        // TS sets the copies' row pitch, which a 2D load wants a multiple of 16 bytes: TS % 16 (a pitch of 1023 keys read
        // garbage)
        GLM_CHECK(kt && TS >= p0 + rows && TS % 16 == 0 && (variant == 13 || vt), "attn_prompt %d: needs the transposed "
                  "copies (TS %d)", variant, TS);
        if (variant == 13) go(::q35fa::Q35Fa<true, false>{qn16, Kc, Vc, o, rows, p0 + rows, AH, AKV, G, p0, nqt});
        else go(::q35fa::Q35Fa<true, true>{qn16, Kc, Vc, o, rows, p0 + rows, AH, AKV, G, p0, nqt});
    } else if (variant == 5) {
        go(::q35fa::Q35Fa<false, false>{qn16, Kc, Vc, o, rows, p0 + rows, AH, AKV, G, p0, nqt});
    } else {
        // kernels/g4_fa.h as Gemma 4 runs it: one thread a head and 8 rows, 64-key blocks
        constexpr float SM = 1.4426950408889634f * 0.0625f;      // log2 e / sqrt(256)
        ::g4fa::FaKernel<AD, QR, false, false, 4, SM> kern{qn16, Kc, Vc, o, nullptr, nullptr, rows, p0 + rows, AH, AKV, G,
                                                           p0, 0, 0, nqt};
        launch(q, (size_t)nqt * AKV * G, kern);
    }
}

void attn_dec(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, const int* pdev, const int* sdev,
              int TS, float* part, int ns) {
    const size_t nm = (size_t)rows * AH * ns;
    ::q35dec::Dec kern{qn16, Kc, Vc, part, part + nm, part + 2 * nm, pdev, sdev, rows, ns, TS};
    launch(q, (size_t)rows * AKV * ns, kern);
}

void attn_bench(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, int p0, half* o, int dbg) {
    constexpr int QR = 8, G = AH / AKV;
    const int nqt = (rows + QR - 1) / QR;
    auto go = [&](auto kern) { launch(q, (size_t)nqt * AKV * G, kern); };
#define Q35_BENCH(D_) case D_: go(::q35fa::Q35Fa<false, false, D_>{qn16, Kc, Vc, o, rows, p0 + rows, AH, AKV, G, p0, nqt}); break;
    switch (dbg) {
        Q35_BENCH(1) Q35_BENCH(2) Q35_BENCH(3) Q35_BENCH(4) Q35_BENCH(8) Q35_BENCH(16)
        default: attn_prompt(q, qn16, Kc, Vc, rows, p0, o, 5);
    }
#undef Q35_BENCH
}

void gdn_prep(sycl::queue& q, const half* qh, const half* kh, const float* beta, const float* g, float* A, half* Mq,
              float* bt, float* gam, float* wl, int n, int npad, int nc) {
    ::q35gdn::GdnPrep kern{qh, kh, beta, g, A, Mq, bt, gam, wl, n, npad, nc};
    launch(q, (size_t)nc * q4::GK * 8, kern);           // a work-group: the 8 row blocks of a chunk and k-head
}

void gdn_scan(sycl::queue& q, const bf16* cv, const half* qh, const half* kh, const float* A, const half* Mq,
              const float* bt, const float* gam, const float* wl, float* rec, bf16* core, int n, int npad, int nc) {
    // a work-group: 8 threads of one v-head (8 of its 16 column blocks), which share each chunk's A through SLM
    ::q35gdn::GdnScan kern{reinterpret_cast<const uint16_t*>(cv), qh, kh, A, Mq, bt, gam, wl, rec,
                           reinterpret_cast<uint16_t*>(core), n, npad, nc};
    launch(q, (size_t)q4::GV * (q4::GDV / ::q35gdn::VB), kern);
}

}  // namespace glm::q35ops
