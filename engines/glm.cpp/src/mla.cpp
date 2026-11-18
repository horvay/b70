#include "mla.h"
#include "ops.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <oneapi/mkl/blas.hpp>

namespace glm {

namespace {
template <class T> float f(T v) { return static_cast<float>(v); }
using sycl::nd_item;
using sycl::nd_range;
namespace blas = oneapi::mkl::blas::row_major;
constexpr auto NT = oneapi::mkl::transpose::nontrans, TR = oneapi::mkl::transpose::trans;

// fp16 bit pattern -> uint16 that sorts like the value (NaN aside)
inline uint32_t okey(half h) {
    uint16_t b = sycl::bit_cast<uint16_t>(h);
    return (b & 0x8000) ? (uint16_t)~b : (uint16_t)(b | 0x8000);
}
}  // namespace

MlaLayer MlaLayer::load(const SafeTensors& st, const std::string& p) {
    MlaLayer L;
    L.q_a = Exl3Linear::load(st, p + ".q_a_proj");
    L.q_b = Exl3Linear::load(st, p + ".q_b_proj");
    L.kv_a = Exl3Linear::load(st, p + ".kv_a_proj_with_mqa");
    L.o = Exl3Linear::load(st, p + ".o_proj");
    L.wq_b = Exl3Linear::load(st, p + ".indexer.wq_b");
    L.q_a_norm = st.load(p + ".q_a_layernorm.weight");
    L.kv_a_norm = st.load(p + ".kv_a_layernorm.weight");
    L.kv_b = st.load(p + ".kv_b_proj.weight");
    L.wk = st.load(p + ".indexer.wk.weight");
    L.k_norm_w = st.load(p + ".indexer.k_norm.weight");
    L.k_norm_b = st.load(p + ".indexer.k_norm.bias");
    L.weights_proj = st.load(p + ".indexer.weights_proj.weight");
    L.gate = st.load(p + ".indexer.index_kpool_compress_gate");
    L.ape = st.load(p + ".indexer.index_kpool_compress_ape");
    GLM_CHECK(L.kv_a.out == DC && L.kv_b.dt == DType::F16 && L.kv_b.size(0) == H * 2 * DN && L.ape.dt == DType::F32,
              "%s: unexpected MLA tensors", p.c_str());
    return L;
}

void MlaCache::init(int T) {
    max_tokens = T;
    latent = Tensor::zeros({T, MlaLayer::DC}, DType::F16);
    kidx = Tensor::zeros({T, 2 * MlaLayer::DI}, DType::F16);
    kpool = Tensor::zeros({T / MlaLayer::POOL + 1, MlaLayer::DI}, DType::F16);
}

void mla_forward(sycl::queue& q, Workspace& lin, Workspace& act, const MlaLayer& L, MlaCache& c, const half* x, int Q,
                 int pos0, const int* pos_dev, float* out) {
    constexpr int H = MlaLayer::H, DC = MlaLayer::DC, DN = MlaLayer::DN, DV = MlaLayer::DV, QR = MlaLayer::QR;
    constexpr int HI = MlaLayer::HI, DI = MlaLayer::DI, P = MlaLayer::POOL, KSEL = MlaLayer::TOPK / P;
    constexpr int NSEL = KSEL * P + P - 1;     // 2051 token slots per row
    GLM_CHECK(pos0 + Q <= c.max_tokens, "mla: position %d past the cache (%d)", pos0 + Q, c.max_tokens);
    act.reset();
    half* qa = act.take<half>((size_t)Q * QR);
    half* qres = act.take<half>((size_t)Q * QR);
    half* qh = act.take<half>((size_t)Q * H * DN);
    half* kva = act.take<half>((size_t)Q * DC);
    half* qlat = act.take<half>((size_t)H * Q * DC);
    half* kk = act.take<half>((size_t)Q * DI);
    half* kg = act.take<half>((size_t)Q * DI);
    half* qidx = act.take<half>((size_t)Q * HI * DI);
    half* wts = act.take<half>((size_t)Q * HI);
    const int npool_max = c.max_tokens / P + 1;      // every pool slot of the cache (masked past the context)
    half* sc = act.take<half>((size_t)Q * npool_max);
    int* sel = act.take<int>((size_t)Q * NSEL);
    half* olat = act.take<half>((size_t)H * Q * DC);
    half* o16 = act.take<half>((size_t)Q * H * DV);
    // prompts (more than 16 rows): absorb, indexer scores, attention and unfold as oneMKL fp16 GEMMs on the XMX units,
    // same math and roundings as the per-row kernels below, summed in another order. GLM_MLA_GEMM=0: per-row kernels
    static const bool gemm_on = !getenv("GLM_MLA_GEMM") || atoi(getenv("GLM_MLA_GEMM")) != 0;
    const bool gemm = gemm_on && Q > 16;

    prof_point("mla: (entry)");
    // 1-2. queries, latent
    L.q_a.forward(q, lin, x, Q, qa, DType::F16);
    ops::rms_norm<half, half>(q, qa, L.q_a_norm.ptr<bf16>(), qres, Q, QR, 1e-5f);
    L.q_b.forward(q, lin, qres, Q, qh, DType::F16);
    L.kv_a.forward(q, lin, x, Q, kva, DType::F16);
    half* lat = c.latent.ptr<half>();
    half* ckv = act.take<half>((size_t)Q * DC);
    ops::rms_norm<half, half>(q, kva, L.kv_a_norm.ptr<bf16>(), ckv, Q, DC, 1e-5f);
    q.parallel_for(sycl::range<2>(Q, DC), [=](sycl::item<2> it) {                          // 4. append the latent
        lat[(size_t)(*pos_dev + it[0]) * DC + it[1]] = ckv[(size_t)it[0] * DC + it[1]];
    });

    prof_point("mla: q_a, q_b, kv_a, norms");
    // 3. absorb: q_lat[h, r, c] = sum_j q[r, h, j] * W_UK[h, j, c]  (fp32 acc, fp16 out)
    const half* kvb = L.kv_b.ptr<half>();
    if (gemm)   // per head: q_lat[h] [Q, 512] = q[:, h, :] [Q, 256] @ W_UK[h] [256, 512]
        blas::gemm_batch(q, NT, NT, Q, DC, DN, half(1.0f), qh, H * DN, DN, kvb, DC, 2 * DN * DC, half(0.0f), qlat, DC,
                         (int64_t)Q * DC, H);
    else
    q.parallel_for(sycl::range<3>(H, Q, DC), [=](sycl::item<3> it) {
        int h = it[0], r = it[1], cc = it[2];
        const half* qr = qh + ((size_t)r * H + h) * DN;
        const half* w = kvb + (size_t)h * 2 * DN * DC + cc;
        float acc = 0;
        for (int j = 0; j < DN; j++) acc += f(qr[j]) * f(w[(size_t)j * DC]);
        qlat[((size_t)h * Q + r) * DC + cc] = half(acc);
    });

    prof_point("mla: absorb");
    // 5. indexer key (biased LayerNorm, eps 1e-6) and pool gate (a projection of x), into k_idx rows
    ops::gemv_f16<half>(q, x, Q, 4096, L.wk.ptr<half>(), DI, kk);
    ops::gemv_f16<half>(q, x, Q, 4096, L.gate.ptr<half>(), DI, kg);
    half* kidx = c.kidx.ptr<half>();
    const half* lnw = L.k_norm_w.ptr<half>();
    const half* lnb = L.k_norm_b.ptr<half>();
    q.parallel_for(nd_range<1>((size_t)Q * DI, DI), [=](nd_item<1> it) {
        int r = it.get_group(0), d = it.get_local_id(0);
        float v = f(kk[(size_t)r * DI + d]);
        float mean = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / DI;
        float dv = v - mean;
        float var = sycl::reduce_over_group(it.get_group(), dv * dv, sycl::plus<float>()) / DI;
        float y = dv / sycl::sqrt(var + 1e-6f) * f(lnw[d]) + f(lnb[d]);
        half* row = kidx + (size_t)(*pos_dev + r) * 2 * DI;
        row[d] = half(y);
        row[DI + d] = kg[(size_t)r * DI + d];
    });

    prof_point("mla: indexer key + gate");
    // 6. pooled key of each row's pool: per dim, softmax over the 4 members of (gate + ape), weights on the keys
    half* kpool = c.kpool.ptr<half>();
    const float* ape = L.ape.ptr<float>();
    q.parallel_for(sycl::range<2>(Q, DI), [=](sycl::item<2> it) {
        int r = it[0], d = it[1];
        int pool = (*pos_dev + r) / P;
        float g[P], mx = -INFINITY;
        for (int m = 0; m < P; m++) {
            g[m] = f(kidx[(size_t)(pool * P + m) * 2 * DI + DI + d]) + ape[m * DI + d];
            mx = sycl::fmax(mx, g[m]);
        }
        float se = 0, acc = 0;
        for (int m = 0; m < P; m++) { g[m] = sycl::exp(g[m] - mx); se += g[m]; }
        for (int m = 0; m < P; m++) acc += (g[m] / se) * f(kidx[(size_t)(pool * P + m) * 2 * DI + d]);
        kpool[(size_t)pool * DI + d] = half(acc);
    });

    prof_point("mla: pool key");
    // 7. scores over the visible (complete) pools: sum_h (w_h / sqrt(32)) * relu(q_h . k_pool / sqrt(128)), fp16
    L.wq_b.forward(q, lin, qres, Q, qidx, DType::F16);
    ops::gemv_f16<half>(q, x, Q, 4096, L.weights_proj.ptr<half>(), HI, wts);
    const size_t scratch = act.off;     // the GEMM path's temporaries, reused by each step (in-order queue)
    if (gemm) {
        // per block of rows: dots [rows x 32 heads, pools] = q_idx @ k_pool^T (fp32), then per row the weighted sum of
        // relu(dot / sqrt(128)) over the heads, in head order as below; pools past a row's visible ones: -inf
        const int np_all = (pos0 + Q) / P;
        const int RBI = (int)std::clamp<size_t>(((size_t)96 << 20) / ((size_t)HI * std::max(np_all, 1) * 4), 1, Q);
        float* dots = act.take<float>((size_t)RBI * HI * std::max(np_all, 1));
        for (int r0 = 0; r0 < Q; r0 += RBI) {
            const int nr = std::min(RBI, Q - r0), np = (pos0 + r0 + nr) / P;
            if (np == 0) continue;     // no complete pool yet: the selection reads none
            blas::gemm(q, NT, TR, (int64_t)nr * HI, np, DI, 1.0f, qidx + (size_t)r0 * HI * DI, DI, kpool, DI, 0.0f,
                       dots, np);
            q.parallel_for(sycl::range<2>(nr, np), [=](sycl::item<2> it) {
                int rr = it[0], j = it[1], r = r0 + rr;
                float s = -INFINITY;
                if (j < (pos0 + r + 1) / P) {
                    s = 0;
                    for (int h = 0; h < HI; h++) {
                        float w = f(wts[(size_t)r * HI + h]) * (1.0f / sycl::sqrt((float)HI));
                        s += w * sycl::fmax(dots[((size_t)rr * HI + h) * np + j] * (1.0f / sycl::sqrt((float)DI)), 0.0f);
                    }
                }
                sc[(size_t)r * npool_max + j] = half(s);
            });
        }
        act.off = scratch;
    } else
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HI * DI + HI), hd);
        constexpr int WG = 256;
        size_t groups = (npool_max + WG - 1) / WG;
        hd.parallel_for(nd_range<2>({(size_t)Q, groups * WG}, {1, WG}), [=](nd_item<2> it) {
            int r = it.get_global_id(0), j = it.get_global_id(1), t = it.get_local_id(1);
            for (int i = t; i < HI * DI; i += WG) qs[i] = f(qidx[(size_t)r * HI * DI + i]);
            for (int i = t; i < HI; i += WG) qs[HI * DI + i] = f(wts[(size_t)r * HI + i]) * (1.0f / sycl::sqrt((float)HI));
            sycl::group_barrier(it.get_group());
            if (j >= npool_max) return;
            int vis = (*pos_dev + r + 1) / P;
            float s = -INFINITY;
            if (j < vis) {
                float kp[DI];
                for (int d = 0; d < DI; d++) kp[d] = f(kpool[(size_t)j * DI + d]);
                s = 0;
                for (int h = 0; h < HI; h++) {
                    float dot = 0;
                    for (int d = 0; d < DI; d++) dot += qs[h * DI + d] * kp[d];
                    s += qs[HI * DI + h] * sycl::fmax(dot * (1.0f / sycl::sqrt((float)DI)), 0.0f);
                }
            }
            sc[(size_t)r * npool_max + j] = half(s);
        });
    });

    prof_point("mla: wq_b, weights, scores");
    // 8. selection: the top 512 pools (ties to the lower index), expanded to tokens, plus the incomplete tail pool
    q.submit([&](sycl::handler& hd) {
        constexpr int WG = 1024;
        sycl::local_accessor<int, 1> hist(sycl::range<1>(256), hd);
        sycl::local_accessor<int, 1> misc(sycl::range<1>(8), hd);
        hd.parallel_for(nd_range<1>((size_t)Q * WG, WG), [=](nd_item<1> it) {
            int r = it.get_group(0), t = it.get_local_id(0);
            auto grp = it.get_group();
            const half* s = sc + (size_t)r * npool_max;
            int vis = (*pos_dev + r + 1) / P;
            int* out = sel + (size_t)r * NSEL;
            for (int i = t; i < NSEL; i += WG) out[i] = -1;
            sycl::group_barrier(grp);
            // threshold key T (16 bits, two radix passes): fewer than KSEL keys above T, at least KSEL at or above
            uint32_t prefix = 0;
            int need = KSEL;
            if (vis > KSEL) {
                for (int pass = 0; pass < 2; pass++) {
                    for (int i = t; i < 256; i += WG) hist[i] = 0;
                    sycl::group_barrier(grp);
                    for (int j = t; j < vis; j += WG) {
                        uint32_t k = okey(s[j]);
                        if (pass == 0 || (k >> 8) == prefix)
                            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>(hist[pass == 0 ? k >> 8 : k & 255])
                                .fetch_add(1);
                    }
                    sycl::group_barrier(grp);
                    if (t == 0) {
                        int acc = 0, b = 255;
                        for (; b > 0; b--) {
                            if (acc + hist[b] >= need) break;
                            acc += hist[b];
                        }
                        misc[0] = b;
                        misc[1] = need - acc;     // still needed from bin b
                    }
                    sycl::group_barrier(grp);
                    prefix = pass == 0 ? (uint32_t)misc[0] : (prefix << 8) | (uint32_t)misc[0];
                    need = misc[1];
                    sycl::group_barrier(grp);
                }
            }
            // write: all keys above T, then the lowest-index keys equal to T (need of them); -inf pools never
            if (t == 0) {
                int n = 0;
                if (vis <= KSEL) {
                    for (int j = 0; j < vis; j++) {
                        if (okey(s[j]) == okey(half(-INFINITY))) continue;
                        for (int m = 0; m < P; m++) out[n * P + m] = j * P + m;
                        n++;
                    }
                } else {
                    uint32_t T = prefix;
                    int eq = 0;
                    for (int j = 0; j < vis && n < KSEL; j++) {
                        uint32_t k = okey(s[j]);
                        bool take = k > T || (k == T && eq < need);
                        if (k == T && take) eq++;
                        if (take && k != okey(half(-INFINITY))) {
                            for (int m = 0; m < P; m++) out[n * P + m] = j * P + m;
                            n++;
                        }
                    }
                }
                int pv = *pos_dev + r + 1, tc = pv % P;
                for (int i = 0; i < P - 1; i++) out[KSEL * P + i] = i < tc ? pv - tc + i : -1;
            }
        });
    });

    prof_point("mla: selection");
    // 9. gathered latent attention: s = fp16(q_lat . kv) * 0.0625, softmax fp32, p fp16, o = fp16(sum p kv).
    // The latent cache is one head shared by all 64 query heads, so each selected row is read once per query row
    // (not once per head): scores by token (a sub-group per token, all heads against its row), the softmax per head,
    // then the weighted sum by token chunk for all heads at once, chunk partials summed in order. Rows in blocks of RB.
    constexpr int RB = 16, CH = 256, NCH = (NSEL + CH - 1) / CH, HB = 32;
    if (gemm) {
        // per block of rows: the selected latent rows gathered [rows, 2051, 512] (zero where none), then per row
        // s [64 heads, 2051] = q_lat @ kv^T and o [64, 512] = p @ kv as batched GEMMs (fp32 accumulation)
        const size_t per_row = (size_t)NSEL * DC * 2 + (size_t)H * NSEL * 6 + (size_t)H * DC * 4 + 2048;
        const size_t avail = act.cap > act.off + ((size_t)8 << 20) ? act.cap - act.off - ((size_t)8 << 20) : 0;
        static const int ra_max = getenv("GLM_MLA_RA") ? std::max(1, atoi(getenv("GLM_MLA_RA"))) : 8;   // 8: the gathered rows (16 MB) stay in L2
        const int RA = (int)std::clamp<size_t>(avail / per_row, 1, ra_max);
        half* G = act.take<half>((size_t)RA * NSEL * DC);
        float* S = act.take<float>((size_t)RA * H * NSEL);
        half* Pr = act.take<half>((size_t)RA * H * NSEL);
        float* O = act.take<float>((size_t)RA * H * DC);
        for (int r0 = 0; r0 < Q; r0 += RA) {
            const int nr = std::min(RA, Q - r0);
            using V = sycl::vec<uint32_t, 4>;     // 8 halves
            constexpr int VR = DC / 8;
            q.parallel_for(sycl::range<1>((size_t)nr * NSEL * VR), [=](sycl::id<1> i) {
                size_t rj = i / VR;
                int v = i % VR;
                int tok = sel[(size_t)(r0 + rj / NSEL) * NSEL + rj % NSEL];
                V z(0u);
                reinterpret_cast<V*>(G)[i] = tok < 0 ? z : reinterpret_cast<const V*>(lat + (size_t)tok * DC)[v];
            });
            blas::gemm_batch(q, NT, TR, H, NSEL, DC, 1.0f, qlat + (size_t)r0 * DC, (int64_t)Q * DC, DC, G, DC,
                             (int64_t)NSEL * DC, 0.0f, S, NSEL, (int64_t)H * NSEL, nr);
            // softmax per (row, head) of fp16-rounded scores * 1/16, probabilities rounded to fp16 (none -> 0)
            q.parallel_for(nd_range<1>((size_t)nr * H * 256, 256), [=](nd_item<1> it) {
                size_t g = it.get_group(0);
                int t = it.get_local_id(0);
                const int* sr = sel + (size_t)(r0 + g / H) * NSEL;
                const float* s = S + g * NSEL;
                half* pr = Pr + g * NSEL;
                auto grp = it.get_group();
                auto sv = [&](int j) { return sr[j] < 0 ? -INFINITY : f(half(s[j])) * 0.0625f; };
                float mx = -INFINITY;
                for (int j = t; j < NSEL; j += 256) mx = sycl::fmax(mx, sv(j));
                mx = sycl::reduce_over_group(grp, mx, sycl::maximum<float>());
                float se = 0;
                for (int j = t; j < NSEL; j += 256) { float v = sv(j); se += v == -INFINITY ? 0.0f : sycl::exp(v - mx); }
                se = sycl::reduce_over_group(grp, se, sycl::plus<float>());
                for (int j = t; j < NSEL; j += 256) {
                    float v = sv(j);
                    pr[j] = half(v == -INFINITY ? 0.0f : sycl::exp(v - mx) / se);
                }
            });
            blas::gemm_batch(q, NT, NT, H, DC, NSEL, 1.0f, Pr, NSEL, (int64_t)H * NSEL, G, DC, (int64_t)NSEL * DC, 0.0f,
                             O, DC, (int64_t)H * DC, nr);
            q.parallel_for(sycl::range<3>(nr, H, DC), [=](sycl::item<3> it) {
                olat[((size_t)it[1] * Q + r0 + it[0]) * DC + it[2]] = half(O[((size_t)it[0] * H + it[1]) * DC + it[2]]);
            });
        }
        act.off = scratch;
    }
    float* scores = gemm ? nullptr : act.take<float>((size_t)RB * H * NSEL);
    half* probs = gemm ? nullptr : act.take<half>((size_t)RB * H * NSEL);
    float* part = gemm ? nullptr : act.take<float>((size_t)RB * NCH * H * DC);
    for (int r0 = 0; r0 < (gemm ? 0 : Q); r0 += RB) {
        const int nr = std::min(RB, Q - r0);
        // scores [r][h][j]: work-group = 16 tokens (one sub-group each) x HB heads (q_lat of those heads in SLM)
        q.submit([&](sycl::handler& hd) {
            constexpr int SGS = 16, TPW = 16;
            sycl::local_accessor<half, 1> qs(sycl::range<1>(HB * DC), hd);
            size_t groups = (size_t)nr * ((NSEL + TPW - 1) / TPW) * (H / HB);
            hd.parallel_for(nd_range<1>(groups * TPW * SGS, TPW * SGS), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SGS)]] {
                int g = it.get_group(0), t = it.get_local_id(0);
                int hb = g % (H / HB), rest = g / (H / HB);
                int jb = rest % ((NSEL + TPW - 1) / TPW), r = r0 + rest / ((NSEL + TPW - 1) / TPW);
                for (int i = t; i < HB * DC; i += TPW * SGS) qs[i] = qlat[((size_t)(hb * HB + i / DC) * Q + r) * DC + i % DC];
                sycl::group_barrier(it.get_group());
                auto sg = it.get_sub_group();
                int lane = sg.get_local_id()[0];
                int j = jb * TPW + sg.get_group_id()[0];
                if (j >= NSEL) return;
                int tok = sel[(size_t)r * NSEL + j];
                float* out = scores + (size_t)(r - r0) * H * NSEL;
                if (tok < 0) {
                    for (int h = lane; h < HB; h += SGS) out[(size_t)(hb * HB + h) * NSEL + j] = -INFINITY;
                    return;
                }
                const half* kv = lat + (size_t)tok * DC;
                float kr[DC / SGS];
                for (int i = 0; i < DC / SGS; i++) kr[i] = f(kv[i * SGS + lane]);
                for (int h = 0; h < HB; h++) {
                    float acc = 0;
                    const half* qh = &qs[h * DC];
                    for (int i = 0; i < DC / SGS; i++) acc += f(qh[i * SGS + lane]) * kr[i];
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) out[(size_t)(hb * HB + h) * NSEL + j] = f(half(acc)) * 0.0625f;
                }
            });
        });
        // softmax per (row, head), probabilities rounded to fp16 (-inf -> 0)
        q.parallel_for(nd_range<1>((size_t)nr * H * 256, 256), [=](nd_item<1> it) {
            size_t g = it.get_group(0);
            int t = it.get_local_id(0);
            const float* s = scores + g * NSEL;
            half* pr = probs + g * NSEL;
            auto grp = it.get_group();
            float mx = -INFINITY;
            for (int j = t; j < NSEL; j += 256) mx = sycl::fmax(mx, s[j]);
            mx = sycl::reduce_over_group(grp, mx, sycl::maximum<float>());
            float se = 0;
            for (int j = t; j < NSEL; j += 256) se += s[j] == -INFINITY ? 0.0f : sycl::exp(s[j] - mx);
            se = sycl::reduce_over_group(grp, se, sycl::plus<float>());
            for (int j = t; j < NSEL; j += 256)
                pr[j] = half(s[j] == -INFINITY ? 0.0f : sycl::exp(s[j] - mx) / se);
        });
        // weighted sum by token chunk: thread = latent dim, all 64 heads accumulated, chunk probabilities in SLM
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<half, 1> pc(sycl::range<1>(H * CH), hd);
            sycl::local_accessor<int, 1> ic(sycl::range<1>(CH), hd);
            hd.parallel_for(nd_range<1>((size_t)nr * NCH * DC, DC), [=](nd_item<1> it) {
                int g = it.get_group(0), cc = it.get_local_id(0);
                int ch = g % NCH, rr = g / NCH, r = r0 + rr;
                int j0 = ch * CH, n = std::min(CH, NSEL - j0);
                for (int i = cc; i < CH; i += DC) ic[i] = i < n ? sel[(size_t)r * NSEL + j0 + i] : -1;
                for (int i = cc; i < H * CH; i += DC) {
                    int h = i / CH, jj = i % CH;
                    pc[i] = jj < n ? probs[((size_t)rr * H + h) * NSEL + j0 + jj] : half(0.0f);
                }
                sycl::group_barrier(it.get_group());
                float acc[H];
                for (int h = 0; h < H; h++) acc[h] = 0;
                for (int jj = 0; jj < n; jj++) {
                    int tok = ic[jj];
                    if (tok < 0) continue;
                    float v = f(lat[(size_t)tok * DC + cc]);
                    for (int h = 0; h < H; h++) acc[h] += f(pc[h * CH + jj]) * v;
                }
                float* o = part + ((size_t)rr * NCH + ch) * H * DC;
                for (int h = 0; h < H; h++) o[(size_t)h * DC + cc] = acc[h];
            });
        });
        q.parallel_for(sycl::range<2>((size_t)nr * H, DC), [=](sycl::item<2> it) {
            int rr = it[0] / H, h = it[0] % H, cc = it[1];
            float acc = 0;
            for (int ch = 0; ch < NCH; ch++) acc += part[(((size_t)rr * NCH + ch) * H + h) * DC + cc];
            olat[((size_t)h * Q + r0 + rr) * DC + cc] = half(acc);
        });
    }

    prof_point("mla: attention");
    // 10. unfold: o[r, h, v] = sum_c o_lat[h, r, c] * W_UV[h, v, c]  (fp32 acc, fp16 out)
    if (gemm)   // per head: o[:, h, :] [Q, 256] = o_lat[h] [Q, 512] @ W_UV[h]^T (one call each: the heads' outputs
                // interleave, which gemm_batch refuses)
        for (int h = 0; h < H; h++)
            blas::gemm(q, NT, TR, Q, DV, DC, half(1.0f), olat + (size_t)h * Q * DC, DC,
                       kvb + ((size_t)h * 2 * DN + DN) * DC, DC, half(0.0f), o16 + (size_t)h * DV, H * DV);
    else
    q.parallel_for(nd_range<1>((size_t)Q * H * DV * 16, 128), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        size_t o = it.get_global_id(0) / 16;
        int l = sg.get_local_id()[0];
        int v = o % DV, h = (o / DV) % H, r = o / (DV * H);
        const half* w = kvb + ((size_t)h * 2 * DN + DN + v) * DC;
        const half* ol = olat + ((size_t)h * Q + r) * DC;
        float acc = 0;
        for (int cc = l; cc < DC; cc += 16) acc += f(ol[cc]) * f(w[cc]);
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (l == 0) o16[((size_t)r * H + h) * DV + v] = half(acc);
    });

    prof_point("mla: unfold");
    // 11. o_proj, fp32 out
    L.o.forward(q, lin, o16, Q, out, DType::F32);
    prof_point("mla: o_proj");
}

}  // namespace glm
