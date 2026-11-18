#include "kda.h"
#include "ops.h"

namespace glm {

namespace {
template <class T> float f(T v) { return static_cast<float>(v); }
}  // namespace

KdaLayer KdaLayer::load(const SafeTensors& st, const std::string& p) {
    KdaLayer L;
    L.qkv = Exl3Linear::load(st, p + ".qkv_proj");
    L.o = Exl3Linear::load(st, p + ".o_proj");
    L.conv_w = st.load(p + ".conv1d.weight");
    L.conv_w.shape = {F, CONV};
    L.b_proj = st.load(p + ".b_proj.weight");
    L.f_a = st.load(p + ".f_a_proj.weight");
    L.f_b = st.load(p + ".f_b_proj.weight");
    L.g_a = st.load(p + ".g_a_proj.weight");
    L.g_b = st.load(p + ".g_b_proj.weight");
    L.A_log = st.load(p + ".A_log");
    L.dt_bias = st.load(p + ".dt_bias");
    L.o_norm = st.load(p + ".o_norm.weight");
    GLM_CHECK(L.conv_w.dt == DType::BF16 && L.b_proj.dt == DType::F16 && L.A_log.dt == DType::F32 &&
              L.o_norm.dt == DType::BF16, "%s: unexpected KDA tensor dtypes", p.c_str());
    return L;
}

void KdaLayer::fuse() {
    if (lga || lb || lfa) return;
    w3 = Tensor::empty({320, 4096}, DType::F16);
    auto& q = Dev::get().q;
    q.memcpy(w3.ptr<half>(), g_a.p, g_a.bytes());
    q.memcpy(w3.ptr<half>() + 128 * 4096, b_proj.p, b_proj.bytes());
    q.memcpy(w3.ptr<half>() + 192 * 4096, f_a.p, f_a.bytes());
    q.wait();
}

void KdaState::init(int max_history) {
    hist = max_history;
    conv = Tensor::zeros({KdaLayer::F, KdaLayer::CONV + hist}, DType::BF16);
    rec = Tensor::zeros({hist + 1, KdaLayer::H, KdaLayer::DK, KdaLayer::DV}, DType::F32);
    last_history = 0;
}

void KdaState::zero() {
    auto& q = Dev::get().q;
    q.memset(conv.p, 0, conv.bytes());
    q.memset(rec.p, 0, rec.bytes());
    last_history = 0;
}

void kda_forward(sycl::queue& q, Workspace& ws, Workspace& own, const KdaLayer& L, KdaState& st, const half* y, int S,
                 float* out, bool history) {
    constexpr int H = KdaLayer::H, DK = KdaLayer::DK, DV = KdaLayer::DV, F = KdaLayer::F, CK = KdaLayer::CONV;
    GLM_CHECK(!history || S <= st.hist + 1, "kda: %d tokens with history %d", S, st.hist);
    own.reset();
    float* qkv = own.take<float>((size_t)S * F);
    half* ga = own.take<half>((size_t)S * 128);
    half* fa = own.take<half>((size_t)S * 128);
    float* z = own.take<float>((size_t)S * H * DV);
    float* fg = own.take<float>((size_t)S * H * DK);
    float* bt = own.take<float>((size_t)S * H);
    bf16* conv_out = own.take<bf16>((size_t)S * F);
    bf16* core = own.take<bf16>((size_t)S * H * DV);
    half* normed = own.take<half>((size_t)S * H * DV);

    prof_point("kda: (entry)");
    // 1. projections (fp16 linears: fp16-rounded, held in fp32; a LoRA'd f_a / g_a output: fp32 + delta, then fp16)
    L.qkv.forward(q, ws, y, S, qkv, DType::F32);
    auto low = [&](const Tensor& W, const Lora* l, half* out16) {
        if (!l) { ops::gemv_f16<half>(q, y, S, 4096, W.ptr<half>(), 128, out16); return; }
        float* t32 = own.take<float>((size_t)S * 128);
        ops::gemv_f16<float>(q, y, S, 4096, W.ptr<half>(), 128, t32, true);
        ws.reset();
        l->apply(q, ws, y, S, t32, DType::F32);
        ops::f32_to_f16(q, t32, out16, (size_t)S * 128);
    };
    auto lora = [&](const Lora* l, const half* in, float* out) {
        if (!l) return;
        ws.reset();
        l->apply(q, ws, in, S, out, DType::F32);
    };
    prof_point("kda: qkv_proj (+LoRA)");
    if (L.w3.p) {
        // g_a | b | f_a in one GEMV (the same dot products, fp16-rounded), split; beta = bf16(sigmoid(b)) on the way
        half* y3 = own.take<half>((size_t)S * 320);
        ops::gemv_f16<half>(q, y, S, 4096, L.w3.ptr<half>(), 320, y3);
        q.parallel_for(sycl::range<2>(S, 320), [=](sycl::item<2> it) {
            int s_ = it[0], c = it[1];
            half v = y3[(size_t)s_ * 320 + c];
            if (c < 128) ga[(size_t)s_ * 128 + c] = v;
            else if (c < 192) bt[(size_t)s_ * H + c - 128] = f(bf16(1.0f / (1.0f + sycl::exp(-f(v)))));
            else fa[(size_t)s_ * 128 + c - 192] = v;
        });
    } else {
        low(L.g_a, L.lga, ga);
        prof_point("kda: g_a (+LoRA)");
        ops::gemv_f16<float>(q, y, S, 4096, L.b_proj.ptr<half>(), H, bt, true);
        prof_point("kda: b_proj");
        lora(L.lb, y, bt);
        prof_point("kda: b_proj LoRA");
        low(L.f_a, L.lfa, fa);
        prof_point("kda: f_a (+LoRA)");
    }
    ops::gemv_f16<float>(q, ga, S, 128, L.g_b.ptr<half>(), H * DV, z, true);
    prof_point("kda: g_b");
    lora(L.lgb, ga, z);
    prof_point("kda: g_b LoRA");
    ops::gemv_f16<float>(q, fa, S, 128, L.f_b.ptr<half>(), H * DK, fg, true);
    prof_point("kda: f_b");
    lora(L.lfb, fa, fg);

    prof_point("kda: small projections");
    // 2. gates: beta = bf16(sigmoid(b)); g = -5 * sigmoid((f + dt_bias) * exp(A_log[h]))
    const float* A_log = L.A_log.ptr<float>();
    const float* dtb = L.dt_bias.ptr<float>();
    q.parallel_for(sycl::range<1>((size_t)S * H * DK), [=](sycl::id<1> i) {
        int c = i % (H * DK), h = c / DK;
        float v = (fg[i] + dtb[c]) * sycl::exp(A_log[h]);
        fg[i] = -5.0f * (1.0f / (1.0f + sycl::exp(-v)));
    });
    if (!L.w3.p)
        q.parallel_for(sycl::range<1>((size_t)S * H), [=](sycl::id<1> i) {
            bt[i] = f(bf16(1.0f / (1.0f + sycl::exp(-bt[i]))));
        });

    prof_point("kda: gates");
    // 3. short conv (depthwise, causal, kernel 4, SiLU), state shift; inputs rounded to bf16 first
    bf16* cs = st.conv.ptr<bf16>();
    const bf16* cw = L.conv_w.ptr<bf16>();
    const int width = CK + st.hist;
    q.parallel_for(sycl::range<1>(F), [=](sycl::id<1> ci) {
        int c = ci;
        bf16* sc = cs + (size_t)c * width;
        float wk[CK];
        for (int k = 0; k < CK; k++) wk[k] = f(cw[c * CK + k]);
        // window over concat(state[0:4], x[0:S]); token s reads entries s+1 .. s+4
        auto at = [&](int t) { return t < CK ? sc[t] : bf16(qkv[(size_t)(t - CK) * F + c]); };   // inputs to bf16
        for (int s = 0; s < S; s++) {
            float acc = 0;
            for (int k = 0; k < CK; k++) acc += f(bf16(f(at(s + k + 1)) * wk[k]));   // bf16 x bf16 product (Triton)
            conv_out[(size_t)s * F + c] = bf16(acc * (1.0f / (1.0f + sycl::exp(-acc))));
        }
        int tot = CK + S;
        if (history) {
            // the last min(width, 4 + S) entries of concat(state[0:4], x) at the tail of the buffer
            int n = tot < width ? tot : width;
            bf16 tmp[64];
            for (int j = 0; j < n; j++) tmp[j] = at(tot - n + j);
            for (int j = 0; j < n; j++) sc[width - n + j] = tmp[j];
        } else {
            bf16 tmp[CK];
            for (int j = 0; j < CK; j++) tmp[j] = at(S + j);
            for (int j = 0; j < CK; j++) sc[j] = tmp[j];
        }
    });

    prof_point("kda: conv");
    // 4. delta rule, one work-group per (head, 32 v columns); thread i owns state row k = i
    float* rec = st.rec.ptr<float>();
    const float scale = 1.0f / sycl::sqrt((float)DK);
    // reductions over k: within each 16-lane sub-group by shuffles, then the 8 sub-groups' partials through local
    // memory (3 barriers per token instead of a work-group reduction per column)
    q.submit([&](sycl::handler& hd) {
        constexpr int SG = 16, NSG = DK / SG, VB = 32;
        sycl::local_accessor<float, 2> part(sycl::range<2>(NSG, VB), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(2 * NSG), hd);
        hd.parallel_for(sycl::nd_range<1>((size_t)H * (DV / VB) * DK, DK),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            int g = it.get_group(0), i = it.get_local_id(0);
            int h = g / (DV / VB), vb = g % (DV / VB);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            int sgi = sg.get_group_id()[0], lane = sg.get_local_id()[0];
            float* Sp = rec + ((size_t)h * DK) * DV;                  // state j = 0
            float Sr[VB];
            for (int v = 0; v < VB; v++) Sr[v] = Sp[(size_t)i * DV + vb * VB + v];
            for (int t = 0; t < S; t++) {
                const bf16* row = conv_out + (size_t)t * F;
                float qi = f(row[h * DK + i]), ki = f(row[(H + h) * DK + i]);
                float qq = sycl::reduce_over_group(sg, qi * qi, sycl::plus<float>());
                float kk2 = sycl::reduce_over_group(sg, ki * ki, sycl::plus<float>());
                if (lane == 0) { red[sgi] = qq; red[NSG + sgi] = kk2; }
                float a = sycl::exp(fg[((size_t)t * H + h) * DK + i]);
                float beta = bt[(size_t)t * H + h];
                for (int v = 0; v < VB; v++) Sr[v] *= a;
                sycl::group_barrier(grp);
                float qn = 0, kn = 0;
                for (int j = 0; j < NSG; j++) { qn += red[j]; kn += red[NSG + j]; }
                qi = qi / sycl::sqrt(qn + 1e-6f);
                ki = ki / sycl::sqrt(kn + 1e-6f);
                // u = v - S^T k (with the decayed S)
                for (int v = 0; v < VB; v++) {
                    float p = sycl::reduce_over_group(sg, Sr[v] * ki, sycl::plus<float>());
                    if (lane == 0) part[sgi][v] = p;
                }
                sycl::group_barrier(grp);
                for (int v = 0; v < VB; v++) {
                    float sk = 0;
                    for (int j = 0; j < NSG; j++) sk += part[j][v];
                    float u = f(row[2 * H * DK + h * DV + vb * VB + v]) - sk;
                    Sr[v] += beta * ki * u;
                }
                sycl::group_barrier(grp);
                // o = S^T q / sqrt(Dk), from the updated S
                for (int v = 0; v < VB; v++) {
                    float p = sycl::reduce_over_group(sg, Sr[v] * qi, sycl::plus<float>());
                    if (lane == 0) part[sgi][v] = p;
                }
                sycl::group_barrier(grp);
                if (i < VB) {
                    float o = 0;
                    for (int j = 0; j < NSG; j++) o += part[j][i];
                    core[((size_t)t * H + h) * DV + vb * VB + i] = bf16(o * scale);
                }
                if (history && t < S - 1) {
                    float* Sh = rec + (((size_t)(t + 1) * H + h) * DK) * DV;
                    for (int v = 0; v < VB; v++) Sh[(size_t)i * DV + vb * VB + v] = Sr[v];
                }
                sycl::group_barrier(grp);      // part / red are rewritten by the next token
            }
            for (int v = 0; v < VB; v++) Sp[(size_t)i * DV + vb * VB + v] = Sr[v];
        });
    });
    if (history) st.last_history = S - 1;

    prof_point("kda: delta rule");
    // 5. gated RMSNorm per head (o_norm weight [128] shared), sigmoid gate after the weight, fp16 out
    const bf16* onw = L.o_norm.ptr<bf16>();
    q.parallel_for(sycl::nd_range<1>((size_t)S * H * DV, DV), [=](sycl::nd_item<1> it) {
        size_t base = it.get_group(0) * DV;
        int d = it.get_local_id(0);
        float hv = f(core[base + d]);
        float ss = sycl::reduce_over_group(it.get_group(), hv * hv, sycl::plus<float>());
        hv = hv * (1.0f / sycl::sqrt(ss / DV + 1e-5f));
        hv = hv * (f(onw[d]) + 0.0f);
        hv = hv * (1.0f / (1.0f + sycl::exp(-z[base + d])));
        normed[base + d] = half(hv);
    });

    prof_point("kda: gated norm");
    // 6. o_proj, fp32 out
    L.o.forward(q, ws, normed, S, out, DType::F32);
    prof_point("kda: o_proj (+LoRA)");
}

void kda_rewind(sycl::queue& q, KdaState& st, int r) {
    constexpr int H = KdaLayer::H, DK = KdaLayer::DK, DV = KdaLayer::DV, F = KdaLayer::F, CK = KdaLayer::CONV;
    size_t n = (size_t)H * DK * DV;
    float* rec = st.rec.ptr<float>();
    if (r > 0) q.memcpy(rec, rec + (size_t)(st.last_history + 1 - r) * n, n * sizeof(float));
    if (st.last_history > 0) {
        int width = CK + st.hist, p = width - r;
        bf16* cs = st.conv.ptr<bf16>();
        q.parallel_for(sycl::range<1>(F), [=](sycl::id<1> c) {
            bf16* sc = cs + (size_t)c * width;
            bf16 tmp[CK];
            for (int j = 0; j < CK; j++) tmp[j] = sc[p - CK + j];
            for (int j = 0; j < CK; j++) sc[j] = tmp[j];
        });
    }
    st.last_history = 0;
}

}  // namespace glm
