#include "q35v.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <oneapi/mkl/blas.hpp>

namespace glm {

using namespace q35v;
using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
const std::string VP = "model.visual.";
template <class T> float f(T v) { return static_cast<float>(v); }

std::vector<float> host_f32(const SafeTensors& st, const std::string& name) {
    const StTensor& t = st.at(name);
    size_t n = 1;
    for (auto s : t.shape) n *= s;
    std::vector<float> v(n);
    GLM_CHECK(t.dt == DType::BF16, "%s: expected bf16", name.c_str());
    const bf16* p = static_cast<const bf16*>(t.data);
    for (size_t i = 0; i < n; i++) v[i] = f(p[i]);
    return v;
}
Tensor dev_f32(const std::vector<float>& v) {
    Tensor t = Tensor::empty({(int64_t)v.size()}, DType::F32);
    upload(t.p, v.data(), v.size() * 4);
    return t;
}
Tensor dev_f16(const std::vector<float>& v, int64_t rows, int64_t cols) {
    Tensor t = Tensor::empty({rows, cols}, DType::F16);
    std::vector<half> h(v.begin(), v.end());
    upload(t.p, h.data(), h.size() * 2);
    return t;
}
Tensor load_f32(const SafeTensors& st, const std::string& n) { return dev_f32(host_f32(st, n)); }
Tensor load_f16(const SafeTensors& st, const std::string& n) {
    const StTensor& t = st.at(n);
    return dev_f16(host_f32(st, n), t.shape[0], t.shape[1]);
}

// LayerNorm over the last dim: fp16((x - mean) / sqrt(var + 1e-6) * w + b)
void layer_norm(sycl::queue& q, const float* x, const float* w, const float* b, half* y, int rows, int dim) {
    constexpr int WG = 64;
    q.parallel_for(nd_range<1>((size_t)rows * WG, WG), [=](nd_item<1> it) {
        size_t r = it.get_group(0);
        int l = it.get_local_id(0);
        const float* xr = x + r * dim;
        half* o = y + r * dim;
        float s = 0;
        for (int d = l; d < dim; d += WG) s += xr[d];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        float mean = s / dim, v = 0;
        for (int d = l; d < dim; d += WG) { float e = xr[d] - mean; v += e * e; }
        v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
        float rs = sycl::rsqrt(v / dim + 1e-6f);
        for (int d = l; d < dim; d += WG) o[d] = half((xr[d] - mean) * rs * w[d] + b[d]);
    });
}

// the rotated q and k of one row: dim i rotates by ang[i] (i < 36) or ang[i - 36]
void rope_qkv(sycl::queue& q, const float* qkv, const float* cs, half* Q, half* K, half* V, int rows) {
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for(range<3>(rows, H, HD), [=](sycl::item<3> it) {
            size_t r = it[0];
            int h = it[1], d = it[2], a = d % (HD / 2), other = (d + HD / 2) % HD;
            float c = cs[(r * (HD / 2) + a) * 2], s = cs[(r * (HD / 2) + a) * 2 + 1];
            float sign = d < HD / 2 ? -1.0f : 1.0f;
            const float* row = qkv + r * 3 * D;
            size_t oo = (r * H + h) * HD + d;
            Q[oo] = half(row[h * HD + d] * c + sign * row[h * HD + other] * s);
            K[oo] = half(row[D + h * HD + d] * c + sign * row[D + h * HD + other] * s);
            V[oo] = half(row[2 * D + h * HD + d]);
        });
    });
}

// row-wise softmax in place, fp16 scores [rows, n] scaled by sc
void softmax_rows(sycl::queue& q, half* S, int rows, int n, float sc) {
    constexpr int WG = 256;
    q.parallel_for(nd_range<1>((size_t)rows * WG, WG), [=](nd_item<1> it) {
        size_t r = it.get_group(0);
        int l = it.get_local_id(0);
        half* s = S + r * n;
        float m = -INFINITY;
        for (int t = l; t < n; t += WG) m = sycl::fmax(m, f(s[t]));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0;
        for (int t = l; t < n; t += WG) sum += sycl::exp((f(s[t]) - m) * sc);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        float inv = 1.0f / sum;
        for (int t = l; t < n; t += WG) s[t] = half(sycl::exp((f(s[t]) - m) * sc) * inv);
    });
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + sycl::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// the bicubic filter of torch's antialiased interpolate (a = -0.5), its support scaled by the downscale factor
struct Weights {
    std::vector<int> start, count;
    std::vector<float> w;
    int taps = 0;
};
Weights resample_weights(int in, int out) {
    Weights t;
    double scale = (double)in / out, fs = std::max(1.0, scale), sup = 2.0 * fs;
    t.taps = (int)std::ceil(sup) * 2 + 1;
    t.start.resize(out);
    t.count.resize(out);
    t.w.assign((size_t)out * t.taps, 0.0f);
    auto cubic = [](double x) {
        x = std::fabs(x);
        const double a = -0.5;
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    };
    for (int o = 0; o < out; o++) {
        double c = (o + 0.5) * scale;
        int lo = std::max(0, (int)(c - sup + 0.5)), hi = std::min(in, (int)(c + sup + 0.5));
        double total = 0;
        std::vector<double> tmp;
        for (int i = lo; i < hi; i++) {
            double v = cubic((i + 0.5 - c) / fs);
            tmp.push_back(v);
            total += v;
        }
        t.start[o] = lo;
        t.count[o] = (int)tmp.size();
        for (size_t i = 0; i < tmp.size(); i++) t.w[(size_t)o * t.taps + i] = (float)(tmp[i] / total);
    }
    return t;
}

void par_for(int n, const std::function<void(int)>& fn) {
    int th = std::min(12u, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (int k = 0; k < th; k++)
        pool.emplace_back([=, &fn] { for (int i = (int)((int64_t)n * k / th); i < (int)((int64_t)n * (k + 1) / th); i++) fn(i); });
    for (auto& t : pool) t.join();
}
}  // namespace

// ------------------------------------------------------------------------------------------------ preprocessing

void q35_target_size(int h, int w, int max_pixels, int* th, int* tw) {
    constexpr int F = P * MERGE;
    constexpr double MIN_PIXELS = 65536;
    double hb = std::nearbyint((double)h / F) * F, wb = std::nearbyint((double)w / F) * F;   // Python round: ties to even
    if (hb * wb > max_pixels) {
        double beta = std::sqrt((double)h * w / max_pixels);
        hb = std::max<double>(F, std::floor(h / beta / F) * F);
        wb = std::max<double>(F, std::floor(w / beta / F) * F);
    } else if (hb * wb < MIN_PIXELS) {
        double beta = std::sqrt(MIN_PIXELS / ((double)h * w));
        hb = std::ceil(h * beta / F) * F;
        wb = std::ceil(w * beta / F) * F;
    }
    *th = (int)hb;
    *tw = (int)wb;
}

Q35Pixels q35_rgb_to_pixels(const unsigned char* rgb, int h, int w, int max_pixels) {
    int th, tw;
    q35_target_size(h, w, max_pixels, &th, &tw);
    std::vector<float> a((size_t)h * tw * 3), b((size_t)th * tw * 3);
    if (tw == w && th == h) {
        for (size_t i = 0; i < b.size(); i++) b[i] = rgb[i];
    } else {
        Weights wx = resample_weights(w, tw), wy = resample_weights(h, th);
        par_for(h, [&](int y) {
            const unsigned char* src = rgb + (size_t)y * w * 3;
            float* dst = a.data() + (size_t)y * tw * 3;
            for (int x = 0; x < tw; x++) {
                const float* wt = wx.w.data() + (size_t)x * wx.taps;
                int s = wx.start[x], n = wx.count[x];
                float c0 = 0, c1 = 0, c2 = 0;
                for (int i = 0; i < n; i++) {
                    const unsigned char* px = src + (size_t)(s + i) * 3;
                    c0 += wt[i] * px[0];
                    c1 += wt[i] * px[1];
                    c2 += wt[i] * px[2];
                }
                dst[x * 3] = c0;
                dst[x * 3 + 1] = c1;
                dst[x * 3 + 2] = c2;
            }
        });
        par_for(th, [&](int y) {
            const float* wt = wy.w.data() + (size_t)y * wy.taps;
            int s = wy.start[y], n = wy.count[y];
            float* dst = b.data() + (size_t)y * tw * 3;
            for (int x = 0; x < tw * 3; x++) {
                float acc = 0;
                for (int i = 0; i < n; i++) acc += wt[i] * a[(size_t)(s + i) * tw * 3 + x];
                dst[x] = std::nearbyint(std::min(255.0f, std::max(0.0f, acc)));
            }
        });
    }
    Q35Pixels out;
    out.gh = th / P;
    out.gw = tw / P;
    out.patch.resize((size_t)out.gh * out.gw * PIX);
    const int mh = out.gh / MERGE, mw = out.gw / MERGE;
    par_for(mh, [&](int i) {
        for (int j = 0; j < mw; j++)
            for (int ma = 0; ma < MERGE; ma++)
                for (int mb = 0; mb < MERGE; mb++) {
                    int py = (i * MERGE + ma) * P, px = (j * MERGE + mb) * P;
                    float* o = out.patch.data() + (((size_t)i * mw + j) * MERGE * MERGE + ma * MERGE + mb) * PIX;
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < P; y++)
                            for (int x = 0; x < P; x++)
                                o[(c * P + y) * P + x] = (b[((size_t)(py + y) * tw + px + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
                }
    });
    return out;
}

// ------------------------------------------------------------------------------------------------ tower

Q35Vision::Q35Vision(const SafeTensors& st) {
    blocks.resize(DEPTH);
    for (int i = 0; i < DEPTH; i++) {
        Block& b = blocks[i];
        std::string p = VP + "blocks." + std::to_string(i) + ".";
        b.n1w = load_f32(st, p + "norm1.weight");
        b.n1b = load_f32(st, p + "norm1.bias");
        b.n2w = load_f32(st, p + "norm2.weight");
        b.n2b = load_f32(st, p + "norm2.bias");
        b.qkv = load_f16(st, p + "attn.qkv.weight");
        b.qkvb = load_f32(st, p + "attn.qkv.bias");
        b.proj = load_f16(st, p + "attn.proj.weight");
        b.projb = load_f32(st, p + "attn.proj.bias");
        b.fc1 = load_f16(st, p + "mlp.linear_fc1.weight");
        b.fc1b = load_f32(st, p + "mlp.linear_fc1.bias");
        b.fc2 = load_f16(st, p + "mlp.linear_fc2.weight");
        b.fc2b = load_f32(st, p + "mlp.linear_fc2.bias");
    }
    // the two temporal slices of the Conv3d see the same frame: their kernels add
    auto pw = host_f32(st, VP + "patch_embed.proj.weight");      // [D, 3, 2, 16, 16]
    std::vector<float> sum((size_t)D * PIX);
    for (int e = 0; e < D; e++)
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < P * P; y++)
                sum[((size_t)e * 3 + c) * (P * P) + y] = pw[(((size_t)e * 3 + c) * 2 + 0) * (P * P) + y] + pw[(((size_t)e * 3 + c) * 2 + 1) * (P * P) + y];
    patch_w = dev_f16(sum, D, PIX);
    patch_b = load_f32(st, VP + "patch_embed.proj.bias");
    pos_table = host_f32(st, VP + "pos_embed.weight");
    mg_nw = load_f32(st, VP + "merger.norm.weight");
    mg_nb = load_f32(st, VP + "merger.norm.bias");
    mg_f1 = load_f16(st, VP + "merger.linear_fc1.weight");
    mg_f1b = load_f32(st, VP + "merger.linear_fc1.bias");
    mg_f2 = load_f16(st, VP + "merger.linear_fc2.weight");
    mg_f2b = load_f32(st, VP + "merger.linear_fc2.bias");
    fprintf(stderr, "vision tower: %.2f GiB VRAM in all\n", Dev::get().vram_used / 1073741824.0);
}

// y fp32 [rows, out] = x fp16 [rows, in] @ W^T + b, by pieces (GEMM's fp16 output staged in fp32)
void Q35Vision::linear(const Tensor& w, const Tensor& b, const half* x, int rows, int in, int out, float* y) {
    auto& q = Dev::get().q;
    const size_t need = (size_t)ROWS * out;
    static thread_local std::pair<half*, size_t> tmp{nullptr, 0};
    if (tmp.second < need) {
        q.wait();
        if (tmp.first) sycl::free(tmp.first, q);
        tmp = {sycl::malloc_device<half>(need, q), need};
        GLM_CHECK(tmp.first, "vision linear: staging allocation failed");
    }
    half* o16 = tmp.first;
    const half* wp = w.ptr<half>();
    const float* bp = b.ptr<float>();
    for (int r0 = 0; r0 < rows; r0 += ROWS) {
        int nr = std::min(ROWS, rows - r0);
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, nr, out, in,
                                           half(1.0f), x + (size_t)r0 * in, in, wp, in, half(0.0f), o16, out);
        float* yr = y + (size_t)r0 * out;
        q.parallel_for(range<2>(nr, out), [=](sycl::item<2> it) {
            size_t i = (size_t)it[0] * out + it[1];
            yr[i] = f(o16[i]) + bp[it[1]];
        });
    }
}

Q35Image Q35Vision::encode(const Q35Pixels& px) {
    auto& q = Dev::get().q;
    const int gh = px.gh, gw = px.gw, N = gh * gw;
    GLM_CHECK(gh % MERGE == 0 && gw % MERGE == 0, "vision: patch grid %d x %d", gh, gw);
    auto dev16 = [&](size_t n) {
        half* p = sycl::malloc_device<half>(n, q);
        GLM_CHECK(p, "vision: %zu fp16 values", n);
        return p;
    };
    auto dev32 = [&](size_t n) {
        float* p = sycl::malloc_device<float>(n, q);
        GLM_CHECK(p, "vision: %zu fp32 values", n);
        return p;
    };
    float* X = dev32((size_t)N * D);
    float* tmp32 = dev32((size_t)N * std::max(3 * D, I));
    half* y16 = dev16((size_t)N * std::max(D, I));
    half* Q16 = dev16((size_t)N * D);
    half* K16 = dev16((size_t)N * D);
    half* V16 = dev16((size_t)N * D);
    half* O16 = dev16((size_t)N * D);
    half* S16 = dev16((size_t)ROWS * N);

    // patch embedding + the position table, bilinear in the grid (linspace over 0 .. 47, in merge-unit order)
    {
        std::vector<half> ph(px.patch.begin(), px.patch.end());
        half* pd = dev16(ph.size());
        q.memcpy(pd, ph.data(), ph.size() * 2).wait();
        linear(patch_w, patch_b, pd, N, PIX, D, X);
        q.wait();
        sycl::free(pd, q);
    }
    std::vector<float> pos((size_t)N * D), cs((size_t)N * (HD / 2) * 2);
    {
        auto lin = [](int n, int i) { return n > 1 ? (float)i * (NPOS - 1) / (float)(n - 1) : 0.0f; };
        std::vector<float> inv(HD / 4);
        for (int j = 0; j < HD / 4; j++) inv[j] = 1.0f / std::pow(10000.0f, (float)(2 * j) / (float)(HD / 2));
        par_for(N, [&](int i) {
            int unit = i / (MERGE * MERGE), sub = i % (MERGE * MERGE);
            int mh = unit / (gw / MERGE), mw = unit % (gw / MERGE);
            int y = mh * MERGE + sub / MERGE, x = mw * MERGE + sub % MERGE;
            float hg = lin(gh, y), wg = lin(gw, x);
            int h0 = (int)hg, w0 = (int)wg, h1 = std::min(h0 + 1, NPOS - 1), w1 = std::min(w0 + 1, NPOS - 1);
            float hf = hg - h0, wf = wg - w0;
            const float *a = &pos_table[(size_t)(h0 * NPOS + w0) * D], *b = &pos_table[(size_t)(h0 * NPOS + w1) * D],
                        *c = &pos_table[(size_t)(h1 * NPOS + w0) * D], *d = &pos_table[(size_t)(h1 * NPOS + w1) * D];
            float* o = pos.data() + (size_t)i * D;
            float wa = (1 - hf) * (1 - wf), wb = (1 - hf) * wf, wc = hf * (1 - wf), wd = hf * wf;
            for (int e = 0; e < D; e++) o[e] = a[e] * wa + b[e] * wb + c[e] * wc + d[e] * wd;
            float* co = cs.data() + (size_t)i * (HD / 2) * 2;
            for (int j = 0; j < HD / 4; j++) {
                float ty = (float)y * inv[j], tx = (float)x * inv[j];
                co[j * 2] = std::cos(ty);
                co[j * 2 + 1] = std::sin(ty);
                co[(j + HD / 4) * 2] = std::cos(tx);
                co[(j + HD / 4) * 2 + 1] = std::sin(tx);
            }
        });
    }
    float* posd = dev32(pos.size());
    float* csd = dev32(cs.size());
    q.memcpy(posd, pos.data(), pos.size() * 4).wait();
    q.memcpy(csd, cs.data(), cs.size() * 4).wait();
    q.parallel_for(range<1>((size_t)N * D), [=](sycl::id<1> i) { X[i] += posd[i]; });
    auto grab = [&]() {
        if (!dump) return;
        q.wait();
        size_t o = dump->size();
        dump->resize(o + D);
        q.memcpy(dump->data() + o, X, D * 4).wait();
    };
    grab();

    const float scale = 1.0f / std::sqrt((float)HD);
    for (auto& b : blocks) {
        layer_norm(q, X, b.n1w.ptr<float>(), b.n1b.ptr<float>(), y16, N, D);
        linear(b.qkv, b.qkvb, y16, N, D, 3 * D, tmp32);
        rope_qkv(q, tmp32, csd, Q16, K16, V16, N);
        for (int r0 = 0; r0 < N; r0 += ROWS) {
            int nr = std::min(ROWS, N - r0);
            for (int h = 0; h < H; h++) {
                oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, nr, N, HD,
                                                   half(1.0f), Q16 + ((size_t)r0 * H + h) * HD, D, K16 + (size_t)h * HD, D,
                                                   half(0.0f), S16, N);
                softmax_rows(q, S16, nr, N, scale);
                oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, nr, HD, N,
                                                   half(1.0f), S16, N, V16 + (size_t)h * HD, D, half(0.0f),
                                                   O16 + ((size_t)r0 * H + h) * HD, D);
            }
        }
        linear(b.proj, b.projb, O16, N, D, D, tmp32);
        q.parallel_for(range<1>((size_t)N * D), [=](sycl::id<1> i) { X[i] += tmp32[i]; });
        layer_norm(q, X, b.n2w.ptr<float>(), b.n2b.ptr<float>(), y16, N, D);
        linear(b.fc1, b.fc1b, y16, N, D, I, tmp32);
        q.parallel_for(range<1>((size_t)N * I), [=](sycl::id<1> i) { y16[i] = half(gelu_tanh(tmp32[i])); });
        linear(b.fc2, b.fc2b, y16, N, I, D, tmp32);
        q.parallel_for(range<1>((size_t)N * D), [=](sycl::id<1> i) { X[i] += tmp32[i]; });
        q.wait();
        grab();
    }

    // merger: norm each patch, four to a unit, fc1, exact GELU, fc2
    const int NU = N / (MERGE * MERGE), UD = D * MERGE * MERGE, OD = q35::D;
    layer_norm(q, X, mg_nw.ptr<float>(), mg_nb.ptr<float>(), y16, N, D);
    float* m32 = dev32((size_t)NU * std::max(UD, OD));
    linear(mg_f1, mg_f1b, y16, NU, UD, UD, m32);
    q.parallel_for(range<1>((size_t)NU * UD), [=](sycl::id<1> i) {
        float x = m32[i];
        y16[i] = half(0.5f * x * (1.0f + sycl::erf(x * 0.70710678f)));
    });
    linear(mg_f2, mg_f2b, y16, NU, UD, OD, m32);
    q.wait();
    Q35Image img;
    img.rows = gh / MERGE;
    img.cols = gw / MERGE;
    img.emb.resize((size_t)NU * OD);
    q.memcpy(img.emb.data(), m32, img.emb.size() * 4).wait();
    for (void* p : {(void*)X, (void*)tmp32, (void*)y16, (void*)Q16, (void*)K16, (void*)V16, (void*)O16, (void*)S16, (void*)posd,
                    (void*)csd, (void*)m32})
        sycl::free(p, q);
    return img;
}

}  // namespace glm
