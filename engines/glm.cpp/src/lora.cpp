#include "lora.h"
#include <oneapi/mkl/blas.hpp>
#include <cmath>
#include <cstring>
#include <sstream>
#include "json.h"
#include "ops.h"

namespace glm {

namespace {
float bf16_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

std::string module_name(std::string k) {
    // base_model.model.<module>.lora_{A,B}.weight -> <module>; ".forget_gate." dropped; model.layers. -> language model
    auto strip = [&](const std::string& p) { if (k.compare(0, p.size(), p) == 0) k = k.substr(p.size()); };
    strip("base_model.model.");
    size_t fg;
    while ((fg = k.find(".forget_gate.")) != std::string::npos) k.replace(fg, 13, ".");
    if (k.compare(0, 13, "model.layers.") == 0) k = "model.language_model.layers." + k.substr(13);
    return k;
}
}  // namespace

LoraSet::LoraSet(const std::string& dirs, float scale) {
    std::stringstream ss(dirs);
    std::string dir;
    while (std::getline(ss, dir, ',')) {
        if (dir.empty()) continue;
        Json cfg = Json::parse(read_file(dir + "/adapter_config.json"));
        GLM_CHECK(!cfg.get_b("use_dora", false), "LoRA %s: DoRA not supported", dir.c_str());
        int r = cfg["r"].i();
        float alpha = (float)cfg.get_f("lora_alpha", r);
        float s = (cfg.get_b("use_rslora", false) ? alpha / std::sqrt((float)r) : alpha / r) * scale;
        SafeTensors st(dir);
        for (auto& name : st.names_with_prefix("")) {
            size_t pa = name.find(".lora_A.weight");
            if (pa == std::string::npos) continue;
            std::string base = name.substr(0, pa);
            const StTensor& a = st.at(name);
            const StTensor& b = st.at(base + ".lora_B.weight");
            GLM_CHECK(a.dt == DType::BF16 && b.dt == DType::BF16, "LoRA %s: bf16 expected", base.c_str());
            int rr = a.shape[0], in = a.shape[1], out = b.shape[0];
            GLM_CHECK(b.shape[1] == rr, "LoRA %s: rank mismatch", base.c_str());
            std::string mod = module_name(base);
            Raw& rw = raw_[mod];
            const uint16_t* ah = static_cast<const uint16_t*>(a.data);
            const uint16_t* bh = static_cast<const uint16_t*>(b.data);
            if (rw.r == 0) {
                rw.r = rr; rw.in = in; rw.out = out;
                rw.A.resize((size_t)rr * in);
                rw.B.resize((size_t)out * rr);
                for (size_t i = 0; i < rw.A.size(); i++) rw.A[i] = bf16_to_f(ah[i]);
                for (size_t i = 0; i < rw.B.size(); i++) rw.B[i] = bf16_to_f(bh[i]) * s;
            } else {
                // a second adapter on the same module: stack the ranks (deltas add)
                GLM_CHECK(rw.in == in && rw.out == out, "LoRA %s: shapes differ between adapters", mod.c_str());
                std::vector<float> A2((size_t)(rw.r + rr) * in), B2((size_t)out * (rw.r + rr));
                std::copy(rw.A.begin(), rw.A.end(), A2.begin());
                for (size_t i = 0; i < (size_t)rr * in; i++) A2[(size_t)rw.r * in + i] = bf16_to_f(ah[i]);
                for (int o = 0; o < out; o++) {
                    for (int j = 0; j < rw.r; j++) B2[(size_t)o * (rw.r + rr) + j] = rw.B[(size_t)o * rw.r + j];
                    for (int j = 0; j < rr; j++) B2[(size_t)o * (rw.r + rr) + rw.r + j] = bf16_to_f(bh[(size_t)o * rr + j]) * s;
                }
                rw.A.swap(A2);
                rw.B.swap(B2);
                rw.r += rr;
            }
        }
        fprintf(stderr, "LoRA %s: r=%d, alpha %.0f, scale %.3f\n", dir.c_str(), r, alpha, s);
    }
}

Lora* LoraSet::upload(const std::string& name, const Raw& rw) {
    auto l = std::make_unique<Lora>();
    l->r = rw.r; l->in = rw.in; l->out = rw.out; l->blocks = rw.blocks;
    std::vector<half> a(rw.A.size()), b(rw.B.size());
    for (size_t i = 0; i < a.size(); i++) a[i] = half(rw.A[i]);
    for (size_t i = 0; i < b.size(); i++) b[i] = half(rw.B[i]);
    l->A = Tensor::empty({rw.r, rw.in}, DType::F16);
    l->sB = Tensor::empty({rw.out, rw.r / rw.blocks}, DType::F16);
    glm::upload(l->A.p, a.data(), a.size() * 2);
    glm::upload(l->sB.p, b.data(), b.size() * 2);
    {
        int R = rw.r / rw.blocks, NB = rw.out / rw.blocks;
        std::vector<half> bt(b.size());
        for (int i = 0; i < rw.blocks; i++)
            for (int n = 0; n < NB; n++)
                for (int j = 0; j < R; j++) bt[((size_t)i * R + j) * NB + n] = b[((size_t)i * NB + n) * R + j];
        l->sBT = Tensor::empty({rw.out, R}, DType::F16);
        glm::upload(l->sBT.p, bt.data(), bt.size() * 2);
    }
    Lora* p = l.get();
    mods_[name] = std::move(l);
    return p;
}

const Lora* LoraSet::get(const std::string& module) const {
    auto it = mods_.find(module);
    if (it != mods_.end()) return it->second.get();
    auto r = raw_.find(module);
    if (r == raw_.end()) return nullptr;
    return const_cast<LoraSet*>(this)->upload(module, r->second);
}

const Lora* LoraSet::fused_qkv(const std::string& p) {
    const Raw* parts[3] = {nullptr, nullptr, nullptr};
    const char* nm[3] = {".q_proj", ".k_proj", ".v_proj"};
    int any = 0;
    for (int i = 0; i < 3; i++) {
        auto it = raw_.find(p + nm[i]);
        if (it != raw_.end()) { parts[i] = &it->second; any++; }
    }
    if (!any) return nullptr;
    // rank 3r: A stacked [3r, in]; sB block-diagonal [3 * out, 3r] (q, k, v order; a missing part is zero)
    int r = 0, in = 0, out = 0;
    for (auto* x : parts) if (x) { r = std::max(r, x->r); in = x->in; out = x->out; }
    Raw f;
    f.r = 3 * r; f.in = in; f.out = 3 * out;
    f.A.assign((size_t)f.r * in, 0.0f);
    f.blocks = 3;                      // B kept as [3 * out, r]: output row o of part i uses t columns i*r .. i*r+r-1
    f.B.assign((size_t)f.out * r, 0.0f);
    for (int i = 0; i < 3; i++) {
        const Raw* x = parts[i];
        if (!x) continue;
        GLM_CHECK(x->in == in && x->out == out, "LoRA %s: q/k/v shapes differ", p.c_str());
        for (int j = 0; j < x->r; j++)
            std::copy(x->A.begin() + (size_t)j * in, x->A.begin() + (size_t)(j + 1) * in, f.A.begin() + (size_t)(i * r + j) * in);
        for (int o = 0; o < out; o++)
            for (int j = 0; j < x->r; j++) f.B[(size_t)(i * out + o) * r + j] = x->B[(size_t)o * x->r + j];
    }
    return upload(p + ".qkv_proj", f);
}

bool LoraSet::merge_into(const std::string& module, Tensor& W) const {
    auto it = raw_.find(module);
    if (it == raw_.end()) return false;
    const Raw& rw = it->second;
    GLM_CHECK(W.dt == DType::F16 && W.size(0) == rw.out && W.size(1) == rw.in, "LoRA merge %s: shape", module.c_str());
    auto& q = Dev::get().q;
    // delta = sB @ A in fp32 on the device, then W = fp16(W + fp16(delta))
    float* B = sycl::malloc_device<float>(rw.B.size(), q);
    float* A = sycl::malloc_device<float>(rw.A.size(), q);
    float* D = sycl::malloc_device<float>((size_t)rw.out * rw.in, q);
    q.memcpy(B, rw.B.data(), rw.B.size() * 4);
    q.memcpy(A, rw.A.data(), rw.A.size() * 4);
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, rw.out,
                                       rw.in, rw.r, 1.0f, B, rw.r, A, rw.in, 0.0f, D, rw.in);
    half* w = W.ptr<half>();
    q.parallel_for(sycl::range<1>((size_t)rw.out * rw.in), [=](sycl::id<1> i) {
        w[i] = half((float)w[i] + (float)half(D[i]));
    });
    q.wait();
    sycl::free(B, q);
    sycl::free(A, q);
    sycl::free(D, q);
    return true;
}

void LoraSet::group(const std::vector<const Lora*>& members, int rows) {
    std::vector<Lora*> m;
    for (auto* l : members) if (l) m.push_back(const_cast<Lora*>(l));
    if (m.size() < 2) return;
    auto g = std::make_unique<LoraGroup>();
    g->in = m[0]->in;
    g->rows = rows;
    for (auto* l : m) {
        GLM_CHECK(l->in == g->in && !l->group, "LoRA group: inputs differ");
        l->goff = g->R;
        g->R += l->r;
    }
    g->A = Tensor::empty({g->R, g->in}, DType::F16);
    g->t = Tensor::empty({rows, g->R}, DType::F16);
    auto& q = Dev::get().q;
    for (auto* l : m) q.memcpy(g->A.ptr<half>() + (size_t)l->goff * g->in, l->A.p, l->A.bytes());
    q.wait();
    for (size_t i = 0; i < m.size(); i++) { m[i]->group = g.get(); m[i]->lead = i == 0; }
    groups_.push_back(std::move(g));
}

void Lora::apply(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt) const {
    if (group && M <= group->rows) {
        half* t = group->t.ptr<half>();
        if (lead) ops::gemv_f16<half>(q, x, M, in, group->A.ptr<half>(), group->R, t);   // every member's t at once
        apply_delta(q, ws, t + goff, group->R, M, y, ydt);
        return;
    }
    half* t = ws.take<half>((size_t)M * r);
    ops::gemv_f16<half>(q, x, M, in, A.ptr<half>(), r, t);        // t = fp16(x @ A^T)
    apply_delta(q, ws, t, r, M, y, ydt);
}

void Lora::apply_delta(sycl::queue& q, Workspace& ws, const half* t, int ts, int M, void* y, DType ydt) const {
    const half* b = sB.ptr<half>();
    const half* bt = sBT.ptr<half>();
    const int R = r / blocks, N = out, NB = out / blocks;     // R: columns per output row
    if (M > 16) {
        // a prompt: y (+)= t @ sB^T as GEMMs, one per diagonal block (fp16 out: beta = 1, one rounding; fp32 out: the
        // product rounded to fp16, then added)
        for (int i = 0; i < blocks; i++) {
            const half* tb = t + i * R;
            const half* bb = b + (size_t)i * NB * R;
            if (ydt == DType::F16) {
                oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                                                   M, NB, R, half(1.0f), tb, ts, bb, R, half(1.0f),
                                                   static_cast<half*>(y) + (size_t)i * NB, N);
            } else {
                half* d = ws.take<half>((size_t)M * NB);
                oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                                                   M, NB, R, half(1.0f), tb, ts, bb, R, half(0.0f), d, NB);
                float* yf = static_cast<float*>(y) + (size_t)i * NB;
                q.parallel_for(sycl::range<2>(M, NB), [=](sycl::item<2> it) {
                    yf[(size_t)it[0] * N + it[1]] += (float)d[(size_t)it[0] * NB + it[1]];
                });
            }
        }
        return;
    }
    if (ydt == DType::F16) {
        half* yh = static_cast<half*>(y);
        q.parallel_for(sycl::range<2>(M, N), [=](sycl::item<2> it) {
            int m = it[0], n = it[1];
            float acc = (float)yh[(size_t)m * N + n];
            int blk = n / NB, nn = n % NB;
            const half* tr = t + (size_t)m * ts + blk * R;
            const half* bc = bt + (size_t)blk * R * NB + nn;
            for (int j = 0; j < R; j++) acc += (float)tr[j] * (float)bc[(size_t)j * NB];
            yh[(size_t)m * N + n] = half(acc);
        });
    } else {
        float* yf = static_cast<float*>(y);
        q.parallel_for(sycl::range<2>(M, N), [=](sycl::item<2> it) {
            int m = it[0], n = it[1];
            float acc = 0;
            int blk = n / NB, nn = n % NB;
            const half* tr = t + (size_t)m * ts + blk * R;
            const half* bc = bt + (size_t)blk * R * NB + nn;
            for (int j = 0; j < R; j++) acc += (float)tr[j] * (float)bc[(size_t)j * NB];
            yf[(size_t)m * N + n] += (float)half(acc);
        });
    }
}

}  // namespace glm
