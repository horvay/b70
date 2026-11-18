// Qwen3.8-Flash-Next routed experts: the MoE kernels on q4_build_record's records (in VRAM and read zero-copy from
// pinned host memory) vs the three EXL3 linears run separately (exl3_check checks those against the format):
// out = w0 * expert_a(x) + w1 * expert_b(x), expert(x) = down(fp16(silu(gate x) * up x)).
//   q4_moe_check <model dir> [layer expert_a expert_b]...
#include <cmath>
#include <cstdio>
#include <array>
#include <cstring>
#include <random>
#include "../src/moe.h"
#include "../src/q4.h"

using namespace glm;
using namespace q4;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: q4_moe_check <model dir> [layer ea eb]...");
    SafeTensors st(argv[1]);
    std::vector<std::array<int, 3>> cases;
    for (int i = 2; i + 2 < argc; i += 3) cases.push_back({atoi(argv[i]), atoi(argv[i + 1]), atoi(argv[i + 2])});
    if (cases.empty()) cases = {{0, 5, 300}, {27, 140, 6}, {47, 511, 0}};
    auto& q = Dev::get().q;
    Workspace lin, mws;
    lin.init(256 << 20);
    mws.init(1024 << 20);
    const size_t rec = moe_record_bytes(D, I, 3);
    char* dev = static_cast<char*>(sycl::malloc_device(2 * rec, q));
    char* host = static_cast<char*>(sycl::malloc_host(2 * rec, q));
    bool ok = true;
    for (auto [li, ea, eb] : cases) {
        std::vector<char> buf(2 * rec);
        q4_build_record(st, li, ea, buf.data());
        q4_build_record(st, li, eb, buf.data() + rec);
        q.memcpy(dev, buf.data(), 2 * rec).wait();
        memcpy(host, buf.data(), 2 * rec);
        std::string pa = "model.language_model.layers." + std::to_string(li) + ".mlp.experts.";
        Exl3Linear g[2], u[2], d[2];
        int ee[2] = {ea, eb};
        for (int k = 0; k < 2; k++) {
            std::string p = pa + std::to_string(ee[k]) + ".";
            g[k] = Exl3Linear::load(st, p + "gate_proj");
            u[k] = Exl3Linear::load(st, p + "up_proj");
            d[k] = Exl3Linear::load(st, p + "down_proj");
        }
        for (int M : {1, 2, 5, 40, 300}) {
            std::mt19937 rng(M + li);
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<half> x((size_t)M * D);
            for (auto& v : x) v = half(nd(rng));
            Tensor xd = Tensor::empty({M, D}, DType::F16);
            upload(xd.p, x.data(), x.size() * 2);
            // reference
            const float wts[2] = {0.3f, 0.7f};
            std::vector<double> ref((size_t)M * D, 0.0);
            Tensor gd = Tensor::empty({M, I}, DType::F16), ud = Tensor::empty({M, I}, DType::F16),
                   ad = Tensor::empty({M, I}, DType::F16), od = Tensor::empty({M, D}, DType::F32);
            for (int k = 0; k < 2; k++) {
                g[k].forward(q, lin, xd.ptr<half>(), M, gd.p, DType::F16);
                u[k].forward(q, lin, xd.ptr<half>(), M, ud.p, DType::F16);
                std::vector<half> gh((size_t)M * I), uh((size_t)M * I), ah((size_t)M * I);
                q.wait();
                download(gh.data(), gd.p, gh.size() * 2);
                download(uh.data(), ud.p, uh.size() * 2);
                for (size_t i = 0; i < ah.size(); i++) {
                    float gv = gh[i], uv = uh[i];
                    ah[i] = half(gv / (1.0f + std::exp(-gv)) * uv);
                }
                upload(ad.p, ah.data(), ah.size() * 2);
                d[k].forward(q, lin, ad.ptr<half>(), M, od.p, DType::F32);
                q.wait();
                std::vector<float> o((size_t)M * D);
                download(o.data(), od.p, o.size() * 4);
                for (size_t i = 0; i < o.size(); i++) ref[i] += wts[k] * o[i];
            }
            for (int where = 0; where < 2; where++) {
                char* base = where ? host : dev;
                std::vector<int64_t> ptrs(E, (int64_t)base);
                ptrs[7] = (int64_t)base;
                ptrs[9] = (int64_t)(base + rec);
                Tensor pd = Tensor::empty({E}, DType::I64);
                upload(pd.p, ptrs.data(), E * 8);
                std::vector<int> sel((size_t)M * 2);
                std::vector<float> w((size_t)M * 2);
                for (int m = 0; m < M; m++) { sel[m * 2] = 7; sel[m * 2 + 1] = 9; w[m * 2] = wts[0]; w[m * 2 + 1] = wts[1]; }
                Tensor sd = Tensor::empty({M, 2}, DType::I32), wd = Tensor::empty({M, 2}, DType::F32);
                upload(sd.p, sel.data(), sel.size() * 4);
                upload(wd.p, w.data(), w.size() * 4);
                Tensor out = Tensor::empty({M, D}, DType::F16);
                moe_forward(q, mws, xd.ptr<half>(), M, sd.ptr<int>(), wd.ptr<float>(), 2, pd.ptr<int64_t>(), D, I, 3, E,
                            out.ptr<half>(), 0.0f);
                q.wait();
                std::vector<half> oh((size_t)M * D);
                download(oh.data(), out.p, oh.size() * 2);
                double num = 0, den = 0;
                for (size_t i = 0; i < oh.size(); i++) {
                    double dd = (double)(float)oh[i] - ref[i];
                    num += dd * dd;
                    den += ref[i] * ref[i];
                }
                double rel = std::sqrt(num / den);
                bool good = rel < 5e-3;
                ok &= good;
                printf("layer %2d experts %3d,%3d M=%-3d %s: rel err %.2e  %s\n", li, ea, eb, M, where ? "RAM " : "VRAM", rel,
                       good ? "ok" : "BAD");
            }
        }
    }
    printf(ok ? "Q4 MOE CHECK OK\n" : "Q4 MOE CHECK FAILED\n");
    return ok ? 0 : 1;
}
