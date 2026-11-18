// quant_dflash2: EXL3 for the big linears of a DFlash2 drafter (Ripple-65B's: 7.5 GiB as fp16 on a 32 GB card), everything else copied.
// No calibration by default: a drafter only proposes, the target verifies every token, so its error costs acceptance and never
// quality; the Hessian is the identity and there is no refit. Layers finished in <out>.work are kept, a killed run resumes; after each
// layer it exits 99, so under `b70q submit --chunked` the next piece goes back in line and others get the card between layers.
//   quant_dflash2 -i <dflash2 dir> -o <out dir> [-b K_MLP] [--attn-bits K]
#include "exl3/quantize.hpp"
#include "io/safetensors.hpp"
#include "io/st_writer.hpp"
#include "nn/ops.hpp"
#include "rt/rt.hpp"
#include "rt/timer.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using half = sycl::half;
using exl3::DBuf;

static DBuf<float> load_w(sycl::queue& q, const st::TensorView& t)
{
    // HF [out, in] bf16 -> (in, out) fp32, rounded through fp16
    const int64_t out = t.shape[0], in = t.shape[1];
    std::vector<half> h((size_t) t.numel());
    const uint16_t* b = (const uint16_t*) t.data;
    for (size_t i = 0; i < h.size(); ++i) h[i] = (half) st::bf16_to_float(b[i]);
    DBuf<half> wh(q, h.size());
    wh.upload(h.data(), h.size());
    DBuf<float> W(q, (size_t) in * out);
    nn::transpose_h2f(q, wh.get(), out, in, W.get()).wait();
    return W;
}

int main(int argc, char** argv)
{
    fs::path in, out;
    int K_mlp = 4, K_attn = 4;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto val = [&] { if (++i >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[i]); };
        if (a == "-i") in = val();
        else if (a == "-o") out = val();
        else if (a == "-b") K_mlp = std::stoi(val());
        else if (a == "--attn-bits") K_attn = std::stoi(val());
        else throw std::runtime_error("unknown argument " + a);
    }
    if (in.empty() || out.empty()) throw std::runtime_error("usage: quant_dflash2 -i <dir> -o <dir> [-b K] [--attn-bits K]");
    sycl::queue q = rt::make_queue();
    st::Model m(in);
    const fs::path work = out.string() + ".work";
    fs::create_directories(work);
    fs::create_directories(out);

    auto group = [&](const std::vector<std::string>& keys, int K, std::vector<st::OutTensor>& ts) {
        rt::Timer t;
        const auto& t0 = m.get(keys[0] + ".weight");
        const int64_t k = t0.shape[1];
        DBuf<float> H(q, (size_t) k * k);
        float* hp = H.get();
        q.parallel_for(sycl::range<1>((size_t) k * k), [=](sycl::id<1> i) { hp[i] = i % (k + 1) == 0 ? 1.0f : 0.0f; }).wait();
        std::vector<DBuf<float>> Ws;
        std::vector<exl3::QuantIn> ins;
        for (auto& key : keys)
        {
            Ws.push_back(load_w(q, m.get(key + ".weight")));
            ins.push_back({key, m.get(key + ".weight").shape[0], Ws.back().get()});
        }
        const uint64_t seed = exl3::mix64(exl3::hash_str(keys[0]));
        exl3::Hessian hs = exl3::finalize_hessian(q, std::move(H), k, 1, seed);
        exl3::QuantArgs qa;
        qa.K = K;
        qa.refit = false;
        auto outs = exl3::quantize_group(q, hs, ins, qa, seed);
        for (auto& o : outs)
        {
            std::printf("  %-40s %6lld x %-6lld K %d  g %.3f  err %.6f -> %.6f\n", o.key.c_str(), (long long) o.k, (long long) o.n, K, o.g_scale, o.err_before, o.err_after);
            const int32_t marker = (int32_t) exl3::MUL1_MULT;
            ts.push_back(st::OutTensor::from_vector(o.key + ".suh", "F16", {o.k}, std::move(o.suh)));
            ts.push_back(st::OutTensor::from_vector(o.key + ".svh", "F16", {o.n}, std::move(o.svh)));
            ts.push_back(st::OutTensor::from_vector(o.key + ".mul1", "I32", {}, std::vector<int32_t>{marker}));
            ts.push_back(st::OutTensor::from_vector(o.key + ".trellis", "I16", {o.k / 16, o.n / 16, 16 * o.K}, std::move(o.trellis)));
        }
        std::printf("  [%.1f s]\n", t.s());
        std::fflush(stdout);
    };

    std::vector<std::string> quantized;
    for (int li = 0; li < 5; ++li)
    {
        const std::string p = "layers." + std::to_string(li) + ".";
        const fs::path f = work / ("layer_" + std::to_string(li) + ".safetensors");
        quantized.insert(quantized.end(), {p + "self_attn.q_proj.weight", p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight",
                                           p + "mlp.up_proj.weight", p + "mlp.down_proj.weight"});
        if (fs::exists(f)) { std::printf("layer %d: done earlier\n", li); continue; }
        std::printf("layer %d\n", li);
        std::vector<st::OutTensor> ts;
        group({p + "mlp.down_proj"}, K_mlp, ts);
        group({p + "mlp.gate_proj", p + "mlp.up_proj"}, K_mlp, ts);
        group({p + "self_attn.q_proj"}, K_attn, ts);
        group({p + "self_attn.o_proj"}, K_attn, ts);
        st::write_file(f, ts);
        if (li + 1 < 5) { std::printf("layer %d done; back in line\n", li); return 99; }
    }

    // the output: the layers' files merged, every other tensor as it is
    std::vector<std::unique_ptr<st::File>> files;
    std::vector<st::OutTensor> all;
    for (int li = 0; li < 5; ++li)
    {
        files.push_back(std::make_unique<st::File>(work / ("layer_" + std::to_string(li) + ".safetensors")));
        for (const auto& name : files.back()->order())
        {
            const auto& t = files.back()->tensors().at(name);
            all.push_back(st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes));
        }
    }
    for (const auto& name : m.names())
    {
        if (std::find(quantized.begin(), quantized.end(), name) != quantized.end()) continue;
        const auto& t = m.get(name);
        all.push_back(st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes));
    }
    st::write_file(out / "model.safetensors", all);
    fs::copy_file(in / "config.json", out / "config.json", fs::copy_options::overwrite_existing);
    fs::remove_all(work);
    std::printf("wrote %s\n", out.c_str());
}
