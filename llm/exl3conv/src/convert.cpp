// exl3conv: convert a Hugging Face BF16 checkpoint to EXL3 on the Intel Arc Pro B70. Gemma 4 dense, Ripple-65B
// (convert_ripple.hpp), Qwen3.5 / Qwen3.8 dense (convert_qwen35.hpp), Qwen3.8-Flash-Next (convert_qwen4exp.hpp).
//
//   exl3conv -i <hf model dir> -o <output dir> [-b 4] [-w <work dir>] [--max-minutes 25] [--rows 250] [--cols 2048]
//            [--cal-dir <exllamav3 standard_cal_data> | --cal-data <trace.safetensors>] [--layers N]
//            [--out-scales auto|always] [--scaled-hessian on|off] [--recipe R.json] [--compile-only]
//            Ripple only: [--head-bits 16|3..8] [--group-bits qkg=4,v=4,o=4,gu=3,d=4]
//            qwen3_5 only: [--head-bits 3..8 (default 6)] [--mtp-bits 1..8 (default 4)]
// --cal-data replaces the corpus mix with a self-sampled trace (exllamav3's -cd, notes/optimize-recipes.md §1.7):
// int64 input_ids [rows, cols] and lengths [rows]; each row runs at its own length and only its real tokens enter
// the Hessians.
//
// The loop (notes/gemma4-forward.md §3, exl3-algorithm.md §1): the calibration rows' hidden states stay on the card
// (250 x 2048 x 5376 fp32 = 11 GB). For each decoder layer: load it (BF16 -> fp16), run every row through the
// original layer while accumulating one Hessian per input group, quantize the layer's linears (exl3/quantize.hpp),
// dequantize the stored result and run every row through that to get the next layer's input. Each finished layer
// is written to <work>/layer_NN.safetensors at once.
//
// Queue turns: with --max-minutes, after a layer that ends past the limit the state is saved to <work>/state.bin
// and the process exits 99 ("more to do"); run the same command again (b70q submit --chunked does) to continue.
// When every layer is done it writes the output model: shards, index, config.json with quantization_config,
// quantization_config.json, and the source's other files. Exit 0.
//
// Output = exllamav3 1.5.3 with -hb 16 -vb 16, out_scales always, mul1: every decoder linear EXL3 at K bits, the
// embedding / norms / layer scalars / vision tower copied as BF16, lm_head an fp16 copy of the tied embedding.
// Except: output scales are exllamav3 1.6.1's "auto" by default (the MLP's gate and up projections drop theirs where
// the calibration's sensitivity predicts a lower error without them, exl3/quantize.hpp); --out-scales always is 1.5.3's.
#include "arch/gemma4.hpp"
#include "exl3/dequant.hpp"
#include "exl3/quantize.hpp"
#include "io/caldata.hpp"
#include "io/safetensors.hpp"
#include "io/st_writer.hpp"
#include "io/tokenizer.hpp"
#include "nn/ops.hpp"
#include "rt/rt.hpp"

#include <oneapi/mkl/blas.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using exl3::DBuf;
using half = sycl::half;
using namespace g4;

constexpr const char* VERSION = "exl3conv 0.1";

struct Args
{
    fs::path in, out, work, cal_dir;
    double bits = 4.0;
    int head_bits = 16;
    bool head_bits_set = false;     // --head-bits given (qwen3_5's default is 6, the others' 16)
    int mtp_bits = 4;               // --mtp-bits: qwen3_5's MTP block (exllamav3's default 4)
    int rows = 250, cols = 2048;
    double max_minutes = 0;
    int layers = -1;
    bool compile_only = false;
    bool out_scales_auto = true;
    fs::path recipe;                // --recipe: per-tensor bits (see read_recipe)
    fs::path cal_data;              // --cal-data: calibration rows from a trace (tools/make_trace.py pack) instead of the corpus mix    // --out-scales auto (exllamav3 1.6.1's default); --out-scales always is 1.5.3's
    std::string group_bits;         // --group-bits: Ripple's bits per Hessian group, over -b
    bool scaled_hessian = false;    // --scaled-hessian on: one Hessian factorization per tensor with its input scales (exllamav3's default since 1.6.6)
};
static bool g_scaled_hessian = false;

static void usage()
{
    std::fprintf(stderr, "usage: exl3conv -i <hf dir> -o <out dir> [-b 4] [-w <work dir>] [--max-minutes M] [--rows R] "
                         "[--cols C] [--cal-dir D | --cal-data F.safetensors] [--layers N] [--out-scales auto|always] [--scaled-hessian on|off] [--recipe R.json] [--compile-only] "
                         "[--head-bits 16|3..8] [--group-bits qkg=K,v=K,o=K,gu=K,d=K] [--mtp-bits K]\n");
    std::exit(2);
}

// ---------------------------------------------------------------------------------------------------------------
// Quantize one layer: returns the output tensors of its linears, and replaces L's weights by the dequantized ones.

struct LinSpec { std::string key; Lin* lin; };

// Bits per input group of a layer: q/k/v, o, gate/up, down (one K per group: the B70 server fuses q/k/v and gate/up
// and needs one K across each).
struct LayerK { int qkv = 4, o = 4, gu = 4, d = 4; };

// --recipe R.json: {"bits": {"<linear key>": K, ...}}, keys as in the checkpoint without ".weight"
// ("model.language_model.layers.12.mlp.down_proj"); linears it doesn't name get -b. One K per group is required.
static std::vector<LayerK> read_recipe(const fs::path& path, const G4& g, int K)
{
    std::vector<LayerK> lks(g.L, LayerK{K, K, K, K});
    if (path.empty()) return lks;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read recipe " + path.string());
    const json r = json::parse(f);
    const auto& bits = r.at("bits");
    size_t used = 0;
    for (int64_t li = 0; li < g.L; ++li)
    {
        const auto p = g.lp(li);
        auto get = [&](const std::vector<std::string>& names, int& out) {
            int k = -1;
            for (const auto& n : names)
            {
                if (!bits.contains(p + n)) continue;
                const int v = bits.at(p + n).get<int>();
                if (v < 1 || v > 8) throw std::runtime_error("recipe: " + p + n + ": bits must be an integer 1..8");
                if (k >= 0 && v != k) throw std::runtime_error("recipe: " + p + names[0] + " and its group differ (" + std::to_string(k) + " vs " + std::to_string(v) + "); the group shares one K");
                k = v; ++used;
            }
            if (k >= 0) out = k;
        };
        get({"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}, lks[li].qkv);
        get({"self_attn.o_proj"}, lks[li].o);
        get({"mlp.gate_proj", "mlp.up_proj"}, lks[li].gu);
        get({"mlp.down_proj"}, lks[li].d);
    }
    if (used != bits.size()) throw std::runtime_error("recipe: " + std::to_string(bits.size() - used) + " keys name no decoder linear of this model");
    return lks;
}

static std::vector<exl3::QuantOut> quantize_layer(sycl::queue& q, const G4& g, int64_t li, Layer& L, Capture& cap, const LayerK& lk,
                                                  const DBuf<int>& d_perm, bool out_scales_auto)
{
    const auto p = g.lp(li);
    std::vector<exl3::QuantOut> all;
    // --out-scales auto: per intermediate channel, the mean squared slope of gelu(g) * u along u and along g, times
    // the energy of the channel's row of the (still unquantized) down projection (finalize_out_sensitivity).
    std::vector<float> sens_u, sens_g;
    if (out_scales_auto && cap.n_sens > 0)
    {
        const int64_t I = g.I, H = g.H;
        DBuf<float> de(q, I);
        const half* wd = L.d.w.get();
        float* dp = de.get();
        q.parallel_for(sycl::range<1>(I), [=](sycl::id<1> j) {
            float s = 0.0f;
            for (int64_t h = 0; h < H; ++h) { const float v = (float) wd[h * I + j[0]]; s += v * v; }
            dp[j] = s;
        }).wait();
        std::vector<float> down(I);
        std::vector<double> su(I), sg(I);
        de.download(down.data(), I); cap.sens_u.download(su.data(), I); cap.sens_g.download(sg.data(), I);
        sens_u.resize(I); sens_g.resize(I);
        for (int64_t j = 0; j < I; ++j)
        {
            sens_u[j] = (float) (su[j] / cap.n_sens * down[j]);
            sens_g[j] = (float) (sg[j] / cap.n_sens * down[j]);
        }
    }
    auto group = [&](DBuf<float>& H, int64_t count, int64_t k, std::vector<LinSpec> specs, const char* tag, int K) {
        rt::Timer t;
        std::vector<DBuf<float>> Ws;
        std::vector<exl3::QuantIn> ins;
        for (auto& sp : specs)
        {
            DBuf<float> W(q, (size_t) sp.lin->in * sp.lin->out);
            nn::transpose_h2f(q, sp.lin->w.get(), sp.lin->out, sp.lin->in, W.get()).wait();
            const float* sens = sp.lin == &L.u && !sens_u.empty() ? sens_u.data() : sp.lin == &L.g && !sens_g.empty() ? sens_g.data() : nullptr;
            ins.push_back({sp.key, sp.lin->out, W.get(), sens});
            Ws.push_back(std::move(W));
        }
        const uint64_t seed = exl3::mix64(exl3::hash_str(p + tag));
        exl3::Hessian hs = exl3::finalize_hessian(q, std::move(H), k, count, seed, !g_scaled_hessian);
        exl3::QuantArgs qa; qa.K = K; qa.scaled_hessian = g_scaled_hessian;
        if (out_scales_auto) { qa.out_scales = false; qa.out_scales_auto = true; }
        auto outs = exl3::quantize_group(q, hs, ins, qa, seed);
        // Replace each weight by its dequantized EXL3 version (in, out).
        DBuf<float> tmp(q, 1);
        for (size_t i = 0; i < specs.size(); ++i)
        {
            auto& o = outs[i];
            Lin& l = *specs[i].lin;
            DBuf<uint16_t> tr(q, o.trellis.size()), su(q, o.suh.size()), sv(q, o.svh.size());
            tr.upload(o.trellis.data(), o.trellis.size()); su.upload(o.suh.data(), o.suh.size()); sv.upload(o.svh.data(), o.svh.size());
            if (tmp.n < (size_t) (o.k * o.n)) tmp = DBuf<float>(q, (size_t) (o.k * o.n));
            DBuf<half> w(q, (size_t) (o.k * o.n));
            exl3::dequant_linear(q, tr.get(), su.get(), sv.get(), o.k, o.n, K, exl3::CB_MUL1, d_perm.get(), tmp.get(), w.get());
            l.w = std::move(w); l.hf = false;
            std::printf("    %-34s %6lld x %-6lld  g %.3f  err %.6f -> %.6f", o.key.c_str() + p.size(), (long long) o.k,
                        (long long) o.n, o.g_scale, o.err_before, o.err_after);
            if (o.out_ratio >= 0) std::printf("  out scales %s (loss ratio %.4f)", o.out_scaled ? "on" : "off", o.out_ratio);
            std::printf("\n");
            all.push_back(std::move(o));
        }
        std::printf("    [%s: K=%d, %.1f s, Hessian over %lld rows%s]\n", tag, K, t.s(), (long long) count, hs.retries ? ", damping retried" : "");
        std::fflush(stdout);
    };
    std::vector<LinSpec> qkv = {{p + "self_attn.q_proj", &L.q}, {p + "self_attn.k_proj", &L.k}};
    if (!(L.full && g.k_eq_v)) qkv.push_back({p + "self_attn.v_proj", &L.v});
    group(cap.qkv, cap.n_qkv, g.H, qkv, "attn.input", lk.qkv);
    group(cap.o, cap.n_o, g.NH * L.D, {{p + "self_attn.o_proj", &L.o}}, "attn.o", lk.o);
    group(cap.gu, cap.n_gu, g.H, {{p + "mlp.gate_proj", &L.g}, {p + "mlp.up_proj", &L.u}}, "mlp.input", lk.gu);
    group(cap.d, cap.n_d, g.I, {{p + "mlp.down_proj", &L.d}}, "mlp.down", lk.d);
    return all;
}

// The stored tensors of one layer: EXL3 linears plus the BF16 norms and scalar from the source.
static std::vector<st::OutTensor> layer_tensors(const st::Model& m, const G4& g, int64_t li, std::vector<exl3::QuantOut>& outs)
{
    std::vector<st::OutTensor> ts;
    const auto p = g.lp(li);
    auto copy = [&](const std::string& name) {
        const auto& t = m.get(name);
        ts.push_back(st::OutTensor::from_vector(name, st::dtype_name(t.dtype), t.shape, std::vector<uint8_t>(t.data, t.data + t.nbytes)));
    };
    copy(p + "input_layernorm.weight");
    for (auto& o : outs)
    {
        const int32_t marker = (int32_t) exl3::MUL1_MULT;
        ts.push_back(st::OutTensor::from_vector(o.key + ".suh", "F16", {o.k}, std::move(o.suh)));
        ts.push_back(st::OutTensor::from_vector(o.key + ".svh", "F16", {o.n}, std::move(o.svh)));
        ts.push_back(st::OutTensor::from_vector(o.key + ".mul1", "I32", {}, std::vector<int32_t>{marker}));
        ts.push_back(st::OutTensor::from_vector(o.key + ".trellis", "I16", {o.k / 16, o.n / 16, 16 * o.K}, std::move(o.trellis)));
    }
    for (const char* n : {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight",
                          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar"})
        copy(p + n);
    return ts;
}

// ---------------------------------------------------------------------------------------------------------------
// Output model

// An fp16 copy of a BF16 tensor, streamed in pieces.
static st::OutTensor fp16_copy(const std::string& name, const st::TensorView& e)
{
    const uint16_t* src = (const uint16_t*) e.data;
    const uint64_t n = (uint64_t) e.numel();
    return {name, "F16", e.shape, n * 2, [src, n](std::ostream& o) {
        std::vector<uint16_t> buf(1 << 20);
        for (uint64_t i = 0; i < n; i += buf.size())
        {
            const uint64_t c = std::min<uint64_t>(buf.size(), n - i);
            for (uint64_t j = 0; j < c; ++j) buf[j] = exl3::f16_from_float(st::bf16_to_float(src[i + j]));
            o.write((const char*) buf.data(), (std::streamsize) (c * 2));
        }
    }};
}

// The output model from the tensors of the first shard, the finished layer files (moved into place) and the tensors of the
// last shard: shards, index, config.json with quantization_config (plus qc_extra), quantization_config.json, the source's
// other files. L: progress.json's next_layer from now on.
static void write_model(const Args& a, const std::vector<std::unique_ptr<st::File>>& layer_files, const std::vector<st::OutTensor>& first,
                        const std::vector<st::OutTensor>& last, int K, int64_t L, const json& qc_extra)
{
    fs::create_directories(a.out);
    std::vector<st::OutTensor> all(first);
    for (const auto& f : layer_files)
        for (const auto& name : f->order())
        {
            const auto& t = f->tensors().at(name);
            all.push_back(st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes));
        }
    for (const auto& t : last) all.push_back(t);

    // Shards: [embedding], one per layer (the work folder's layer files, moved into place, not copied: for large
    // models the copy would need the output's size twice on disk), [norm, lm_head, vision].
    const size_t nlayers = layer_files.size();
    const size_t nshards = nlayers + 2;
    auto shard_name = [&](size_t i) { char fn[64]; std::snprintf(fn, sizeof fn, "model-%05zu-of-%05zu.safetensors", i + 1, nshards); return std::string(fn); };
    json index = {{"metadata", {{"total_size", 0}}}, {"weight_map", json::object()}};
    uint64_t total = 0;
    {
        rt::Timer t;
        st::write_file(a.out / shard_name(0), first);
        for (auto& x : first) { index["weight_map"][x.name] = shard_name(0); total += x.nbytes; }
        std::printf("  wrote %s (%.1f s)\n", shard_name(0).c_str(), t.s());
    }
    for (size_t li = 0; li < nlayers; ++li)
    {
        const auto& f = *layer_files[li];
        for (const auto& name : f.order()) { index["weight_map"][name] = shard_name(li + 1); total += f.tensors().at(name).nbytes; }
        const fs::path dst = a.out / shard_name(li + 1);
        std::error_code ec;
        fs::rename(f.path(), dst, ec);
        if (ec) { fs::copy_file(f.path(), dst, fs::copy_options::overwrite_existing); fs::remove(f.path()); }   // other disk
    }
    std::printf("  moved %zu layer files into place\n", nlayers);
    {
        rt::Timer t;
        st::write_file(a.out / shard_name(nshards - 1), last);
        for (auto& x : last) { index["weight_map"][x.name] = shard_name(nshards - 1); total += x.nbytes; }
        std::printf("  wrote %s (%zu tensors, %.1f s)\n", shard_name(nshards - 1).c_str(), last.size(), t.s());
    }
    index["metadata"]["total_size"] = total;
    std::ofstream(a.out / "model.safetensors.index.json") << index.dump(4);
    { std::ofstream pf(a.work / "progress.json"); pf << json{{"next_layer", L}, {"compiled", true}}.dump(); }

    // config.json with quantization_config; quantization_config.json; other files copied.
    std::ifstream cf(a.in / "config.json");
    json cfg = json::parse(cf);
    // Average bits over the quantized linears, weighted by their sizes (the trellis is k/16 x n/16 x 16K int16).
    double wsum = 0, bsum = 0;
    for (auto& t : all)
        if (t.name.size() > 8 && t.name.compare(t.name.size() - 8, 8, ".trellis") == 0)
        {
            const double w = (double) t.shape[0] * t.shape[1] * 256;
            wsum += w; bsum += w * (double) (t.shape[2] / 16);
        }
    const double avg_bits = wsum > 0 ? std::round(bsum / wsum * 1000) / 1000 : (double) K;
    json qc = {{"quant_method", "exl3"}, {"version", "1.5.3"}, {"bits", avg_bits}, {"head_bits", a.head_bits},
               {"calibration", {{"rows", a.rows}, {"cols", a.cols}}}, {"out_scales", a.out_scales_auto ? "auto" : "always"}, {"codebook", "mul1"},
               {"scaled_hessian", a.scaled_hessian}, {"quantizer", VERSION}};
    qc.update(qc_extra);
    if (cfg.contains("quantization_config") && cfg["quantization_config"].value("quant_method", "") != "exl3")
        cfg["original_quantization_config"] = cfg["quantization_config"];
    cfg["quantization_config"] = qc;
    std::ofstream(a.out / "config.json") << cfg.dump(4);
    json qfile = qc;
    json storage = json::object();
    for (auto& t : all)
    {
        const auto dot = t.name.rfind('.');
        const std::string mod = t.name.substr(0, dot), leaf = t.name.substr(dot + 1);
        static const std::map<std::string, std::string> tn = {{"F16", "torch.float16"}, {"BF16", "torch.bfloat16"}, {"F32", "torch.float32"},
                                                              {"I16", "torch.int16"}, {"I32", "torch.int32"}};
        storage[mod]["stored_tensors"][t.name] = {{"shape", t.shape}, {"n_bytes", t.nbytes}, {"dtype", tn.count(t.dtype) ? tn.at(t.dtype) : t.dtype}};
        if (leaf == "trellis") { storage[mod]["quant_format"] = "exl3"; storage[mod]["bits_per_weight"] = (int) (t.shape[2] / 16); storage[mod]["mul1_multiplier"] = exl3::MUL1_MULT; }
    }
    qfile["tensor_storage"] = storage;
    std::ofstream(a.out / "quantization_config.json") << qfile.dump(4);
    for (auto& e : fs::directory_iterator(a.in))
    {
        if (!e.is_regular_file()) continue;
        const auto n = e.path().filename().string(), ext = e.path().extension().string();
        if (ext == ".safetensors" || ext == ".gguf" || n == "config.json" || n == "model.safetensors.index.json" || ext == ".bin" || ext == ".pt" ||
            ext == ".pth" || ext == ".ckpt")
            continue;
        fs::copy_file(e.path(), a.out / n, fs::copy_options::overwrite_existing);
    }
}

static void compile_output(const Args& a, const st::Model& m, const G4& g, int K)
{
    std::vector<std::unique_ptr<st::File>> layer_files;
    auto copy_src = [&](const std::string& name) {
        const auto& t = m.get(name);
        return st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes);
    };
    for (int64_t li = 0; li < g.L; ++li)
    {
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        layer_files.push_back(std::make_unique<st::File>(a.work / fn));
    }
    std::vector<st::OutTensor> first = {copy_src(g.prefix + "embed_tokens.weight")};
    std::vector<st::OutTensor> last = {copy_src(g.prefix + "norm.weight")};
    // lm_head: fp16 copy of the tied embedding, [vocab, hidden].
    last.push_back(fp16_copy("lm_head.weight", m.get(g.prefix + "embed_tokens.weight")));
    for (const auto& name : m.names())
        if (name.rfind("model.vision_tower.", 0) == 0 || name.rfind("model.embed_vision.", 0) == 0) last.push_back(copy_src(name));
    write_model(a, layer_files, first, last, K, g.L, json::object());
}

// --cal-data: the trace's rows, each at its own length. Its row count and width replace a.rows and a.cols.
static std::vector<std::vector<int32_t>> read_trace_rows(Args& a, int64_t vocab)
{
    std::vector<std::vector<int32_t>> trace_rows;
    if (a.cal_data.empty()) return trace_rows;
    st::File cf2(a.cal_data);
    const auto& ids = cf2.tensors().at("input_ids");
    if (ids.dtype != st::DType::I64 || ids.shape.size() != 2) throw std::runtime_error("--cal-data: input_ids must be int64 [rows, cols]");
    const int64_t n = ids.shape[0], c = ids.shape[1];
    const int64_t* p = (const int64_t*) ids.data;
    const int64_t* lp = cf2.tensors().count("lengths") ? (const int64_t*) cf2.tensors().at("lengths").data : nullptr;
    if (n < a.rows) std::printf("note: --cal-data has %lld rows, fewer than --rows %d\n", (long long) n, a.rows);
    a.rows = (int) std::min<int64_t>(n, a.rows); a.cols = (int) c;
    for (int64_t r = 0; r < a.rows; ++r)
    {
        const int64_t len = lp ? std::min<int64_t>(lp[r], c) : c;
        if (len < 1) throw std::runtime_error("--cal-data: empty row " + std::to_string(r));
        std::vector<int32_t> row(c, 0);
        for (int64_t t = 0; t < len; ++t)
        {
            if (p[r * c + t] < 0 || p[r * c + t] >= vocab) throw std::runtime_error("--cal-data: token id out of range");
            row[t] = (int32_t) p[r * c + t];
        }
        row.resize(len);
        trace_rows.push_back(std::move(row));
    }
    return trace_rows;
}

#include "convert_qwen4exp.hpp"
#include "convert_ripple.hpp"
#include "convert_qwen35.hpp"

// ---------------------------------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (s == "-i") a.in = next(); else if (s == "-o") a.out = next(); else if (s == "-w") a.work = next();
        else if (s == "-b") a.bits = std::stod(next()); else if (s == "--max-minutes") a.max_minutes = std::stod(next());
        else if (s == "--rows") a.rows = std::stoi(next()); else if (s == "--cols") a.cols = std::stoi(next());
        else if (s == "--cal-dir") a.cal_dir = next(); else if (s == "--layers") a.layers = std::stoi(next());
        else if (s == "--compile-only") a.compile_only = true; else if (s == "--recipe") a.recipe = next();
        else if (s == "--cal-data") a.cal_data = next();
        else if (s == "--head-bits") { a.head_bits = std::stoi(next()); a.head_bits_set = true; } else if (s == "--group-bits") a.group_bits = next();
        else if (s == "--mtp-bits") a.mtp_bits = std::stoi(next());
        else if (s == "--scaled-hessian") { const auto v = next(); if (v != "on" && v != "off") usage(); a.scaled_hessian = g_scaled_hessian = v == "on"; }
        else if (s == "--out-scales") { const auto v = next(); if (v != "always" && v != "auto") usage(); a.out_scales_auto = v == "auto"; }
        else usage();
    }
    if (a.in.empty() || a.out.empty()) usage();
    if (a.work.empty()) a.work = a.out.string() + ".work";
    if (a.cal_dir.empty())
        a.cal_dir = fs::path(argv[0]).parent_path() / "../../dsv41/exllamav3/exllamav3/conversion/standard_cal_data";
    const int K = (int) a.bits;
    if (K != a.bits || K < 1 || K > 8) { std::fprintf(stderr, "only integer bits 1..8 for now\n"); return 2; }
    try
    {
        rt::Timer total;
        std::ifstream cf(a.in / "config.json");
        if (!cf) throw std::runtime_error("no config.json in " + a.in.string());
        const json cfg = json::parse(cf);
        if (cfg.value("model_type", "") == "qwen4_exp")
        {
            if (K != 4) std::printf("note: qwen4_exp uses its fixed recipe (DESIGN-qwen4exp.md); -b is ignored\n");
            qx::QxArgs qa{a.in, a.out, a.work, a.cal_dir, a.rows, a.cols, a.max_minutes, a.layers};
            return qx::convert(qa, cfg);
        }
        if (cfg.value("model_type", "") == "ricursive") return rpc::convert(a, cfg, K);
        if (cfg.value("model_type", "") == "qwen3_5" || cfg.value("model_type", "") == "qwen3_5_text") return q35c::convert(a, cfg, K);
        if (a.head_bits != 16 || !a.group_bits.empty()) throw std::runtime_error("--head-bits and --group-bits are for Ripple only");
        const G4 g(cfg);
        st::Model m(a.in);
        fs::create_directories(a.work);
        const auto lks = read_recipe(a.recipe, g, K);
        auto layer_k = [&](int64_t li) { return lks[li]; };
        {
            // The recipe (or its absence) is part of the work dir's identity: a resume must use the same one.
            std::string h = "none";
            if (!a.recipe.empty()) { std::ifstream rf(a.recipe); std::stringstream rs; rs << rf.rdbuf(); h = std::to_string(exl3::hash_str(rs.str())); }
            if (!a.cal_data.empty()) h += "|cal:" + std::to_string(fs::file_size(a.cal_data)) + ":" + a.cal_data.filename().string();
            const auto hp = a.work / "recipe.hash";
            if (fs::exists(hp)) { std::ifstream hf(hp); std::string old; std::getline(hf, old); if (old != h) throw std::runtime_error("this work dir was started with a different --recipe or --cal-data"); }
            else std::ofstream(hp) << h;
        }
        // Calibration rows from a trace: their count and width come from the file, each row has its own length.
        const std::vector<std::vector<int32_t>> trace_rows = read_trace_rows(a, g.vocab);
        const int64_t T = a.cols, R = a.rows;
        std::vector<int64_t> rlen(R, T);
        for (size_t r = 0; r < trace_rows.size(); ++r) rlen[r] = (int64_t) trace_rows[r].size();
        int64_t cal_tokens = 0;
        for (auto l : rlen) cal_tokens += l;
        std::printf("%s: %s, %lld layers, hidden %lld, %d-bit EXL3%s, calibration %lld rows x %lld (%lld tokens%s)\n", VERSION,
                    a.in.filename().c_str(), (long long) g.L, (long long) g.H, K, a.recipe.empty() ? "" : " (recipe)", (long long) R,
                    (long long) T, (long long) cal_tokens, a.cal_data.empty() ? "" : (", trace " + a.cal_data.filename().string()).c_str());

        // Progress.
        int64_t start = 0;
        bool compiled = false;
        const auto prog = a.work / "progress.json";
        if (fs::exists(prog))
        {
            std::ifstream pf(prog);
            const auto pj = json::parse(pf);
            start = pj.value("next_layer", 0); compiled = pj.value("compiled", false);
            const std::string os = pj.value("out_scales", "always");
            if (!compiled && start > 0 && os != (a.out_scales_auto ? "auto" : "always"))
                throw std::runtime_error("this work dir was started with --out-scales " + os + "; resume with the same setting");
        }
        if (compiled) { std::printf("%s is already complete (the layer files were moved into it)\n", a.out.c_str()); return 0; }
        const int64_t stop = a.layers > 0 ? std::min<int64_t>(g.L, a.layers) : g.L;
        if (a.compile_only || start >= g.L)
        {
            std::printf("compiling the output model\n");
            compile_output(a, m, g, K);
            std::printf("done in %.1f min\n", total.s() / 60);
            return 0;
        }

        // progress.json's next_layer is where state.bin resumes, so it only moves with state.bin (a turn's end) or once
        // every layer file is written; a kill in between redoes the layers since (their files are overwritten).
        auto save_progress = [&](int64_t next) {
            { std::ofstream pf(prog.string() + ".tmp"); pf << json{{"next_layer", next}, {"out_scales", a.out_scales_auto ? "auto" : "always"}}.dump(); }
            fs::rename(prog.string() + ".tmp", prog);
        };

        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        Upload up(q);
        DBuf<int> d_perm(q, 256);
        d_perm.upload(exl3::tile_perm().data(), 256);

        // Calibration state: rows x T x H fp32 on the card.
        DBuf<float> state(q, (size_t) R * T * g.H);
        const size_t row_floats = (size_t) T * g.H;
        const auto state_file = a.work / "state.bin";
        if (start > 0)
        {
            rt::Timer t;
            std::ifstream sf(state_file, std::ios::binary);
            if (!sf) throw std::runtime_error("progress says layer " + std::to_string(start) + " but there is no state.bin");
            std::vector<float> buf(row_floats);
            for (int64_t r = 0; r < R; ++r)
            {
                sf.read((char*) buf.data(), (std::streamsize) (row_floats * 4));
                if (!sf) throw std::runtime_error("state.bin is truncated");
                up(state.get() + r * row_floats, buf.data(), row_floats * 4);
            }
            std::printf("resumed at layer %lld (state loaded in %.1f s)\n", (long long) start, t.s());
        }
        else
        {
            rt::Timer t;
            tok::BPE tk((a.in / "tokenizer.json").string());
            auto rows = trace_rows.empty() ? cal::default_rows(tk, a.cal_dir, (int) R, (int) T, g.bos, g.eos, (int) g.vocab) : trace_rows;
            for (auto& r : rows) r.resize(T, 0);   // trace rows: padding past their length, never read
            const auto& e = m.get(g.prefix + "embed_tokens.weight");
            const uint16_t* E = (const uint16_t*) e.data;
            const float mult = st::bf16_to_float(nn::f2bf16(std::sqrt((float) g.H)));
            std::vector<uint16_t> gath(row_floats);
            DBuf<uint16_t> dg(q, row_floats);
            for (int64_t r = 0; r < R; ++r)
            {
                for (int64_t t2 = 0; t2 < T; ++t2) std::memcpy(&gath[t2 * g.H], E + (int64_t) rows[r][t2] * g.H, g.H * 2);
                up(dg.get(), gath.data(), row_floats * 2);
                nn::embed(q, dg.get(), T, g.H, mult, state.get() + r * row_floats).wait();
            }
            std::printf("calibration rows tokenized and embedded in %.1f s (embedding scale %.4f)\n", t.s(), mult);
        }

        Scratch s(q, g, T);
        DBuf<uint16_t> wscratch(q, 1);
        DBuf<float> xt(q, row_floats);
        const auto inv_s = g.inv_freq(false), inv_f = g.inv_freq(true);
        DBuf<float> d_inv_s(q, inv_s.size()), d_inv_f(q, inv_f.size());
        d_inv_s.upload(inv_s.data(), inv_s.size()); d_inv_f.upload(inv_f.data(), inv_f.size());
        const int nref = (int) std::min<int64_t>(5, R);
        std::vector<DBuf<float>> refs;
        for (int i = 0; i < nref; ++i) refs.emplace_back(q, row_floats);

        rt::Timer chunk;
        for (int64_t li = start; li < stop; ++li)
        {
            rt::Timer tl;
            std::printf("layer %lld/%lld (%s attention)\n", (long long) li, (long long) g.L, g.full[li] ? "full" : "sliding");
            Layer L = load_layer(q, m, g, li, up, wscratch);
            const float* inv = L.full ? d_inv_f.get() : d_inv_s.get();
            // Capture.
            Capture cap;
            cap.qkv = DBuf<float>(q, (size_t) g.H * g.H); cap.qkv.zero();
            cap.o = DBuf<float>(q, (size_t) (g.NH * L.D) * (g.NH * L.D)); cap.o.zero();
            cap.gu = DBuf<float>(q, (size_t) g.H * g.H); cap.gu.zero();
            cap.d = DBuf<float>(q, (size_t) g.I * g.I); cap.d.zero();
            if (a.out_scales_auto) { cap.sens_u = DBuf<double>(q, g.I); cap.sens_u.zero(); cap.sens_g = DBuf<double>(q, g.I); cap.sens_g.zero(); }
            rt::Timer tc;
            for (int64_t r = 0; r < R; ++r)
            {
                q.memcpy(xt.get(), state.get() + r * row_floats, row_floats * 4);
                forward_layer(q, g, L, xt.get(), rlen[r], s, inv, &cap);
                if (r < nref) q.memcpy(refs[r].get(), xt.get(), row_floats * 4).wait();
            }
            const int64_t full_n = cal_tokens;
            std::printf("  capture %.1f s%s\n", tc.s(),
                        (cap.n_qkv + cap.n_o + cap.n_gu + cap.n_d < 4 * full_n) ? "  (some rows had non-finite activations and were left out)" : "");
            // Quantize.
            auto outs = quantize_layer(q, g, li, L, cap, layer_k(li), d_perm, a.out_scales_auto);
            // Advance through the quantized layer.
            rt::Timer ta;
            double rfn = 0;
            for (int64_t r = 0; r < R; ++r)
            {
                forward_layer(q, g, L, state.get() + r * row_floats, rlen[r], s, inv, nullptr);
                if (r < nref)
                {
                    std::vector<float> a1(row_floats), b1(row_floats);
                    q.memcpy(a1.data(), state.get() + r * row_floats, row_floats * 4).wait();
                    q.memcpy(b1.data(), refs[r].get(), row_floats * 4).wait();
                    double num = 0, den = 0;
                    for (size_t i = 0; i < (size_t) rlen[r] * g.H; ++i) { const double d = a1[i] - b1[i]; num += d * d; den += (double) b1[i] * b1[i]; }
                    rfn += std::sqrt(num / std::max(den, 1e-30)) / nref;
                }
            }
            std::printf("  advance %.1f s; layer output vs unquantized (rows 0-%d): rfn %.5f\n", ta.s(), nref - 1, rfn);
            // Write the layer.
            char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
            st::write_file(a.work / fn, layer_tensors(m, g, li, outs));
            std::printf("  layer done in %.1f s (total %.1f min)\n", tl.s(), total.s() / 60);
            std::fflush(stdout);
            if (li + 1 < stop && a.max_minutes > 0 && chunk.s() > a.max_minutes * 60)
            {
                rt::Timer t;
                std::ofstream sf(state_file.string() + ".tmp", std::ios::binary);
                std::vector<float> buf(row_floats);
                for (int64_t r = 0; r < R; ++r)
                {
                    up.download(buf.data(), state.get() + r * row_floats, row_floats * 4);
                    sf.write((const char*) buf.data(), (std::streamsize) (row_floats * 4));
                }
                sf.close();
                fs::rename(state_file.string() + ".tmp", state_file);
                save_progress(li + 1);
                std::printf("turn over after layer %lld: state saved in %.1f s; run again to continue\n", (long long) li, t.s());
                return 99;
            }
        }
        if (stop < g.L) { std::printf("stopped after %lld layers (--layers)\n", (long long) stop); return 0; }
        save_progress(g.L);
        std::printf("all layers quantized; compiling the output model\n");
        compile_output(a, m, g, K);
        fs::remove(state_file);
        std::printf("done in %.1f min\n", total.s() / 60);
        return 0;
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
