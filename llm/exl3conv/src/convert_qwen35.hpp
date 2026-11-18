// exl3conv: the conversion loop for Qwen3.5 / Qwen3.8 dense (qwen3_5, e.g. Qwen3.8-27B and the 40B upscales). Included by
// convert.cpp, which dispatches on model_type. The output is exllamav3's layout for qwen3_5 (architecture/qwen3_5.py,
// qwen3_5_mtp.py), checked against turboderp/Qwen3.8-27B-exl3:
//
//   decoder linears   EXL3 at -b K: GDN in_proj_qkv, in_proj_z, out_proj; attention q/k/v/o; MLP gate/up/down
//   in_proj_a, _b     F16 (exllamav3 keeps them unquantized, qmap None)
//   norms, conv1d, A_log, dt_bias, embed_tokens   BF16 copies
//   lm_head           EXL3 at --head-bits (default 6), calibrated on the final norm's output over every row
//   mtp.*             EXL3 at --mtp-bits (default 4), uncalibrated (exllamav3 quantizes the MTP model as a side model without
//                     a Hessian); its norms BF16
//   vision tower      not written (text only)
//
// The loop is Gemma's (convert.cpp) with Ripple's checks (plan, VRAM ledger, disk): the 250 rows' hidden states (10.5 GB for
// hidden 5120) stay on the card; per layer capture through the original layer, quantize, dequantize, advance through the
// quantized layer. Then, each resumable from <work>/progress.json:
//   head     the Hessian of the final norm's output (saved to <work>/head_H.bin, so the state is no longer needed), then
//            lm_head through exl3/quantize_wide.hpp (a 248,320-wide head does not fit quantize_group's working set)
//   mtp      the MTP block's linears with identity Hessians -> <work>/mtp.safetensors
//   compile  [embedding], one shard per layer (moved from the work folder), [norm, lm_head, mtp.*]; index, config.json,
//            quantization_config.json, the source's other files
// --layers N stops at N and writes a model of N layers (quantization_config "layers": N), for format checks on parts of a
// checkpoint.
#pragma once

#include "arch/qwen35.hpp"
#include "exl3/quantize_wide.hpp"

#include <sys/stat.h>

namespace q35c {

using q35::Config; using q35::Layer; using q35::Capture; using q35::Scratch; using q35::Lin; using q35::Upload;
using exl3::DBuf;
using half = sycl::half;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double GiB = 1073741824.0;

struct Progress
{
    fs::path path;
    json j = json::object();
    explicit Progress(fs::path p) : path(std::move(p)) { if (fs::exists(path)) { std::ifstream f(path); j = json::parse(f); } }
    void save() { const auto t = path.string() + ".tmp"; { std::ofstream f(t); f << j.dump(); } fs::rename(t, path); }
};

inline uint64_t linear_bytes(uint64_t k, uint64_t n, int K) { return k * n * K / 8 + 2 * (k + n) + 4; }

// ---------------------------------------------------------------------------------------------------------------
// Quantization of a layer's groups.

struct Groups { std::vector<rpc::LinSpec> in, o, gu, d; std::string tag_in, tag_o; };

inline void quantize_layer(sycl::queue& q, const Config& c, const std::string& p, Layer& L, Capture& cap, int K, const DBuf<int>& d_perm,
                           bool out_scales_auto, std::vector<exl3::QuantOut>& all)
{
    // --out-scales auto: per intermediate channel, the mean squared slope along up and along gate, times the energy of the channel's
    // row of the (still unquantized) down projection.
    std::vector<float> sens_u, sens_g;
    if (out_scales_auto && cap.n_sens > 0)
    {
        const int64_t I = c.I, D = c.D;
        DBuf<float> de(q, I);
        const half* wd = L.d.w.get();
        float* dp = de.get();
        q.parallel_for(sycl::range<1>(I), [=](sycl::id<1> j) {
            float s = 0.0f;
            for (int64_t h = 0; h < D; ++h) { const float v = (float) wd[h * I + j[0]]; s += v * v; }
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
    auto group = [&](DBuf<float>& H, int64_t count, int64_t k, std::vector<rpc::LinSpec> specs, const char* tag) {
        rpc::quant_group(q, H, count, k, specs, p, tag, K, d_perm, out_scales_auto, all);
    };
    if (L.full)
    {
        const auto a = p + "self_attn.";
        group(cap.in, cap.n_in, c.D, {{a + "q_proj", &L.q}, {a + "k_proj", &L.k}, {a + "v_proj", &L.v}}, "block.attn.input");
        group(cap.o, cap.n_o, c.NH * c.Dh, {{a + "o_proj", &L.o}}, "block.attn.output");
    }
    else
    {
        const auto a = p + "linear_attn.";
        group(cap.in, cap.n_in, c.D, {{a + "in_proj_qkv", &L.qkv}, {a + "in_proj_z", &L.z}}, "block.attn.input");
        group(cap.o, cap.n_o, c.HV * c.DV, {{a + "out_proj", &L.out}}, "block.attn.output");
    }
    group(cap.gu, cap.n_gu, c.D, {{p + "mlp.gate_proj", &L.g, sens_g.empty() ? nullptr : sens_g.data()},
                                  {p + "mlp.up_proj", &L.u, sens_u.empty() ? nullptr : sens_u.data()}}, "block.mlp.input");
    group(cap.d, cap.n_d, c.I, {{p + "mlp.down_proj", &L.d}}, "block.mlp.down");
}

// The stored tensors of a block (a decoder layer, or the MTP block with prefix "mtp.layers.0."), from the quantized linears in outs
// and the source's small tensors.
inline std::vector<st::OutTensor> block_tensors(const st::Model& m, const std::string& p, bool full, std::vector<exl3::QuantOut>& outs)
{
    std::vector<st::OutTensor> ts;
    std::map<std::string, exl3::QuantOut*> by;
    for (auto& o : outs) by[o.key] = &o;
    auto bf16 = [&](const std::string& n, std::vector<int64_t> shape = {}) {
        const auto& t = m.get(n);
        ts.push_back(st::OutTensor::from_vector(n, "BF16", shape.empty() ? t.shape : shape, q35::host_bf16(m, n)));
    };
    auto f16 = [&](const std::string& n) { ts.push_back(st::OutTensor::from_vector(n, "F16", m.get(n).shape, q35::host_f16(m, n))); };
    auto lin = [&](const std::string& key) {
        auto it = by.find(key);
        if (it == by.end()) throw std::runtime_error("no quantized tensor for " + key);
        rpc::push_exl3(ts, *it->second);
    };
    bf16(p + "input_layernorm.weight");
    if (full)
    {
        const auto a = p + "self_attn.";
        lin(a + "q_proj"); lin(a + "k_proj"); lin(a + "v_proj"); lin(a + "o_proj");
        bf16(a + "q_norm.weight"); bf16(a + "k_norm.weight");
    }
    else
    {
        const auto a = p + "linear_attn.";
        const auto& cw = m.get(a + "conv1d.weight");
        const int64_t ks = cw.shape.back();
        lin(a + "in_proj_qkv"); lin(a + "in_proj_z");
        f16(a + "in_proj_b.weight"); f16(a + "in_proj_a.weight");
        lin(a + "out_proj");
        bf16(a + "A_log", {m.get(a + "A_log").numel()});
        bf16(a + "conv1d.weight", {cw.numel() / ks, 1, ks});
        bf16(a + "dt_bias", {m.get(a + "dt_bias").numel()});
        bf16(a + "norm.weight");
    }
    bf16(p + "post_attention_layernorm.weight");
    lin(p + "mlp.up_proj"); lin(p + "mlp.gate_proj"); lin(p + "mlp.down_proj");
    return ts;
}

// ---------------------------------------------------------------------------------------------------------------
// Uncalibrated quantization (the MTP block): an identity Hessian, so LDLQ has no error feedback and the refit fits plain least
// squares, which is what exllamav3's fallback (no activations) does.

inline exl3::QuantOut quantize_uncalibrated(sycl::queue& q, const std::string& key, const Lin& l, int K)
{
    const int64_t k = l.in;
    DBuf<float> H(q, (size_t) k * k);
    H.zero();
    float* h = H.get();
    q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { h[i * k + i] = 1.0f; }).wait();
    const uint64_t seed = exl3::mix64(exl3::hash_str(key));
    exl3::Hessian hs = exl3::finalize_hessian(q, std::move(H), k, 1, seed);
    DBuf<float> W(q, (size_t) l.in * l.out);
    nn::transpose_h2f(q, l.w.get(), l.out, l.in, W.get()).wait();
    exl3::QuantArgs qa; qa.K = K;
    auto outs = exl3::quantize_group(q, hs, {{key, l.out, W.get()}}, qa, seed);
    std::printf("    %-40s %6lld x %-6lld  g %.3f  err %.6f -> %.6f\n", key.c_str(), (long long) outs[0].k, (long long) outs[0].n, outs[0].g_scale,
                outs[0].err_before, outs[0].err_after);
    return std::move(outs[0]);
}

// ---------------------------------------------------------------------------------------------------------------
// Output model.

// A source tensor as a BF16 output tensor, streamed (the embedding is 2.5 GB).
inline st::OutTensor bf16_stream(const st::Model& m, const std::string& name)
{
    const auto& t = m.get(name);
    if (t.dtype == st::DType::BF16) return st::OutTensor::from_bytes(name, "BF16", t.shape, t.data, t.nbytes);
    const st::TensorView* tp = &t;
    const uint64_t n = (uint64_t) t.numel();
    return {name, "BF16", t.shape, n * 2, [tp, n](std::ostream& o) {
        std::vector<uint16_t> buf(1 << 20);
        for (uint64_t i = 0; i < n; i += buf.size())
        {
            const uint64_t c = std::min<uint64_t>(buf.size(), n - i);
            for (uint64_t j = 0; j < c; ++j) buf[j] = nn::f2bf16(q35::elem_f32(*tp, i + j));
            o.write((const char*) buf.data(), (std::streamsize) (c * 2));
        }
    }};
}

inline void compile(const Args& a, const st::Model& m, const Config& c, int64_t nl, int K, int head_bits, int mtp_bits)
{
    std::printf("compiling the output model\n");
    rt::Timer tc;
    std::vector<std::unique_ptr<st::File>> layer_files;
    uint64_t layer_bytes = 0;
    for (int64_t li = 0; li < nl; ++li)
    {
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        layer_files.push_back(std::make_unique<st::File>(a.work / fn));
        layer_bytes += fs::file_size(a.work / fn);
    }
    st::File hf(a.work / "head.safetensors");
    std::unique_ptr<st::File> mf;
    if (fs::exists(a.work / "mtp.safetensors")) mf = std::make_unique<st::File>(a.work / "mtp.safetensors");
    std::vector<st::OutTensor> first = {bf16_stream(m, c.prefix + "embed_tokens.weight")};
    std::vector<st::OutTensor> last = {bf16_stream(m, c.prefix + "norm.weight")};
    for (const st::File* f : {&hf, mf.get()})
        if (f)
            for (const auto& n : f->order())
            {
                const auto& t = f->tensors().at(n);
                last.push_back(st::OutTensor::from_bytes(n, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes));
            }
    // Room on the output's disk: the layer files are copied when the work folder is on another disk; keep 10 GB beyond that.
    fs::create_directories(a.out);
    {
        uint64_t need = 0;
        for (auto& t : first) need += t.nbytes;
        for (auto& t : last) need += t.nbytes;
        struct stat so{}, sw{};
        const bool same_fs = ::stat(a.out.c_str(), &so) == 0 && ::stat(a.work.c_str(), &sw) == 0 && so.st_dev == sw.st_dev;
        if (!same_fs) need += layer_bytes;
        const uint64_t avail = fs::space(a.out).available;
        if (avail < need + (10ull << 30))
            throw std::runtime_error("not enough disk for the output in " + a.out.string() + ": it needs " + std::to_string(need >> 30) + " GiB plus 10 GiB kept free, " +
                                     std::to_string(avail >> 30) + " GiB are free");
    }
    const size_t nshards = layer_files.size() + 2;
    auto shard = [&](size_t i) { char fn[64]; std::snprintf(fn, sizeof fn, "model-%05zu-of-%05zu.safetensors", i + 1, nshards); return std::string(fn); };
    json index = {{"metadata", {{"total_size", 0}}}, {"weight_map", json::object()}};
    uint64_t total = 0;
    nlohmann::ordered_json storage = nlohmann::ordered_json::object();
    uint64_t lin_bits = 0, lin_weights = 0;
    auto note = [&](const std::string& name, const std::string& dt, const std::vector<int64_t>& shape, uint64_t nb, const std::string& file) {
        index["weight_map"][name] = file; total += nb;
        // tensor_storage: the main model's modules only (exllamav3 lists neither mtp.* nor the GDN's own A_log / dt_bias / conv1d).
        const auto dot = name.rfind('.');
        const std::string mod = name.substr(0, dot), leaf = name.substr(dot + 1);
        if (name.rfind("mtp.", 0) == 0 || leaf == "A_log" || leaf == "dt_bias" || mod.size() >= 7 && mod.compare(mod.size() - 7, 7, ".conv1d") == 0) return;
        static const std::map<std::string, std::string> tn = {{"F16", "torch.float16"}, {"BF16", "torch.bfloat16"}, {"F32", "torch.float32"},
                                                              {"I16", "torch.int16"}, {"I32", "torch.int32"}};
        storage[mod]["stored_tensors"][name] = {{"shape", shape}, {"n_bytes", nb}, {"dtype", tn.count(dt) ? tn.at(dt) : dt}};
        if (leaf == "trellis")
        {
            storage[mod]["quant_format"] = "exl3"; storage[mod]["bits_per_weight"] = shape[2] / 16; storage[mod]["mul1_multiplier"] = exl3::MUL1_MULT;
            if (mod != "lm_head") { const uint64_t w = (uint64_t) shape[0] * 16 * shape[1] * 16; lin_weights += w; lin_bits += w * (uint64_t) (shape[2] / 16); }
        }
    };
    {
        rt::Timer t;
        st::write_file(a.out / shard(0), first);
        for (auto& x : first) note(x.name, x.dtype, x.shape, x.nbytes, shard(0));
        std::printf("  wrote %s (%.1f s)\n", shard(0).c_str(), t.s());
    }
    for (size_t li = 0; li < layer_files.size(); ++li)
    {
        const auto& f = *layer_files[li];
        for (const auto& n : f.order()) { const auto& t = f.tensors().at(n); note(n, st::dtype_name(t.dtype), t.shape, t.nbytes, shard(li + 1)); }
    }
    {
        rt::Timer t;
        st::write_file(a.out / shard(nshards - 1), last);
        for (auto& x : last) note(x.name, x.dtype, x.shape, x.nbytes, shard(nshards - 1));
        std::printf("  wrote %s (%zu tensors, %.1f s)\n", shard(nshards - 1).c_str(), last.size(), t.s());
    }
    {
        rt::Timer t;
        std::vector<fs::path> paths;
        for (const auto& f : layer_files) paths.push_back(f->path());
        layer_files.clear();
        for (size_t li = 0; li < paths.size(); ++li)
        {
            const fs::path dst = a.out / shard(li + 1);
            std::error_code ec;
            fs::rename(paths[li], dst, ec);
            if (ec) { fs::copy_file(paths[li], dst, fs::copy_options::overwrite_existing); fs::remove(paths[li]); }   // other disk
        }
        std::printf("  moved %zu layer files into place (%.1f s)\n", paths.size(), t.s());
    }
    index["metadata"]["total_size"] = total;
    std::ofstream(a.out / "model.safetensors.index.json") << index.dump(4);

    std::ifstream cf(a.in / "config.json");
    json cfg = json::parse(cf);
    const double bpw = lin_weights ? std::round(1000.0 * (double) lin_bits / (double) lin_weights) / 1000.0 : (double) K;
    nlohmann::ordered_json qc = {{"quant_method", "exl3"}, {"version", "1.5.3"}, {"bits", bpw}, {"head_bits", head_bits},
                                 {"calibration", {{"rows", a.rows}, {"cols", a.cols}}}, {"out_scales", a.out_scales_auto ? "auto" : "always"},
                                 {"codebook", "mul1"}, {"mtp_bits", mtp_bits}, {"scaled_hessian", a.scaled_hessian}, {"quantizer", VERSION}};
    if (nl < c.L) qc["layers"] = nl;
    if (cfg.contains("quantization_config") && cfg["quantization_config"].value("quant_method", "") != "exl3")
        cfg["original_quantization_config"] = cfg["quantization_config"];
    cfg["quantization_config"] = json::parse(qc.dump());
    std::ofstream(a.out / "config.json") << cfg.dump(4);
    nlohmann::ordered_json qf = qc;
    qf["tensor_storage"] = storage;
    std::ofstream(a.out / "quantization_config.json") << qf.dump(4);
    for (auto& e : fs::directory_iterator(a.in))
    {
        if (!e.is_regular_file()) continue;
        const auto n = e.path().filename().string(), ext = e.path().extension().string();
        if (ext == ".safetensors" || ext == ".gguf" || n == "config.json" || n == "model.safetensors.index.json" || ext == ".bin" || ext == ".pt" ||
            ext == ".pth" || ext == ".ckpt")
            continue;
        fs::copy_file(e.path(), a.out / n, fs::copy_options::overwrite_existing);
    }
    std::printf("  %zu shards; decoder linears at %.3f bpw, lm_head %d, MTP %d; total %.2f GB (%.1f s)\n", nshards, bpw, head_bits, mtp_bits, total / 1e9, tc.s());
}

// ---------------------------------------------------------------------------------------------------------------

inline void save_state(Upload& up, const DBuf<float>& state, int64_t R, size_t row_floats, const fs::path& path)
{
    const auto tmp = path.string() + ".tmp";
    const uint64_t need = (uint64_t) R * row_floats * 4;
    const auto room = fs::space(path.parent_path()).available;
    if (room < need + (1ull << 30))
        throw std::runtime_error("no room for the calibration state: it needs " + std::to_string(need >> 30) + " GiB and " + std::to_string(room >> 30) + " GiB are free on disk");
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    std::vector<float> buf(row_floats);
    bool ok = true;
    for (int64_t r = 0; r < R && ok; ++r)
    {
        up.download(buf.data(), state.get() + r * row_floats, row_floats * 4);
        ok = std::fwrite(buf.data(), 4, row_floats, f) == row_floats;
    }
    ok = ok && std::fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = std::fclose(f) == 0 && ok;
    if (!ok) { fs::remove(tmp); throw std::runtime_error("writing the calibration state failed (disk full?); the previous state.bin is untouched"); }
    fs::rename(tmp, path);
}

inline void load_state(Upload& up, DBuf<float>& state, int64_t R, size_t row_floats, const fs::path& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("no state.bin to resume from in " + path.parent_path().string());
    std::vector<float> buf(row_floats);
    bool ok = true;
    for (int64_t r = 0; r < R && ok; ++r)
    {
        ok = std::fread(buf.data(), 4, row_floats, f) == row_floats;
        if (ok) up(state.get() + r * row_floats, buf.data(), row_floats * 4);
    }
    std::fclose(f);
    if (!ok) throw std::runtime_error("state.bin is truncated");
}

inline int convert(const Args& a0, const json& cfg, int K)
{
    Args a = a0;
    rt::Timer total;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const Config c(cfg);
    const int HB = a.head_bits_set ? a.head_bits : 6, MB = a.mtp_bits;
    if (HB < 3 || HB > 8) throw std::runtime_error("--head-bits: 3..8 for qwen3_5 (the engine reads an EXL3 head)");
    if (MB < 1 || MB > 8) throw std::runtime_error("--mtp-bits: 1..8");
    if (!a.recipe.empty() || !a.group_bits.empty()) throw std::runtime_error("--recipe and --group-bits are not supported for qwen3_5 yet");
    st::Model m(a.in);
    fs::create_directories(a.work);
    {
        // The settings are part of the work dir's identity: a resume must use the same ones.
        std::string h = "qwen3_5|b" + std::to_string(K) + "|hb" + std::to_string(HB) + "|mb" + std::to_string(MB) + "|rows" + std::to_string(a.rows) + "x" +
                        std::to_string(a.cols) + "|os" + (a.out_scales_auto ? "auto" : "always") + "|sh" + (a.scaled_hessian ? "1" : "0");
        if (!a.cal_data.empty()) h += "|cal:" + std::to_string(fs::file_size(a.cal_data)) + ":" + a.cal_data.filename().string();
        const auto hp = a.work / "recipe.hash";
        if (fs::exists(hp)) { std::ifstream hf(hp); std::string old; std::getline(hf, old); if (old != h) throw std::runtime_error("this work dir was started with different settings (" + old + "); now " + h); }
        else std::ofstream(hp) << h;
    }
    const std::vector<std::vector<int32_t>> trace_rows = read_trace_rows(a, c.vocab);
    const int64_t T = a.cols, R = a.rows, D = c.D;
    const int64_t stop = a.layers > 0 ? std::min<int64_t>(c.L, a.layers) : c.L;
    Progress pg(a.work / "progress.json");
    if (pg.j.value("compiled", false)) { std::printf("%s is already complete (the layer files were moved into it)\n", a.out.c_str()); return 0; }
    const int64_t start = pg.j.value("next_layer", (int64_t) 0);
    const double max_s = a.max_minutes * 60;
    rt::Timer turn;

    // The plan: what the run needs on the card and on disk.
    const int nref = (int) std::min<int64_t>(5, R);
    const uint64_t QD = c.NH * c.Dh, KD = c.NKV * c.Dh, VO = c.HV * c.DV, C = c.C();
    const uint64_t w_gdn = D * C + D * VO + VO * D + 2 * D * c.HV, w_att = D * 2 * QD + 2 * D * KD + QD * D, w_mlp = 3 * D * c.I;
    const uint64_t base = 2 * (std::max(w_gdn, w_att) + w_mlp) + Scratch::bytes(c, T) + (uint64_t) (nref + 1) * T * D * 4 + (64ull << 20);
    const uint64_t state_b = (uint64_t) R * T * D * 4;
    const uint64_t hD = 4ull * D * D, hO = 4ull * std::max(VO, QD) * std::max(VO, QD), hI = 4ull * c.I * c.I;
    const uint64_t p_capture = base + state_b + Capture::bytes(c);
    const uint64_t p_quant = base + state_b + std::max({rpc::group_bytes(D, C + VO, C) + hO + hD + hI, rpc::group_bytes(D, 2 * QD + 2 * KD, 2 * QD) + hO + hD + hI,
                                                         rpc::group_bytes(VO, D, D) + hD + hI, rpc::group_bytes(D, 2 * c.I, c.I) + hI, rpc::group_bytes(c.I, D, D)});
    const uint64_t p_head = 2ull * c.vocab * D + 8 * 4ull * D * 16384 + 3 * hD + (1ull << 30);
    const uint64_t p_mtp = 2 * (w_att + w_mlp + 2 * D * D) + rpc::group_bytes(c.I, D, D);
    const uint64_t peak = std::max({p_capture, p_quant, p_head, p_mtp});
    const uint64_t disk_layer = (uint64_t) ((double) (w_gdn * 3 + w_att) / 4 / 8 * K) + 2 * linear_bytes(D, c.I, K) + linear_bytes(c.I, D, K) + 4 * D * 4;
    const uint64_t state_disk = a.max_minutes > 0 ? state_b * (fs::exists(a.work / "state.bin") ? 1 : 2) : 0;
    const uint64_t disk_need = (uint64_t) std::max<int64_t>(0, stop - start) * disk_layer + state_disk + (2ull << 30);
    const auto disk_free = fs::space(a.work).available;
    std::printf("exl3conv: %s, qwen3_5, layers %lld..%lld of %lld (%lld full attention), hidden %lld, EXL3 K=%d, head %d, MTP %d, calibration %lld rows x %lld\n",
                a.in.filename().c_str(), (long long) start, (long long) stop - 1, (long long) c.L, (long long) std::count(c.full.begin(), c.full.end(), true),
                (long long) D, K, HB, MB, (long long) R, (long long) T);
    std::printf("plan: VRAM %.1f GiB capture, %.1f quantize, %.1f head, %.1f MTP; disk %.1f GiB in %s (%.2f per layer, state %.1f), %.1f free\n",
                p_capture / GiB, p_quant / GiB, p_head / GiB, p_mtp / GiB, disk_need / GiB, a.work.c_str(), disk_layer / GiB, state_disk / GiB, disk_free / GiB);
    if (start < stop && disk_free < disk_need)
        throw std::runtime_error("not enough disk in " + a.work.string() + ": the rest of this conversion needs " + std::to_string(disk_need >> 30) + " GiB and " +
                                 std::to_string(disk_free >> 30) + " GiB are free");

    // Calibration rows (Qwen: BOS = EOS = <|endoftext|>; random rows over the ids the tokenizer has).
    std::vector<int64_t> rlen(R, T);
    std::vector<int32_t> ids((size_t) R * T, 0);
    const auto rows_file = a.work / "rows.bin";
    if (start > 0 || pg.j.value("head_hessian", false))
    {
        std::ifstream rf(rows_file, std::ios::binary);
        rf.read((char*) rlen.data(), R * 8);
        rf.read((char*) ids.data(), (std::streamsize) (ids.size() * 4));
        if (!rf) throw std::runtime_error("no good rows.bin to resume from in " + a.work.string());
    }
    else
    {
        rt::Timer t;
        std::vector<std::vector<int32_t>> rows = trace_rows;
        if (rows.empty())
        {
            std::ifstream tf(a.in / "tokenizer.json");
            if (!tf) throw std::runtime_error("no tokenizer.json in " + a.in.string());
            const auto tj = json::parse(tf);
            int64_t maxid = 0;
            for (auto& [k2, v] : tj["model"]["vocab"].items()) maxid = std::max<int64_t>(maxid, v.get<int64_t>());
            for (auto& tkn : tj["added_tokens"]) maxid = std::max<int64_t>(maxid, tkn["id"].get<int64_t>());
            tok::BPE tk((a.in / "tokenizer.json").string());
            rows = cal::default_rows(tk, a.cal_dir, (int) R, (int) T, c.bos, c.eos, (int) std::min<int64_t>(maxid + 1, c.vocab));
        }
        for (int64_t r = 0; r < R; ++r)
        {
            rlen[r] = std::min<int64_t>(T, (int64_t) rows[r].size());
            for (int64_t t2 = 0; t2 < rlen[r]; ++t2) ids[r * T + t2] = rows[r][t2];
        }
        const auto tmp = rows_file.string() + ".tmp";
        {
            std::ofstream rf(tmp, std::ios::binary);
            rf.write((const char*) rlen.data(), R * 8);
            rf.write((const char*) ids.data(), (std::streamsize) (ids.size() * 4));
            if (!rf) throw std::runtime_error("cannot write " + tmp);
        }
        fs::rename(tmp, rows_file);
        std::printf("calibration rows tokenized in %.1f s\n", t.s());
    }
    int64_t cal_tokens = 0;
    for (auto l : rlen) cal_tokens += l;

    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    {
        const uint64_t total_b = q.get_device().get_info<sycl::info::device::global_mem_size>();
        const uint64_t others = rt::card_vram().used, reserve = 1ull << 30;
        exl3::Vram::budget = total_b - std::min(total_b, others + reserve);
        std::printf("plan: VRAM peak %.1f GiB; budget %.1f (%.1f card, %.1f in use by others, %.1f reserve)\n", peak / GiB, exl3::Vram::budget / GiB, total_b / GiB,
                    others / GiB, reserve / GiB);
        if (peak > exl3::Vram::budget)
            throw std::runtime_error("this conversion needs " + std::to_string(peak >> 20) + " MiB on the card and only " + std::to_string(exl3::Vram::budget >> 20) + " MiB are free");
    }
    rt::VramWatch vw;
    Upload up(q);
    DBuf<int> d_perm(q, 256);
    d_perm.upload(exl3::tile_perm().data(), 256);
    const size_t row_floats = (size_t) T * D, row_bytes = row_floats * 4;
    const auto state_file = a.work / "state.bin";
    const auto inv = c.inv_freq();
    DBuf<float> d_inv(q, inv.size());
    d_inv.upload(inv.data(), inv.size());
    DBuf<uint16_t> wscratch(q, 1);

    // ---- Layers.
    if (start < stop)
    {
        DBuf<float> state(q, (size_t) R * row_floats);
        if (start > 0)
        {
            rt::Timer t;
            load_state(up, state, R, row_floats, state_file);
            std::printf("resumed at layer %lld (state loaded in %.1f s)\n", (long long) start, t.s());
        }
        else
        {
            rt::Timer t;
            const auto& e = m.get(c.prefix + "embed_tokens.weight");
            if (e.shape.size() != 2 || e.shape[0] != c.vocab || e.shape[1] != D) throw std::runtime_error("embedding: expected [vocab, hidden]");
            std::vector<float> buf(row_floats);
            for (int64_t r = 0; r < R; ++r)
            {
                for (int64_t t2 = 0; t2 < T; ++t2)
                    for (int64_t d = 0; d < D; ++d) buf[t2 * D + d] = q35::elem_f32(e, (size_t) ids[r * T + t2] * D + d);
                up(state.get() + r * row_floats, buf.data(), row_bytes);
            }
            std::printf("calibration rows embedded in %.1f s\n", t.s());
        }
        Scratch s(q, c, T);
        DBuf<float> xt(q, row_floats);
        std::vector<DBuf<float>> refs;
        for (int i = 0; i < nref; ++i) refs.emplace_back(q, row_floats);
        double sum_capture = 0, sum_advance = 0, sum_layer = 0;
        int64_t done = 0;
        for (int64_t li = start; li < stop; ++li)
        {
            rt::Timer tl;
            std::printf("layer %lld/%lld (%s)\n", (long long) li, (long long) c.L, c.full[li] ? "full attention" : "Gated DeltaNet");
            Layer L = q35::load_layer(q, m, c.lp(li), c.full[li], up, wscratch);
            wscratch = DBuf<uint16_t>(q, 1);
            Capture cap;
            cap.alloc(q, c, a.out_scales_auto);
            rt::Timer tc;
            for (int64_t r = 0; r < R; ++r)
            {
                q.memcpy(xt.get(), state.get() + r * row_floats, row_bytes);
                q35::forward_layer(q, c, L, xt.get(), rlen[r], s, d_inv.get(), &cap);
                if (r < nref) q.memcpy(refs[r].get(), xt.get(), row_bytes).wait();
            }
            q.wait();
            const double cap_s = tc.s();
            std::printf("  capture %.1f s%s (VRAM peak %.1f GiB, ledger %.1f)\n", cap_s,
                        (cap.n_in + cap.n_o + cap.n_gu + cap.n_d < 4 * cal_tokens) ? "  (some rows had non-finite activations and were left out)" : "",
                        vw.take() / GiB, exl3::Vram::peak / GiB);
            std::vector<exl3::QuantOut> outs;
            quantize_layer(q, c, c.lp(li), L, cap, K, d_perm, a.out_scales_auto, outs);
            const uint64_t quant_peak = vw.take();
            cap = Capture();
            rt::Timer ta;
            double rfn = 0;
            for (int64_t r = 0; r < R; ++r)
            {
                q35::forward_layer(q, c, L, state.get() + r * row_floats, rlen[r], s, d_inv.get(), nullptr);
                if (r < nref)
                {
                    std::vector<float> a1(row_floats), b1(row_floats);
                    q.memcpy(a1.data(), state.get() + r * row_floats, row_bytes).wait();
                    q.memcpy(b1.data(), refs[r].get(), row_bytes).wait();
                    double num = 0, den = 0;
                    for (size_t i = 0; i < (size_t) rlen[r] * D; ++i) { const double d = a1[i] - b1[i]; num += d * d; den += (double) b1[i] * b1[i]; }
                    rfn += std::sqrt(num / std::max(den, 1e-30)) / nref;
                }
            }
            q.wait();
            const double adv_s = ta.s();
            std::printf("  advance %.1f s; layer output vs unquantized (rows 0-%d): rfn %.5f; VRAM peak in quantize %.1f GiB, ledger %.1f\n", adv_s, nref - 1, rfn,
                        quant_peak / GiB, exl3::Vram::peak / GiB);
            char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
            st::write_file(a.work / fn, block_tensors(m, c.lp(li), c.full[li], outs));
            const double layer_s = tl.s();
            sum_capture += cap_s; sum_advance += adv_s; sum_layer += layer_s; ++done;
            std::printf("  layer done in %.1f s (total %.1f min)\n", layer_s, total.s() / 60);
            if (li + 1 < stop && max_s > 0 && turn.s() > max_s)
            {
                rt::Timer t;
                save_state(up, state, R, row_floats, state_file);
                pg.j["next_layer"] = li + 1; pg.save();
                std::printf("turn over after layer %lld: state saved in %.1f s; run again to continue\n", (long long) li, t.s());
                return 99;
            }
        }
        std::printf("%lld layers: mean %.1f s each (capture %.1f, advance %.1f); %lld layers at this rate: %.2f h\n", (long long) done, sum_layer / done,
                    sum_capture / done, sum_advance / done, (long long) c.L, sum_layer / done * c.L / 3600);
        // The head's Hessian from the last layer's output, then the state is no longer needed.
        rt::Timer t;
        DBuf<float> Hh(q, (size_t) D * D);
        Hh.zero();
        const auto normw = q35::load_norm1p(q, m, c.prefix + "norm.weight", up);
        int64_t n = 0;
        for (int64_t r = 0; r < R; ++r)
        {
            nn::rmsnorm_w(q, state.get() + r * row_floats, normw.get(), rlen[r], D, (float) c.eps, s.y.get());
            n += q35::accumulate(q, s.y.get(), rlen[r], D, Hh.get(), s);
        }
        q35::symmetrize(q, Hh.get(), D);
        {
            std::vector<float> h((size_t) D * D);
            Hh.download(h.data(), h.size());
            const auto tmp = (a.work / "head_H.bin").string() + ".tmp";
            { std::ofstream hf(tmp, std::ios::binary); hf.write((const char*) h.data(), (std::streamsize) (h.size() * 4)); if (!hf) throw std::runtime_error("cannot write " + tmp); }
            fs::rename(tmp, a.work / "head_H.bin");
        }
        pg.j["next_layer"] = stop; pg.j["head_hessian"] = true; pg.j["head_rows"] = n; pg.save();
        if (fs::exists(state_file)) fs::remove(state_file);
        std::printf("lm_head: Hessian over %lld tokens in %.1f s\n", (long long) n, t.s());
    }

    // ---- Head.
    if (!pg.j.value("head_done", false))
    {
        rt::Timer t;
        const int64_t n = pg.j.value("head_rows", (int64_t) 0);
        DBuf<float> Hh(q, (size_t) D * D);
        {
            std::vector<float> h((size_t) D * D);
            std::ifstream hf(a.work / "head_H.bin", std::ios::binary);
            hf.read((char*) h.data(), (std::streamsize) (h.size() * 4));
            if (!hf) throw std::runtime_error("head_H.bin is missing or short");
            Hh.upload(h.data(), h.size());
        }
        Lin head = q35::load_lin(q, m, "lm_head.weight", up, wscratch, false);
        wscratch = DBuf<uint16_t>(q, 1);
        if (head.in != D || head.out != c.vocab) throw std::runtime_error("lm_head.weight: expected [vocab, hidden]");
        const uint64_t seed = exl3::mix64(exl3::hash_str("lm_head"));
        exl3::Hessian hs = exl3::finalize_hessian(q, std::move(Hh), D, n, seed);
        exl3::QuantArgs qa; qa.K = HB; qa.verbose = true;
        const half* hw = head.w.get();
        auto src = [&](int64_t c0, int64_t Cn, float* W) { nn::transpose_h2f(q, hw + c0 * D, Cn, D, W).wait(); };
        exl3::QuantOut o = exl3::quantize_wide(q, hs, "lm_head", c.vocab, src, qa, seed);
        std::printf("    lm_head %lld x %lld  K=%d  g %.3f  err %.6f -> %.6f  (%.1f s, VRAM peak %.1f GiB, ledger %.1f)\n", (long long) o.k, (long long) o.n, HB, o.g_scale,
                    o.err_before, o.err_after, t.s(), vw.take() / GiB, exl3::Vram::peak / GiB);
        std::vector<st::OutTensor> ts;
        rpc::push_exl3(ts, o);
        st::write_file(a.work / "head.safetensors", ts);
        pg.j["head_done"] = true; pg.save();
        if (max_s > 0 && turn.s() > max_s) { std::printf("turn over after the head; run again to continue\n"); return 99; }
    }

    // ---- MTP: uncalibrated.
    if (!pg.j.value("mtp_done", false))
    {
        if (c.mtp_layers > 0 && m.has("mtp.fc.weight"))
        {
            rt::Timer t;
            if (c.mtp_layers != 1) throw std::runtime_error("qwen3_5: only one MTP layer is supported");
            std::printf("MTP block (K %d, uncalibrated)\n", MB);
            std::vector<exl3::QuantOut> outs;
            {
                Lin fc = q35::load_lin(q, m, "mtp.fc.weight", up, wscratch);
                outs.push_back(quantize_uncalibrated(q, "mtp.fc", fc, MB));
            }
            Layer L = q35::load_layer(q, m, "mtp.layers.0.", true, up, wscratch);
            wscratch = DBuf<uint16_t>(q, 1);
            const std::string p = "mtp.layers.0.";
            for (auto [key, lin] : std::vector<std::pair<std::string, Lin*>>{{p + "self_attn.q_proj", &L.q}, {p + "self_attn.k_proj", &L.k}, {p + "self_attn.v_proj", &L.v},
                                                                              {p + "self_attn.o_proj", &L.o}, {p + "mlp.gate_proj", &L.g}, {p + "mlp.up_proj", &L.u},
                                                                              {p + "mlp.down_proj", &L.d}})
            {
                outs.push_back(quantize_uncalibrated(q, key, *lin, MB));
                lin->w.reset();
            }
            std::vector<st::OutTensor> ts;
            rpc::push_exl3(ts, outs[0]);
            for (const char* nme : {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"})
                ts.push_back(st::OutTensor::from_vector(nme, "BF16", m.get(nme).shape, q35::host_bf16(m, nme)));
            for (auto& x : block_tensors(m, p, true, outs)) ts.push_back(std::move(x));
            st::write_file(a.work / "mtp.safetensors", ts);
            std::printf("  MTP done in %.1f s\n", t.s());
        }
        else std::printf("no MTP block in the source\n");
        pg.j["mtp_done"] = true; pg.save();
    }

    // ---- Output.
    compile(a, m, c, stop, K, HB, MB);
    pg.j["compiled"] = true; pg.save();
    if (stop < c.L) std::printf("stopped after %lld layers (--layers): wrote a model of %lld layers\n", (long long) stop, (long long) stop);
    std::printf("done in %.1f min (VRAM ledger peak %.1f GiB)\n", total.s() / 60, exl3::Vram::peak / GiB);
    return 0;
}

} // namespace q35c
