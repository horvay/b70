// exl3conv: the conversion loop for Ripple-65B (ricursive). Included by convert.cpp, which dispatches on model_type.
//
// The loop is Gemma's (convert.cpp): the rows' hidden states on the card; per layer capture through the original layer, quantize,
// dequantize, advance through the quantized layer. What Ripple changes:
//
//   Memory. The card has 31.9 GiB: state 9.8 + down's Hessian 6.25 + its LDL factor 6.25 + working set 4 + the layer 1.5 + scratch
//   does not fit. While a layer quantizes the state is parked in pinned host memory (<1 s each way at 28 GB/s; state_park below),
//   so the peak is 21.8 GiB (measured), in the refit of the largest group. Before anything runs, the plan's peak is checked against what is
//   free on the card, and every allocation goes through a ledger with that budget (rt/dbuf.hpp): past it the job stops with an
//   error rather than a driver fault. The host needs the state's size free, checked too.
//   Disk. The plan also checks the disk: what is left of the output, and with --max-minutes the state twice (the new state.bin is
//   written beside the old one, which stays good until the rename).
//   Resume. v_proj reads the rows' embedding, so the token ids go to <work>/rows.bin next to state.bin.
//   Groups. q/k/g share the input norm's Hessian, v has its own, then o, gate/up, down. One K per group: -b, or --group-bits, or
//   --recipe (per layer).
//   Head. lm_head is its own weight: fp16 by default, --head-bits 3..8 for EXL3 with the Hessian of the final norm's output.
//   --layers N. Stops at N and writes a model of N layers, for parts of a checkpoint; config.json stays as it is (the engine reads such a
//   quant with RP_TEST_LAYERS=N), quantization_config gains "layers": N.
#pragma once

#include "arch/ripple.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace rpc {

// Explicit using-declarations: convert.cpp also pulls in the Gemma 4 names (Layer, Capture, ...) with a using-directive.
using rp::Rp; using rp::Layer; using rp::Capture; using rp::Scratch; using rp::Lin; using rp::Upload;
using exl3::DBuf;
using half = sycl::half;
using json = nlohmann::json;
namespace fs = std::filesystem;

constexpr double GiB = 1073741824.0;

// Bits per input group of a layer.
struct LayerK { int qkg = 4, v = 4, o = 4, gu = 4, d = 4; };

inline int check_k(int k, const std::string& what)
{
    if (k < 1 || k > 8) throw std::runtime_error(what + ": bits must be an integer 1..8");
    return k;
}

// --group-bits qkg=4,v=4,o=4,gu=3,d=4: any of them, over -b.
inline LayerK group_bits(const std::string& s, int K)
{
    LayerK k{K, K, K, K, K};
    std::stringstream ss(s);
    for (std::string it; std::getline(ss, it, ',');)
    {
        const auto eq = it.find('=');
        if (eq == std::string::npos) throw std::runtime_error("--group-bits: " + it + ": expected name=K");
        const std::string n = it.substr(0, eq);
        const int v = check_k(std::stoi(it.substr(eq + 1)), "--group-bits " + n);
        if (n == "qkg") k.qkg = v; else if (n == "v") k.v = v; else if (n == "o") k.o = v; else if (n == "gu") k.gu = v; else if (n == "d") k.d = v;
        else throw std::runtime_error("--group-bits: unknown group " + n + " (qkg, v, o, gu, d)");
    }
    return k;
}

// --recipe R.json: {"bits": {"<linear key>": K, ...}}, keys as in the checkpoint without ".weight"; linears it doesn't name get
// the flags' K. One K per group is required.
inline std::vector<LayerK> read_recipe(const fs::path& path, const Rp& g, const LayerK& base)
{
    std::vector<LayerK> lks(g.L, base);
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
                const int v = check_k(bits.at(p + n).get<int>(), "recipe: " + p + n);
                if (k >= 0 && v != k) throw std::runtime_error("recipe: " + p + names[0] + " and its group differ (" + std::to_string(k) + " vs " + std::to_string(v) + "); the group shares one K");
                k = v; ++used;
            }
            if (k >= 0) out = k;
        };
        get({"self_attn.q_proj", "self_attn.k_proj", "self_attn.g_proj"}, lks[li].qkg);
        get({"self_attn.v_proj"}, lks[li].v);
        get({"self_attn.o_proj"}, lks[li].o);
        get({"mlp.gate_proj", "mlp.up_proj"}, lks[li].gu);
        get({"mlp.down_proj"}, lks[li].d);
    }
    if (used != bits.size()) throw std::runtime_error("recipe: " + std::to_string(bits.size() - used) + " keys name no decoder linear of this model");
    return lks;
}

// ---------------------------------------------------------------------------------------------------------------
// The plan: what the run needs on the card, in host memory and on disk, checked before it starts.

// The quantizer's most VRAM for a group of tensors over k inputs, N outputs in all, widest n (quantize_group): Hd + L (or the refit's A),
// the fp32 weights, W5 and the error products in LDLQ, or the refit's Q0, HW, HQ; Qm, the encoder's states and history. The rest is small.
inline uint64_t group_bytes(uint64_t k, uint64_t N, uint64_t n)
{
    const uint64_t kk = 4 * k * k, kN = 4 * k * N, kn = 4 * k * n;
    const uint64_t common = 2 * kk + 2 * kN + kN / 2 + (640ull << 20);
    return common + std::max(2 * kN, 3 * kn);
}

struct Plan
{
    uint64_t base = 0, state = 0, capture = 0, quant = 0, head = 0, disk_layer = 0, disk_rest = 0;
    uint64_t peak() const { return std::max({capture, quant, head}); }
};

inline uint64_t linear_bytes(uint64_t k, uint64_t n, int K) { return k * n * K / 8 + 2 * (k + n) + 4; }

inline Plan make_plan(const Rp& g, int64_t R, int64_t T, const LayerK& lk, int head_bits, int nref)
{
    Plan p;
    const uint64_t H = g.H, QD = g.NH * g.D, KD = g.NKV * g.D, I = g.I, V = g.vocab;
    const uint64_t weights = 2 * (2 * H * QD + 2 * H * KD + QD * H + 3 * H * I);
    p.state = (uint64_t) R * T * H * 4;
    p.base = weights + Scratch::bytes(g, T) + 2 * V * H + (uint64_t) R * T * 4 + (uint64_t) (nref + 1) * T * H * 4;
    const uint64_t hq = 4 * H * H, ho = 4 * QD * QD, hd = 4 * I * I;
    p.capture = p.base + p.state + 3 * hq + ho + hd;
    // Each group finalized and quantized with the later groups' Hessians still waiting.
    p.quant = p.base + std::max({group_bytes(H, 2 * QD + KD, QD) + 2 * hq + ho + hd, group_bytes(H, KD, KD) + hq + ho + hd,
                                 group_bytes(QD, H, H) + hq + hd, group_bytes(H, 2 * I, I) + hd, group_bytes(I, H, H)});
    p.head = head_bits < 16 ? p.base + 2 * V * H + group_bytes(H, V, V) : 0;
    p.disk_layer = 2 * linear_bytes(H, QD, lk.qkg) + linear_bytes(H, KD, lk.qkg) + linear_bytes(H, KD, lk.v) + linear_bytes(QD, H, lk.o) +
                   2 * linear_bytes(H, I, lk.gu) + linear_bytes(I, H, lk.d) + 2 * 4 * H + 4 * g.D + 2;
    p.disk_rest = 2 * V * H + (head_bits < 16 ? linear_bytes(H, V, head_bits) : 2 * V * H) + 2 * H;
    return p;
}

// ---------------------------------------------------------------------------------------------------------------
// The calibration state in pinned host memory, while a layer quantizes and for state.bin. Pieces of 1 GiB (a pinned allocation above ~4 GB
// silently returns null) and copies of at most 32 MiB (notes/toolchain-and-reuse.md §6.1).
class StatePark
{
public:
    StatePark(sycl::queue& q_, int64_t R_, size_t row_bytes_) : q(q_), R(R_), row_bytes(row_bytes_)
    {
        per = std::max<size_t>(1, (size_t) (1ull << 30) / row_bytes);
        for (int64_t r = 0; r < R; r += per)
        {
            pieces.push_back((uint8_t*) sycl::malloc_host(std::min<size_t>(per, R - r) * row_bytes, q));
            if (!pieces.back()) throw std::runtime_error("pinned host allocation for the parked state failed");
        }
    }
    ~StatePark() { for (auto* p : pieces) if (p) sycl::free(p, q); }
    StatePark(const StatePark&) = delete;
    StatePark& operator=(const StatePark&) = delete;
    uint8_t* row(int64_t r) const { return pieces[r / per] + (r % per) * row_bytes; }
    void from_card(const float* state) { copy(state, true); }
    void to_card(float* state) { copy(state, false); }

private:
    void copy(const float* state, bool out)
    {
        constexpr size_t CH = 32ull << 20;
        for (int64_t r = 0; r < R; ++r)
        {
            const uint8_t* dev = (const uint8_t*) state + r * row_bytes;
            for (size_t o = 0; o < row_bytes; o += CH)
            {
                const size_t n = std::min(CH, row_bytes - o);
                if (out) q.memcpy(row(r) + o, dev + o, n);
                else q.memcpy((uint8_t*) dev + o, row(r) + o, n);
            }
        }
        q.wait();
    }
    sycl::queue& q;
    int64_t R;
    size_t row_bytes, per;
    std::vector<uint8_t*> pieces;
};

// The parked state to <work>/state.bin, durable before the rename: the old file stays good until the new one is whole.
inline void write_state(const StatePark& park, int64_t R, size_t row_bytes, const fs::path& path)
{
    const auto tmp = path.string() + ".tmp";
    const uint64_t need = (uint64_t) R * row_bytes;
    const auto room = fs::space(path.parent_path()).available;
    if (room < need + (1ull << 30))
        throw std::runtime_error("no room for the calibration state: it needs " + std::to_string(need >> 30) + " GiB and " + std::to_string(room >> 30) + " GiB are free on disk");
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    bool ok = true;
    for (int64_t r = 0; r < R && ok; ++r) ok = std::fwrite(park.row(r), 1, row_bytes, f) == row_bytes;
    ok = ok && std::fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = std::fclose(f) == 0 && ok;
    if (!ok) { fs::remove(tmp); throw std::runtime_error("writing the calibration state failed (disk full?); the previous state.bin is untouched"); }
    fs::rename(tmp, path);
}

inline void read_state(StatePark& park, int64_t R, size_t row_bytes, const fs::path& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("no state.bin to resume from in " + path.parent_path().string());
    bool ok = true;
    for (int64_t r = 0; r < R && ok; ++r) ok = std::fread(park.row(r), 1, row_bytes, f) == row_bytes;
    std::fclose(f);
    if (!ok) throw std::runtime_error("state.bin is truncated");
}

// ---------------------------------------------------------------------------------------------------------------
// Quantize one layer.

struct LinSpec { std::string key; Lin* lin; const float* sens = nullptr; };   // sens: per-output sensitivities for out-scales auto

// One Hessian's tensors: quantized as a group, each weight replaced by its dequantized EXL3 version (in, out).
inline void quant_group(sycl::queue& q, DBuf<float>& H, int64_t count, int64_t k, const std::vector<LinSpec>& specs, const std::string& p, const char* tag, int K,
                        const DBuf<int>& d_perm, bool out_scales_auto, std::vector<exl3::QuantOut>& all)
{
    rt::Timer t;
    std::vector<DBuf<float>> Ws;
    std::vector<exl3::QuantIn> ins;
    for (const auto& sp : specs)
    {
        DBuf<float> W(q, (size_t) sp.lin->in * sp.lin->out);
        nn::transpose_h2f(q, sp.lin->w.get(), sp.lin->out, sp.lin->in, W.get()).wait();
        ins.push_back({sp.key, sp.lin->out, W.get(), sp.sens});
        Ws.push_back(std::move(W));
    }
    const uint64_t seed = exl3::mix64(exl3::hash_str(p + tag));
    rp::symmetrize(q, H.get(), k);
    rt::Timer tf;
    exl3::Hessian hs = exl3::finalize_hessian(q, std::move(H), k, count, seed, !g_scaled_hessian);
    const double final_s = tf.s();
    exl3::QuantArgs qa; qa.K = K; qa.verbose = true; qa.scaled_hessian = g_scaled_hessian;
    if (out_scales_auto) { qa.out_scales = false; qa.out_scales_auto = true; }
    auto outs = exl3::quantize_group(q, hs, ins, qa, seed);
    Ws.clear();
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
    std::printf("    [%s: K=%d, %.1f s (finalize %.1f), Hessian over %lld rows%s]\n", tag, K, t.s(), final_s, (long long) count, hs.retries ? ", damping retried" : "");
    std::fflush(stdout);
}

inline std::vector<exl3::QuantOut> quantize_layer(sycl::queue& q, const Rp& g, int64_t li, Layer& L, Capture& cap, const LayerK& lk,
                                                  const DBuf<int>& d_perm, bool out_scales_auto)
{
    const auto p = g.lp(li), a = p + "self_attn.";
    std::vector<exl3::QuantOut> all;
    // --out-scales auto: per intermediate channel, the mean squared slope along up and along gate, times the energy of the channel's
    // row of the (still unquantized) down projection.
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
    const int64_t QD = g.NH * g.D;
    auto group = [&](DBuf<float>& H, int64_t count, int64_t k, std::vector<LinSpec> specs, const char* tag, int K) {
        quant_group(q, H, count, k, specs, p, tag, K, d_perm, out_scales_auto, all);
    };
    group(cap.qkg, cap.n_qkg, g.H, {{a + "q_proj", &L.q}, {a + "k_proj", &L.k}, {a + "g_proj", &L.gt}}, "attn.input", lk.qkg);
    group(cap.v, cap.n_v, g.H, {{a + "v_proj", &L.v}}, "attn.value", lk.v);
    group(cap.o, cap.n_o, QD, {{a + "o_proj", &L.o}}, "attn.o", lk.o);
    group(cap.gu, cap.n_gu, g.H, {{p + "mlp.gate_proj", &L.g, sens_g.empty() ? nullptr : sens_g.data()}, {p + "mlp.up_proj", &L.u, sens_u.empty() ? nullptr : sens_u.data()}}, "mlp.input", lk.gu);
    group(cap.d, cap.n_d, g.I, {{p + "mlp.down_proj", &L.d}}, "mlp.down", lk.d);
    return all;
}

inline void push_exl3(std::vector<st::OutTensor>& ts, exl3::QuantOut& o)
{
    const int32_t marker = (int32_t) exl3::MUL1_MULT;
    ts.push_back(st::OutTensor::from_vector(o.key + ".suh", "F16", {o.k}, std::move(o.suh)));
    ts.push_back(st::OutTensor::from_vector(o.key + ".svh", "F16", {o.n}, std::move(o.svh)));
    ts.push_back(st::OutTensor::from_vector(o.key + ".mul1", "I32", {}, std::vector<int32_t>{marker}));
    ts.push_back(st::OutTensor::from_vector(o.key + ".trellis", "I16", {o.k / 16, o.n / 16, 16 * o.K}, std::move(o.trellis)));
}

// The stored tensors of one layer: EXL3 linears plus the BF16 norms and value_res_lambda from the source.
inline std::vector<st::OutTensor> layer_tensors(const st::Model& m, const Rp& g, int64_t li, std::vector<exl3::QuantOut>& outs)
{
    std::vector<st::OutTensor> ts;
    const auto p = g.lp(li);
    auto copy = [&](const std::string& name) {
        const auto& t = m.get(name);
        ts.push_back(st::OutTensor::from_vector(name, st::dtype_name(t.dtype), t.shape, std::vector<uint8_t>(t.data, t.data + t.nbytes)));
    };
    copy(p + "input_layernorm.weight");
    for (auto& o : outs) push_exl3(ts, o);
    for (const char* n : {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.value_res_lambda", "post_attention_layernorm.weight",
                          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"})
        copy(p + n);
    return ts;
}

// ---------------------------------------------------------------------------------------------------------------
// Output model

inline void compile(const Args& a, const st::Model& m, const Rp& g, int64_t nl, int K)
{
    std::vector<std::unique_ptr<st::File>> layer_files;
    auto copy_src = [&](const std::string& name) {
        const auto& t = m.get(name);
        return st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes);
    };
    for (int64_t li = 0; li < nl; ++li)
    {
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        layer_files.push_back(std::make_unique<st::File>(a.work / fn));
    }
    std::vector<st::OutTensor> first = {copy_src(g.prefix + "embed_tokens.weight")};
    std::vector<st::OutTensor> last = {copy_src(g.prefix + "norm.weight")};
    std::unique_ptr<st::File> head;
    if (a.head_bits < 16)
    {
        head = std::make_unique<st::File>(a.work / "head.safetensors");
        for (const auto& name : head->order())
        {
            const auto& t = head->tensors().at(name);
            last.push_back(st::OutTensor::from_bytes(name, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes));
        }
    }
    else last.push_back(fp16_copy("lm_head.weight", m.get("lm_head.weight")));
    json extra = json::object();
    if (nl < g.L) extra["layers"] = nl;
    write_model(a, layer_files, first, last, K, g.L, extra);
}

// ---------------------------------------------------------------------------------------------------------------

inline int convert(const Args& a0, const json& cfg, int K)
{
    Args a = a0;
    rt::Timer total;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const Rp g(cfg);
    if (a.head_bits != 16 && (a.head_bits < 3 || a.head_bits > 8)) throw std::runtime_error("--head-bits: 16, or 3..8 for an EXL3 head");
    st::Model m(a.in);
    fs::create_directories(a.work);
    const LayerK base = group_bits(a.group_bits, K);
    const auto lks = read_recipe(a.recipe, g, base);
    {
        // The settings are part of the work dir's identity: a resume must use the same ones.
        std::string h = "b" + std::to_string(K) + "|gb:" + a.group_bits + "|hb" + std::to_string(a.head_bits) + "|rows" + std::to_string(a.rows) + "x" + std::to_string(a.cols);
        if (!a.recipe.empty()) { std::ifstream rf(a.recipe); std::stringstream rs; rs << rf.rdbuf(); h += "|r:" + std::to_string(exl3::hash_str(rs.str())); }
        if (!a.cal_data.empty()) h += "|cal:" + std::to_string(fs::file_size(a.cal_data)) + ":" + a.cal_data.filename().string();
        const auto hp = a.work / "recipe.hash";
        if (fs::exists(hp)) { std::ifstream hf(hp); std::string old; std::getline(hf, old); if (old != h) throw std::runtime_error("this work dir was started with different settings (" + old + "); now " + h); }
        else std::ofstream(hp) << h;
    }
    const std::vector<std::vector<int32_t>> trace_rows = read_trace_rows(a, g.vocab);
    const int64_t T = a.cols, R = a.rows;
    const int64_t stop = a.layers > 0 ? std::min<int64_t>(g.L, a.layers) : g.L;

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
    if (a.compile_only || start >= stop)
    {
        std::printf("compiling the output model\n");
        compile(a, m, g, stop, K);
        std::printf("done in %.1f min\n", total.s() / 60);
        return 0;
    }
    auto save_progress = [&](int64_t next) {
        { std::ofstream pf(prog.string() + ".tmp"); pf << json{{"next_layer", next}, {"out_scales", a.out_scales_auto ? "auto" : "always"}}.dump(); }
        fs::rename(prog.string() + ".tmp", prog);
    };

    const int nref = (int) std::min<int64_t>(5, R);
    LayerK kmax = base;
    for (const auto& l : lks) { kmax.qkg = std::max(kmax.qkg, l.qkg); kmax.v = std::max(kmax.v, l.v); kmax.o = std::max(kmax.o, l.o); kmax.gu = std::max(kmax.gu, l.gu); kmax.d = std::max(kmax.d, l.d); }
    const Plan plan = make_plan(g, R, T, kmax, a.head_bits, nref);
    const auto state_file = a.work / "state.bin";

    // Host and disk are checked before the card is opened.
    const uint64_t ram = rt::host_mem_available();
    if (ram < plan.state + (4ull << 30))
        throw std::runtime_error("not enough host memory to park the calibration state: it needs " + std::to_string(plan.state >> 30) + " GiB plus a margin of 4, and " + std::to_string(ram >> 30) + " GiB are available");
    const uint64_t state_disk = a.max_minutes > 0 ? plan.state * (fs::exists(state_file) ? 1 : 2) : 0;
    const uint64_t disk_need = (uint64_t) (stop - start) * plan.disk_layer + plan.disk_rest + state_disk;
    const auto disk_free = fs::space(a.work).available;
    std::printf("exl3conv: %s, Ripple-65B, layers %lld..%lld of %lld, hidden %lld, EXL3 K qkg=%d v=%d o=%d gu=%d d=%d%s, head %d bits, calibration %lld rows x %lld\n",
                a.in.filename().c_str(), (long long) start, (long long) stop - 1, (long long) g.L, (long long) g.H, base.qkg, base.v, base.o, base.gu, base.d,
                a.recipe.empty() ? "" : " (recipe)", a.head_bits, (long long) R, (long long) T);
    std::printf("plan: disk %.1f GiB (%.2f per layer, state %.1f), %.1f free; host %.1f pinned, %.1f available\n", disk_need / GiB, plan.disk_layer / GiB, state_disk / GiB,
                disk_free / GiB, plan.state / GiB, ram / GiB);
    if (disk_free < disk_need)
        throw std::runtime_error("not enough disk in " + a.work.string() + ": the rest of this conversion needs " + std::to_string(disk_need >> 30) + " GiB and " + std::to_string(disk_free >> 30) + " GiB are free");

    // Rows, before the card too.
    std::vector<int64_t> rlen(R, T);
    std::vector<int32_t> ids((size_t) R * T);
    const auto rows_file = a.work / "rows.bin";
    if (start > 0)
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
            tok::BPE tk((a.in / "tokenizer.json").string());
            rows = cal::default_rows(tk, a.cal_dir, (int) R, (int) T, g.bos, g.eos, (int) g.vocab);
        }
        for (int64_t r = 0; r < R; ++r)
        {
            rlen[r] = (int64_t) rows[r].size();
            for (int64_t c = 0; c < rlen[r]; ++c) ids[r * T + c] = rows[r][c];
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
        // What the others hold at our turn's start, and what the plan needs of the rest. 1 GiB stays free for the driver and oneMKL.
        const uint64_t total_b = q.get_device().get_info<sycl::info::device::global_mem_size>();
        const uint64_t others = rt::card_vram().used, reserve = 1ull << 30;
        exl3::Vram::budget = total_b - std::min(total_b, others + reserve);
        std::printf("plan: VRAM %.1f GiB capture, %.1f quantize, %.1f head; budget %.1f (%.1f card, %.1f in use by others, %.1f reserve)\n", plan.capture / GiB, plan.quant / GiB,
                    plan.head / GiB, exl3::Vram::budget / GiB, total_b / GiB, others / GiB, reserve / GiB);
        if (plan.peak() > exl3::Vram::budget)
            throw std::runtime_error("this conversion needs " + std::to_string(plan.peak() >> 20) + " MiB on the card and only " + std::to_string(exl3::Vram::budget >> 20) + " MiB are free");
    }
    std::fflush(stdout);
    rt::VramWatch vw;

    Upload up(q);
    DBuf<int> d_perm(q, 256);
    d_perm.upload(exl3::tile_perm().data(), 256);
    const size_t row_floats = (size_t) T * g.H, row_bytes = row_floats * 4;
    StatePark park(q, R, row_bytes);
    const auto& et = m.get(g.prefix + "embed_tokens.weight");
    if (et.dtype != st::DType::BF16 || et.shape[0] != g.vocab || et.shape[1] != g.H) throw std::runtime_error("embedding: expected BF16 [vocab, hidden]");
    DBuf<uint16_t> E(q, (size_t) et.numel());
    up(E.get(), et.data, et.nbytes);
    DBuf<int32_t> dids(q, ids.size());
    dids.upload(ids.data(), ids.size());
    auto row_ids = [&](int64_t r) { return dids.get() + r * T; };

    // Calibration state: rows x T x H fp32 on the card, in pinned host memory while a layer quantizes.
    DBuf<float> state(q, (size_t) R * row_floats);
    if (start > 0)
    {
        rt::Timer t;
        read_state(park, R, row_bytes, state_file);
        park.to_card(state.get());
        std::printf("resumed at layer %lld (state loaded in %.1f s)\n", (long long) start, t.s());
    }
    else
    {
        for (int64_t r = 0; r < R; ++r) nn::embed_row(q, E.get(), row_ids(r), g.H, g.mult, T, state.get() + r * row_floats);
        q.wait();
    }

    Scratch s(q, g, T);
    DBuf<uint16_t> wscratch(q, 1);
    DBuf<float> xt(q, row_floats);
    const auto inv = g.inv_freq();
    DBuf<float> d_inv(q, inv.size());
    d_inv.upload(inv.data(), inv.size());
    std::vector<DBuf<float>> refs;
    for (int i = 0; i < nref; ++i) refs.emplace_back(q, row_floats);

    rt::Timer chunk;
    double sum_capture = 0, sum_advance = 0, sum_layer = 0;
    int64_t done = 0;
    for (int64_t li = start; li < stop; ++li)
    {
        rt::Timer tl;
        std::printf("layer %lld/%lld (%s attention)\n", (long long) li, (long long) g.L, g.full[li] ? "full" : "sliding");
        Layer L = rp::load_layer(q, m, g, li, up, wscratch);
        wscratch = DBuf<uint16_t>(q, 1);
        // Capture.
        Capture cap;
        const int64_t QD = g.NH * g.D;
        cap.qkg = DBuf<float>(q, (size_t) g.H * g.H); cap.qkg.zero();
        cap.v = DBuf<float>(q, (size_t) g.H * g.H); cap.v.zero();
        cap.o = DBuf<float>(q, (size_t) QD * QD); cap.o.zero();
        cap.gu = DBuf<float>(q, (size_t) g.H * g.H); cap.gu.zero();
        cap.d = DBuf<float>(q, (size_t) g.I * g.I); cap.d.zero();
        if (a.out_scales_auto) { cap.sens_u = DBuf<double>(q, g.I); cap.sens_u.zero(); cap.sens_g = DBuf<double>(q, g.I); cap.sens_g.zero(); }
        rt::Timer tc;
        for (int64_t r = 0; r < R; ++r)
        {
            q.memcpy(xt.get(), state.get() + r * row_floats, row_bytes);
            rp::forward_layer(q, g, L, xt.get(), E.get(), row_ids(r), rlen[r], s, d_inv.get(), &cap);
            if (r < nref) q.memcpy(refs[r].get(), xt.get(), row_bytes).wait();
        }
        const double cap_s = tc.s();
        std::printf("  capture %.1f s%s (VRAM peak %.1f GiB, ledger %.1f)\n", cap_s,
                    (cap.n_qkg + cap.n_v + cap.n_o + cap.n_gu + cap.n_d < 5 * cal_tokens) ? "  (some rows had non-finite activations and were left out)" : "",
                    vw.take() / GiB, exl3::Vram::peak / GiB);
        // Quantize, with the state parked in host memory.
        rt::Timer tp;
        park.from_card(state.get());
        state.reset();
        const double park_s = tp.s();
        auto outs = quantize_layer(q, g, li, L, cap, lks[li], d_perm, a.out_scales_auto);
        const uint64_t quant_peak = vw.take();
        // Advance through the quantized layer.
        rt::Timer tu;
        state = DBuf<float>(q, (size_t) R * row_floats);
        park.to_card(state.get());
        const double unpark_s = tu.s();
        rt::Timer ta;
        double rfn = 0;
        for (int64_t r = 0; r < R; ++r)
        {
            rp::forward_layer(q, g, L, state.get() + r * row_floats, E.get(), row_ids(r), rlen[r], s, d_inv.get(), nullptr);
            if (r < nref)
            {
                std::vector<float> a1(row_floats), b1(row_floats);
                q.memcpy(a1.data(), state.get() + r * row_floats, row_bytes).wait();
                q.memcpy(b1.data(), refs[r].get(), row_bytes).wait();
                double num = 0, den = 0;
                for (size_t i = 0; i < (size_t) rlen[r] * g.H; ++i) { const double d = a1[i] - b1[i]; num += d * d; den += (double) b1[i] * b1[i]; }
                rfn += std::sqrt(num / std::max(den, 1e-30)) / nref;
            }
        }
        const double adv_s = ta.s();
        std::printf("  advance %.1f s; layer output vs unquantized (rows 0-%d): rfn %.5f; state parked in %.2f s, back in %.2f s; VRAM peak in quantize %.1f GiB, ledger %.1f\n", adv_s, nref - 1, rfn, park_s, unpark_s,
                    quant_peak / GiB, exl3::Vram::peak / GiB);
        // Write the layer.
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        st::write_file(a.work / fn, layer_tensors(m, g, li, outs));
        const double layer_s = tl.s();
        sum_capture += cap_s; sum_advance += adv_s; sum_layer += layer_s; ++done;
        std::printf("  layer done in %.1f s (total %.1f min)\n", layer_s, total.s() / 60);
        std::fflush(stdout);
        if (li + 1 < stop && a.max_minutes > 0 && chunk.s() > a.max_minutes * 60)
        {
            rt::Timer t;
            park.from_card(state.get());
            write_state(park, R, row_bytes, state_file);
            save_progress(li + 1);
            std::printf("turn over after layer %lld: state saved in %.1f s; run again to continue\n", (long long) li, t.s());
            return 99;
        }
    }
    std::printf("%lld layers: mean %.1f s each (capture %.1f, advance %.1f); %lld layers at this rate: %.2f h\n", (long long) done, sum_layer / done, sum_capture / done,
                sum_advance / done, (long long) g.L, sum_layer / done * g.L / 3600);
    // lm_head: the final norm's output over every row, then an EXL3 linear like any other.
    if (a.head_bits < 16)
    {
        rt::Timer t;
        DBuf<float> Hh(q, (size_t) g.H * g.H);
        Hh.zero();
        const auto normw = rp::load_bf16_vec(q, m, g.prefix + "norm.weight", up);
        int64_t n = 0;
        for (int64_t r = 0; r < R; ++r)
        {
            nn::rmsnorm_h(q, state.get() + r * row_floats, normw.get(), rlen[r], g.H, g.eps, s.h.get());
            n += rp::accumulate(q, s.h.get(), rlen[r], g.H, Hh.get(), s);
        }
        std::printf("lm_head: Hessian over %lld rows in %.1f s\n", (long long) n, t.s());
        state.reset();
        Lin head = rp::load_linear(q, m, "lm_head.weight", up, wscratch);
        wscratch = DBuf<uint16_t>(q, 1);
        std::vector<exl3::QuantOut> all;
        quant_group(q, Hh, n, g.H, {{"lm_head", &head}}, "", "lm_head", a.head_bits, d_perm, a.out_scales_auto, all);
        std::vector<st::OutTensor> ts;
        push_exl3(ts, all[0]);
        st::write_file(a.work / "head.safetensors", ts);
        std::printf("lm_head done in %.1f s (VRAM peak %.1f GiB, ledger %.1f)\n", t.s(), vw.take() / GiB, exl3::Vram::peak / GiB);
    }
    if (stop < g.L) std::printf("stopped after %lld layers (--layers); writing a model of %lld layers\n", (long long) stop, (long long) stop);
    save_progress(stop);
    std::printf("all layers quantized; compiling the output model\n");
    compile(a, m, g, stop, K);
    if (fs::exists(state_file)) fs::remove(state_file);
    std::printf("done in %.1f min (VRAM ledger peak %.1f GiB)\n", total.s() / 60, exl3::Vram::peak / GiB);
    return 0;
}

} // namespace rpc
