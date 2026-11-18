// exl3conv: the conversion loop for Qwen3.8-Flash-Next (qwen4_exp). Included by convert.cpp, which dispatches on
// model_type. Stages, each resumable from <work>/progress.json (exit 99 between them when --max-minutes runs out):
//
//   ngram   the hashed n-gram table -> <out>/ngram_embedding.safetensors (written in place, shard by shard), and the
//           packed rows the calibration tokens hash to -> <work>/ple_rows.bin (the PLE layer reads those, quantized)
//   layers  per decoder layer: capture (all-experts MoE), quantize to the recipe, dequantize, advance; the 250 rows'
//           4-stream state (21 GB fp32) lives in pinned host memory, one row on the card at a time
//   head    the final mixer, lm_head's Hessian over all rows, lm_head quantized
//   compile shards (layer files moved into place), index, config, quantization_config.json, copied files
//
// Recipe (DESIGN-qwen4exp.md, after the 3.05 bpw model the B70 server runs): routed experts K 4 in layer 0, else 3;
// GDN 5 in layers 0-1, else 4; attention and the indexer 5; shared expert 5 in layer 0, else 4; lm_head 5; table 3.
#pragma once

#include "arch/qwen4exp.hpp"
#include "arch/qwen4exp_ngram.hpp"
#include "exl3/ngram.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace qx {

// Explicit using-declarations: convert.cpp also pulls in the Gemma 4 names (Layer, Capture, ...) with a using-directive.
using qwen4exp::Config; using qwen4exp::Layer; using qwen4exp::Capture; using qwen4exp::Scratch; using qwen4exp::Lin;
using qwen4exp::Upload; using qwen4exp::ngram_ids;
using exl3::DBuf;
using half = sycl::half;
using json = nlohmann::json;
namespace fs = std::filesystem;

struct Recipe
{
    int experts(int64_t l) const { return l == 0 ? 4 : 3; }
    int gdn(int64_t l) const { return l <= 1 ? 5 : 4; }
    int attn() const { return 5; }
    int shared(int64_t l) const { return l == 0 ? 5 : 4; }
    int head() const { return 5; }
    int ngram() const { return 3; }
};

struct Progress
{
    fs::path path;
    json j = json::object();
    explicit Progress(fs::path p) : path(std::move(p)) { if (fs::exists(path)) { std::ifstream f(path); j = json::parse(f); } }
    void save() { const auto t = path.string() + ".tmp"; { std::ofstream f(t); f << j.dump(); } fs::rename(t, path); }
};

// ---------------------------------------------------------------------------------------------------------------
// Stage: the n-gram table.

struct Ngram
{
    std::vector<int64_t> offsets, sizes;
    int64_t mult[3];
    int64_t shard_rows = 0, nshards = 0;
    std::string prefix;    // ...layers.1.ple.ple_embedding.
};

inline Ngram ngram_info(const st::Model& m, const Config& c)
{
    Ngram n;
    n.prefix = c.lp(c.ple_layer) + "ple.ple_embedding.";
    n.offsets.resize(16); n.sizes.resize(16);
    std::memcpy(n.offsets.data(), m.get(n.prefix + "ngram_heads_offsets").data, 128);
    std::memcpy(n.sizes.data(), m.get(n.prefix + "ngram_heads_vocab_sizes").data, 128);
    std::memcpy(n.mult, m.get(n.prefix + "layer_multipliers").data, 24);
    n.shard_rows = m.get(n.prefix + "ngram_embedding.shard_0.weight").shape[0];
    while (m.has(n.prefix + "ngram_embedding.shard_" + std::to_string(n.nshards) + ".weight")) ++n.nshards;
    return n;
}

// The calibration tokens' n-gram rows: sorted unique global ids, and per token x head the index into them.
struct PleIndex { std::vector<int64_t> ids; std::vector<int32_t> tok; };

inline PleIndex ple_index(const Config& c, const Ngram& ng, const std::vector<cal::Row>& rows)
{
    const int64_t T = (int64_t) rows[0].size(), R = (int64_t) rows.size();
    std::vector<int64_t> all((size_t) R * T * 16);
    for (int64_t r = 0; r < R; ++r) ngram_ids(rows[r].data(), T, c.eos, ng.mult, ng.sizes.data(), ng.offsets.data(), &all[r * T * 16]);
    PleIndex p;
    p.ids = all;
    std::sort(p.ids.begin(), p.ids.end());
    p.ids.erase(std::unique(p.ids.begin(), p.ids.end()), p.ids.end());
    p.tok.resize(all.size());
    for (size_t i = 0; i < all.size(); ++i) p.tok[i] = (int32_t) (std::lower_bound(p.ids.begin(), p.ids.end(), all[i]) - p.ids.begin());
    return p;
}

// Quantize the table into <out>/ngram_embedding.safetensors, resuming at progress["ngram_shard"], and keep the packed
// rows listed in pi.ids in <work>/ple_rows.bin (filled in shard by shard). Returns false if time ran out.
inline bool ngram_stage(sycl::queue& q, const st::Model& m, const Config& c, const Ngram& ng, const PleIndex& pi, const fs::path& in_dir,
                        const fs::path& out, const fs::path& work, Progress& pg, int K, const rt::Timer& turn, double max_s)
{
    const int words = exl3::ngram::words_per_row(K);
    const int64_t R = ng.shard_rows;
    const auto file = out / "ngram_embedding.safetensors";
    const auto rows_file = work / "ple_rows.bin";
    const std::string P = c.lp(c.ple_layer) + "ple.ple_embedding.ngram_embedding.";
    // Head bias: computed once, kept in the progress file.
    std::vector<uint16_t> bias;
    if (pg.j.contains("ngram_bias")) bias = pg.j["ngram_bias"].get<std::vector<uint16_t>>();
    else
    {
        rt::Timer t;
        auto read = [&](int64_t lo, int64_t hi, float* dst) {
            for (int64_t r = lo; r < hi; ++r)
            {
                const auto& sh = m.get(ng.prefix + "ngram_embedding.shard_" + std::to_string(r / R) + ".weight");
                const uint16_t* src = (const uint16_t*) sh.data + (r % R) * exl3::ngram::ROW;
                for (int i = 0; i < exl3::ngram::ROW; ++i) dst[(r - lo) * exl3::ngram::ROW + i] = st::bf16_to_float(src[i]);
            }
        };
        bias = exl3::ngram::head_bias(ng.offsets, ng.sizes, read);
        pg.j["ngram_bias"] = bias; pg.save();
        std::printf("  n-gram head bias in %.1f s\n", t.s());
    }
    // File layout: header, then head_bias, offsets, sizes, multipliers, then the shards back to back.
    nlohmann::ordered_json h = nlohmann::ordered_json::object();
    h["__metadata__"] = {{"format", "exl3_ngram_trellis"}, {"version", "1"}, {"K", std::to_string(K)}, {"codebook", "mul1"},
                         {"codebook_scale", "heuristic"}, {"row_dim", "160"}, {"rows", std::to_string(R * ng.nshards)},
                         {"shard_rows", std::to_string(R)}, {"source", fs::absolute(in_dir).string()}};
    uint64_t off = 0;
    auto add = [&](const std::string& name, const char* dt, std::vector<int64_t> shape, uint64_t n) {
        h[P + name] = {{"dtype", dt}, {"shape", shape}, {"data_offsets", {off, off + n}}}; off += n;
    };
    add("head_bias", "F16", {16, 160}, 16 * 160 * 2);
    add("head_offsets", "I64", {16}, 128);
    add("head_vocab_sizes", "I64", {16}, 128);
    add("layer_multipliers", "I64", {3}, 24);
    const uint64_t shard_bytes = (uint64_t) R * words * 2;
    for (int64_t s = 0; s < ng.nshards; ++s) add("shard_" + std::to_string(s) + ".trellis", "I16", {R, words}, shard_bytes);
    std::string hs = h.dump();
    while ((hs.size() + 8) % 8) hs += ' ';
    const uint64_t data0 = 8 + hs.size();
    int64_t s0 = pg.j.value("ngram_shard", (int64_t) 0);
    const int fd = ::open(file.c_str(), O_RDWR | O_CREAT, 0644);
    const int rfd = ::open(rows_file.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0 || rfd < 0) throw std::runtime_error("cannot open the n-gram output files");
    // Full sizes up front (sparse until written): the table file, and one record per calibration row id.
    if (::ftruncate(rfd, (off_t) (pi.ids.size() * words * 2)) != 0 || ::ftruncate(fd, (off_t) (data0 + 16 * 160 * 2 + 280 + (uint64_t) ng.nshards * shard_bytes)) != 0)
        throw std::runtime_error("cannot size the n-gram output files");
    auto pw = [&](int f, const void* p, size_t n, uint64_t at) {
        for (size_t d = 0; d < n;) { const ssize_t w = ::pwrite(f, (const char*) p + d, n - d, (off_t) (at + d)); if (w <= 0) throw std::runtime_error("write failed"); d += (size_t) w; }
    };
    if (s0 == 0)
    {
        const uint64_t n = hs.size();
        pw(fd, &n, 8, 0); pw(fd, hs.data(), hs.size(), 8);
        pw(fd, bias.data(), bias.size() * 2, data0);
        pw(fd, ng.offsets.data(), 128, data0 + 16 * 160 * 2);
        pw(fd, ng.sizes.data(), 128, data0 + 16 * 160 * 2 + 128);
        pw(fd, ng.mult, 24, data0 + 16 * 160 * 2 + 256);
    }
    exl3::ngram::RowQuantizer rq(q, K, 262144);
    DBuf<uint16_t> dbias(q, bias.size()), drows(q, 262144 * exl3::ngram::ROW), dpk(q, 262144 * words);
    DBuf<int64_t> doffs(q, 16);
    dbias.upload(bias.data(), bias.size()); doffs.upload(ng.offsets.data(), 16);
    std::vector<uint16_t> packed((size_t) 262144 * words);
    // EXL3CONV_NGRAM_SHARDS=N (debug): quantize only the first N table shards; the rest of the file stays zero.
    const int64_t nsh = std::getenv("EXL3CONV_NGRAM_SHARDS") ? std::min<int64_t>(ng.nshards, std::atoll(std::getenv("EXL3CONV_NGRAM_SHARDS"))) : ng.nshards;
    for (int64_t s = s0; s < nsh; ++s)
    {
        rt::Timer t;
        const auto& src = m.get(ng.prefix + "ngram_embedding.shard_" + std::to_string(s) + ".weight");
        const uint16_t* rows = (const uint16_t*) src.data;
        for (int64_t r0 = 0; r0 < R; r0 += 262144)
        {
            const int64_t n = std::min<int64_t>(262144, R - r0);
            drows.upload(rows + r0 * exl3::ngram::ROW, n * exl3::ngram::ROW);
            rq.run(drows.get(), s * R + r0, n, dbias.get(), doffs.get(), 16, dpk.get());
            dpk.download(packed.data(), n * words);
            pw(fd, packed.data(), (size_t) n * words * 2, data0 + 16 * 160 * 2 + 280 + (uint64_t) s * shard_bytes + (uint64_t) r0 * words * 2);
            // The calibration's rows in [s R + r0, s R + r0 + n)
            const int64_t g0 = s * R + r0, g1 = g0 + n;
            auto lo = std::lower_bound(pi.ids.begin(), pi.ids.end(), g0), hi = std::lower_bound(pi.ids.begin(), pi.ids.end(), g1);
            for (auto it = lo; it != hi; ++it)
                pw(rfd, &packed[(size_t) (*it - g0) * words], (size_t) words * 2, (uint64_t) (it - pi.ids.begin()) * words * 2);
        }
        src.nbytes ? (void) 0 : (void) 0;
        m.file_of(ng.prefix + "ngram_embedding.shard_" + std::to_string(s) + ".weight").dont_need(src);
        ::fsync(fd); ::fsync(rfd);
        pg.j["ngram_shard"] = s + 1; pg.save();
        std::printf("  n-gram shard %lld/%lld in %.1f s\n", (long long) s + 1, (long long) ng.nshards, t.s());
        std::fflush(stdout);
        if (s + 1 < ng.nshards && max_s > 0 && turn.s() > max_s) { ::close(fd); ::close(rfd); return false; }
    }
    ::close(fd); ::close(rfd);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Pinned host state: one allocation per row.

struct HostState
{
    sycl::queue& q;
    std::vector<float*> rows;
    size_t row_floats;
    HostState(sycl::queue& q_, int64_t R, size_t rf) : q(q_), row_floats(rf)
    {
        for (int64_t r = 0; r < R; ++r)
        {
            float* p = sycl::malloc_host<float>(rf, q);
            if (!p) throw std::runtime_error("pinned host allocation for the calibration state failed (row " + std::to_string(r) + ")");
            rows.push_back(p);
        }
    }
    ~HostState() { for (auto* p : rows) sycl::free(p, q); }
    void save(const fs::path& f) const
    {
        std::ofstream o(f.string() + ".tmp", std::ios::binary);
        for (auto* p : rows) o.write((const char*) p, (std::streamsize) (row_floats * 4));
        o.close();
        fs::rename(f.string() + ".tmp", f);
    }
    void load(const fs::path& f)
    {
        std::ifstream i(f, std::ios::binary);
        for (auto* p : rows) { i.read((char*) p, (std::streamsize) (row_floats * 4)); if (!i) throw std::runtime_error("state file truncated"); }
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Quantization of one layer.

struct LayerOut { std::vector<exl3::QuantOut> q; };

inline DBuf<float> transpose_f32(sycl::queue& q, const half* w, int64_t out, int64_t in)
{
    DBuf<float> W(q, (size_t) out * in);
    nn::transpose_h2f(q, w, out, in, W.get()).wait();
    return W;
}

// Quantize a list of (key, HF [out, in] fp16 pointer, out, in) sharing one Hessian at bitrate K, in column chunks of at
// most 32768; the finalized Hessian (with its LDL factor) is reused across chunks and calls.
struct Item { std::string key; const half* w; int64_t out, in; };

inline void quantize_items(sycl::queue& q, exl3::Hessian& hs, const std::vector<Item>& items, int K, uint64_t seed, std::vector<exl3::QuantOut>& outs)
{
    size_t i = 0;
    while (i < items.size())
    {
        std::vector<DBuf<float>> Ws;
        std::vector<exl3::QuantIn> ins;
        int64_t N = 0;
        while (i < items.size() && (ins.empty() || N + items[i].out <= 32768))
        {
            Ws.push_back(transpose_f32(q, items[i].w, items[i].out, items[i].in));
            ins.push_back({items[i].key, items[i].out, Ws.back().get()});
            N += items[i].out; ++i;
        }
        exl3::QuantArgs qa; qa.K = K; qa.keep_L = true;
        for (auto& o : exl3::quantize_group(q, hs, ins, qa, seed)) outs.push_back(std::move(o));
    }
}

inline exl3::Hessian finalize(sycl::queue& q, DBuf<float>& H, int64_t k, int64_t count, uint64_t seed)
{
    return exl3::finalize_hessian(q, std::move(H), k, count, seed);
}

// Dequantize an output to a fp16 (in, out) weight on the card.
inline DBuf<half> dequant(sycl::queue& q, const exl3::QuantOut& o, const DBuf<int>& perm, DBuf<float>& tmp)
{
    DBuf<uint16_t> tr(q, o.trellis.size()), su(q, o.suh.size()), sv(q, o.svh.size());
    tr.upload(o.trellis.data(), o.trellis.size()); su.upload(o.suh.data(), o.suh.size()); sv.upload(o.svh.data(), o.svh.size());
    if (tmp.n < (size_t) (o.k * o.n)) tmp = DBuf<float>(q, (size_t) (o.k * o.n));
    DBuf<half> w(q, (size_t) (o.k * o.n));
    exl3::dequant_linear(q, tr.get(), su.get(), sv.get(), o.k, o.n, o.K, exl3::CB_MUL1, perm.get(), tmp.get(), w.get());
    return w;
}

inline void quantize_layer(sycl::queue& q, const Config& c, int64_t li, Layer& L, Capture& cap, const Recipe& rc, const DBuf<int>& perm,
                           std::vector<exl3::QuantOut>& outs)
{
    const auto p = c.lp(li);
    const uint64_t base = exl3::mix64(exl3::hash_str(p));
    const int64_t D = c.D, F = c.F;
    auto log = [&](const char* tag, size_t from, rt::Timer& t) {
        double eb = 0, ea = 0; for (size_t i = from; i < outs.size(); ++i) { eb += outs[i].err_before; ea += outs[i].err_after; }
        const double n = std::max<size_t>(1, outs.size() - from);
        std::printf("    %-12s %5zu tensors, mean err %.6f -> %.6f, %.1f s\n", tag, outs.size() - from, eb / n, ea / n, t.s());
        std::fflush(stdout);
    };
    // Attention-site input group.
    { rt::Timer t; const size_t from = outs.size();
      auto hs = finalize(q, cap.attn_in, D, cap.n_attn_in, base ^ 1);
      if (L.full)
          quantize_items(q, hs, {{p + "self_attn.q_proj", L.q.w.get(), L.q.out, L.q.in}, {p + "self_attn.k_proj", L.k.w.get(), L.k.out, L.k.in},
                                 {p + "self_attn.v_proj", L.v.w.get(), L.v.out, L.v.in},
                                 {p + "self_attn.indexer.index_qk_proj", L.idx.w.get(), L.idx.out, L.idx.in}}, rc.attn(), base ^ 1, outs);
      else
          quantize_items(q, hs, {{p + "linear_attn.in_proj_qkv", L.qkv.w.get(), L.qkv.out, L.qkv.in}, {p + "linear_attn.in_proj_z", L.z.w.get(), L.z.out, L.z.in}},
                         rc.gdn(li), base ^ 1, outs);
      log(L.full ? "attn.input" : "gdn.input", from, t); }
    // Attention-site output group.
    { rt::Timer t; const size_t from = outs.size();
      const Lin& o = L.full ? L.o : L.out;
      auto hs = finalize(q, cap.attn_o, o.in, cap.n_attn_o, base ^ 2);
      quantize_items(q, hs, {{p + (L.full ? "self_attn.o_proj" : "linear_attn.out_proj"), o.w.get(), o.out, o.in}}, L.full ? rc.attn() : rc.gdn(li), base ^ 2, outs);
      log(L.full ? "attn.o" : "gdn.out", from, t); }
    // MoE input group: every expert's gate and up, and the shared expert's.
    { rt::Timer t; const size_t from = outs.size();
      auto hs = finalize(q, cap.mlp_in, D, cap.n_mlp_in, base ^ 3);
      std::vector<Item> items;
      for (int64_t e = 0; e < c.E; ++e)
      {
          const half* gu = L.ex.gu.get() + e * 2 * F * D;
          items.push_back({p + "mlp.experts." + std::to_string(e) + ".gate_proj", gu, F, D});
          items.push_back({p + "mlp.experts." + std::to_string(e) + ".up_proj", gu + F * D, F, D});
      }
      quantize_items(q, hs, items, rc.experts(li), base ^ 3, outs);
      quantize_items(q, hs, {{p + "mlp.shared_expert.gate_proj", L.s_gate.w.get(), L.s_gate.out, L.s_gate.in},
                             {p + "mlp.shared_expert.up_proj", L.s_up.w.get(), L.s_up.out, L.s_up.in}}, rc.shared(li), base ^ 3, outs);
      log("mlp.input", from, t); }
    // Shared expert down.
    { rt::Timer t; const size_t from = outs.size();
      auto hs = finalize(q, cap.s_down, c.SF, cap.n_s_down, base ^ 4);
      quantize_items(q, hs, {{p + "mlp.shared_expert.down_proj", L.s_down.w.get(), L.s_down.out, L.s_down.in}}, rc.shared(li), base ^ 4, outs);
      log("shared.down", from, t); }
    // Every routed expert's down, each with its own Hessian.
    { rt::Timer t; const size_t from = outs.size();
      for (int64_t e = 0; e < c.E; ++e)
      {
          DBuf<float> H(q, (size_t) F * F);
          q.memcpy(H.get(), cap.e_down.get() + e * F * F, sizeof(float) * F * F).wait();
          auto hs = finalize(q, H, F, cap.n_e_down, base ^ (5 + (uint64_t) e));
          quantize_items(q, hs, {{p + "mlp.experts." + std::to_string(e) + ".down_proj", L.ex.dn.get() + e * D * F, D, F}}, rc.experts(li), base ^ (5 + (uint64_t) e), outs);
      }
      log("experts.down", from, t); }
}

// Replace a layer's weights by the dequantized EXL3 ones.
inline void install_quantized(sycl::queue& q, const Config& c, int64_t li, Layer& L, const std::vector<exl3::QuantOut>& outs, const DBuf<int>& perm)
{
    const auto p = c.lp(li);
    const int64_t D = c.D, F = c.F;
    std::map<std::string, const exl3::QuantOut*> by;
    for (auto& o : outs) by[o.key] = &o;
    DBuf<float> tmp(q, 1);
    auto put = [&](Lin& l, const std::string& key) { const auto& o = *by.at(p + key); l.w = dequant(q, o, perm, tmp); l.in = o.k; l.out = o.n; l.hf = false; };
    if (L.full) { put(L.q, "self_attn.q_proj"); put(L.k, "self_attn.k_proj"); put(L.v, "self_attn.v_proj"); put(L.o, "self_attn.o_proj"); }
    else { put(L.qkv, "linear_attn.in_proj_qkv"); put(L.z, "linear_attn.in_proj_z"); put(L.out, "linear_attn.out_proj"); }
    put(L.s_gate, "mlp.shared_expert.gate_proj"); put(L.s_up, "mlp.shared_expert.up_proj"); put(L.s_down, "mlp.shared_expert.down_proj");
    DBuf<half> gu(q, (size_t) c.E * D * 2 * F), dn(q, (size_t) c.E * F * D);
    for (int64_t e = 0; e < c.E; ++e)
    {
        const auto es = p + "mlp.experts." + std::to_string(e) + ".";
        auto g = dequant(q, *by.at(es + "gate_proj"), perm, tmp), u = dequant(q, *by.at(es + "up_proj"), perm, tmp);
        auto d = dequant(q, *by.at(es + "down_proj"), perm, tmp);
        half* blk = gu.get() + e * D * 2 * F;
        q.ext_oneapi_memcpy2d(blk, 2 * F * sizeof(half), g.get(), F * sizeof(half), F * sizeof(half), D);
        q.ext_oneapi_memcpy2d(blk + F, 2 * F * sizeof(half), u.get(), F * sizeof(half), F * sizeof(half), D);
        q.memcpy(dn.get() + e * F * D, d.get(), sizeof(half) * F * D).wait();
    }
    L.ex.gu = std::move(gu); L.ex.dn = std::move(dn); L.ex.hf = false;
}

// ---------------------------------------------------------------------------------------------------------------
// Stored tensors.

inline std::vector<uint16_t> to_f16(const st::TensorView& t)
{
    std::vector<uint16_t> v((size_t) t.numel());
    const uint16_t* p = (const uint16_t*) t.data;
    if (t.dtype == st::DType::F16) std::memcpy(v.data(), p, t.nbytes);
    else for (size_t i = 0; i < v.size(); ++i) v[i] = exl3::f16_from_float(st::bf16_to_float(p[i]));
    return v;
}

inline void add_exl3(std::vector<st::OutTensor>& ts, exl3::QuantOut& o)
{
    ts.push_back(st::OutTensor::from_vector(o.key + ".suh", "F16", {o.k}, std::move(o.suh)));
    ts.push_back(st::OutTensor::from_vector(o.key + ".svh", "F16", {o.n}, std::move(o.svh)));
    ts.push_back(st::OutTensor::from_vector(o.key + ".mul1", "I32", {}, std::vector<int32_t>{(int32_t) exl3::MUL1_MULT}));
    ts.push_back(st::OutTensor::from_vector(o.key + ".trellis", "I16", {o.k / 16, o.n / 16, 16 * o.K}, std::move(o.trellis)));
}

// One layer's tensors (notes/qwen4exp-format.md §3): EXL3 linears, fp16 tables and small linears, BF16 norms/scalars.
inline std::vector<st::OutTensor> layer_tensors(const st::Model& m, const Config& c, int64_t li, std::vector<exl3::QuantOut>& outs)
{
    std::vector<st::OutTensor> ts;
    const auto p = c.lp(li);
    auto copy = [&](const std::string& n) { const auto& t = m.get(n); ts.push_back(st::OutTensor::from_vector(n, st::dtype_name(t.dtype), t.shape, std::vector<uint8_t>(t.data, t.data + t.nbytes))); };
    auto f16 = [&](const std::string& n) { const auto& t = m.get(n); ts.push_back(st::OutTensor::from_vector(n, "F16", t.shape, to_f16(t))); };
    std::map<std::string, exl3::QuantOut*> by;
    for (auto& o : outs) by[o.key] = &o;
    auto q = [&](const std::string& key) { add_exl3(ts, *by.at(p + key)); };
    if (li == c.ple_layer)
    {
        f16(p + "ple.key_proj.weight"); f16(p + "ple.value_proj.weight"); f16(p + "ple.conv1d.weight");
        copy(p + "ple.norm_key.weight"); copy(p + "ple.norm_query.weight"); copy(p + "ple.norm_conv.weight");
    }
    for (const char* site : {"attn_hyper_connection.", "mlp_hyper_connection."})
    {
        f16(p + site + std::string("input_mix_weight_down.weight")); f16(p + site + std::string("input_mix_weight_up.weight"));
        f16(p + site + std::string("block_inject_weight.weight")); f16(p + site + std::string("hc_norm.weight"));
        if (std::string(site) == "attn_hyper_connection.")
        {
            if (!c.full[li])
            {
                q("linear_attn.in_proj_qkv"); q("linear_attn.in_proj_z"); q("linear_attn.out_proj");
                f16(p + "linear_attn.in_proj_a.weight"); f16(p + "linear_attn.in_proj_b.weight");
                copy(p + "linear_attn.conv1d.weight"); copy(p + "linear_attn.A_log"); copy(p + "linear_attn.dt_bias"); copy(p + "linear_attn.norm.weight");
            }
            else
            {
                q("self_attn.indexer.index_qk_proj"); copy(p + "self_attn.indexer.q_layernorm.weight"); copy(p + "self_attn.indexer.k_layernorm.weight");
                q("self_attn.q_proj"); q("self_attn.k_proj"); q("self_attn.v_proj"); q("self_attn.o_proj");
                copy(p + "self_attn.q_norm.weight"); copy(p + "self_attn.k_norm.weight");
            }
        }
    }
    f16(p + "mlp.gate.weight"); f16(p + "mlp.shared_expert_gate.weight");
    for (int64_t e = 0; e < c.E; ++e)
        for (const char* w : {"up_proj", "gate_proj", "down_proj"}) q("mlp.experts." + std::to_string(e) + "." + w);
    q("mlp.shared_expert.up_proj"); q("mlp.shared_expert.gate_proj"); q("mlp.shared_expert.down_proj");
    return ts;
}

// Decoded n-gram rows of one calibration row: comp[t * 16 + h] indexes the compact packed rows, ids the global rows.
inline void ple_rows(sycl::queue& q, const DBuf<uint16_t>& packed, const PleIndex& pi, int64_t row, int64_t T, int K,
                     const DBuf<uint16_t>& bias, const DBuf<int64_t>& offs, DBuf<int32_t>& dcomp, DBuf<int64_t>& dids, DBuf<half>& E)
{
    const int words = exl3::ngram::words_per_row(K);
    const int64_t n = T * 16;
    std::vector<int64_t> gid(n);
    const int32_t* comp = &pi.tok[(size_t) row * n];
    for (int64_t i = 0; i < n; ++i) gid[i] = pi.ids[comp[i]];
    dcomp.upload(comp, n); dids.upload(gid.data(), n);
    const uint16_t* pk = packed.get(); const int32_t* cp = dcomp.get(); const int64_t* gp = dids.get();
    const uint16_t* bp = bias.get(); const int64_t* op = offs.get(); half* out = E.get();
    q.parallel_for(sycl::range<2>(n, exl3::ngram::ROW), [=](sycl::id<2> id) {
        const int64_t r = id[0]; const int i = (int) id[1], ROW = exl3::ngram::ROW;
        const uint16_t* pr = pk + (int64_t) cp[r] * words;
        auto sym = [&](int pp) {
            pp = (pp % ROW + ROW) % ROW;
            uint32_t v = 0;
            for (int j = 0; j < K; ++j) { const int b = pp * K + j; v |= ((pr[1 + (b >> 4)] >> (b & 15)) & 1u) << j; }
            return v;
        };
        uint32_t st_ = 0;
        for (int mm = 0; mm * K < 16; ++mm) st_ |= sym(i - mm) << (mm * K);
        st_ &= 0xFFFF;
        int h = 0;
        for (int k2 = 1; k2 < 16; ++k2) if (gp[r] >= op[k2]) h = k2;
        const float v = (float) exl3::dev::decode<2>(st_) * (float) sycl::bit_cast<half>(pr[0]) + (float) sycl::bit_cast<half>(bp[h * ROW + i]);
        out[r * ROW + i] = half(v);
    }).wait();
}

// ---------------------------------------------------------------------------------------------------------------
// Output: [embed], one shard per layer (moved from the work folder), [mixer, lm_head, vision]; the n-gram table file
// is already in place and stays out of the index (the loaders open it by name).

struct QxArgs;
inline void compile(const struct QxArgs& a, const st::Model& m, const Config& c, const Recipe& rc);

struct QxArgs { fs::path in, out, work, cal_dir; int rows = 250, cols = 2048; double max_minutes = 0; int layers = -1; };

inline int convert(const QxArgs& a, const json& cfg)
{
    rt::Timer total;
    const Config c(cfg);
    const Recipe rc;
    st::Model m(a.in);
    fs::create_directories(a.out); fs::create_directories(a.work);
    Progress pg(a.work / "progress.json");
    if (pg.j.value("compiled", false)) { std::printf("%s is already complete\n", a.out.c_str()); return 0; }
    const int64_t T = a.cols, R = a.rows, S = c.S, D = c.D;
    const double max_s = a.max_minutes * 60;
    rt::Timer turn;
    std::printf("exl3conv (qwen4_exp): %s, %lld layers, %lld experts, %lld x %lld calibration tokens\n", a.in.filename().c_str(),
                (long long) c.L, (long long) c.E, (long long) R, (long long) T);

    // Calibration rows (Qwen: BOS = EOS = <|endoftext|>; random rows over the ids in use, 1 + the highest added id).
    std::vector<cal::Row> rows;
    {
        std::ifstream tf(a.in / "tokenizer.json");
        const auto tj = json::parse(tf);
        int64_t maxid = 0;
        for (auto& [k, v] : tj["model"]["vocab"].items()) maxid = std::max<int64_t>(maxid, v.get<int64_t>());
        for (auto& t : tj["added_tokens"]) maxid = std::max<int64_t>(maxid, t["id"].get<int64_t>());
        tok::BPE tk((a.in / "tokenizer.json").string());
        rows = cal::default_rows(tk, a.cal_dir, (int) R, (int) T, (int) c.eos, (int) c.eos, (int) (maxid + 1));
    }
    const Ngram ng = ngram_info(m, c);
    const PleIndex pi = ple_index(c, ng, rows);
    std::printf("  %zu distinct n-gram rows feed the calibration\n", pi.ids.size());
    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    const int KN = rc.ngram();

    // Stage 1: the n-gram table.
    if (!pg.j.value("ngram_done", false))
    {
        std::printf("n-gram table (K %d)\n", KN);
        if (!ngram_stage(q, m, c, ng, pi, a.in, a.out, a.work, pg, KN, turn, max_s)) { std::printf("turn over; run again to continue\n"); return 99; }
        pg.j["ngram_done"] = true; pg.save();
    }
    const auto bias_h = pg.j["ngram_bias"].get<std::vector<uint16_t>>();
    const int words = exl3::ngram::words_per_row(KN);

    // Stage 2: the state.
    Upload up(q);
    DBuf<int> perm(q, 256); perm.upload(exl3::tile_perm().data(), 256);
    const size_t rf = (size_t) T * S * D;
    HostState hs(q, R, rf);
    int64_t start = pg.j.value("next_layer", (int64_t) 0);
    const auto state_file = a.work / "state.bin";
    if (start > 0 || pg.j.value("layers_done", false)) { rt::Timer t; hs.load(state_file); std::printf("state loaded (%.1f s)\n", t.s()); }
    else
    {
        const uint16_t* E = (const uint16_t*) m.get(c.prefix + "embed_tokens.weight").data;
        for (int64_t r = 0; r < R; ++r)
            for (int64_t t = 0; t < T; ++t)
                for (int64_t s = 0; s < S; ++s)
                    for (int64_t d = 0; d < D; ++d) hs.rows[r][(t * S + s) * D + d] = st::bf16_to_float(E[(int64_t) rows[r][t] * D + d]);
    }
    const int64_t stop = a.layers > 0 ? std::min<int64_t>(c.L, a.layers) : c.L;
    Scratch sc(q, c, T);
    DBuf<uint16_t> ws(q, 1);
    DBuf<float> xr(q, rf);
    const auto inv = c.inv_freq();
    DBuf<float> dinv(q, inv.size()); dinv.upload(inv.data(), inv.size());
    DBuf<uint16_t> dbias(q, bias_h.size()); dbias.upload(bias_h.data(), bias_h.size());
    DBuf<int64_t> doffs(q, 16); doffs.upload(ng.offsets.data(), 16);
    DBuf<int32_t> dcomp(q, T * 16); DBuf<int64_t> dids(q, T * 16); DBuf<half> Erow(q, T * D);
    const int nref = (int) std::min<int64_t>(5, R);

    // Stage 3: the layers.
    for (int64_t li = start; li < stop && !pg.j.value("layers_done", false); ++li)
    {
        rt::Timer tl;
        std::printf("layer %lld/%lld (%s)\n", (long long) li, (long long) c.L, c.full[li] ? "full attention" : "Gated DeltaNet");
        Layer L = qwen4exp::load_layer(q, m, c, li, up, ws);
        DBuf<uint16_t> ple_packed;
        if (L.ple)
        {
            std::vector<uint16_t> h(pi.ids.size() * words);
            std::ifstream f(a.work / "ple_rows.bin", std::ios::binary);
            f.read((char*) h.data(), (std::streamsize) (h.size() * 2));
            if (!f) throw std::runtime_error("ple_rows.bin is incomplete");
            ple_packed = DBuf<uint16_t>(q, h.size()); ple_packed.upload(h.data(), h.size());
        }
        auto rowE = [&](int64_t r) -> const half* {
            if (!L.ple) return nullptr;
            ple_rows(q, ple_packed, pi, r, T, KN, dbias, doffs, dcomp, dids, Erow);
            return Erow.get();
        };
        Capture cap; cap.alloc(q, c, L.full);
        std::vector<std::vector<float>> refs(nref, std::vector<float>(rf));
        rt::Timer tc;
        for (int64_t r = 0; r < R; ++r)
        {
            q.memcpy(xr.get(), hs.rows[r], rf * 4).wait();
            qwen4exp::forward_layer(q, c, L, xr.get(), T, sc, dinv.get(), &cap, rowE(r));
            if (r < nref)
            {
                q.memcpy(xr.get(), hs.rows[r], rf * 4).wait();
                qwen4exp::forward_layer(q, c, L, xr.get(), T, sc, dinv.get(), nullptr, rowE(r));
                q.memcpy(refs[r].data(), xr.get(), rf * 4).wait();
            }
        }
        std::printf("  capture %.1f s\n", tc.s());
        std::vector<exl3::QuantOut> outs;
        quantize_layer(q, c, li, L, cap, rc, perm, outs);
        cap = Capture();
        install_quantized(q, c, li, L, outs, perm);
        rt::Timer ta;
        double rfn = 0;
        for (int64_t r = 0; r < R; ++r)
        {
            q.memcpy(xr.get(), hs.rows[r], rf * 4).wait();
            qwen4exp::forward_layer(q, c, L, xr.get(), T, sc, dinv.get(), nullptr, rowE(r));
            q.memcpy(hs.rows[r], xr.get(), rf * 4).wait();
            if (r < nref)
            {
                double num = 0, den = 0;
                for (size_t i = 0; i < rf; ++i) { const double d = hs.rows[r][i] - refs[r][i]; num += d * d; den += (double) refs[r][i] * refs[r][i]; }
                rfn += std::sqrt(num / std::max(den, 1e-30)) / nref;
            }
        }
        std::printf("  advance %.1f s; streams vs unquantized (rows 0-%d): rfn %.5f\n", ta.s(), nref - 1, rfn);
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        st::write_file(a.work / fn, layer_tensors(m, c, li, outs));
        // next_layer is where state.bin resumes: it moves only with state.bin (a kill in between redoes those layers).
        std::printf("  layer done in %.1f s (total %.1f min)\n", tl.s(), total.s() / 60);
        std::fflush(stdout);
        if (li + 1 < stop && max_s > 0 && turn.s() > max_s)
        {
            rt::Timer t; hs.save(state_file);
            pg.j["next_layer"] = li + 1; pg.save();
            std::printf("turn over after layer %lld: state saved in %.1f s; run again to continue\n", (long long) li, t.s());
            return 99;
        }
    }
    if (stop < c.L) { std::printf("stopped after %lld layers (--layers)\n", (long long) stop); return 0; }
    if (!pg.j.value("layers_done", false)) { hs.save(state_file); pg.j["next_layer"] = c.L; pg.j["layers_done"] = true; pg.save(); }

    // Stage 4: final mixer and lm_head.
    if (!pg.j.value("head_done", false))
    {
        rt::Timer th;
        std::printf("lm_head (K %d)\n", rc.head());
        qwen4exp::Site mix = qwen4exp::load_site(q, m, c.prefix + "hyper_connection_mixer.", up);
        DBuf<float> H(q, (size_t) D * D); H.zero();
        for (int64_t r = 0; r < R; ++r)
        {
            q.memcpy(xr.get(), hs.rows[r], rf * 4).wait();
            qwen4exp::site_mix(q, c, mix, xr.get(), T, sc, false);
            qwen4exp::hess(q, sc.y.get(), T, D, D, H.get());
        }
        q.wait();
        const auto& lh = m.get("lm_head.weight");
        DBuf<half> lw(q, (size_t) lh.numel());
        {
            const int64_t piece = 64ll << 20;
            DBuf<uint16_t> stg(q, piece);
            for (int64_t o = 0; o < lh.numel(); o += piece)
            {
                const int64_t n = std::min(piece, lh.numel() - o);
                up(stg.get(), (const uint16_t*) lh.data + o, n * 2);
                nn::bf16_to_half(q, stg.get(), n, lw.get() + o).wait();
            }
        }
        auto hss = finalize(q, H, D, R * T, exl3::mix64(exl3::hash_str("lm_head")));
        std::vector<exl3::QuantOut> outs;
        quantize_items(q, hss, {{"lm_head", lw.get(), lh.shape[0], lh.shape[1]}}, rc.head(), exl3::mix64(exl3::hash_str("lm_head")), outs);
        // quantize_items gives an item wider than its 32768-column chunk a pass of its own, so the head is one tensor.
        if (outs.size() != 1) throw std::runtime_error("lm_head came back in pieces");
        exl3::QuantOut head = std::move(outs[0]);
        std::vector<st::OutTensor> ts;
        for (const char* n : {"input_mix_weight_down.weight", "input_mix_weight_up.weight", "hc_norm.weight"})
        {
            const auto name = c.prefix + "hyper_connection_mixer." + n;
            ts.push_back(st::OutTensor::from_vector(name, "F16", m.get(name).shape, to_f16(m.get(name))));
        }
        add_exl3(ts, head);
        st::write_file(a.work / "head.safetensors", ts);
        pg.j["head_done"] = true; pg.save();
        std::printf("  lm_head done in %.1f s\n", th.s());
    }
    // Stage 5: the output model.
    compile(a, m, c, rc);
    pg.j["compiled"] = true; pg.save();
    fs::remove(state_file);
    std::printf("done in %.1f min\n", total.s() / 60);
    return 0;
}

inline void compile(const QxArgs& a, const st::Model& m, const Config& c, const Recipe& rc)
{
    std::printf("compiling the output model\n");
    const size_t nshards = (size_t) c.L + 2;
    auto shard = [&](size_t i) { char fn[64]; std::snprintf(fn, sizeof fn, "model-%05zu-of-%05zu.safetensors", i + 1, nshards); return std::string(fn); };
    json index = {{"metadata", {{"total_size", 0}}}, {"weight_map", json::object()}};
    uint64_t total = 0;
    json storage = json::object();
    auto note = [&](const std::string& name, const std::string& dt, const std::vector<int64_t>& shape, uint64_t n, const std::string& file) {
        index["weight_map"][name] = file; total += n;
        const auto dot = name.rfind('.');
        const std::string mod = name.substr(0, dot), leaf = name.substr(dot + 1);
        static const std::map<std::string, std::string> tn = {{"F16", "torch.float16"}, {"BF16", "torch.bfloat16"}, {"F32", "torch.float32"}, {"I16", "torch.int16"}, {"I32", "torch.int32"}, {"I64", "torch.int64"}};
        storage[mod]["stored_tensors"][name] = {{"shape", shape}, {"n_bytes", n}, {"dtype", tn.count(dt) ? tn.at(dt) : dt}};
        if (leaf == "trellis") { storage[mod]["quant_format"] = "exl3"; storage[mod]["bits_per_weight"] = shape[2] / 16; storage[mod]["mul1_multiplier"] = exl3::MUL1_MULT; }
    };
    // Shard 0: the embedding.
    {
        const auto& e = m.get(c.prefix + "embed_tokens.weight");
        std::vector<st::OutTensor> ts = {st::OutTensor::from_bytes(e.name, st::dtype_name(e.dtype), e.shape, e.data, e.nbytes)};
        st::write_file(a.out / shard(0), ts);
        note(e.name, st::dtype_name(e.dtype), e.shape, e.nbytes, shard(0));
    }
    // Layers: move each file into place.
    uint64_t lin_bits = 0, lin_weights = 0;
    for (int64_t li = 0; li < c.L; ++li)
    {
        char fn[64]; std::snprintf(fn, sizeof fn, "layer_%03lld.safetensors", (long long) li);
        {
            st::File f(a.work / fn);
            for (const auto& n : f.order())
            {
                const auto& t = f.tensors().at(n);
                note(n, st::dtype_name(t.dtype), t.shape, t.nbytes, shard((size_t) li + 1));
                if (n.size() > 8 && n.compare(n.size() - 8, 8, ".trellis") == 0)
                { const uint64_t w = (uint64_t) t.shape[0] * 16 * t.shape[1] * 16; lin_weights += w; lin_bits += w * (uint64_t) (t.shape[2] / 16); }
            }
        }
        std::error_code ec;
        fs::rename(a.work / fn, a.out / shard((size_t) li + 1), ec);
        if (ec) { fs::copy_file(a.work / fn, a.out / shard((size_t) li + 1), fs::copy_options::overwrite_existing); fs::remove(a.work / fn); }
    }
    // Last shard: mixer + lm_head (from the head file), then the vision tower as BF16.
    {
        st::File hf(a.work / "head.safetensors");
        std::vector<st::OutTensor> ts;
        for (const auto& n : hf.order()) { const auto& t = hf.tensors().at(n); ts.push_back(st::OutTensor::from_bytes(n, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes)); }
        for (const auto& n : m.names())
            if (n.rfind("model.visual.", 0) == 0) { const auto& t = m.get(n); ts.push_back(st::OutTensor::from_bytes(n, st::dtype_name(t.dtype), t.shape, t.data, t.nbytes)); }
        st::write_file(a.out / shard(nshards - 1), ts);
        for (auto& t : ts) note(t.name, t.dtype, t.shape, t.nbytes, shard(nshards - 1));
    }
    index["metadata"]["total_size"] = total;
    std::ofstream(a.out / "model.safetensors.index.json") << index.dump(4);
    std::ifstream cf(a.in / "config.json");
    json cfg = json::parse(cf);
    const double bpw = lin_weights ? std::round(100.0 * (double) lin_bits / (double) lin_weights) / 100.0 : 0.0;
    json qc = {{"quant_method", "exl3"}, {"version", "1.5.3"}, {"bits", bpw}, {"head_bits", rc.head()},
               {"calibration", {{"rows", a.rows}, {"cols", a.cols}}}, {"out_scales", "always"}, {"codebook", "mul1"}, {"quantizer", "exl3conv 0.1"}};
    cfg["quantization_config"] = qc;
    std::ofstream(a.out / "config.json") << cfg.dump(4);
    json qf = qc; qf["tensor_storage"] = storage;
    std::ofstream(a.out / "quantization_config.json") << qf.dump(4);
    for (auto& e : fs::directory_iterator(a.in))
    {
        if (!e.is_regular_file()) continue;
        const auto n = e.path().filename().string(), ext = e.path().extension().string();
        if (ext == ".safetensors" || ext == ".gguf" || n == "config.json" || n == "model.safetensors.index.json" || ext == ".bin" || ext == ".pt" || ext == ".pth")
            continue;
        fs::copy_file(e.path(), a.out / n, fs::copy_options::overwrite_existing);
    }
    std::printf("  %zu shards + ngram_embedding.safetensors; decoder linears at %.2f bpw\n", nshards, bpw);
}

} // namespace qx
