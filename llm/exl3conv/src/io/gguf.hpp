// exl3conv: GGUF reading, so a llama.cpp GGUF can stand in for the Hugging Face checkpoint it was made from.
//
// gguf::File maps a GGUF file read-only and parses its metadata and tensor table. gguf::HfMap names every tensor
// the way the original Hugging Face checkpoint does (name, shape, memory layout, values) and undoes what llama.cpp's
// converter did to it; materialize() writes one tensor in HF layout as BF16. st::Model (safetensors.hpp) uses both
// to read a directory holding one GGUF as if it were the BF16 safetensors checkpoint.
//
// Tensor types read: F32, F16, BF16, Q8_0. Anything else is an error that names the tensor and its type.
//
// Supported architectures: qwen35 (Qwen3_5ForConditionalGeneration's text model, as written by
// engines/llama.cpp/conversion/qwen.py: Qwen3_5TextModel <- _LinearAttentionVReorderBase <- Qwen3NextModel, with
// _QwenMtpMixin). The inverted transforms, per GGUF tensor (HF name in brackets):
//   blk.N.attn_norm, post_attention_norm, attn_q_norm, attn_k_norm, output_norm, blk.MTP.nextn.{enorm,hnorm,
//     shared_head_norm}  -> w - 1  (llama.cpp added 1 to every "*norm.weight" except linear_attn.norm)
//   blk.N.ssm_a [linear_attn.A_log]          -> log(-x)  (llama.cpp stored -exp(A_log)), V heads un-tiled
//   blk.N.ssm_dt.bias [linear_attn.dt_bias]  -> renamed back, V heads un-tiled
//   blk.N.ssm_conv1d [linear_attn.conv1d]    -> [C, 4] back to [C, 1, 4], V channels (after q and k) un-tiled
//   blk.N.attn_qkv [in_proj_qkv]             -> V rows (after q and k) un-tiled
//   blk.N.attn_gate [in_proj_z], ssm_alpha [in_proj_a], ssm_beta [in_proj_b] -> rows un-tiled
//   blk.N.ssm_out [out_proj]                 -> columns un-tiled
//   blk.<n_layer + j>.*                      -> mtp.layers.j.*, nextn.eh_proj -> mtp.fc, nextn.enorm ->
//     mtp.pre_fc_norm_embedding, nextn.hnorm -> mtp.pre_fc_norm_hidden, nextn.shared_head_norm -> mtp.norm
// "Un-tiled": llama.cpp reorders the 48 V heads of a linear-attention layer from HF's grouped order (the value heads
// of key head g together: g*3 + j) to a tiled order (j*16 + g) so ggml can broadcast; this reverses it.
// Everything else (full-attention q/k/v/o with q and its gate interleaved per head, the MLP, embedding, lm_head) is
// stored as is: GGUF dims are the HF dims reversed with the same memory layout, no q/k RoPE permute for Qwen.
#pragma once

#include "json.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gguf {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ggml_type values (ggml.h) for the types read here.
enum Type : uint32_t { F32 = 0, F16 = 1, Q8_0 = 8, BF16 = 30 };

inline const char* type_name(uint32_t t)
{
    static const char* names[] = {"F32", "F16", "Q4_0", "Q4_1", "Q4_2", "Q4_3", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
                                  "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
                                  "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
                                  "BF16", "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0"};
    return t < sizeof(names) / sizeof(names[0]) ? names[t] : "unknown";
}

inline float f16_to_float(uint16_t h)
{
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t u;
    if (e == 0)
    {
        if (m == 0) u = s;
        else { float f = std::ldexp((float) m, -24); std::memcpy(&u, &f, 4); u |= s; }
    }
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

inline float bf16_to_float(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }

// Round to nearest even, as torch's .to(torch.bfloat16) does.
inline uint16_t float_to_bf16(float f)
{
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

struct Tensor
{
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;          // ggml order: ne[0] is the innermost (contiguous) dimension
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t row_len() const { return ne.empty() ? 1 : ne[0]; }
    int64_t rows() const { int64_t r = 1; for (size_t i = 1; i < ne.size(); ++i) r *= ne[i]; return r; }
    int64_t numel() const { return row_len() * rows(); }
};

// Bytes of one row of n elements, or 0 for a type this reader does not decode.
inline size_t row_bytes(uint32_t type, int64_t n)
{
    switch (type)
    {
        case F32: return (size_t) n * 4;
        case F16: case BF16: return (size_t) n * 2;
        case Q8_0: return n % 32 ? 0 : (size_t) (n / 32) * 34;
        default: return 0;
    }
}

// Decode row r of t (ne[0] values) into out.
inline void dequant_row(const Tensor& t, int64_t r, float* out)
{
    const int64_t n = t.row_len();
    const uint8_t* p = t.data + (size_t) r * row_bytes(t.type, n);
    switch (t.type)
    {
        case F32: std::memcpy(out, p, (size_t) n * 4); break;
        case F16: for (int64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, p + 2 * i, 2); out[i] = f16_to_float(h); } break;
        case BF16: for (int64_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, p + 2 * i, 2); out[i] = bf16_to_float(h); } break;
        case Q8_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 34)
            {
                uint16_t h; std::memcpy(&h, p, 2);
                const float d = f16_to_float(h);
                const int8_t* q = (const int8_t*) (p + 2);
                for (int i = 0; i < 32; ++i) out[b * 32 + i] = d * (float) q[i];
            }
            break;
        default: throw std::runtime_error("gguf: " + t.name + " has type " + type_name(t.type) + "; only F32, F16, BF16 and Q8_0 are read");
    }
}

class File
{
public:
    explicit File(const fs::path& path) : path_(path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat sb;
        if (fstat(fd_, &sb) != 0) throw std::runtime_error("cannot stat " + path.string());
        size_ = (size_t) sb.st_size;
        map_ = (const uint8_t*) mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) throw std::runtime_error("cannot mmap " + path.string());
        parse();
    }
    ~File() { if (map_ && map_ != MAP_FAILED) munmap((void*) map_, size_); if (fd_ >= 0) ::close(fd_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    // Metadata: scalars and strings as JSON values; arrays longer than 4096 entries (the vocabulary) are left out.
    const json& meta() const { return meta_; }
    const std::map<std::string, Tensor>& tensors() const { return tensors_; }
    const Tensor* find(const std::string& n) const { auto it = tensors_.find(n); return it == tensors_.end() ? nullptr : &it->second; }
    const fs::path& path() const { return path_; }

    void advise(const Tensor& t, int how) const
    {
        const uintptr_t page = 4096, a = (uintptr_t) t.data & ~(page - 1);
        const uintptr_t b = ((uintptr_t) t.data + t.nbytes + page - 1) & ~(page - 1);
        madvise((void*) a, b - a, how);
    }

private:
    size_t pos_ = 0;
    void need(size_t n) const { if (pos_ + n > size_) throw std::runtime_error("truncated GGUF header in " + path_.string()); }
    template <class T> T rd() { need(sizeof(T)); T v; std::memcpy(&v, map_ + pos_, sizeof(T)); pos_ += sizeof(T); return v; }
    std::string rd_str() { const uint64_t n = rd<uint64_t>(); need(n); std::string s((const char*) map_ + pos_, n); pos_ += n; return s; }
    json rd_value(uint32_t vt, bool keep)
    {
        switch (vt)
        {
            case 0: return rd<uint8_t>();  case 1: return rd<int8_t>();
            case 2: return rd<uint16_t>(); case 3: return rd<int16_t>();
            case 4: return rd<uint32_t>(); case 5: return rd<int32_t>();
            case 6: return rd<float>();    case 7: return rd<uint8_t>() != 0;
            case 8: return rd_str();
            case 10: return rd<uint64_t>(); case 11: return rd<int64_t>();
            case 12: return rd<double>();
            case 9:
            {
                const uint32_t et = rd<uint32_t>();
                const uint64_t n = rd<uint64_t>();
                keep = keep && n <= 4096;
                json a = json::array();
                for (uint64_t i = 0; i < n; ++i) { json v = rd_value(et, keep); if (keep) a.push_back(std::move(v)); }
                return keep ? a : json(nullptr);
            }
            default: throw std::runtime_error("unknown GGUF metadata type " + std::to_string(vt) + " in " + path_.string());
        }
    }
    void parse()
    {
        if (size_ < 24 || std::memcmp(map_, "GGUF", 4) != 0) throw std::runtime_error(path_.string() + " is not a GGUF file");
        pos_ = 4;
        const uint32_t version = rd<uint32_t>();
        if (version < 2 || version > 3) throw std::runtime_error("unsupported GGUF version " + std::to_string(version));
        const uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
        for (uint64_t i = 0; i < nkv; ++i)
        {
            const std::string k = rd_str();
            const uint32_t vt = rd<uint32_t>();
            json v = rd_value(vt, true);
            if (!v.is_null()) meta_[k] = std::move(v);
        }
        struct Info { std::string name; uint32_t type; std::vector<int64_t> ne; uint64_t off; };
        std::vector<Info> infos;
        for (uint64_t i = 0; i < nt; ++i)
        {
            Info in;
            in.name = rd_str();
            const uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd; ++d) in.ne.push_back(rd<int64_t>());
            in.type = rd<uint32_t>();
            in.off = rd<uint64_t>();
            infos.push_back(std::move(in));
        }
        const uint64_t align = meta_.contains("general.alignment") ? meta_["general.alignment"].get<uint64_t>() : 32;
        const uint64_t base = (pos_ + align - 1) / align * align;
        for (auto& in : infos)
        {
            Tensor t;
            t.name = in.name; t.type = in.type; t.ne = in.ne;
            const size_t rb = row_bytes(t.type, t.row_len());
            // Size is only known for the decoded types; others are kept (so the error names them) with no data.
            t.nbytes = rb * (size_t) t.rows();
            if (base + in.off + t.nbytes > size_) throw std::runtime_error("gguf: tensor past end of file: " + t.name);
            t.data = map_ + base + in.off;
            tensors_.emplace(t.name, std::move(t));
        }
    }

    fs::path path_;
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* map_ = nullptr;
    json meta_ = json::object();
    std::map<std::string, Tensor> tensors_;
};

// llama.cpp's V-head reorder over one span of an axis: HF index -> GGUF index. The span starts at `off` and holds
// nk * nvpk heads of hd entries each; HF orders them (g, j) (key head g, its j-th value head), GGUF (j, g).
struct Untile
{
    int64_t off = 0, nk = 0, nvpk = 0, hd = 0;
    bool on() const { return nk > 0; }
    int64_t operator()(int64_t i) const
    {
        if (!on() || i < off || i >= off + nk * nvpk * hd) return i;
        const int64_t rel = i - off, d = rel % hd, j = (rel / hd) % nvpk, g = rel / (hd * nvpk);
        return off + (j * nk + g) * hd + d;
    }
};

enum class Op { Copy, Minus1, LogNeg };

// One HF tensor and how to get it from the GGUF.
struct Source
{
    std::string hf_name, gguf_name;
    std::vector<int64_t> hf_shape;
    Op op = Op::Copy;
    Untile rows, cols;     // rows: over the GGUF rows (HF dim 0 .. n-2 flattened); cols: within a row (HF last dim)
    const Tensor* t = nullptr;
    size_t out_bytes() const { size_t n = 2; for (auto s : hf_shape) n *= (size_t) s; return n; }
};

// The HF view of a GGUF: every GGUF tensor mapped to its HF name. Throws for an architecture it does not know or a
// GGUF tensor it cannot place, so nothing is silently dropped.
class HfMap
{
public:
    explicit HfMap(const File& f, const fs::path& config_json = {})
    {
        const auto& m = f.meta();
        const std::string arch = m.value("general.architecture", "");
        if (arch != "qwen35")
            throw std::runtime_error("gguf: architecture '" + arch + "' is not supported as a stand-in for a HF checkpoint (only qwen35)");
        auto u = [&](const std::string& k) -> int64_t {
            if (!m.contains(arch + "." + k)) throw std::runtime_error("gguf: missing metadata " + arch + "." + k);
            return m[arch + "." + k].get<int64_t>();
        };
        const int64_t blocks = u("block_count"), nextn = m.value(arch + ".nextn_predict_layers", (int64_t) 0);
        n_layer_ = blocks - nextn;
        const int64_t nk = u("ssm.group_count"), nv = u("ssm.time_step_rank"), hk = u("ssm.state_size");
        const int64_t hv = u("ssm.inner_size") / nv, H = u("embedding_length");
        if (nv % nk) throw std::runtime_error("gguf: value heads not a multiple of key heads");
        if (!config_json.empty() && fs::exists(config_json)) check_config(config_json, H, nk, nv, hk, hv);

        const Untile v_rows{2 * nk * hk, nk, nv / nk, hv}, v_all{0, nk, nv / nk, hv}, v_one{0, nk, nv / nk, 1};
        for (auto& [gname, t] : f.tensors())
        {
            Source s;
            s.gguf_name = gname;
            s.t = &t;
            for (auto it = t.ne.rbegin(); it != t.ne.rend(); ++it) s.hf_shape.push_back(*it);
            std::string suf;
            int64_t bid = -1;
            if (gname.rfind("blk.", 0) == 0)
            {
                const size_t dot = gname.find('.', 4);
                bid = std::stoll(gname.substr(4, dot - 4));
                suf = gname.substr(dot + 1);
            }
            const bool mtp = bid >= n_layer_;
            const std::string lp = bid < 0 ? "" : mtp ? "mtp.layers." + std::to_string(bid - n_layer_) + "."
                                                       : "model.language_model.layers." + std::to_string(bid) + ".";
            auto set = [&](const std::string& hf, Op op = Op::Copy) { s.hf_name = hf; s.op = op; };
            if (bid < 0)
            {
                if (gname == "token_embd.weight") set("model.language_model.embed_tokens.weight");
                else if (gname == "output_norm.weight") set("model.language_model.norm.weight", Op::Minus1);
                else if (gname == "output.weight") set("lm_head.weight");
            }
            else if (suf == "attn_norm.weight") set(lp + "input_layernorm.weight", Op::Minus1);
            else if (suf == "post_attention_norm.weight") set(lp + "post_attention_layernorm.weight", Op::Minus1);
            else if (suf == "ffn_gate.weight") set(lp + "mlp.gate_proj.weight");
            else if (suf == "ffn_up.weight") set(lp + "mlp.up_proj.weight");
            else if (suf == "ffn_down.weight") set(lp + "mlp.down_proj.weight");
            else if (suf == "attn_q.weight") set(lp + "self_attn.q_proj.weight");
            else if (suf == "attn_k.weight") set(lp + "self_attn.k_proj.weight");
            else if (suf == "attn_v.weight") set(lp + "self_attn.v_proj.weight");
            else if (suf == "attn_output.weight") set(lp + "self_attn.o_proj.weight");
            else if (suf == "attn_q_norm.weight") set(lp + "self_attn.q_norm.weight", Op::Minus1);
            else if (suf == "attn_k_norm.weight") set(lp + "self_attn.k_norm.weight", Op::Minus1);
            else if (!mtp && suf == "attn_qkv.weight") { set(lp + "linear_attn.in_proj_qkv.weight"); s.rows = v_rows; }
            else if (!mtp && suf == "attn_gate.weight") { set(lp + "linear_attn.in_proj_z.weight"); s.rows = v_all; }
            else if (!mtp && suf == "ssm_alpha.weight") { set(lp + "linear_attn.in_proj_a.weight"); s.rows = v_one; }
            else if (!mtp && suf == "ssm_beta.weight") { set(lp + "linear_attn.in_proj_b.weight"); s.rows = v_one; }
            else if (!mtp && suf == "ssm_out.weight") { set(lp + "linear_attn.out_proj.weight"); s.cols = v_all; }
            else if (!mtp && suf == "ssm_norm.weight") set(lp + "linear_attn.norm.weight");
            else if (!mtp && suf == "ssm_a") { set(lp + "linear_attn.A_log", Op::LogNeg); s.cols = v_one; }
            else if (!mtp && suf == "ssm_dt.bias") { set(lp + "linear_attn.dt_bias"); s.cols = v_one; }
            else if (!mtp && suf == "ssm_conv1d.weight")
            {
                set(lp + "linear_attn.conv1d.weight");
                s.rows = v_rows;
                s.hf_shape = {t.ne.at(1), 1, t.ne.at(0)};   // HF Conv1d weight [C, 1, K]; llama.cpp squeezed the 1
            }
            else if (mtp && bid == n_layer_ && suf == "nextn.eh_proj.weight") set("mtp.fc.weight");
            else if (mtp && bid == n_layer_ && suf == "nextn.enorm.weight") set("mtp.pre_fc_norm_embedding.weight", Op::Minus1);
            else if (mtp && bid == n_layer_ && suf == "nextn.hnorm.weight") set("mtp.pre_fc_norm_hidden.weight", Op::Minus1);
            else if (mtp && bid == n_layer_ && suf == "nextn.shared_head_norm.weight") set("mtp.norm.weight", Op::Minus1);
            if (s.hf_name.empty()) throw std::runtime_error("gguf: no HF name known for tensor " + gname);
            if (!row_bytes(t.type, t.row_len()))
                throw std::runtime_error("gguf: " + gname + " has type " + type_name(t.type) + "; only F32, F16, BF16 and Q8_0 are read");
            if (sources_.count(s.hf_name)) throw std::runtime_error("gguf: two tensors map to " + s.hf_name);
            sources_.emplace(s.hf_name, std::move(s));
        }
    }

    const std::map<std::string, Source>& sources() const { return sources_; }
    int64_t n_layer() const { return n_layer_; }

private:
    static void check_config(const fs::path& p, int64_t H, int64_t nk, int64_t nv, int64_t hk, int64_t hv)
    {
        std::ifstream f(p);
        json c = json::parse(f);
        if (c.contains("text_config")) c = c["text_config"];
        auto chk = [&](const char* k, int64_t want) {
            if (c.contains(k) && c[k].get<int64_t>() != want)
                throw std::runtime_error(std::string("gguf: config.json ") + k + " = " + std::to_string(c[k].get<int64_t>()) +
                                         " but the GGUF says " + std::to_string(want));
        };
        chk("hidden_size", H); chk("linear_num_key_heads", nk); chk("linear_num_value_heads", nv);
        chk("linear_key_head_dim", hk); chk("linear_value_head_dim", hv);
    }
    int64_t n_layer_ = 0;
    std::map<std::string, Source> sources_;
};

// Write s in HF layout as BF16 into out (s.out_bytes() bytes). Rows are decoded straight into place, so the only
// memory used is the output; big tensors are split over a few threads.
inline void materialize(const Source& s, uint16_t* out)
{
    const Tensor& t = *s.t;
    const int64_t C = t.row_len(), R = t.rows();
    auto work = [&](int64_t r0, int64_t r1) {
        std::vector<float> row(C);
        for (int64_t r = r0; r < r1; ++r)
        {
            dequant_row(t, s.rows(r), row.data());
            uint16_t* o = out + r * C;
            for (int64_t c = 0; c < C; ++c)
            {
                float v = row[s.cols(c)];
                if (s.op == Op::Minus1) v -= 1.0f;
                else if (s.op == Op::LogNeg) v = (float) std::log(-(double) v);
                o[c] = float_to_bf16(v);
            }
        }
    };
    const int64_t elems = R * C;
    const int nth = elems < (1 << 22) ? 1 : (int) std::min<int64_t>({8, (int64_t) std::max(1u, std::thread::hardware_concurrency() / 2), R});
    if (nth <= 1) { work(0, R); return; }
    std::vector<std::thread> th;
    for (int i = 0; i < nth; ++i) th.emplace_back(work, R * i / nth, R * (i + 1) / nth);
    for (auto& x : th) x.join();
}

} // namespace gguf
