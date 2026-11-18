// Checks that st::Model reads a GGUF directory as the Hugging Face checkpoint it came from. CPU only; it compares
// tensors and runs no model.
//
//   tests/gguf_check.sh <gguf dir> <ref dir>
//
// <ref dir> holds some real BF16 shards of the HF checkpoint (model-*.safetensors), its model.safetensors.index.json,
// and optionally hdr/<shard>.json: each shard's JSON header (fetched with HTTP range requests, so shards that were not
// downloaded still give their shapes and dtypes). It reports:
//   1. for every tensor in the downloaded shards (model.visual.* skipped): name, shape and dtype as the reader has them,
//      and per tensor class the relative RMS error, the correlation and the share of bit-identical values;
//   2. for every non-vision name in the index: present in the reader, and shape and dtype as in hdr/ when it is there;
//   3. tensors the reader has that the index does not.
// Exit status 1 if any name, shape or dtype differs, or a class's worst relative RMS error is above 1%.
#include "io/safetensors.hpp"

#include <cmath>
#include <cstdio>
#include <regex>

namespace fs = std::filesystem;
using json = nlohmann::json;

static double to_f(st::DType d, const uint8_t* p, int64_t i)
{
    switch (d)
    {
        case st::DType::BF16: { uint16_t h; std::memcpy(&h, p + 2 * i, 2); return st::bf16_to_float(h); }
        case st::DType::F16: { uint16_t h; std::memcpy(&h, p + 2 * i, 2); return gguf::f16_to_float(h); }
        case st::DType::F32: { float f; std::memcpy(&f, p + 4 * i, 4); return f; }
        default: throw std::runtime_error("unexpected dtype");
    }
}

static std::string shape_str(const std::vector<int64_t>& s)
{
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

// float -> fp16, round to nearest even (GGML_FP32_TO_FP16), normal range only (Q8_0 scales).
static uint16_t f16_rne(float f)
{
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000; int e = (int) ((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffff;
    if (e <= 0) { // subnormal half
        if (e < -10) return (uint16_t) s;
        m |= 0x800000; const int sh = 14 - e; uint32_t h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t) (s | h);
    }
    uint32_t h = (uint32_t) (e << 10) | (m >> 13); const uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t) (s | h);
}

// What the reader should return for a Q8_0 tensor: the HF values through llama.cpp's quantize_row_q8_0_ref and back,
// rounded to BF16. Rows are HF rows (llama.cpp's reorders move whole 128-wide heads, so the 32-blocks are the same).
static double q8_exact(const st::TensorView& g, const st::TensorView& r)
{
    const int64_t C = r.shape.back(), R = r.numel() / C;
    double same = 0; float x[32];
    for (int64_t row = 0; row < R; ++row)
        for (int64_t b = 0; b < C / 32; ++b)
        {
            float amax = 0;
            for (int i = 0; i < 32; ++i) { x[i] = (float) to_f(r.dtype, r.data, row * C + b * 32 + i); amax = std::max(amax, std::fabs(x[i])); }
            const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f, dh = gguf::f16_to_float(f16_rne(d));
            for (int i = 0; i < 32; ++i)
            {
                const float v = dh * (float) (int8_t) std::roundf(x[i] * id);
                uint16_t got; std::memcpy(&got, g.data + 2 * (row * C + b * 32 + i), 2);
                same += gguf::float_to_bf16(v) == got;
            }
        }
    return same;
}

struct Cls { std::string type; int n = 0; double worst = 0, sum = 0, min_corr = 1, exact = 0, elems = 0, q8 = 0; };

int main(int argc, char** argv)
{
    if (argc != 3) { std::fprintf(stderr, "usage: %s <gguf dir> <ref dir>\n", argv[0]); return 2; }
    const fs::path gdir = argv[1], rdir = argv[2];
    int bad = 0;
    try
    {
        st::Model m(gdir);
        if (!m.is_gguf()) throw std::runtime_error(gdir.string() + " was not read as a GGUF");
        std::printf("reader: %zu tensors from %s\n", m.names().size(), m.gguf_file()->path().filename().c_str());

        // 1. Values against the downloaded shards.
        std::vector<fs::path> shards;
        for (auto& e : fs::directory_iterator(rdir)) if (e.path().extension() == ".safetensors") shards.push_back(e.path());
        std::sort(shards.begin(), shards.end());
        std::map<std::string, Cls> cls;
        const std::regex num(R"(\.(\d+)\.)");
        for (auto& sp : shards)
        {
            st::File f(sp);
            std::printf("comparing %s\n", sp.filename().c_str());
            for (auto& name : f.order())
            {
                if (name.rfind("model.visual.", 0) == 0) continue;
                const auto& r = f.tensors().at(name);
                if (!m.has(name)) { std::printf("  MISSING %s\n", name.c_str()); ++bad; continue; }
                const auto& g = m.get(name);
                if (g.shape != r.shape || g.dtype != r.dtype)
                {
                    std::printf("  MISMATCH %s: reader %s %s, HF %s %s\n", name.c_str(), st::dtype_name(g.dtype), shape_str(g.shape).c_str(),
                                st::dtype_name(r.dtype), shape_str(r.shape).c_str());
                    ++bad; m.dont_need(name); continue;
                }
                double se = 0, sr = 0, sg = 0, sgg = 0, srr = 0, sgr = 0, ex = 0;
                const int64_t n = r.numel();
                for (int64_t i = 0; i < n; ++i)
                {
                    const double a = to_f(g.dtype, g.data, i), b = to_f(r.dtype, r.data, i);
                    se += (a - b) * (a - b); srr += b * b; sgg += a * a; sr += b; sg += a; sgr += a * b;
                    ex += (a == b);
                }
                const double rel = srr > 0 ? std::sqrt(se / srr) : std::sqrt(se);
                const double cov = sgr / n - (sg / n) * (sr / n), vg = sgg / n - (sg / n) * (sg / n), vr = srr / n - (sr / n) * (sr / n);
                const double corr = (vg > 0 && vr > 0) ? cov / std::sqrt(vg * vr) : (se == 0 ? 1.0 : 0.0);
                std::string k = std::regex_replace(name, num, ".N.");
                Cls& c = cls[k];
                c.type = gguf::type_name(m.gguf_source(name)->t->type);
                if (m.gguf_source(name)->t->type == gguf::Q8_0) c.q8 += q8_exact(g, r);
                ++c.n; c.worst = std::max(c.worst, rel); c.sum += rel; c.min_corr = std::min(c.min_corr, corr); c.exact += ex; c.elems += n;
                m.dont_need(name);
                f.dont_need(r);
            }
        }
        std::printf("\n%-62s %4s %-6s %10s %10s %11s %8s %10s\n", "tensor class", "n", "type", "relRMS max", "mean", "corr min", "exact", "=Q8_0(HF)");
        for (auto& [k, c] : cls)
        {
            std::printf("%-62s %4d %-6s %10.3e %10.3e %11.7f %7.2f%% %9s\n", k.c_str(), c.n, c.type.c_str(), c.worst, c.sum / c.n, c.min_corr, 100 * c.exact / c.elems,
                        c.type == "Q8_0" ? (std::to_string(100 * c.q8 / c.elems).substr(0, 7) + "%").c_str() : "-");
            if (c.worst > 0.01) { std::printf("  ^ relative RMS error above 1%%\n"); ++bad; }
        }

        // 2. Names and shapes of the whole checkpoint.
        std::ifstream jf(rdir / "model.safetensors.index.json");
        const auto idx = json::parse(jf);
        std::map<std::string, json> hdr;
        if (fs::exists(rdir / "hdr"))
            for (auto& e : fs::directory_iterator(rdir / "hdr"))
            {
                std::ifstream hf(e.path());
                auto h = json::parse(hf);
                for (auto& [n, v] : h.items()) if (n != "__metadata__") hdr[n] = v;
            }
        int checked = 0, shaped = 0;
        std::map<std::string, bool> in_index;
        for (auto& [name, file] : idx["weight_map"].items())
        {
            if (name.rfind("model.visual.", 0) == 0) continue;
            in_index[name] = true;
            ++checked;
            if (!m.has(name)) { std::printf("MISSING from reader: %s\n", name.c_str()); ++bad; continue; }
            auto it = hdr.find(name);
            if (it == hdr.end()) continue;
            ++shaped;
            // The view's metadata: no decoding needed for the shape check.
            const auto shape = it->second["shape"].get<std::vector<int64_t>>();
            const auto dtype = st::parse_dtype(it->second["dtype"].get<std::string>());
            const auto& g = m.view_of(name);
            if (g.shape != shape || g.dtype != dtype)
            {
                std::printf("MISMATCH %s: reader %s %s, HF %s %s\n", name.c_str(), st::dtype_name(g.dtype), shape_str(g.shape).c_str(),
                            st::dtype_name(dtype), shape_str(shape).c_str());
                ++bad;
            }
        }
        int extra = 0;
        for (auto& n : m.names()) if (!in_index.count(n)) { std::printf("EXTRA in reader (not in the HF index): %s\n", n.c_str()); ++extra; }
        std::printf("\nindex: %d non-vision names, all present: %s; %d shape/dtype-checked against shard headers; %d extra\n", checked,
                    bad ? "see above" : "yes", shaped, extra);
        bad += extra;
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::printf("%s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
