// exl3conv: the qwen3_5 calibration forward (arch/qwen35.hpp) on the B70 against llama.cpp's forward of the same model on the
// B70 (tests/llama_dump.cpp, a GGUF of the same weights), on the same tokens.
//
//   test_qwen35_forward <model dir> <tokens.i32> <llama dump dir> <layers, e.g. 0,1,3> [--chain N]
//
// Per listed layer L: the input is llama.cpp's own (embed.f32 for L = 0, else l_out-(L-1).f32), the output is compared with
// l_out-L.f32. Reported: the relative Frobenius error of the layer's contribution (output - input) and of the output, and the
// cosine of the contributions. --chain N also runs layers 0..N-1 in order from this tool's own embedding and compares each
// output with llama.cpp's (errors accumulate). The GGUF's weights differ from the BF16 ones by its rounding (Q8_0: ~0.3% per
// weight), so expect errors of that order, not zero; a wrong kernel shows up as tens of percent.
#include "arch/qwen35.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace q35;
namespace fs = std::filesystem;

static std::vector<float> read_f32(const fs::path& p, size_t n)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::vector<float> v(n);
    f.read((char*) v.data(), (std::streamsize) (n * 4));
    if (!f) throw std::runtime_error(p.string() + ": short");
    return v;
}

struct Cmp { double rel_delta, rel_out, cos; };

static Cmp compare(const std::vector<float>& in, const std::vector<float>& got, const std::vector<float>& ref)
{
    double nd = 0, dd = 0, no = 0, do_ = 0, dot = 0, a2 = 0, b2 = 0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        const double a = (double) got[i] - in[i], b = (double) ref[i] - in[i];
        nd += (a - b) * (a - b); dd += b * b;
        no += ((double) got[i] - ref[i]) * ((double) got[i] - ref[i]); do_ += (double) ref[i] * ref[i];
        dot += a * b; a2 += a * a; b2 += b * b;
    }
    return {std::sqrt(nd / dd), std::sqrt(no / do_), dot / std::sqrt(a2 * b2)};
}

int main(int argc, char** argv)
{
    if (argc < 5) { std::fprintf(stderr, "usage: test_qwen35_forward <model dir> <tokens.i32> <llama dump dir> <layers> [--chain N]\n"); return 2; }
    const fs::path dir = argv[1], ref = argv[3];
    std::vector<int> layers;
    { std::stringstream ss(argv[4]); for (std::string s; std::getline(ss, s, ',');) layers.push_back(std::stoi(s)); }
    int chain = 0;
    for (int i = 5; i + 1 < argc; ++i) if (std::string(argv[i]) == "--chain") chain = std::stoi(argv[i + 1]);
    std::vector<int32_t> toks;
    {
        std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
        toks.resize((size_t) f.tellg() / 4); f.seekg(0); f.read((char*) toks.data(), (std::streamsize) (toks.size() * 4));
    }
    std::ifstream cf(dir / "config.json");
    const Config c(json::parse(cf));
    st::Model m(dir);
    const int64_t T = (int64_t) toks.size(), D = c.D;
    const size_t n = (size_t) T * D;
    std::printf("qwen3_5 forward check: %lld tokens, layers %s, chain %d\n", (long long) T, argv[4], chain);

    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    Upload up(q);
    DBuf<uint16_t> scratch(q, 1);
    Scratch sc(q, c, T);
    const auto inv = c.inv_freq();
    DBuf<float> dinv(q, inv.size()); dinv.upload(inv.data(), inv.size());
    DBuf<float> x(q, n);
    int bad = 0;

    // Our embedding against llama.cpp's.
    std::vector<float> emb(n);
    {
        const auto& e = m.get(c.prefix + "embed_tokens.weight");
        for (int64_t t = 0; t < T; ++t)
            for (int64_t d = 0; d < D; ++d) emb[t * D + d] = elem_f32(e, (size_t) toks[t] * D + d);
        const auto r = read_f32(ref / "embed.f32", n);
        const std::vector<float> zero(n, 0.0f);
        const auto cm = compare(zero, emb, r);
        std::printf("embedding: rel %.5f\n", cm.rel_out);
    }
    for (int li : layers)
    {
        Layer L = load_layer(q, m, c.lp(li), c.full[li], up, scratch);
        const auto in = read_f32(ref / (li == 0 ? std::string("embed.f32") : "l_out-" + std::to_string(li - 1) + ".f32"), n);
        const auto r = read_f32(ref / ("l_out-" + std::to_string(li) + ".f32"), n);
        x.upload(in.data(), n);
        rt::Timer t;
        forward_layer(q, c, L, x.get(), T, sc, dinv.get(), nullptr);
        const double s = t.s();
        std::vector<float> got(n);
        x.download(got.data(), n);
        const auto cm = compare(in, got, r);
        const bool ok = cm.rel_delta < 0.05 && cm.cos > 0.998;
        bad += !ok;
        std::printf("layer %2d (%s): contribution rel err %.5f, cos %.6f; output rel err %.6f  (%.2f s)  %s\n", li, c.full[li] ? "attention" : "GDN      ",
                    cm.rel_delta, cm.cos, cm.rel_out, s, ok ? "ok" : "MISMATCH");
    }
    if (chain > 0)
    {
        x.upload(emb.data(), n);
        std::vector<float> prev = emb;
        for (int li = 0; li < chain; ++li)
        {
            Layer L = load_layer(q, m, c.lp(li), c.full[li], up, scratch);
            forward_layer(q, c, L, x.get(), T, sc, dinv.get(), nullptr);
            std::vector<float> got(n);
            x.download(got.data(), n);
            const auto r = read_f32(ref / ("l_out-" + std::to_string(li) + ".f32"), n);
            const auto cm = compare(prev, got, r);
            std::printf("chain layer %2d: output rel err %.6f\n", li, cm.rel_out);
            prev = got;
        }
    }
    std::printf("%s\n", bad ? "FAILED" : "all layers match");
    return bad ? 1 : 0;
}
