// exl3conv test tool: run a GGUF through llama.cpp (SYCL, on the B70) and dump per-layer hidden states, as the
// reference for the converter's calibration forward (tests/qwen35_check.py compares them).
//
//   llama_dump <model.gguf> <tokens.i32> <out dir> <last layer> [host-layer-from]
//
// tokens.i32: raw int32 token ids, one row. Writes <out>/embed.f32 (the input embedding) and <out>/l_out-N.f32 for
// N = 0..last layer, each [T, n_embd] fp32 as llama.cpp computed it. Every layer is assigned to the GPU; with
// host-layer-from F, the FFN weights of layers F.. stay in host memory and their matmuls are offloaded to the card
// per batch (op_offload), so a model larger than the card still computes there. Build: tests/build_llama_dump.sh.
#include "ggml-backend.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Dump
{
    std::string dir;
    int last = 0;
    int written = 0;
};

static bool wanted(const ggml_tensor* t, const Dump& d, std::string& file)
{
    const std::string n = t->name;
    if (n == "model.input_embed") { file = "embed.f32"; return true; }
    if (n.rfind("l_out-", 0) == 0)
    {
        const int l = std::atoi(n.c_str() + 6);
        if (l <= d.last) { file = n + ".f32"; return true; }
    }
    return false;
}

static bool cb(ggml_tensor* t, bool ask, void* ud)
{
    Dump& d = *(Dump*) ud;
    std::string file;
    if (!wanted(t, d, file)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32) { std::fprintf(stderr, "%s: type %d, not f32\n", t->name, (int) t->type); return true; }
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    if (ggml_is_contiguous(t)) ggml_backend_tensor_get(t, buf.data(), 0, n * 4);
    else { std::fprintf(stderr, "%s: not contiguous\n", t->name); return true; }
    std::ofstream(d.dir + "/" + file, std::ios::binary).write((const char*) buf.data(), (std::streamsize) (n * 4));
    std::printf("dumped %s [%lld, %lld]\n", t->name, (long long) t->ne[1], (long long) t->ne[0]);
    ++d.written;
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 5) { std::fprintf(stderr, "usage: llama_dump <model.gguf> <tokens.i32> <out dir> <last layer> [host-layer-from]\n"); return 2; }
    Dump d; d.dir = argv[3]; d.last = std::atoi(argv[4]);
    const int host_from = argc > 5 ? std::atoi(argv[5]) : -1;
    std::vector<llama_token> toks;
    {
        std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        toks.resize((size_t) f.tellg() / 4);
        f.seekg(0); f.read((char*) toks.data(), (std::streamsize) (toks.size() * 4));
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.use_extra_bufts = false;
    std::string pat;
    std::vector<llama_model_tensor_buft_override> ov;
    if (host_from >= 0)
    {
        // blk.(host_from..999).ffn_* -> host buffer
        std::string alts;
        for (int l = host_from; l < 200; ++l) alts += (alts.empty() ? "" : "|") + std::to_string(l);
        pat = "blk\\.(" + alts + ")\\.ffn_.*";
        ov.push_back({pat.c_str(), ggml_backend_cpu_buffer_type()});
    }
    ov.push_back({nullptr, nullptr});
    mp.tensor_buft_overrides = ov.data();
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + 16;
    cp.n_batch = cp.n_ubatch = (uint32_t) toks.size();
    cp.n_seq_max = 1;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = &d;
    cp.op_offload = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }
    if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int32_t) toks.size()))) { std::fprintf(stderr, "decode failed\n"); return 1; }
    std::printf("%d tensors dumped for %zu tokens\n", d.written, toks.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return d.written == d.last + 2 ? 0 : 1;
}
