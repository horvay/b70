#pragma once
// PEFT LoRA adapters (the GM LoRA), applied at run time as the PyTorch stack does (docs/forward-spec.md §9):
//   t = fp16(x @ A^T)  (x: the linear's raw fp16 input), delta = t @ (sB)^T, s = alpha / r * scale;
//   fp16 output: y = fp16(y + delta) (one rounding); fp32 output: y += fp32(fp16(delta)).
// A is kept fp16 [r, in], sB fp16 [out, r] (B scaled in fp32, then cast).
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "rt.h"
#include "st.h"

namespace glm {

struct LoraGroup;

struct Lora {
    Tensor A, sB;          // sB: [out, r / blocks] (blocks > 1: block-diagonal, only each block's own columns kept)
    Tensor sBT;            // decode layout: [blocks][r / blocks][out / blocks] (neighbouring outputs, neighbouring loads)
    int r = 0, in = 0, out = 0, blocks = 1;
    // decode (M <= group->rows): the first projection comes from a group of LoRAs on the same input, computed once
    // by its first member (lead) for all of them; the others read their slice
    LoraGroup* group = nullptr;
    int goff = 0;
    bool lead = false;
    void apply(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt) const;
    void apply_delta(sycl::queue& q, Workspace& ws, const half* t, int tstride, int M, void* y, DType ydt) const;
    size_t ws_bytes(int M) const { return (size_t)M * r * 2 + (size_t)M * out * 2 + 512; }
};

struct LoraGroup {
    Tensor A;          // fp16 [sum r, in]: the members' A stacked
    Tensor t;          // fp16 [rows, sum r]
    int R = 0, in = 0, rows = 8;
};

class LoraSet {
public:
    // dirs: one or more adapter folders (comma-separated), their deltas summed; scale multiplies alpha / r
    LoraSet(const std::string& dirs, float scale);
    // the adapter of a module (model name, e.g. "model.language_model.layers.3.self_attn.q_a_proj"), or null
    const Lora* get(const std::string& module) const;
    // KDA q / k / v as one rank-3r LoRA on the fused qkv projection (A stacked, sB block-diagonal), or null
    const Lora* fused_qkv(const std::string& attn_prefix);
    // merge a module's delta into an fp16 weight [out, in] (W = fp16(W + fp16(sB @ A))); false if it has none
    bool merge_into(const std::string& module, Tensor& W) const;
    size_t count() const { return mods_.size(); }
    // members applied in this order to the same input (the first computes the shared projection)
    void group(const std::vector<const Lora*>& members, int rows);

private:
    struct Raw { std::vector<float> A, B; int r = 0, in = 0, out = 0, blocks = 1; };   // host fp32, B scaled
    std::map<std::string, Raw> raw_;
    std::map<std::string, std::unique_ptr<Lora>> mods_;
    std::vector<std::unique_ptr<LoraGroup>> groups_;
    Lora* upload(const std::string& name, const Raw& rw);
};

}  // namespace glm
