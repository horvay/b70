#pragma once
// Qwen3.8-27B's vision tower (docs/qwen35-spec.md §5) and its image preprocessing. A page goes: decode (stb_image) ->
// RGB -> smart_resize to multiples of 32 -> bicubic with antialiasing -> (x / 255 - 0.5) / 0.5 -> 16-pixel patches,
// grouped by the 2 x 2 merge unit -> 27 ViT blocks -> merger -> one 5120-wide embedding per merge unit. The checkpoint's
// tower is bf16; here the weights are fp16 and the linears oneMKL's fp16 GEMM, the stream fp32.
#include <string>
#include <vector>
#include "q35.h"

namespace glm {

namespace q35v {
constexpr int D = 1152, H = 16, HD = 72, I = 4304, P = 16, MERGE = 2, NPOS = 48, DEPTH = 27;
constexpr int PIX = 3 * P * P;                  // one patch's values
constexpr int ROWS = 2048;                      // rows of one attention piece
}  // namespace q35v

// A resized, normalised page: pixels fp32 [gh * gw, 768] in merge-unit order, one patch = [c][y][x]
struct Q35Pixels {
    int gh = 0, gw = 0;                         // patch grid (both even)
    std::vector<float> patch;
};

// smart_resize with factor 32: the size a w x h page is read at
void q35_target_size(int h, int w, int max_pixels, int* th, int* tw);
// bytes: a JPEG, PNG, ... file. max_pixels: the area cap (Q35_MAX_PIXELS)
Q35Pixels q35_load_image(const void* bytes, size_t n, int max_pixels);
// the same from RGB8 [h, w, 3] (tests)
Q35Pixels q35_rgb_to_pixels(const unsigned char* rgb, int h, int w, int max_pixels);
// "data:image/...;base64,...", or plain base64
std::string base64_decode_url(const std::string& url);

class Q35Vision {
public:
    explicit Q35Vision(const SafeTensors& st);
    // embeddings fp32 [gh / 2 * gw / 2, 5120]
    Q35Image encode(const Q35Pixels& px);
    // debug: row 0's stream after the patch embedding and after each block
    std::vector<float>* dump = nullptr;

private:
    struct Block {
        Tensor n1w, n1b, n2w, n2b;              // fp32 [D]: LayerNorm weight, bias
        Tensor qkv, proj, fc1, fc2;             // fp16 [out, in]
        Tensor qkvb, projb, fc1b, fc2b;         // fp32 [out]
    };
    std::vector<Block> blocks;
    Tensor patch_w, patch_b, mg_nw, mg_nb, mg_f1, mg_f1b, mg_f2, mg_f2b;
    std::vector<float> pos_table;               // host [2304, D]
    void linear(const Tensor& w, const Tensor& b, const half* x, int rows, int in, int out, float* y);
};

}  // namespace glm
