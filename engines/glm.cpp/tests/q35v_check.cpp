// The C++ vision tower against tests/q35v_ref.py: q35v_check <model dir> <image> <ref dir>
// Reports relative error of the pixels, of row 0's stream after each stage, and of the output embeddings.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "../src/q35v.h"
#include "../src/json.h"

using namespace glm;

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    GLM_CHECK(f.good(), "cannot read %s", p.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
static std::vector<float> floats(const std::string& p) {
    std::string b = slurp(p);
    std::vector<float> v(b.size() / 4);
    memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}
static double rel(const float* a, const float* b, size_t n) {
    double e = 0, s = 0;
    for (size_t i = 0; i < n; i++) { e += (a[i] - b[i]) * (double)(a[i] - b[i]); s += b[i] * (double)b[i]; }
    return std::sqrt(e / std::max(1e-30, s));
}

int main(int argc, char** argv) {
    GLM_CHECK(argc == 4, "usage: q35v_check <model dir> <image> <ref dir>");
    std::string dir = argv[1], image = argv[2], ref = argv[3];
    Json meta = Json::parse(slurp(ref + "/meta.json"));
    std::string file = slurp(image);
    Q35Pixels px = q35_load_image(file.data(), file.size(), (int)meta["max_pixels"].i());
    auto rp = floats(ref + "/pixels.f32");
    printf("grid %d x %d (ref %lld x %lld); pixels rel err %.5f\n", px.gh, px.gw, (long long)meta["grid"][1].i(),
           (long long)meta["grid"][2].i(), px.patch.size() == rp.size() ? rel(px.patch.data(), rp.data(), rp.size()) : -1.0);
    SafeTensors st(dir);
    Q35Vision vis(st);
    // the reference's own pixels, so the tower's error stands alone
    Q35Pixels rpx = px;
    if (rp.size() == px.patch.size()) rpx.patch = rp;
    std::vector<float> stream;
    vis.dump = &stream;
    auto t0 = std::chrono::steady_clock::now();
    Q35Image img = vis.encode(rpx);
    printf("encode %.2f s (%d patches)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), px.gh * px.gw);
    auto rs = floats(ref + "/stream0.f32");
    for (size_t i = 0; i * q35v::D < std::min(stream.size(), rs.size()); i += i < 3 ? 1 : 6)
        printf("stage %2zu rel err %.5f\n", i, rel(&stream[i * q35v::D], &rs[i * q35v::D], q35v::D));
    auto ro = floats(ref + "/out.f32");
    printf("output: %zu vs %zu values, rel err %.5f\n", img.emb.size(), ro.size(), img.emb.size() == ro.size() ? rel(img.emb.data(), ro.data(), ro.size()) : -1.0);
    if (std::ifstream(ref + "/out32.f32").good()) {
        auto r3 = floats(ref + "/out32.f32");
        printf("output against the fp32 tower: rel err %.5f\n", rel(img.emb.data(), r3.data(), r3.size()));
    }
    // a second encode: the tower's speed once its kernels are loaded
    vis.dump = nullptr;
    t0 = std::chrono::steady_clock::now();
    img = vis.encode(rpx);
    printf("encode again %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
