// Host-only: image file decoding for the vision tower (stb_image from engines/sd.cpp/thirdparty), base64.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA
#include "stb_image.h"
#include "q35v.h"

namespace glm {

Q35Pixels q35_load_image(const void* bytes, size_t n, int max_pixels) {
    int w = 0, h = 0, c = 0;
    unsigned char* rgb = stbi_load_from_memory(static_cast<const stbi_uc*>(bytes), (int)n, &w, &h, &c, 3);
    GLM_CHECK(rgb, "cannot decode the image: %s", stbi_failure_reason());
    Q35Pixels p = q35_rgb_to_pixels(rgb, h, w, max_pixels);
    stbi_image_free(rgb);
    return p;
}

std::string base64_decode_url(const std::string& url) {
    size_t i = 0;
    if (url.rfind("data:", 0) == 0) {
        size_t comma = url.find(',');
        GLM_CHECK(comma != std::string::npos, "malformed data URL");
        i = comma + 1;
    }
    static const auto table = [] {
        std::vector<int8_t> t(256, -1);
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int k = 0; k < 64; k++) t[(unsigned char)a[k]] = (int8_t)k;
        t[(unsigned char)'-'] = 62;
        t[(unsigned char)'_'] = 63;
        return t;
    }();
    std::string out;
    out.reserve((url.size() - i) * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (; i < url.size(); i++) {
        int v = table[(unsigned char)url[i]];
        if (v < 0) continue;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char)((acc >> bits) & 255));
        }
    }
    return out;
}

}  // namespace glm
