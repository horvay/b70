// exl3conv: safetensors writing. A file is written in one pass: the header (minified JSON, no padding, like
// exllamav3's writer) then every tensor's bytes in header order. Each tensor is a name, dtype, shape, byte count and a
// callback that streams its bytes, so multi-GB tensors (the embedding table, lm_head) never sit in RAM whole.
#pragma once

#include "json.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

struct OutTensor
{
    std::string name, dtype;              // safetensors dtype: F16, BF16, F32, I16, I32, ...
    std::vector<int64_t> shape;
    uint64_t nbytes = 0;
    std::function<void(std::ostream&)> write;

    static OutTensor from_bytes(std::string name, std::string dtype, std::vector<int64_t> shape, const void* p, uint64_t n)
    {
        return {std::move(name), std::move(dtype), std::move(shape), n, [p, n](std::ostream& o) { o.write((const char*) p, (std::streamsize) n); }};
    }
    template <typename T>
    static OutTensor from_vector(std::string name, std::string dtype, std::vector<int64_t> shape, std::vector<T> v)
    {
        auto keep = std::make_shared<std::vector<T>>(std::move(v));
        const uint64_t n = keep->size() * sizeof(T);
        return {std::move(name), std::move(dtype), std::move(shape), n, [keep, n](std::ostream& o) { o.write((const char*) keep->data(), (std::streamsize) n); }};
    }
};

inline void write_file(const std::filesystem::path& path, const std::vector<OutTensor>& ts)
{
    nlohmann::ordered_json h = nlohmann::ordered_json::object();
    uint64_t off = 0;
    for (const auto& t : ts)
    {
        h[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {off, off + t.nbytes}}};
        off += t.nbytes;
    }
    const std::string hs = h.dump();
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary);
        if (!o) throw std::runtime_error("cannot write " + tmp);
        const uint64_t n = hs.size();
        o.write((const char*) &n, 8);
        o.write(hs.data(), (std::streamsize) hs.size());
        for (const auto& t : ts)
        {
            const auto p0 = o.tellp();
            t.write(o);
            if ((uint64_t) (o.tellp() - p0) != t.nbytes) throw std::runtime_error("tensor " + t.name + ": wrote a different size than declared");
        }
        if (!o) throw std::runtime_error("write failed: " + tmp);
    }
    std::filesystem::rename(tmp, path);
}

} // namespace st
