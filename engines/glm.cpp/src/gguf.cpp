#include "gguf.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include "rt.h"

namespace glm {

size_t ggml_row_bytes(uint32_t type, int64_t n) {
    switch (type) {
        case GG_F32: case GG_I32: return (size_t)n * 4;
        case GG_F16: case GG_BF16: return (size_t)n * 2;
        case GG_I8: return (size_t)n;
        case GG_Q8_0: GLM_CHECK(n % 32 == 0, "Q8_0 row of %lld", (long long)n); return (size_t)n / 32 * 34;
        case GG_NVFP4: GLM_CHECK(n % 64 == 0, "NVFP4 row of %lld", (long long)n); return (size_t)n / 64 * 36;
    }
    GLM_CHECK(false, "ggml type %u not supported", type);
    return 0;
}

namespace {
struct Reader {
    int fd;
    uint64_t at = 0;
    std::string buf;                  // read-ahead window
    uint64_t buf_at = 0;
    void fill(size_t n) {
        if (at >= buf_at && at + n <= buf_at + buf.size()) return;
        size_t want = std::max<size_t>(n, (size_t)1 << 22);
        buf.resize(want);
        ssize_t got = pread(fd, buf.data(), want, (off_t)at);
        GLM_CHECK(got >= (ssize_t)n, "gguf: short read at %llu", (unsigned long long)at);
        buf.resize(got);
        buf_at = at;
    }
    void raw(void* dst, size_t n) {
        fill(n);
        memcpy(dst, buf.data() + (at - buf_at), n);
        at += n;
    }
    template <class T> T get() { T v; raw(&v, sizeof v); return v; }
    std::string str() {
        uint64_t n = get<uint64_t>();
        GLM_CHECK(n < ((uint64_t)1 << 30), "gguf: string of %llu bytes", (unsigned long long)n);
        std::string s(n, '\0');
        if (n) raw(s.data(), n);
        return s;
    }
    GgufValue value(uint32_t t) {
        GgufValue v;
        switch (t) {
            case 0: v.i = get<uint8_t>(); break;
            case 1: v.i = get<int8_t>(); break;
            case 2: v.i = get<uint16_t>(); break;
            case 3: v.i = get<int16_t>(); break;
            case 4: v.i = get<uint32_t>(); break;
            case 5: v.i = get<int32_t>(); break;
            case 6: v.kind = GgufValue::Float; v.f = get<float>(); break;
            case 7: v.kind = GgufValue::Bool; v.i = get<uint8_t>(); break;
            case 8: v.kind = GgufValue::Str; v.s = str(); break;
            case 9: {
                v.kind = GgufValue::Arr;
                uint32_t at_ = get<uint32_t>();
                uint64_t n = get<uint64_t>();
                v.arr.reserve(n);
                for (uint64_t k = 0; k < n; k++) v.arr.push_back(value(at_));
                break;
            }
            case 10: v.i = (int64_t)get<uint64_t>(); break;
            case 11: v.i = get<int64_t>(); break;
            case 12: v.kind = GgufValue::Float; v.f = get<double>(); break;
            default: GLM_CHECK(false, "gguf: value type %u", t);
        }
        if (v.kind == GgufValue::Int) v.f = (double)v.i;
        return v;
    }
};
}  // namespace

Gguf::Gguf(const std::string& p, bool map) : path(p) {
    int fd = open(p.c_str(), O_RDONLY);
    GLM_CHECK(fd >= 0, "cannot open %s", p.c_str());
    struct stat sb;
    GLM_CHECK(fstat(fd, &sb) == 0, "%s: stat", p.c_str());
    Reader r{fd};
    char magic[4];
    r.raw(magic, 4);
    GLM_CHECK(!memcmp(magic, "GGUF", 4), "%s: not a GGUF file", p.c_str());
    uint32_t ver = r.get<uint32_t>();
    GLM_CHECK(ver == 3, "%s: GGUF version %u", p.c_str(), ver);
    uint64_t nt = r.get<uint64_t>(), nkv = r.get<uint64_t>();
    for (uint64_t k = 0; k < nkv; k++) {
        std::string key = r.str();
        uint32_t t = r.get<uint32_t>();
        kv[key] = r.value(t);
    }
    std::vector<GgufTensor> list;
    for (uint64_t k = 0; k < nt; k++) {
        GgufTensor t;
        t.name = r.str();
        uint32_t nd = r.get<uint32_t>();
        for (uint32_t d = 0; d < nd; d++) t.ne.push_back((int64_t)r.get<uint64_t>());
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        list.push_back(std::move(t));
    }
    uint64_t align = kv.count("general.alignment") ? (uint64_t)kv["general.alignment"].i : 32;
    data_offset = (r.at + align - 1) / align * align;
    for (auto& t : list) {
        t.offset += data_offset;
        t.bytes = ggml_row_bytes(t.type, t.cols()) * (size_t)t.rows();
        GLM_CHECK(t.offset + t.bytes <= (uint64_t)sb.st_size, "%s: tensor %s past the end of the file", p.c_str(),
                  t.name.c_str());
        tensors[t.name] = std::move(t);
    }
    if (map) {
        len_ = (size_t)sb.st_size;
        base_ = mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd, 0);
        GLM_CHECK(base_ != MAP_FAILED, "%s: mmap", p.c_str());
        for (auto& [n, t] : tensors) t.data = static_cast<const char*>(base_) + t.offset;
    }
    close(fd);
}

Gguf::~Gguf() {
    if (base_ && base_ != MAP_FAILED) munmap(base_, len_);
}

const GgufTensor& Gguf::at(const std::string& name) const {
    auto it = tensors.find(name);
    GLM_CHECK(it != tensors.end(), "%s: no tensor %s", path.c_str(), name.c_str());
    return it->second;
}

const GgufValue& Gguf::val(const std::string& k) const {
    auto it = kv.find(k);
    GLM_CHECK(it != kv.end(), "%s: no key %s", path.c_str(), k.c_str());
    return it->second;
}

}  // namespace glm
