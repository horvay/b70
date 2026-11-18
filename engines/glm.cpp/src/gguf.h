#pragma once
// A GGUF file: its key/value metadata and tensor table, the data memory-mapped read-only. Enough for the Edge0
// skeleton (F32, F16, BF16, Q8_0, I8 tensors) and the fp8 n-gram table (only its offset is used; rows are pread).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace glm {

enum GgmlType : uint32_t {
    GG_F32 = 0, GG_F16 = 1, GG_Q8_0 = 8, GG_I8 = 24, GG_I32 = 26, GG_BF16 = 30, GG_NVFP4 = 40,
};

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<int64_t> ne;          // ggml order: ne[0] is the contiguous (column) dimension
    uint64_t offset = 0;              // from the start of the file
    size_t bytes = 0;
    const void* data = nullptr;       // inside the mapping (null when the file is opened without one)
    int64_t rows() const { int64_t r = 1; for (size_t i = 1; i < ne.size(); i++) r *= ne[i]; return r; }
    int64_t cols() const { return ne.empty() ? 0 : ne[0]; }
};

struct GgufValue {
    enum Kind { Int, Float, Str, Bool, Arr } kind = Int;
    int64_t i = 0;
    double f = 0;
    std::string s;
    std::vector<GgufValue> arr;
};

class Gguf {
public:
    // map: memory-map the whole file (else only the header is read; tensors keep their offsets)
    explicit Gguf(const std::string& path, bool map = true);
    ~Gguf();
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;

    bool has(const std::string& name) const { return tensors.count(name); }
    const GgufTensor& at(const std::string& name) const;
    bool has_kv(const std::string& k) const { return kv.count(k); }
    const GgufValue& val(const std::string& k) const;
    int64_t get_i(const std::string& k) const { return val(k).i; }
    double get_f(const std::string& k) const { return val(k).f; }
    std::string get_s(const std::string& k) const { return val(k).s; }

    std::string path;
    std::map<std::string, GgufValue> kv;
    std::map<std::string, GgufTensor> tensors;
    uint64_t data_offset = 0;

private:
    void* base_ = nullptr;
    size_t len_ = 0;
};

// bytes of one row of n values of a ggml type (n a whole number of blocks)
size_t ggml_row_bytes(uint32_t type, int64_t n);

}  // namespace glm
