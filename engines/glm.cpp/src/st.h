#pragma once
// Safetensors model directory: the index (model.safetensors.index.json, or one model.safetensors), each shard's
// header, and memory-mapped tensor data.
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "json.h"
#include "rt.h"

namespace glm {

struct StTensor {
    std::string name;
    DType dt;
    std::vector<int64_t> shape;
    const void* data;     // inside the shard's mapping
    size_t bytes;
};

struct SafeTensors {
    explicit SafeTensors(const std::string& dir);
    ~SafeTensors();
    bool has(const std::string& name) const { return idx.count(name); }
    const StTensor& at(const std::string& name) const;
    // device copy; optional dtype check
    Tensor load(const std::string& name) const;
    std::vector<std::string> names_with_prefix(const std::string& prefix) const;
    // After defer(true), load() allocates and only notes the copy; flush() then reads every noted tensor with parallel
    // readers (GLM_LOAD_THREADS, default 8; 32 MB chunks into pinned buffers, a queue each): pages already in the page
    // cache are copied from the mapping, the rest read with O_DIRECT. Returns the bytes copied. The tensors' device
    // memory must not be read before it.
    void defer(bool on) {
        static const bool off = getenv("GLM_LOAD_DEFER") && atoi(getenv("GLM_LOAD_DEFER")) == 0;   // the old one-by-one loads
        defer_ = on && !off;
    }
    size_t flush();
    // the pages of [p, p + bytes) of a mapping read in now, by `threads` threads (MADV_POPULATE_READ)
    static void populate(const void* p, size_t bytes, int threads);

private:
    struct Map { void* base = nullptr; size_t len = 0; };
    std::vector<Map> maps;
    std::vector<std::string> paths;
    std::map<std::string, StTensor> idx;
    struct Pending { int shard; size_t off, bytes; void* dst; };
    mutable std::vector<Pending> pending_;
    bool defer_ = false;
    int shard_of(const void* p) const;
};

std::string read_file(const std::string& path);

}  // namespace glm
