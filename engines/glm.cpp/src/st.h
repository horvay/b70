#pragma once
// Safetensors model directory: the index (model.safetensors.index.json, or one model.safetensors), each shard's
// header, and memory-mapped tensor data.
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

private:
    struct Map { void* base = nullptr; size_t len = 0; };
    std::vector<Map> maps;
    std::map<std::string, StTensor> idx;
};

std::string read_file(const std::string& path);

}  // namespace glm
