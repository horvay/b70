#include "st.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace glm {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GLM_CHECK(f, "cannot open %s", path.c_str());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

static DType parse_dtype(const std::string& s) {
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "F32") return DType::F32;
    if (s == "I16") return DType::I16;
    if (s == "I32") return DType::I32;
    if (s == "I64") return DType::I64;
    if (s == "U8") return DType::U8;
    GLM_CHECK(false, "safetensors dtype %s not supported", s.c_str());
    return DType::U8;
}

SafeTensors::SafeTensors(const std::string& dir) {
    std::vector<std::string> shards;
    std::string index = dir + "/model.safetensors.index.json";
    if (access(index.c_str(), R_OK) == 0) {
        Json j = Json::parse(read_file(index));
        std::set<std::string> seen;
        for (auto& [k, v] : j["weight_map"].obj)
            if (seen.insert(v.s()).second) shards.push_back(v.s());
    } else if (access((dir + "/model.safetensors").c_str(), R_OK) == 0) {
        shards.push_back("model.safetensors");
    } else {
        shards.push_back("adapter_model.safetensors");     // a PEFT adapter folder
    }
    for (auto& sh : shards) {
        std::string path = dir + "/" + sh;
        int fd = open(path.c_str(), O_RDONLY);
        GLM_CHECK(fd >= 0, "cannot open %s", path.c_str());
        struct stat stt;
        fstat(fd, &stt);
        Map m;
        m.len = stt.st_size;
        m.base = mmap(nullptr, m.len, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        GLM_CHECK(m.base != MAP_FAILED, "mmap %s failed", path.c_str());
        maps.push_back(m);
        uint64_t hlen;
        memcpy(&hlen, m.base, 8);
        const char* h = static_cast<const char*>(m.base) + 8;
        Json hdr = Json::parse(std::string(h, hlen));
        const char* data = h + hlen;
        for (auto& [name, v] : hdr.obj) {
            if (name == "__metadata__") continue;
            StTensor t;
            t.name = name;
            t.dt = parse_dtype(v["dtype"].s());
            for (auto& d : v["shape"].arr) t.shape.push_back(d.i());
            size_t a = v["data_offsets"][0].i(), b = v["data_offsets"][1].i();
            t.data = data + a;
            t.bytes = b - a;
            idx.emplace(name, std::move(t));
        }
    }
}

SafeTensors::~SafeTensors() {
    for (auto& m : maps) munmap(m.base, m.len);
}

const StTensor& SafeTensors::at(const std::string& name) const {
    auto it = idx.find(name);
    GLM_CHECK(it != idx.end(), "tensor %s not in the model", name.c_str());
    return it->second;
}

Tensor SafeTensors::load(const std::string& name) const {
    const StTensor& s = at(name);
    Tensor t = Tensor::empty(s.shape, s.dt);
    GLM_CHECK(t.bytes() == s.bytes, "%s: %zu bytes, shape says %zu", name.c_str(), s.bytes, t.bytes());
    upload(t.p, s.data, s.bytes);
    return t;
}

std::vector<std::string> SafeTensors::names_with_prefix(const std::string& prefix) const {
    std::vector<std::string> r;
    for (auto it = idx.lower_bound(prefix); it != idx.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it)
        r.push_back(it->first);
    return r;
}

}  // namespace glm
