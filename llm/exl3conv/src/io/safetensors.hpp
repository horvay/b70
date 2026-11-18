// exl3conv: safetensors reading (mmap, single file or sharded with model.safetensors.index.json), or a directory
// holding one llama.cpp GGUF read as the Hugging Face checkpoint it came from (see gguf.hpp and Model below).
//
// A safetensors file is: u64 little-endian header length N, N bytes of JSON {name: {dtype, shape, data_offsets}},
// then the raw tensor bytes; offsets are relative to the end of the header. Files are mapped read-only and tensors
// are returned as pointers into the mapping (the page cache does the reading; nothing is copied until asked).
#pragma once

#include "io/gguf.hpp"
#include "json.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class DType { F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, F64, OTHER };

inline DType parse_dtype(const std::string& s)
{
    static const std::map<std::string, DType> m = {
        {"F32", DType::F32}, {"F16", DType::F16}, {"BF16", DType::BF16}, {"I64", DType::I64}, {"I32", DType::I32},
        {"I16", DType::I16}, {"I8", DType::I8}, {"U8", DType::U8}, {"BOOL", DType::BOOL}, {"F64", DType::F64}};
    auto it = m.find(s);
    return it == m.end() ? DType::OTHER : it->second;
}

inline const char* dtype_name(DType d)
{
    switch (d)
    {
        case DType::F32: return "F32"; case DType::F16: return "F16"; case DType::BF16: return "BF16";
        case DType::I64: return "I64"; case DType::I32: return "I32"; case DType::I16: return "I16";
        case DType::I8: return "I8"; case DType::U8: return "U8"; case DType::BOOL: return "BOOL";
        case DType::F64: return "F64"; default: return "OTHER";
    }
}

inline size_t dtype_size(DType d)
{
    switch (d)
    {
        case DType::F64: case DType::I64: return 8;
        case DType::F32: case DType::I32: return 4;
        case DType::F16: case DType::BF16: case DType::I16: return 2;
        case DType::I8: case DType::U8: case DType::BOOL: return 1;
        default: return 0;
    }
}

inline float bf16_to_float(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }

struct TensorView
{
    std::string name;
    DType dtype = DType::OTHER;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
};

class File
{
public:
    explicit File(const fs::path& path) : path_(path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat sb;
        if (fstat(fd_, &sb) != 0) throw std::runtime_error("cannot stat " + path.string());
        size_ = (size_t) sb.st_size;
        map_ = (const uint8_t*) mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (map_ == MAP_FAILED) throw std::runtime_error("cannot mmap " + path.string());
        if (size_ < 8) throw std::runtime_error("truncated safetensors " + path.string());
        uint64_t n;
        std::memcpy(&n, map_, 8);
        if (8 + n > size_) throw std::runtime_error("bad header length in " + path.string());
        const auto header = json::parse(std::string((const char*) map_ + 8, n));
        const uint8_t* base = map_ + 8 + n;
        for (auto it = header.begin(); it != header.end(); ++it)
        {
            if (it.key() == "__metadata__") continue;
            TensorView t;
            t.name = it.key();
            t.dtype = parse_dtype(it.value()["dtype"].get<std::string>());
            t.shape = it.value()["shape"].get<std::vector<int64_t>>();
            const auto off = it.value()["data_offsets"].get<std::vector<uint64_t>>();
            if (8 + n + off[1] > size_) throw std::runtime_error("tensor past end of file: " + t.name);
            t.data = base + off[0];
            t.nbytes = off[1] - off[0];
            order_.push_back(t.name);
            tensors_.emplace(t.name, std::move(t));
        }
    }
    ~File() { if (map_ && map_ != MAP_FAILED) munmap((void*) map_, size_); if (fd_ >= 0) ::close(fd_); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    const std::map<std::string, TensorView>& tensors() const { return tensors_; }
    const std::vector<std::string>& order() const { return order_; }
    const fs::path& path() const { return path_; }

    // Tell the kernel we'll read a range soon / no longer need it (keeps the page cache from holding a whole shard).
    void will_need(const TensorView& t) const { advise(t, MADV_WILLNEED); }
    void dont_need(const TensorView& t) const { advise(t, MADV_DONTNEED); }

private:
    void advise(const TensorView& t, int how) const
    {
        const uintptr_t page = 4096, a = (uintptr_t) t.data & ~(page - 1);
        const uintptr_t b = ((uintptr_t) t.data + t.nbytes + page - 1) & ~(page - 1);
        madvise((void*) a, b - a, how);
    }
    fs::path path_;
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t* map_ = nullptr;
    std::map<std::string, TensorView> tensors_;
    std::vector<std::string> order_;
};

// A model directory: every *.safetensors file in it (or the ones its index names).
//
// A directory with no *.safetensors and exactly one *.gguf (an mmproj-*.gguf beside it is ignored) is read through
// gguf::HfMap instead: the same names, shapes, memory layout and dtype (BF16) as the Hugging Face checkpoint the GGUF
// was converted from, with llama.cpp's transforms undone and quantized tensors dequantized.
//
// GGUF tensors are decoded on get() into buffers this Model owns, kept in a cache of EXL3_GGUF_CACHE_GB (default 8)
// GB, least recently got first out. The rule: the data pointer from get(name) stays valid until the tensors got
// after the last get(name), together with name itself, exceed the budget; the pointer of the tensor just got is never
// freed by its own get. The converters read one layer at a time (under 1 GB here) and upload each tensor right after
// get(); the longest-lived pointers are the embedding (2.5 GB, read while the calibration rows are embedded) and the
// output step's embedding + lm_head + final norm + MTP layer (about 6 GB, held as OutTensor sources until written),
// both inside the default budget. An evicted tensor's TensorView keeps its name and shape but data becomes nullptr,
// so a stale reference through the view crashes rather than reading freed memory. dont_need(name) frees it at once.
class Model
{
public:
    explicit Model(const fs::path& dir) : dir_(dir)
    {
        std::vector<fs::path> files;
        const auto index = dir / "model.safetensors.index.json";
        if (fs::exists(index))
        {
            std::ifstream f(index);
            const auto j = json::parse(f);
            std::map<std::string, bool> seen;
            for (auto& [name, file] : j["weight_map"].items())
                if (!seen[file.get<std::string>()]) { seen[file.get<std::string>()] = true; files.push_back(dir / file.get<std::string>()); }
        }
        else
            for (auto& e : fs::directory_iterator(dir))
                if (e.path().extension() == ".safetensors") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        if (files.empty())
        {
            std::vector<fs::path> ggufs;
            for (auto& e : fs::directory_iterator(dir))
                if (e.path().extension() == ".gguf" && e.path().filename().string().rfind("mmproj", 0) != 0) ggufs.push_back(e.path());
            if (ggufs.size() > 1) throw std::runtime_error("more than one .gguf in " + dir.string() + "; keep only the model's");
            if (ggufs.empty()) throw std::runtime_error("no safetensors or gguf files in " + dir.string());
            open_gguf(ggufs[0]);
            return;
        }
        for (auto& p : files)
        {
            files_.push_back(std::make_unique<File>(p));
            for (auto& [name, t] : files_.back()->tensors()) where_[name] = files_.back().get();
        }
    }

    bool has(const std::string& name) const { return gguf_ ? lazy_.count(name) != 0 : where_.count(name) != 0; }
    const TensorView& get(const std::string& name) const
    {
        if (gguf_) return get_gguf(name);
        auto it = where_.find(name);
        if (it == where_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second->tensors().at(name);
    }
    // Name, dtype and shape without reading the data (for a GGUF, data is nullptr unless the tensor is cached).
    const TensorView& view_of(const std::string& name) const
    {
        if (!gguf_) return get(name);
        auto it = lazy_.find(name);
        if (it == lazy_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second.view;
    }
    // GGUF-backed models: where a tensor comes from (GGUF name and type, transform), or nullptr.
    const gguf::Source* gguf_source(const std::string& name) const
    {
        auto it = lazy_.find(name);
        return it == lazy_.end() ? nullptr : it->second.src;
    }
    const File& file_of(const std::string& name) const
    {
        if (gguf_) throw std::runtime_error("file_of(" + name + "): this model is read from a GGUF; use Model::dont_need(name)");
        return *where_.at(name);
    }
    // Done with a tensor for now: drops its pages (safetensors) or frees its decoded copy (GGUF).
    void dont_need(const std::string& name) const
    {
        if (!gguf_) { file_of(name).dont_need(get(name)); return; }
        std::lock_guard<std::mutex> lk(mu_);
        auto it = lazy_.find(name);
        if (it != lazy_.end()) evict(it->second);
    }
    std::vector<std::string> names() const
    {
        std::vector<std::string> v;
        if (gguf_) for (auto& [n, l] : lazy_) v.push_back(n);
        else for (auto& [n, f] : where_) v.push_back(n);
        return v;
    }
    const fs::path& dir() const { return dir_; }
    bool is_gguf() const { return gguf_ != nullptr; }
    const gguf::File* gguf_file() const { return gguf_.get(); }

private:
    struct Lazy
    {
        const gguf::Source* src = nullptr;
        TensorView view;                     // stable address: get() hands out references to it
        std::unique_ptr<uint16_t[]> buf;
        uint64_t stamp = 0;
    };

    void open_gguf(const fs::path& p)
    {
        gguf_ = std::make_unique<gguf::File>(p);
        hf_ = std::make_unique<gguf::HfMap>(*gguf_, dir_ / "config.json");
        for (auto& [name, s] : hf_->sources())
        {
            Lazy& l = lazy_[name];
            l.src = &s;
            l.view.name = name;
            l.view.dtype = DType::BF16;
            l.view.shape = s.hf_shape;
            l.view.nbytes = s.out_bytes();
        }
        if (const char* e = std::getenv("EXL3_GGUF_CACHE_GB")) budget_ = (size_t) (std::atof(e) * 1e9);
    }

    const TensorView& get_gguf(const std::string& name) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = lazy_.find(name);
        if (it == lazy_.end()) throw std::runtime_error("tensor not found: " + name);
        Lazy& l = it->second;
        l.stamp = ++clock_;
        if (!l.buf)
        {
            l.buf.reset(new uint16_t[l.view.nbytes / 2]);
            const gguf::Tensor& t = *l.src->t;
            gguf_->advise(t, MADV_WILLNEED);
            gguf::materialize(*l.src, l.buf.get());
            gguf_->advise(t, MADV_DONTNEED);
            l.view.data = (const uint8_t*) l.buf.get();
            cached_ += l.view.nbytes;
            while (cached_ > budget_)
            {
                Lazy* old = nullptr;
                for (auto& [n, c] : lazy_)
                    if (c.buf && &c != &l && (!old || c.stamp < old->stamp)) old = &c;
                if (!old) break;
                evict(*old);
            }
        }
        return l.view;
    }
    void evict(Lazy& l) const
    {
        if (!l.buf) return;
        l.buf.reset();
        l.view.data = nullptr;
        cached_ -= l.view.nbytes;
    }

    fs::path dir_;
    std::vector<std::unique_ptr<File>> files_;
    std::map<std::string, const File*> where_;
    // GGUF-backed models only.
    std::unique_ptr<gguf::File> gguf_;
    std::unique_ptr<gguf::HfMap> hf_;
    mutable std::map<std::string, Lazy> lazy_;
    mutable std::mutex mu_;
    mutable size_t cached_ = 0;
    mutable uint64_t clock_ = 0;
    size_t budget_ = (size_t) 8e9;
};

} // namespace st
