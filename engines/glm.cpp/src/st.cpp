#include "st.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>

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
        paths.push_back(path);
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
    if (defer_ && s.bytes) {
        const int sh = shard_of(s.data);
        pending_.push_back({sh, (size_t)(static_cast<const char*>(s.data) - static_cast<const char*>(maps[sh].base)), s.bytes, t.p});
        return t;
    }
    upload(t.p, s.data, s.bytes);
    return t;
}

int SafeTensors::shard_of(const void* p) const {
    const char* c = static_cast<const char*>(p);
    for (size_t i = 0; i < maps.size(); i++) {
        const char* b = static_cast<const char*>(maps[i].base);
        if (c >= b && c < b + maps[i].len) return (int)i;
    }
    GLM_CHECK(false, "a tensor outside every shard");
    return -1;
}

void SafeTensors::populate(const void* p, size_t bytes, int threads) {
    const size_t page = 4096;
    uintptr_t b0 = reinterpret_cast<uintptr_t>(p) & ~(page - 1), e = reinterpret_cast<uintptr_t>(p) + bytes;
    const size_t span = ((e - b0) / threads + page - 1) & ~(page - 1);
    std::vector<std::thread> ts;
    for (uintptr_t a = b0; a < e; a += span)
        ts.emplace_back([a, e, span] { madvise(reinterpret_cast<void*>(a), std::min<size_t>(span, e - a), MADV_POPULATE_READ); });
    for (auto& t : ts) t.join();
}

size_t SafeTensors::flush() {
    if (pending_.empty()) return 0;
    auto t0 = std::chrono::steady_clock::now();
    constexpr size_t CH = (size_t)32 << 20, AL = 4096;
    // 32 MB pieces in file order, taken in turn by the readers
    struct Piece { int shard; size_t off, bytes; char* dst; };
    std::sort(pending_.begin(), pending_.end(), [](const Pending& a, const Pending& b) {
        return a.shard != b.shard ? a.shard < b.shard : a.off < b.off;
    });
    std::vector<Piece> pieces;
    size_t total = 0;
    for (auto& p : pending_) {
        for (size_t o = 0; o < p.bytes; o += CH)
            pieces.push_back({p.shard, p.off + o, std::min(CH, p.bytes - o), static_cast<char*>(p.dst) + o});
        total += p.bytes;
    }
    std::vector<int> dfd(maps.size(), -1);
    for (size_t i = 0; i < maps.size(); i++) dfd[i] = open(paths[i].c_str(), O_RDONLY | O_DIRECT);
    const int nt = std::max(1, getenv("GLM_LOAD_THREADS") ? atoi(getenv("GLM_LOAD_THREADS")) : 8);
    auto& d = Dev::get();
    std::atomic<size_t> next{0}, direct{0};
    std::atomic<bool> failed{false};
    auto worker = [&]() {
        sycl::queue q(d.ctx, d.dev, sycl::property::queue::in_order());
        char* buf[2];
        sycl::event ev[2];
        for (auto& b : buf) b = static_cast<char*>(sycl::aligned_alloc_host(AL, CH + 2 * AL, q));
        if (!buf[0] || !buf[1]) { failed = true; return; }
        std::vector<unsigned char> vec(CH / 4096 + 2);
        for (int k = 0;; k ^= 1) {
            const size_t i = next++;
            if (i >= pieces.size()) break;
            const Piece& pc = pieces[i];
            const char* src = static_cast<const char*>(maps[pc.shard].base) + pc.off;
            ev[k].wait();                                     // this buffer's last copy is done
            // in the page cache (nearly all of it): a copy from the mapping; else O_DIRECT
            const uintptr_t pa = reinterpret_cast<uintptr_t>(src) & ~(uintptr_t)4095;
            const size_t np = (reinterpret_cast<uintptr_t>(src) + pc.bytes - pa + 4095) / 4096;
            size_t res = 0;
            if (mincore(reinterpret_cast<void*>(pa), np * 4096, vec.data()) == 0)
                for (size_t j = 0; j < np; j++) res += vec[j] & 1;
            const char* from = nullptr;
            if (dfd[pc.shard] >= 0 && res * 10 < np * 9) {
                const size_t a0 = pc.off & ~(AL - 1), a1 = (pc.off + pc.bytes + AL - 1) & ~(AL - 1);
                size_t got = 0;
                while (got < a1 - a0) {
                    ssize_t r = pread(dfd[pc.shard], buf[k] + got, a1 - a0 - got, a0 + got);
                    if (r <= 0) break;
                    got += r;
                }
                if (got >= pc.off + pc.bytes - a0) { from = buf[k] + (pc.off - a0); direct += pc.bytes; }
            }
            if (!from) {
                memcpy(buf[k], src, pc.bytes);
                from = buf[k];
            }
            ev[k] = q.memcpy(pc.dst, from, pc.bytes);
        }
        ev[0].wait();
        ev[1].wait();
        for (auto& b : buf) sycl::free(b, q);
    };
    std::vector<std::thread> ts;
    for (int t = 0; t < nt; t++) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
    for (int fd : dfd) if (fd >= 0) close(fd);
    GLM_CHECK(!failed, "loader: pinned buffers");
    pending_.clear();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "weights: %.2f GB in %.2f s (%.2f GB/s, %.0f%% read with O_DIRECT, %d readers)\n", total / 1e9, s,
            total / 1e9 / std::max(s, 1e-6), 100.0 * direct / std::max<size_t>(total, 1), nt);
    return total;
}

std::vector<std::string> SafeTensors::names_with_prefix(const std::string& prefix) const {
    std::vector<std::string> r;
    for (auto it = idx.lower_bound(prefix); it != idx.end() && it->first.compare(0, prefix.size(), prefix) == 0; ++it)
        r.push_back(it->first);
    return r;
}

}  // namespace glm
