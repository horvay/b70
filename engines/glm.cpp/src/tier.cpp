#include "tier.h"
#include <fcntl.h>
#include <unistd.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/udmabuf.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include "json.h"
#include "moe.h"
#include "st.h"

namespace glm {

namespace {
using clk = std::chrono::steady_clock;
double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }
bool has(const std::vector<int64_t>& v, int64_t k) { return std::find(v.begin(), v.end(), k) != v.end(); }
}  // namespace

// ------------------------------------------------------------------------------------------------ Lru
void Lru::unlink(int64_t k) {
    int32_t p = prev_[k], n = next_[k];
    if (p >= 0) next_[p] = n; else head_ = n;
    if (n >= 0) prev_[n] = p; else tail_ = p;
    prev_[k] = next_[k] = -1;
    in_[k] = 0;
    --n_;
}
void Lru::link_back(int64_t k) {
    prev_[k] = tail_;
    next_[k] = -1;
    if (tail_ >= 0) next_[tail_] = (int32_t)k; else head_ = (int32_t)k;
    tail_ = (int32_t)k;
    in_[k] = 1;
    ++n_;
}

// ------------------------------------------------------------------------------------------------ Reader
// bounce -> RAM tier with non-temporal stores: the expert goes straight to DRAM and leaves no dirty lines in the CPU
// caches, which the GPU's zero-copy reads would have to snoop (prefetched experts are read over PCIe a layer later:
// with cached stores every RAM-tier read slowed from 0.31 to 0.42 ms per expert)
static void stream_copy(void* dst, const void* src, size_t n) {
    if (((uintptr_t)dst | (uintptr_t)src | n) & 31) { memcpy(dst, src, n); return; }
    auto* d = static_cast<__m256i*>(dst);
    auto* s = static_cast<const __m256i*>(src);
    for (size_t i = 0, m = n / 32; i < m; i += 4) {
        __m256i a = _mm256_load_si256(s + i), b = _mm256_load_si256(s + i + 1);
        __m256i c = _mm256_load_si256(s + i + 2), e = _mm256_load_si256(s + i + 3);
        _mm256_stream_si256(d + i, a);
        _mm256_stream_si256(d + i + 1, b);
        _mm256_stream_si256(d + i + 2, c);
        _mm256_stream_si256(d + i + 3, e);
    }
    _mm_sfence();
}

Reader::Reader(int fd, int threads, int max_spec, int64_t max_piece, Hooks h, bool direct)
    : fd_(fd), max_piece_(max_piece), h_(h), direct_(direct) {
    max_spec_ = std::max(1, std::min(max_spec, threads - 1));   // the rest stay free for urgent reads
    for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
}

Reader::~Reader() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stopping_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : workers_) t.join();
}

int64_t Reader::submit(int64_t key, int64_t off, int64_t n, void* dst, int pieces, int64_t res_slot, bool urgent) {
    GLM_CHECK(n > 0 && n % 4096 == 0 && off % 4096 == 0, "reader: unaligned read");
    int64_t pages = n / 4096, per = (pages + pieces - 1) / pieces * 4096;
    per = std::min(per, max_piece_);
    auto job = std::make_shared<Job>();
    job->key = key;
    job->res_slot = res_slot;
    job->left = (int)((n + per - 1) / per);
    int64_t id;
    {
        std::lock_guard<std::mutex> g(mu_);
        id = next_id_++;
        jobs_[id] = job;
        std::vector<Piece> ps;
        for (int64_t o = 0; o < n; o += per)
            ps.push_back(Piece{job, off + o, std::min(per, n - o), static_cast<uint8_t*>(dst) + o});
        if (urgent) {
            queue_.insert(queue_.begin() + n_urgent_, ps.begin(), ps.end());
            n_urgent_ += ps.size();
        } else {
            queue_.insert(queue_.end(), ps.begin(), ps.end());
        }
    }
    cv_work_.notify_all();
    return id;
}

void Reader::promote(int64_t id) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = jobs_.find(id);
    if (it == jobs_.end() || it->second->fin) return;
    auto jp = it->second;
    auto first = queue_.begin() + (std::ptrdiff_t)n_urgent_;
    auto mid = std::stable_partition(first, queue_.end(), [&](const Piece& p) { return p.job == jp; });
    n_urgent_ += (size_t)(mid - first);
}

bool Reader::done(int64_t id) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = jobs_.find(id);
    return it == jobs_.end() || it->second->fin.load(std::memory_order_acquire);
}

void Reader::wait(int64_t id) {
    std::shared_ptr<Job> j;
    {
        std::lock_guard<std::mutex> g(mu_);
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return;
        j = it->second;
    }
    {
        std::unique_lock<std::mutex> lk(mu_done_);
        cv_done_.wait(lk, [&] { return j->fin.load(std::memory_order_acquire); });
    }
    forget(id);
    int e = j->err.load();
    GLM_CHECK(!e, "expert read %lld failed: %s", (long long)j->key, e > 0 ? strerror(e) : "short read");
}

void Reader::forget(int64_t id) {
    std::lock_guard<std::mutex> g(mu_);
    jobs_.erase(id);
}

void Reader::run() {
    uint8_t* buf = nullptr;
    if (!direct_ && posix_memalign(reinterpret_cast<void**>(&buf), 4096, (size_t)max_piece_)) buf = nullptr;
    for (;;) {
        Piece p;
        bool spec;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stopping_ || (!queue_.empty() && (n_urgent_ > 0 || spec_running_ < max_spec_)); });
            if (queue_.empty()) break;
            spec = n_urgent_ == 0;
            if (spec) ++spec_running_;
            else --n_urgent_;
            p = std::move(queue_.front());
            queue_.pop_front();
        }
        int err = buf || direct_ ? 0 : ENOMEM;
        uint8_t* into = direct_ ? p.dst : buf;
        auto t0 = clk::now();
        int64_t got = 0;
        while (!err && got < p.n) {
            ssize_t r = pread(fd_, into + got, (size_t)(p.n - got), (off_t)(p.off + got));
            if (r < 0) { if (errno != EINTR) err = errno; }
            else if (r == 0) err = -1;
            else got += r;
        }
        ns_pread_ += std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
        if (!err && !direct_) stream_copy(p.dst, buf, (size_t)p.n);
        else { int z = 0; p.job->err.compare_exchange_strong(z, err); }
        if (spec) {
            { std::lock_guard<std::mutex> g(mu_); --spec_running_; }
            cv_work_.notify_one();
        }
        if (p.job->left.fetch_sub(1, std::memory_order_acq_rel) == 1) finish(*p.job);
    }
    free(buf);
}

void Reader::finish(Job& j) {
    if (j.res_slot >= 0 && !j.err.load() && __atomic_load_n(&h_.ram_of[j.key], __ATOMIC_ACQUIRE) == j.res_slot) {
        __atomic_store_n(&h_.landed[j.key], (uint8_t)1, __ATOMIC_RELAXED);
        if (h_.bv > 0 || h_.br > 0) {
            float b = __atomic_load_n(&h_.vram_of[j.key], __ATOMIC_RELAXED) >= 0 ? h_.bv : h_.br;
            __atomic_store(&h_.bonus[j.key], &b, __ATOMIC_RELAXED);
        }
        __atomic_store_n(h_.dirty, (uint8_t)1, __ATOMIC_RELEASE);
    }
    {
        std::lock_guard<std::mutex> g(mu_done_);
        j.fin.store(true, std::memory_order_release);
    }
    cv_done_.notify_all();
}

// ------------------------------------------------------------------------------------------------ HostWindows
bool HostWindows::init(size_t n, size_t rec) {
    constexpr size_t kMax = 64ull << 20;               // udmabuf size_limit_mb
    if (!n) return true;
    rec_ = rec;
    per_ = kMax / rec;
    if (!per_ || rec % 4096) return false;
    win_ = per_ * rec;
    size_t nw = (n + per_ - 1) / per_;
    mapped_ = nw * win_;
    int ufd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (ufd < 0) return false;
    mfd_ = memfd_create("glm-host-tier", MFD_ALLOW_SEALING | MFD_CLOEXEC);
    if (mfd_ < 0 || ftruncate(mfd_, mapped_) || fcntl(mfd_, F_ADD_SEALS, F_SEAL_SHRINK)) { close(ufd); return false; }
    host_ = static_cast<char*>(mmap(nullptr, mapped_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, mfd_, 0));
    if (host_ == MAP_FAILED) { host_ = nullptr; close(ufd); return false; }
    auto& q = Dev::get().q;
    auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    for (size_t w = 0; w < nw; w++) {
        udmabuf_create c{(__u32)mfd_, UDMABUF_FLAGS_CLOEXEC, w * win_, win_};
        int dfd = ioctl(ufd, UDMABUF_CREATE, &c);
        if (dfd < 0) { close(ufd); return false; }
        ze_external_memory_import_fd_t imp{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr,
                                           ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dfd};
        ze_device_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &imp, 0, 0};
        void* p = nullptr;
        ze_result_t zr = zeMemAllocDevice(zctx, &d, win_, 4096, zdev, &p);
        close(dfd);
        if (zr != ZE_RESULT_SUCCESS || !p) { close(ufd); return false; }
        dev_.push_back(static_cast<char*>(p));
    }
    close(ufd);
    return true;
}

HostWindows::~HostWindows() {
    if (!dev_.empty()) {
        auto& q = Dev::get().q;
        q.wait();
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        for (char* p : dev_) zeMemFree(zctx, p);
    }
    if (host_) munmap(host_, mapped_);
    if (mfd_ >= 0) close(mfd_);
}

// ------------------------------------------------------------------------------------------------ Tier
// share a byte budget over the size classes by layer count (tier_layout.split_slots)
// (minimum: slots a class must get, given first; the rest shared as before)
static std::vector<int> split_slots(size_t budget, const std::vector<int>& nl, const std::vector<size_t>& rec,
                                    const std::vector<int>& fixed, const std::vector<int>& minimum) {
    int C = nl.size();
    std::vector<int> s(C, 0);
    size_t used = 0;
    for (int c = 0; c < C; c++)
        if (nl[c] == 0) { s[c] = fixed[c]; used += fixed[c] * rec[c]; }
    GLM_CHECK(used <= budget, "tier: fixed pools exceed the budget");
    std::vector<bool> active(C);
    for (int c = 0; c < C; c++) active[c] = nl[c] > 0;
    size_t remaining = budget - used;
    for (;;) {
        size_t per = 0;
        for (int c = 0; c < C; c++) if (active[c]) per += (size_t)nl[c] * rec[c];
        if (!per) break;
        bool again = false;
        for (int c = 0; c < C; c++) {
            if (!active[c]) continue;
            int prop = (int)((__int128)remaining * nl[c] / per);
            if (prop < minimum[c]) {
                s[c] = minimum[c];
                GLM_CHECK(remaining >= (size_t)minimum[c] * rec[c], "tier: budget below the staging minimum");
                remaining -= (size_t)minimum[c] * rec[c];
                active[c] = false;
                again = true;
            }
        }
        if (again) continue;
        for (int c = 0; c < C; c++) if (active[c]) s[c] = (int)((__int128)remaining * nl[c] / per);
        break;
    }
    return s;
}

Tier::Tier(const std::string& bin, const TierConfig& cfg_in) : cfg_(cfg_in) {
    // reader shape overrides (measurement): GLM_READERS, GLM_SPEC_MAX, GLM_PIECES, GLM_URGENT_PIECES
    auto envi = [](const char* k, int& v) { if (const char* e = getenv(k)) v = atoi(e); };
    envi("GLM_READERS", cfg_.readers);
    envi("GLM_SPEC_MAX", cfg_.max_spec);
    envi("GLM_PIECES", cfg_.pieces);
    envi("GLM_URGENT_PIECES", cfg_.urgent_pieces);
    // admission policy overrides (measurement): GLM_ADMIT_MIN (decayed picks), GLM_ADMIT_PER_CALL, GLM_DECAY_EVERY
    if (const char* e = getenv("GLM_ADMIT_MIN")) cfg_.admit_min = (float)atof(e);
    envi("GLM_ADMIT_PER_CALL", cfg_.admit_per_call);
    envi("GLM_DECAY_EVERY", cfg_.decay_every);
    if (const char* e = getenv("GLM_MEMFD")) cfg_.memfd = atoi(e) != 0;
    const TierConfig& cfg = cfg_;
    auto& q = Dev::get().q;
    Json j = Json::parse(read_file(bin.substr(0, bin.rfind('.')) + ".json"));
    GLM_CHECK(j["format"].s() == "glm53-expert-store-blob-v2", "store format %s", j["format"].s().c_str());
    E = j["experts_per_layer"].i();
    L = j["n_layers"].i();
    n_moe = j.has("mtp_layer_li") && j["mtp_layer_li"].kind == Json::Num ? (int)j["mtp_layer_li"].i() : L;
    std::map<size_t, int, std::greater<size_t>> cls_of_rec;
    for (auto& l : j["layers"].arr) {
        LayerInfo li{(size_t)l["offset"].i(), (size_t)l["record_bytes"].i(), kcode(l["K"].f()), 0};
        GLM_CHECK(li.rec == moe_record_bytes(4096, 2048, li.kc), "store: record size");
        cls_of_rec[li.rec] = 0;
        lay_.push_back(li);
    }
    for (auto& [rec, c] : cls_of_rec) { c = C_++; crec_.push_back(rec); }
    for (auto& l : lay_) l.cls = cls_of_rec[l.rec];
    std::vector<int> nl(C_, 0), fixed(C_, 0);
    for (int li = 0; li < n_moe; li++) nl[lay_[li].cls]++;
    for (int c = 0; c < C_; c++) if (!nl[c]) fixed[c] = cfg.mtp ? 96 : 0;
    fd_ = open(bin.c_str(), O_RDONLY | O_DIRECT);
    GLM_CHECK(fd_ >= 0, "cannot open %s", bin.c_str());

    size_t LE = (size_t)L * E;
    key_cls_.resize(LE);
    for (size_t k = 0; k < LE; k++) key_cls_[k] = lay_[k / E].cls;
    vram_of_.assign(LE, -1);
    ram_of_.assign(LE, -1);
    flags_.assign(LE, 0);
    landed_.assign(LE, 0);
    freq_.assign(LE, 0);
    hits_.assign(LE, 0);
    bonus_h_ = sycl::malloc_host<float>(LE, q);
    ptrs_h_ = sycl::malloc_host<int64_t>(LE, q);
    std::fill(bonus_h_, bonus_h_ + LE, 0.0f);
    std::fill(ptrs_h_, ptrs_h_ + LE, 0);
    ptrs_dev_ = Tensor::zeros({L, E}, DType::I64);
    bonus_dev_ = Tensor::zeros({L, E}, DType::F32);

    // VRAM pools
    // The largest-record main class (sc_2.60: layer 3, 3.5 bits) gets at least E slots, so that layer stays wholly
    // in VRAM (prompts read it in place). Prompts stage the other layers in slots borrowed from the class with the
    // most layers (its least recently used experts are dropped and refilled by admissions afterwards).
    int big = -1;
    stage_cls_ = -1;
    for (int c = 0; c < C_; c++) {
        if (!nl[c]) continue;
        if (big < 0 || crec_[c] > crec_[big]) big = c;
        if (stage_cls_ < 0 || nl[c] > nl[stage_cls_]) stage_cls_ = c;
    }
    std::vector<int> minimum(C_, 0);
    minimum[big] = E;
    auto nv = split_slots(cfg.vram_bytes, nl, crec_, fixed, minimum);
    free_vram_.resize(C_);
    for (int c = 0; c < C_; c++) {
        char* b = nv[c] ? static_cast<char*>(sycl::malloc_device(crec_[c] * nv[c], q)) : nullptr;
        GLM_CHECK(!nv[c] || b, "tier: VRAM pool %d x %zu failed", nv[c], crec_[c]);
        Dev::get().vram_used += crec_[c] * nv[c];
        vbase_.push_back(b);
        for (int i = 0; i < nv[c]; i++) {
            vaddr_.push_back((int64_t)(b + (size_t)i * crec_[c]));
            vcls_.push_back(c);
            vram_key_.push_back(-1);
        }
    }
    for (int s = (int)vaddr_.size() - 1; s >= 0; s--) free_vram_[vcls_[s]].push_back(s);
    for (int c = 0; c < C_; c++) {
        vram_lru_.push_back(std::make_unique<Lru>(LE));
        ram_lru_.push_back(std::make_unique<Lru>(LE));
    }
    // RAM tier: memfd windows (reads in place), else pinned host chunks under 2 GiB (larger pinned allocations have
    // failed on XPU), touched once
    auto nr = split_slots(cfg.ram_bytes, nl, crec_, fixed, std::vector<int>(C_, 0));
    free_ram_.resize(C_);
    auto t0 = clk::now();
    bool direct = cfg.memfd;
    if (direct) {
        for (int c = 0; c < C_ && direct; c++) {
            auto w = std::make_unique<HostWindows>();
            direct = w->init(nr[c], crec_[c]);
            ram_win_.push_back(std::move(w));
        }
        if (!direct) {
            ram_win_.clear();
            fprintf(stderr, "tier: memfd / udmabuf RAM tier unavailable; pinned USM with a bounce copy per read\n");
        }
    }
    for (int c = 0; c < C_; c++) {
        if (direct) {
            for (int i = 0; i < nr[c]; i++) {
                ram_addr_.push_back((int64_t)ram_win_[c]->dev(i));
                ram_host_.push_back(ram_win_[c]->host(i));
                rcls_.push_back(c);
                ram_key_.push_back(-1);
            }
            continue;
        }
        int per_chunk = (int)(((size_t)2 << 30) - 1) / crec_[c], left = nr[c];
        while (left > 0) {
            int n = std::min(per_chunk, left);
            char* ch = static_cast<char*>(sycl::malloc_host(crec_[c] * n, q));
            GLM_CHECK(ch, "tier: pinned RAM chunk failed");
            ram_chunks_.push_back(ch);
            memset(ch, 0, crec_[c] * n);
            for (int i = 0; i < n; i++) {
                ram_addr_.push_back((int64_t)(ch + (size_t)i * crec_[c]));
                ram_host_.push_back(ch + (size_t)i * crec_[c]);
                rcls_.push_back(c);
                ram_key_.push_back(-1);
            }
            left -= n;
        }
    }
    direct_ = direct;
    for (int s = (int)ram_addr_.size() - 1; s >= 0; s--) free_ram_[rcls_[s]].push_back(s);
    vring_ = static_cast<char*>(sycl::malloc_device(48 * crec_[0], q));
    GLM_CHECK(vring_, "tier: victim ring");
    Dev::get().vram_used += 48 * crec_[0];
    for (int i = 47; i >= 0; i--) vring_free_.push_back(i);
    rd_ = std::make_unique<Reader>(fd_, cfg.readers, cfg.max_spec, 4 << 20,
                                   Reader::Hooks{ram_of_.data(), vram_of_.data(), landed_.data(), bonus_h_, &dirty_,
                                                 cfg.bonus_vram, cfg.bonus_ram}, direct_);
    fprintf(stderr, "tier: %zu VRAM slots, %zu RAM slots (%.1f GiB %s, ready in %.1f s), %d classes\n", vaddr_.size(),
            ram_addr_.size(), cfg.ram_bytes / 1073741824.0, direct_ ? "memfd, read in place" : "pinned", secs(t0), C_);
}

Tier::~Tier() {
    rd_.reset();
    if (!ring_win_)
        for (char* r : ring_) sycl::free(r, Dev::get().q);
    ring_win_.reset();
    ram_win_.clear();
    if (sptrs_h_) sycl::free(sptrs_h_, Dev::get().q);
    auto& q = Dev::get().q;
    q.wait();
    for (char* b : vbase_) if (b) sycl::free(b, q);
    for (void* c : ram_chunks_) sycl::free(c, q);
    if (vring_) sycl::free(vring_, q);
    sycl::free(bonus_h_, q);
    sycl::free(ptrs_h_, q);
    if (fd_ >= 0) close(fd_);
}

void Tier::set_bonus(int64_t k) {
    if (!bonus_on()) return;
    bonus_h_[k] = vram_of_[k] >= 0 ? cfg_.bonus_vram : (ram_of_[k] >= 0 && landed_[k] ? cfg_.bonus_ram : 0.0f);
    dirty_ = 1;
}

void Tier::sync_bonus(sycl::queue& q) {
    if (!bonus_on()) return;
    if (__atomic_exchange_n(&dirty_, (uint8_t)0, __ATOMIC_ACQ_REL))
        q.memcpy(bonus_dev_.p, bonus_h_, (size_t)L * E * sizeof(float));
}

void Tier::ram_put(int64_t k, int rs) {
    landed_[k] = 0;
    ram_of_[k] = rs;
    ram_key_[rs] = k;
    ram_lru_[rcls_[rs]]->set(k, rs);
    ram_lru_[rcls_[rs]]->touch(k);
    set_bonus(k);
}

int64_t Tier::submit_bg(int64_t k, int rs, bool urgent) {
    const LayerInfo& l = lay_[k / E];
    flags_[k] |= 1;
    return rd_->submit(k, l.off + (k % E) * l.rec, l.rec, ram_host_[rs], urgent ? cfg_.urgent_pieces : cfg_.pieces,
                       rs, urgent);
}

void Tier::reap_inflight() {
    for (auto it = inflight_.begin(); it != inflight_.end();) {
        if (rd_->done(it->second)) {
            rd_->wait(it->second);          // surfaces read errors
            flags_[it->first] &= ~1;
            demand_.erase(it->first);
            it = inflight_.erase(it);
        } else {
            ++it;
        }
    }
}

void Tier::reap_wb() {
    // the copy queue is in order: reap from the oldest, stop at the first still running (one driver query per call
    // instead of one per pending write-back)
    while (!wb_order_.empty()) {
        auto it = wb_.find(wb_order_.front());
        if (it == wb_.end()) { wb_order_.pop_front(); continue; }
        if (it->second.ev.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete)
            break;
        vring_free_.push_back(it->second.ring);
        flags_[it->first] &= ~2;
        landed_[it->first] = 1;
        set_bonus(it->first);
        wb_.erase(it);
        wb_order_.pop_front();
    }
}

int Tier::take_ram_slot(const std::vector<int64_t>& protect, int c, bool wait) {
    if (!free_ram_[c].empty()) { int s = free_ram_[c].back(); free_ram_[c].pop_back(); return s; }
    for (int attempt = 0; attempt < 2; attempt++) {
        Lru& lru = *ram_lru_[c];
        for (int32_t k = lru.head(); k >= 0; k = lru.next_of(k)) {
            if (has(protect, k) || (flags_[k] & 3)) continue;
            int rs = (int)lru.pop(k);
            ram_of_[k] = -1;
            ram_key_[rs] = -1;
            landed_[k] = 0;
            set_bonus(k);
            return rs;
        }
        if (attempt || inflight_.empty()) break;
        reap_inflight();
        if (wait)
            for (int32_t k = lru.head(); k >= 0; k = lru.next_of(k))
                if (!has(protect, k) && (flags_[k] & 1) && !(flags_[k] & 2)) {
                    rd_->wait(inflight_[k]);
                    inflight_.erase(k);
                    flags_[k] &= ~1;
                    demand_.erase(k);
                    break;
                }
    }
    return -1;
}

void Tier::write_back(sycl::queue& q, int64_t k, int s, const std::vector<int64_t>& protect) {
    static const bool off = getenv("GLM_TIER_NO_WB") && atoi(getenv("GLM_TIER_NO_WB"));   // experiment switch
    if (ram_addr_.empty() || off) return;
    if (vring_free_.empty()) reap_wb();
    if (vring_free_.empty()) { stats.wb_dropped++; return; }
    int rs = take_ram_slot(protect, key_cls_[k], false);
    if (rs < 0) return;
    int j = vring_free_.back();
    vring_free_.pop_back();
    size_t rec = lay_[k / E].rec;
    // VRAM slot -> ring (compute queue, before the slot is overwritten), ring -> RAM slot (copy queue)
    sycl::event e1 = q.memcpy(vring_ + (size_t)j * crec_[0], (void*)vaddr_[s], rec);
    sycl::event e2 = Dev::get().copy.memcpy((void*)ram_addr_[rs], vring_ + (size_t)j * crec_[0], rec, e1);
    ram_put(k, rs);
    flags_[k] |= 2;
    auto old = wb_.find(k);
    if (old != wb_.end()) {          // an earlier write-back of the same expert still pending: its ring slot is ours
        old->second.ev.wait();
        vring_free_.push_back(old->second.ring);
    }
    wb_[k] = Wb{e2, j};
    wb_order_.push_back(k);
    stats.writeback++;
}

int Tier::take_vram_slot(sycl::queue& q, const std::vector<int64_t>& protect, int c) {
    if (!free_vram_[c].empty()) { int s = free_vram_[c].back(); free_vram_[c].pop_back(); return s; }
    Lru& lru = *vram_lru_[c];
    for (int32_t k = lru.head(); k >= 0; k = lru.next_of(k)) {
        if (has(protect, k)) continue;
        int s = (int)lru.pop(k);
        write_back(q, k, s, protect);
        ptrs_h_[k] = 0;                    // other layers' device rows go stale; only picked rows are read
        vram_of_[k] = -1;
        vram_key_[s] = -1;
        set_bonus(k);
        return s;
    }
    GLM_CHECK(false, "tier: VRAM pool too small for one layer's picks");
    return -1;
}

void Tier::prefetch(int li, const std::vector<int64_t>& es, const std::vector<int64_t>& protect) {
    static const bool off = getenv("GLM_TIER_NO_PF") && atoi(getenv("GLM_TIER_NO_PF"));   // experiment switch
    if (off) return;
    for (int64_t e : es) {
        int64_t k = (int64_t)li * E + e;
        if (vram_of_[k] >= 0 || ram_of_[k] >= 0 || (flags_[k] & 1)) continue;
        int rs = take_ram_slot(protect, key_cls_[k], false);
        if (rs < 0) continue;
        ram_put(k, rs);
        inflight_[k] = submit_bg(k, rs, false);
        stats.prefetch++;
    }
}

void Tier::after_sync(int li, int nli, const int64_t* sel, const int64_t* exact, int nsel, const int64_t* pred,
                      int npred) {
    int64_t base = (int64_t)li * E;
    std::vector<int64_t> keys;
    for (int i = 0; i < nsel; i++) if (!has(keys, base + sel[i])) keys.push_back(base + sel[i]);
    // begin_demand: this layer's NVMe reads first (urgent), queued prefetches of its picks promoted
    for (int64_t k : keys) {
        if (vram_of_[k] >= 0) continue;
        auto it = inflight_.find(k);
        if (it != inflight_.end()) { rd_->promote(it->second); continue; }
        if (ram_of_[k] >= 0) continue;
        int rs = take_ram_slot(keys, key_cls_[k], true);
        if (rs < 0) continue;
        ram_put(k, rs);
        inflight_[k] = submit_bg(k, rs, true);
        demand_.insert(k);
    }
    // substitutions: an exact pick the bonus router passed over still counts for admission
    for (int i = 0; i < nsel; i++) {
        bool chosen = false;
        for (int j = 0; j < nsel; j++) chosen |= sel[j] == exact[i];
        if (!chosen) { stats.substituted++; freq_[base + exact[i]] += 1.0f; }
    }
    // next layer's predicted picks into RAM
    if (nli >= 0 && npred) {
        std::vector<int64_t> v(pred, pred + npred);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        prefetch(nli, v, keys);
    }
}

bool Tier::want_admit(int64_t k, int c, int n_adm, const std::vector<int64_t>& protect, std::vector<int64_t>* consumed) {
    if (n_adm >= cfg_.admit_per_call || freq_[k] < cfg_.admit_min) return false;
    if (!free_vram_[c].empty()) return true;
    Lru& lru = *vram_lru_[c];
    for (int32_t v = lru.head(); v >= 0; v = lru.next_of(v)) {
        if (has(protect, v) || (consumed && has(*consumed, v))) continue;
        bool adm = freq_[v] < freq_[k];
        if (adm && consumed) consumed->push_back(v);
        return adm;
    }
    return false;
}

void Tier::ensure(sycl::queue& q, int li, const int64_t* sel, int n,
                  const std::function<void(const std::vector<int64_t>& late)>& early) {
    if (!inflight_.empty() && tick_ % 64 == 0) reap_inflight();
    reap_wb();
    tick_++;
    stats.calls++;
    int64_t base = (int64_t)li * E;
    std::vector<int64_t> u(sel, sel + n);
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    std::vector<int64_t> protect;
    for (int64_t e : u) protect.push_back(base + e);
    for (int64_t k : protect) { freq_[k] += 1.0f; hits_[k] += 1.0; }
    if (tick_ % cfg_.decay_every == 0)
        for (auto& f : freq_) f *= 0.5f;
    stats.picks += u.size();
    bool changed = false;
    int n_adm = 0;
    std::vector<int64_t> consumed, slow, admit;
    int64_t* row = ptrs_h_ + base;
    for (int64_t e : u) {
        int64_t k = base + e;
        int64_t s = vram_of_[k];
        if (s >= 0) {
            vram_lru_[vcls_[s]]->touch(k);
            stats.vram_hit++;
            if (row[e] != vaddr_[s]) { row[e] = vaddr_[s]; changed = true; }
            continue;
        }
        int64_t rs = ram_of_[k];
        if (rs < 0 || (flags_[k] & 3) || !landed_[k]) { slow.push_back(e); continue; }
        int c = key_cls_[k];
        if (want_admit(k, c, n_adm, protect, &consumed)) { n_adm++; admit.push_back(e); continue; }
        ram_lru_[c]->touch(k);
        if (row[e] != ram_addr_[rs]) { row[e] = ram_addr_[rs]; changed = true; }
        stats.ram_hit++;
        stats.zero_copy++;
    }
    // RAM -> VRAM on the compute queue; the RAM slot is free once that copy has run (host writers only reuse it after
    // the next layer's routing sync, or after the q.wait() below)
    auto admit_one = [&](int64_t e) {
        int64_t k = base + e;
        int c = key_cls_[k];
        int s = take_vram_slot(q, protect, c);
        int rs = (int)ram_of_[k];
        auto w = wb_.find(k);
        if (w != wb_.end()) q.ext_oneapi_submit_barrier({w->second.ev});
        q.memcpy((void*)vaddr_[s], (void*)ram_addr_[rs], lay_[li].rec);
        ram_of_[k] = -1;
        ram_key_[rs] = -1;
        ram_lru_[c]->pop(k);
        free_ram_[c].push_back(rs);
        vram_of_[k] = s;
        vram_key_[s] = k;
        vram_lru_[c]->set(k, s);
        row[e] = vaddr_[s];
        changed = true;
        set_bonus(k);
        stats.admit++;
    };
    if (!slow.empty()) {
        // in flight (prefetch or demand), write-back pending, or NVMe only
        auto t0 = clk::now();
        std::vector<int64_t> waits, forced;
        for (size_t i = 0; i < slow.size(); i++) {
            int64_t e = slow[i], k = base + e;
            auto it = inflight_.find(k);
            if (it != inflight_.end()) {
                rd_->promote(it->second);
                waits.push_back(it->second);
                inflight_.erase(it);
                if (demand_.erase(k)) stats.nvme++;
                else stats.prefetch_used++;
                continue;
            }
            if (ram_of_[k] >= 0) continue;
            int rs = take_ram_slot(protect, key_cls_[k], true);
            if (rs < 0) {
                // more picks than this class has RAM slots (a prompt piece): finish the reads so far and move those
                // experts to VRAM now, which frees their RAM slots
                for (int64_t id : waits) rd_->wait(id);
                waits.clear();
                for (size_t j = 0; j < i; j++) {
                    int64_t e2 = slow[j], k2 = base + e2;
                    if (ram_of_[k2] < 0 || has(forced, e2) || key_cls_[k2] != key_cls_[k]) continue;
                    flags_[k2] &= ~1;
                    landed_[k2] = 1;
                    admit_one(e2);
                    forced.push_back(e2);
                }
                q.wait();
                rs = take_ram_slot(protect, key_cls_[k], true);
                GLM_CHECK(rs >= 0, "tier: no RAM or VRAM slot left for layer %d's picks", li);
            }
            ram_put(k, rs);
            flags_[k] |= 1;
            const LayerInfo& l = lay_[li];
            waits.push_back(rd_->submit(k, l.off + e * l.rec, l.rec, ram_host_[rs], cfg_.urgent_pieces, rs, true));
            stats.nvme++;
        }
        if (early && !waits.empty() && forced.empty()) {
            // the ready picks' rows (and their admissions) now, so their experts run while the reads finish
            for (int64_t e : admit) admit_one(e);
            admit.clear();
            if (changed) q.memcpy(ptrs_dev_.ptr<int64_t>() + base, row, (size_t)E * sizeof(int64_t));
            changed = false;
            early(slow);
            stats.early++;
        }
        for (int64_t id : waits) rd_->wait(id);
        if (!waits.empty()) stats.t_nvme += secs(t0);
        for (int64_t e : slow) {
            if (has(forced, e)) continue;
            int64_t k = base + e;
            flags_[k] &= ~1;
            if (!(flags_[k] & 2)) landed_[k] = 1;
            int c = key_cls_[k];
            if (want_admit(k, c, n_adm, protect, nullptr)) { n_adm++; admit.push_back(e); continue; }
            auto w = wb_.find(k);
            if (w != wb_.end()) q.ext_oneapi_submit_barrier({w->second.ev});    // its RAM copy must have landed
            ram_lru_[c]->touch(k);
            row[e] = ram_addr_[ram_of_[k]];
            changed = true;
            stats.zero_copy++;
            set_bonus(k);
        }
    }
    for (int64_t e : admit) admit_one(e);
    if (changed) q.memcpy(ptrs_dev_.ptr<int64_t>() + base, row, (size_t)E * sizeof(int64_t));
}

void Tier::warm(const std::vector<std::pair<int, int>>& order) {
    auto& q = Dev::get().q;
    auto t0 = clk::now();
    std::vector<int> nv(C_), nr(C_);
    for (int c = 0; c < C_; c++) { nv[c] = free_vram_[c].size(); nr[c] = free_ram_[c].size(); }
    std::vector<int64_t> vjobs, rjobs;
    for (auto [li, e] : order) {
        int64_t k = (int64_t)li * E + e;
        int c = key_cls_[k];
        if (vram_of_[k] >= 0 || ram_of_[k] >= 0) continue;
        if (nv[c] > 0) { vjobs.push_back(k); nv[c]--; }
        else if (nr[c] > 0) { rjobs.push_back(k); nr[c]--; }
    }
    // VRAM: read through RAM slots in batches (the RAM tier is still empty), copy up; a batch is flushed when it is
    // full or its class has no free RAM slot left
    std::vector<std::pair<int64_t, int>> batch;
    std::vector<int64_t> ids;
    auto flush = [&] {
        for (int64_t id : ids) rd_->wait(id);
        for (auto [k, rs] : batch) {
            int c = key_cls_[k];
            int s = free_vram_[c].back();
            free_vram_[c].pop_back();
            q.memcpy((void*)vaddr_[s], (void*)ram_addr_[rs], lay_[k / E].rec);
            vram_of_[k] = s;
            vram_key_[s] = k;
            vram_lru_[c]->set(k, s);
            ptrs_h_[k] = vaddr_[s];
            set_bonus(k);
        }
        q.wait();
        for (auto [k, rs] : batch) free_ram_[key_cls_[k]].push_back(rs);
        batch.clear();
        ids.clear();
    };
    for (int64_t k : vjobs) {
        int c = key_cls_[k];
        if (batch.size() >= 64 || free_ram_[c].empty()) flush();
        GLM_CHECK(!free_ram_[c].empty(), "warm: class %d has no RAM slot to stage through", c);
        int rs = free_ram_[c].back();
        free_ram_[c].pop_back();
        const LayerInfo& l = lay_[k / E];
        ids.push_back(rd_->submit(k, l.off + (k % E) * l.rec, l.rec, ram_host_[rs], cfg_.pieces, -1, false));
        batch.push_back({k, rs});
    }
    flush();
    double tv = secs(t0);
    auto t1 = clk::now();
    ids.clear();
    for (int64_t k : rjobs) {
        int c = key_cls_[k];
        int rs = free_ram_[c].back();
        free_ram_[c].pop_back();
        ram_put(k, rs);
        ids.push_back(submit_bg(k, rs, false));
    }
    for (int64_t id : ids) rd_->wait(id);
    for (int64_t k : rjobs) { flags_[k] &= ~1; landed_[k] = 1; set_bonus(k); }
    q.memcpy(ptrs_dev_.p, ptrs_h_, (size_t)L * E * sizeof(int64_t)).wait();
    fprintf(stderr, "tier warm: %zu VRAM experts in %.1f s, %zu RAM experts in %.1f s\n", vjobs.size(), tv, rjobs.size(),
            secs(t1));
}

// ------------------------------------------------------------------------------------------------ prompts
void* Tier::lend(size_t bytes) {
    GLM_CHECK(!prompt_ && lent_.empty(), "tier: lend() goes before begin_prompt, once per prompt");
    const int c = stage_cls_;
    int k = (int)((bytes + crec_[c] - 1) / crec_[c]), first = -1, n = 0;
    for (int s = 0; s < (int)vcls_.size(); s++)
        if (vcls_[s] == c) { if (first < 0) first = s; n++; }
    // beside the run: begin_prompt's E staging slots and room for one layer's decode picks
    GLM_CHECK(k + E + 64 <= n, "tier: a %.2f GiB prompt buffer does not fit beside the VRAM pool's staging slots",
              bytes / 1073741824.0);
    Lru& lru = *vram_lru_[c];
    auto& fv = free_vram_[c];
    for (int s = first + n - k; s < first + n; s++) {
        int64_t key = vram_key_[s];
        if (key >= 0) {
            if (lru.contains(key)) lru.pop(key);
            ptrs_h_[key] = 0;
            vram_of_[key] = -1;
            vram_key_[s] = -1;
            set_bonus(key);
        } else {
            auto it = std::find(fv.begin(), fv.end(), s);
            if (it != fv.end()) fv.erase(it);
        }
        lent_.push_back(s);
    }
    return (void*)vaddr_[first + n - k];
}

void Tier::prepare_prompts() {
    auto& q = Dev::get().q;
    if (!sptrs_h_) {
        // ring slots of the staging class's record size (the largest main-model record)
        sptrs_h_ = sycl::malloc_host<int64_t>((size_t)L * E, q);
        sptrs_dev_ = Tensor::zeros({L, E}, DType::I64);
        size_t rec = crec_[stage_cls_], n_ring = std::max<size_t>(64, cfg_.ring_bytes / rec);
        if (direct_) {
            ring_win_ = std::make_unique<HostWindows>();
            GLM_CHECK(ring_win_->init(n_ring, rec), "tier: memfd read-ahead ring");
            for (size_t i = 0; i < n_ring; i++) { ring_.push_back(ring_win_->host(i)); ring_dev_.push_back(ring_win_->dev(i)); }
        } else {
            for (size_t i = 0; i < n_ring; i++) {
                char* r = static_cast<char*>(sycl::malloc_host(rec, q));
                GLM_CHECK(r, "tier: read-ahead ring");
                memset(r, 0, rec);
                ring_.push_back(r);
                ring_dev_.push_back(r);
            }
        }
        rs_.resize(ring_.size());
        fprintf(stderr, "tier: prompt read-ahead ring %zu x %.1f MiB\n", ring_.size(), rec / 1048576.0);
    }
}

void Tier::begin_prompt(int first_li, bool dbl) {
    auto& q = Dev::get().q;
    prepare_prompts();
    // borrow E slots of the staging class (2E for dbl): free ones first, then the least recently used (their experts
    // dropped). dbl keeps 64 for one layer's decode picks, as lend() does
    auto& fv = free_vram_[stage_cls_];
    Lru& lru = *vram_lru_[stage_cls_];
    dbl_ = dbl && (int)(fv.size() + lru.size()) >= 2 * E + 64;
    auto borrow = [&](std::vector<int>& set) {
        set.clear();
        while ((int)set.size() < E && !fv.empty()) { set.push_back(fv.back()); fv.pop_back(); }
        while ((int)set.size() < E && lru.size()) {
            int64_t k = lru.head();
            int s = (int)lru.pop(k);
            ptrs_h_[k] = 0;
            vram_of_[k] = -1;
            vram_key_[s] = -1;
            set_bonus(k);
            set.push_back(s);
        }
        GLM_CHECK((int)set.size() == E, "tier: not enough VRAM slots to stage a layer");
    };
    borrow(stage_slots_);
    if (dbl_) {
        borrow(stage2_slots_);
        for (int i = 0; i < 2; i++) {
            pre_li_[i] = -1;
            pre_done_[i].assign(E, 0);
            set_free_[i] = q.ext_oneapi_submit_barrier();
        }
        copied_ = false;
    }
    // decode pointer rows of the dropped experts: refreshed by ensure() before any decode reads them
    q.memcpy(ptrs_dev_.p, ptrs_h_, (size_t)L * E * sizeof(int64_t));
    prompt_ = true;
    plan_li_ = first_li;
    plan_e_ = 0;
    plan_reads();
}

int Tier::free_ring_slot(bool wait) {
    for (int pass = 0; pass < 2; pass++) {
        for (size_t i = 0; i < rs_.size(); i++) {
            RingSlot& r = rs_[i];
            if (!r.busy) return (int)i;
        }
        // slots whose staging copy has run are free again
        for (size_t i = 0; i < rs_.size(); i++) {
            RingSlot& r = rs_[i];
            if (r.busy && r.job < 0 && r.ev.get_info<sycl::info::event::command_execution_status>() ==
                                           sycl::info::event_command_status::complete) {
                ring_of_.erase(r.key);
                r = RingSlot{};
                return (int)i;
            }
        }
        if (!wait) return -1;
        // wait for the oldest staging copy
        bool waited = false;
        for (size_t i = 0; i < rs_.size(); i++)
            if (rs_[i].busy && rs_[i].job < 0) { rs_[i].ev.wait(); waited = true; break; }
        if (waited) continue;
        // every slot holds a planned read: take the one for the farthest layer (re-planned later)
        int far = -1;
        for (size_t i = 0; i < rs_.size(); i++)
            if (rs_[i].busy && rs_[i].job >= 0 && (far < 0 || rs_[i].key > rs_[far].key)) far = (int)i;
        if (far >= 0) {
            rd_->wait(rs_[far].job);
            int li = (int)(rs_[far].key / E);
            if (li < plan_li_ || (li == plan_li_ && rs_[far].key % E < plan_e_)) { plan_li_ = li; plan_e_ = 0; }
            ring_of_.erase(rs_[far].key);
            rs_[far] = RingSlot{};
            return far;
        }
    }
    return -1;
}

void Tier::plan_reads() {
    // queue reads of the coming layers' experts that are in neither VRAM nor RAM, layer by layer, while ring slots last
    for (; plan_li_ < n_moe; plan_li_++, plan_e_ = 0) {
        if (lay_[plan_li_].rec > crec_[stage_cls_]) continue;
        for (; plan_e_ < E; plan_e_++) {
            int64_t k = (int64_t)plan_li_ * E + plan_e_;
            if (vram_of_[k] >= 0 || (ram_of_[k] >= 0 && !(flags_[k] & 1)) || ring_of_.count(k)) continue;
            size_t n_free = 0;      // keep some slots for the current layer's unplanned reads
            for (auto& r : rs_) n_free += !r.busy || (r.job < 0 && r.ev.get_info<sycl::info::event::command_execution_status>() ==
                                                                  sycl::info::event_command_status::complete);
            if (n_free <= 32) return;
            int r = free_ring_slot(false);
            if (r < 0) return;
            const LayerInfo& l = lay_[plan_li_];
            rs_[r].key = k;
            rs_[r].busy = true;
            rs_[r].job = rd_->submit(k, l.off + (size_t)plan_e_ * l.rec, l.rec, ring_[r], cfg_.pieces, -1, false);
            ring_of_[k] = r;
        }
    }
}

void Tier::prestage(int li) {
    if (!prompt_ || !dbl_ || li < 0 || li >= n_moe) return;
    const LayerInfo& l = lay_[li];
    if (l.rec > crec_[stage_cls_]) return;
    const int set = li % 2;
    if (pre_li_[set] != li) {
        pre_li_[set] = li;
        std::fill(pre_done_[set].begin(), pre_done_[set].end(), 0);
    }
    auto& slots = set ? stage2_slots_ : stage_slots_;
    auto& cq = Dev::get().copy;
    int64_t base = (int64_t)li * E;
    for (int e = 0; e < E; e++) {
        if (pre_done_[set][e]) continue;
        int64_t k = base + e;
        if (vram_of_[k] >= 0) continue;
        void* dst = (void*)vaddr_[slots[e]];
        if (ram_of_[k] >= 0) {
            auto it = inflight_.find(k);
            if (it != inflight_.end()) {              // a decode-path read into its RAM slot
                if (!rd_->done(it->second)) continue;
                rd_->wait(it->second);
                inflight_.erase(it);
                flags_[k] &= ~1;
                landed_[k] = 1;
            }
            last_copy_ = cq.memcpy(dst, (void*)ram_addr_[ram_of_[k]], l.rec, set_free_[set]);
            stats.staged_ram++;
        } else {
            auto rit = ring_of_.find(k);
            if (rit == ring_of_.end()) continue;
            int r = rit->second;
            if (rs_[r].job < 0 || !rd_->done(rs_[r].job)) continue;
            rd_->wait(rs_[r].job);
            stats.readahead_used++;
            rs_[r].job = -1;
            last_copy_ = rs_[r].ev = cq.memcpy(dst, ring_dev_[r], l.rec, set_free_[set]);
            stats.staged_nvme++;
        }
        pre_done_[set][e] = 1;
        copied_ = true;
    }
    plan_reads();
}

void Tier::moe_done(sycl::queue& q, int li) {
    if (dbl_ && li >= 0) set_free_[li % 2] = q.ext_oneapi_submit_barrier();
}

const int64_t* Tier::stage(sycl::queue& q, int li, const std::vector<int>& experts) {
    GLM_CHECK(prompt_, "tier: stage() outside a prompt");
    if (dbl_ && lay_[li].rec <= crec_[stage_cls_]) {
        // set li % 2: what prestage has not copied yet goes the same way, reads waited for
        prestage(li);
        const int set = li % 2;
        auto& slots = set ? stage2_slots_ : stage_slots_;
        auto& cq = Dev::get().copy;
        const LayerInfo& l = lay_[li];
        int64_t base = (int64_t)li * E;
        int64_t* row = sptrs_h_ + base;
        std::fill(row, row + E, 0);
        auto t0 = clk::now();
        for (int e : experts) {
            int64_t k = base + e;
            if (vram_of_[k] >= 0) { row[e] = vaddr_[vram_of_[k]]; stats.staged_vram++; continue; }
            row[e] = vaddr_[slots[e]];
            if (pre_done_[set][e]) continue;
            void* dst = (void*)vaddr_[slots[e]];
            auto it = inflight_.find(k);
            if (it != inflight_.end()) {
                rd_->wait(it->second);
                inflight_.erase(it);
                flags_[k] &= ~1;
                landed_[k] = 1;
            }
            if (ram_of_[k] >= 0) {
                last_copy_ = cq.memcpy(dst, (void*)ram_addr_[ram_of_[k]], l.rec, set_free_[set]);
                stats.staged_ram++;
            } else {
                auto rit = ring_of_.find(k);
                int r;
                if (rit != ring_of_.end()) {
                    r = rit->second;
                    if (rs_[r].job >= 0) {
                        rd_->promote(rs_[r].job);
                        rd_->wait(rs_[r].job);
                        stats.readahead_used++;
                    }
                } else {
                    r = free_ring_slot(true);
                    GLM_CHECK(r >= 0, "tier: read-ahead ring exhausted");
                    rs_[r].key = k;
                    rs_[r].busy = true;
                    ring_of_[k] = r;
                    rd_->wait(rd_->submit(k, l.off + (size_t)e * l.rec, l.rec, ring_[r], cfg_.urgent_pieces, -1, true));
                }
                rs_[r].job = -1;
                last_copy_ = rs_[r].ev = cq.memcpy(dst, ring_dev_[r], l.rec, set_free_[set]);
                stats.staged_nvme++;
            }
            pre_done_[set][e] = 1;
            copied_ = true;
        }
        stats.t_stage_wait += secs(t0);
        if (copied_) { q.ext_oneapi_submit_barrier({last_copy_}); copied_ = false; }
        q.memcpy(sptrs_dev_.ptr<int64_t>() + base, row, (size_t)E * sizeof(int64_t));
        plan_reads();
        return sptrs_dev_.ptr<int64_t>() + base;
    }
    int64_t base = (int64_t)li * E;
    int64_t* row = sptrs_h_ + base;
    std::fill(row, row + E, 0);
    const LayerInfo& l = lay_[li];
    bool stage_ok = l.rec <= crec_[stage_cls_];
    size_t next = 0;
    auto t0 = clk::now();
    for (int e : experts) {
        int64_t k = base + e;
        if (vram_of_[k] >= 0) { row[e] = vaddr_[vram_of_[k]]; stats.staged_vram++; continue; }
        if (!stage_ok) {
            // a class without staging rows (its pool holds the whole layer): the decode path makes it resident
            int64_t ek = e;
            ensure(q, li, &ek, 1);
            row[e] = ptrs_h_[k];
            continue;
        }
        GLM_CHECK(next < stage_slots_.size(), "tier: more experts than staging slots");
        int s = stage_slots_[next++];
        void* dst = (void*)vaddr_[s];
        auto it = inflight_.find(k);
        if (it != inflight_.end()) {            // a decode-path read into its RAM slot
            rd_->wait(it->second);
            inflight_.erase(it);
            flags_[k] &= ~1;
            landed_[k] = 1;
        }
        if (ram_of_[k] >= 0) {
            auto w = wb_.find(k);
            if (w != wb_.end()) q.ext_oneapi_submit_barrier({w->second.ev});
            q.memcpy(dst, (void*)ram_addr_[ram_of_[k]], l.rec);
            stats.staged_ram++;
        } else {
            auto rit = ring_of_.find(k);
            int r;
            if (rit != ring_of_.end()) {
                r = rit->second;
                rd_->promote(rs_[r].job);
                rd_->wait(rs_[r].job);
                stats.readahead_used++;
            } else {
                r = free_ring_slot(true);
                GLM_CHECK(r >= 0, "tier: read-ahead ring exhausted");
                rs_[r].key = k;
                rs_[r].busy = true;
                ring_of_[k] = r;
                rd_->wait(rd_->submit(k, l.off + (size_t)e * l.rec, l.rec, ring_[r], cfg_.urgent_pieces, -1, true));
            }
            rs_[r].job = -1;
            rs_[r].ev = q.memcpy(dst, ring_dev_[r], l.rec);
            stats.staged_nvme++;
        }
        row[e] = vaddr_[s];
    }
    stats.t_stage_wait += secs(t0);
    q.memcpy(sptrs_dev_.ptr<int64_t>() + base, row, (size_t)E * sizeof(int64_t));
    plan_reads();
    return sptrs_dev_.ptr<int64_t>() + base;
}

void Tier::end_prompt() {
    auto& q = Dev::get().q;
    q.wait();
    Dev::get().copy.wait();      // dbl: copies into the sets
    // reads still queued for layers the prompt did not reach
    for (auto& r : rs_)
        if (r.busy && r.job >= 0) rd_->wait(r.job);
    for (auto& r : rs_) r = RingSlot{};
    ring_of_.clear();
    for (int s : stage_slots_) free_vram_[stage_cls_].push_back(s);
    stage_slots_.clear();
    for (int s : stage2_slots_) free_vram_[stage_cls_].push_back(s);
    stage2_slots_.clear();
    dbl_ = false;
    for (int s : lent_) free_vram_[stage_cls_].push_back(s);
    lent_.clear();
    prompt_ = false;
}

std::string Tier::summary() const {
    char b[512];
    double c = std::max<size_t>(stats.calls, 1) / 42.0;   // ~ tokens
    snprintf(b, sizeof b,
             "per token: vram %.1f, ram zero-copy %.1f, nvme %.1f, prefetch %.1f (used %.1f), admit %.1f, writeback %.1f, "
             "substituted %.1f, NVMe wait %.1f ms (%.1f layers split around it)",
             stats.vram_hit / c, stats.zero_copy / c, stats.nvme / c, stats.prefetch / c, stats.prefetch_used / c,
             stats.admit / c, stats.writeback / c, stats.substituted / c, stats.t_nvme * 1e3 / c, stats.early / c);
    return b;
}

}  // namespace glm
