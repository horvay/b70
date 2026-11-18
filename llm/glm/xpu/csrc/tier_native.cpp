// Native NVMe expert reader for the GLM expert tier (glm_xpu/tier.py), replacing a Python thread pool.
//
// A record read is split into page-aligned pieces; worker threads pread() each piece with O_DIRECT into a thread-local
// aligned bounce buffer and memcpy it to the destination (pinned USM host memory: O_DIRECT straight into it fails with
// EFAULT). No Python and no GIL on the read path, so readers no longer compete with the decode loop's host step.
//
// A background read (submit with res_slot >= 0) marks its expert resident when every piece has landed and the expert
// still owns that RAM slot: res[key] = 0 and bonus[key] = bonus_ram (bonus_vram if it was admitted to VRAM meanwhile),
// then sets *dirty so the host uploads the tables before the next router (Tier.sync_res).
//
// Python sees a job id: done(id), wait(id) (GIL released while waiting; raises on a read error), forget(id).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <errno.h>
#include <unistd.h>

namespace py = pybind11;

namespace {

struct Job {
    int64_t key = -1;
    int64_t res_slot = -1;               // RAM slot whose landing marks the expert resident; -1: no residency update
    std::atomic<int> left{0};            // pieces still to read
    std::atomic<int> err{0};             // errno of the first failed piece (or -1 for a short read)
    std::atomic<bool> done{false};
};

struct Piece {
    std::shared_ptr<Job> job;
    int64_t file_off;
    int64_t n;
    uint8_t* dst;
};

class Reader {
public:
    Reader(int fd, int threads, int64_t max_piece, uintptr_t res, uintptr_t bonus, uintptr_t vram_of, uintptr_t ram_of,
           uintptr_t dirty, double bonus_vram, double bonus_ram, bool bonus_on)
        : fd_(fd), max_piece_(max_piece), res_(reinterpret_cast<float*>(res)), bonus_(reinterpret_cast<float*>(bonus)),
          vram_of_(reinterpret_cast<const int64_t*>(vram_of)), ram_of_(reinterpret_cast<const int64_t*>(ram_of)),
          dirty_(reinterpret_cast<uint8_t*>(dirty)), bonus_vram_((float)bonus_vram), bonus_ram_((float)bonus_ram),
          bonus_on_(bonus_on) {
        if (threads < 1) throw std::invalid_argument("tier_native.Reader: threads must be >= 1");
        max_spec_ = threads > 4 ? threads - 4 : threads;    // 4 threads stay free for urgent reads
        if (max_piece <= 0 || max_piece % 4096)
            throw std::invalid_argument("tier_native.Reader: max_piece must be a positive multiple of 4096");
        try {
            for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
        } catch (...) {
            // joinable threads in a destroyed vector would std::terminate: stop and join the ones that started
            {
                std::lock_guard<std::mutex> g(mu_);
                stopping_ = true;
            }
            cv_work_.notify_all();
            for (auto& t : workers_)
                if (t.joinable()) t.join();
            throw;
        }
    }

    ~Reader() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> g(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    // Read record bytes [file_off, file_off + n) into dst in `pieces` page-aligned pieces. Returns a job id.
    // urgent: a read the decode step is about to wait on goes ahead of queued speculative prefetches
    int64_t submit(int64_t key, int64_t file_off, int64_t n, uintptr_t dst, int pieces, int64_t res_slot, bool urgent) {
        if (n <= 0 || n % 4096 || file_off % 4096) throw std::invalid_argument("tier_native: unaligned read");
        if (pieces < 1) pieces = 1;
        int64_t pages = n / 4096;
        int64_t per = (pages + pieces - 1) / pieces * 4096;
        if (per > max_piece_) per = max_piece_;
        auto job = std::make_shared<Job>();
        job->key = key;
        job->res_slot = res_slot;
        int count = (int)((n + per - 1) / per);
        job->left.store(count);
        int64_t id;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (stopping_) throw std::runtime_error("tier_native: reader stopped");
            id = next_id_++;
            jobs_[id] = job;
            std::vector<Piece> ps;
            for (int64_t o = 0; o < n; o += per)
                ps.push_back(Piece{job, file_off + o, std::min(per, n - o), reinterpret_cast<uint8_t*>(dst) + o});
            if (urgent)
                queue_.insert(queue_.begin() + n_urgent_q(), ps.begin(), ps.end());   // after earlier urgent ones
            else
                queue_.insert(queue_.end(), ps.begin(), ps.end());
            if (urgent) urgent_jobs_.push_back(job);
        }
        cv_work_.notify_all();
        return id;
    }

    // Move a queued (speculative) job's pieces to the front, after other urgent work: the decode step now waits on it.
    void promote(int64_t id) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = jobs_.find(id);
        if (it == jobs_.end() || it->second->done.load()) return;
        std::shared_ptr<Job> jp = it->second;
        Job* j = jp.get();
        size_t nu = n_urgent_q();
        // its queued pieces move (in order) to just after the urgent prefix; nothing moves when none are queued
        auto first = queue_.begin() + (std::ptrdiff_t)nu;
        if (std::none_of(first, queue_.end(), [j](const Piece& p) { return p.job.get() == j; })) return;
        std::stable_partition(first, queue_.end(), [j](const Piece& p) { return p.job.get() == j; });
        urgent_jobs_.push_back(jp);
    }

    bool done(int64_t id) {
        auto j = get(id);
        return !j || j->done.load(std::memory_order_acquire);
    }

    // Wait for a job (GIL released); raises if a piece failed. The job stays registered until forget().
    void wait(int64_t id) {
        auto j = get(id);
        if (!j) return;
        if (!j->done.load(std::memory_order_acquire)) {
            py::gil_scoped_release nogil;
            std::unique_lock<std::mutex> lk(mu_done_);
            cv_done_.wait(lk, [&] { return j->done.load(std::memory_order_acquire); });
        }
        int e = j->err.load();
        if (e) throw std::runtime_error("tier_native: read of expert record " + std::to_string(j->key) + " failed: " +
                                        (e > 0 ? std::string(strerror(e)) : std::string("short read")));
    }

    void forget(int64_t id) {
        std::lock_guard<std::mutex> g(mu_);
        jobs_.erase(id);
    }

    size_t pending() {
        std::lock_guard<std::mutex> g(mu_);
        return queue_.size();
    }

    py::dict stats() {
        py::dict d;
        d["t_pread"] = ns_pread_.load() * 1e-9;
        d["t_copy"] = ns_copy_.load() * 1e-9;
        d["pieces"] = (int64_t)pieces_.load();
        return d;
    }

private:
    // queued pieces at the front that belong to urgent jobs (caller holds mu_)
    size_t n_urgent_q() {
        // forget finished urgent jobs
        urgent_jobs_.erase(std::remove_if(urgent_jobs_.begin(), urgent_jobs_.end(),
                                          [](const std::shared_ptr<Job>& j) { return j->left.load() == 0; }),
                           urgent_jobs_.end());
        size_t n = 0;
        while (n < queue_.size() && std::any_of(urgent_jobs_.begin(), urgent_jobs_.end(),
                                                [&](const std::shared_ptr<Job>& j) { return j == queue_[n].job; }))
            ++n;
        return n;
    }

    std::shared_ptr<Job> get(int64_t id) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    void run() {
        uint8_t* buf = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&buf), 4096, (size_t)max_piece_) != 0) buf = nullptr;
        for (;;) {
            Piece p;
            bool spec = false;
            {
                // a speculative piece may only start while fewer than max_spec_ are running, so urgent reads (the
                // decode step waiting) always find a free thread instead of queueing behind in-progress prefetches
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&] {
                    return stopping_ || (!queue_.empty() && (n_urgent_q() > 0 || spec_running_ < max_spec_));
                });
                if (queue_.empty()) break;               // stopping and drained
                spec = n_urgent_q() == 0;
                if (spec) ++spec_running_;
                p = std::move(queue_.front());
                queue_.pop_front();
            }
            int err = 0;
            if (!buf) err = ENOMEM;
            auto t0 = std::chrono::steady_clock::now();
            int64_t got = 0;
            while (!err && got < p.n) {
                ssize_t r = pread(fd_, buf + got, (size_t)(p.n - got), (off_t)(p.file_off + got));
                if (r < 0) {
                    if (errno == EINTR) continue;
                    err = errno;
                } else if (r == 0) {
                    err = -1;
                } else {
                    got += r;
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            if (!err) std::memcpy(p.dst, buf, (size_t)p.n);
            auto t2 = std::chrono::steady_clock::now();
            ns_pread_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            ns_copy_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
            pieces_ += 1;
            if (err) {
                int z = 0;
                p.job->err.compare_exchange_strong(z, err);
            }
            if (spec) {
                {
                    std::lock_guard<std::mutex> g(mu_);
                    --spec_running_;
                }
                cv_work_.notify_one();                   // a speculative piece may wait for this slot
            }
            if (p.job->left.fetch_sub(1, std::memory_order_acq_rel) == 1) finish(*p.job);
        }
        free(buf);
    }

    void finish(Job& j) {
        // Residency publication into the Tier's arrays, which the main thread also reads and writes: each entry is one
        // aligned 4- or 8-byte word, accessed atomically here; the dirty byte is stored last with release ordering so
        // the main thread (Tier.sync_res: test, clear, then upload) never uploads before the entries it announces.
        // A completion racing the main thread's clear is still uploaded: its entries were stored before the flag.
        if (j.res_slot >= 0 && !j.err.load() && __atomic_load_n(&ram_of_[j.key], __ATOMIC_ACQUIRE) == j.res_slot) {
            float zero = 0.0f;
            __atomic_store(&res_[j.key], &zero, __ATOMIC_RELAXED);
            if (bonus_on_) {
                float b = __atomic_load_n(&vram_of_[j.key], __ATOMIC_RELAXED) >= 0 ? bonus_vram_ : bonus_ram_;
                __atomic_store(&bonus_[j.key], &b, __ATOMIC_RELAXED);
            }
            __atomic_store_n(dirty_, (uint8_t)1, __ATOMIC_RELEASE);
        }
        {
            std::lock_guard<std::mutex> g(mu_done_);
            j.done.store(true, std::memory_order_release);
        }
        cv_done_.notify_all();
    }

    int fd_;
    int64_t max_piece_;
    float* res_;
    float* bonus_;
    const int64_t* vram_of_;
    const int64_t* ram_of_;
    uint8_t* dirty_;
    float bonus_vram_, bonus_ram_;
    bool bonus_on_;

    std::mutex mu_;
    std::condition_variable cv_work_;
    std::deque<Piece> queue_;
    std::unordered_map<int64_t, std::shared_ptr<Job>> jobs_;
    std::vector<std::shared_ptr<Job>> urgent_jobs_;   // jobs whose queued pieces sit at the front of queue_
    int spec_running_ = 0;                   // speculative (non-urgent) pieces being read (under mu_)
    int max_spec_ = 1;
    int64_t next_id_ = 1;
    bool stopping_ = false;
    std::vector<std::thread> workers_;

    std::mutex mu_done_;
    std::condition_variable cv_done_;

    std::atomic<int64_t> ns_pread_{0}, ns_copy_{0};
    std::atomic<uint64_t> pieces_{0};
};

// ------------------------------------------------------------------------------------------------------------------
// Lru: an insertion/recency-ordered set of expert keys with a slot value each, the subset of collections.OrderedDict the
// tier uses (move_to_end, item get/set, pop, membership, len, oldest-first iteration). Keys are 0..n_keys-1, linked
// through flat arrays, so the C++ decode step can touch and scan it without Python.
class Lru {
public:
    explicit Lru(int64_t n_keys)
        : prev_(n_keys, -1), next_(n_keys, -1), val_(n_keys, 0), in_(n_keys, 0) {}

    bool contains(int64_t k) const { return k >= 0 && k < (int64_t)in_.size() && in_[k]; }
    size_t size() const { return n_; }
    int32_t head() const { return head_; }
    int32_t next_of(int32_t k) const { return next_[k]; }
    int64_t value(int64_t k) const { return val_[k]; }

    void move_to_end(int64_t k) {
        check(k);
        if (!in_[k]) throw py::key_error(std::to_string(k));
        if (tail_ == k) return;
        unlink((int32_t)k);
        link_back((int32_t)k);
    }
    void set(int64_t k, int64_t v) {           // OrderedDict semantics: an existing key keeps its position
        check(k);
        val_[k] = v;
        if (!in_[k]) link_back((int32_t)k);
    }
    int64_t get(int64_t k) const {
        if (!contains(k)) throw py::key_error(std::to_string(k));
        return val_[k];
    }
    int64_t pop(int64_t k) {
        if (!contains(k)) throw py::key_error(std::to_string(k));
        unlink((int32_t)k);
        return val_[k];
    }
    py::object pop_default(int64_t k, py::object d) {
        if (!contains(k)) return d;
        unlink((int32_t)k);
        return py::int_(val_[k]);
    }
    void clear() {
        for (int32_t k = head_; k >= 0;) {
            int32_t nx = next_[k];
            in_[k] = 0; prev_[k] = next_[k] = -1;
            k = nx;
        }
        head_ = tail_ = -1; n_ = 0;
    }

    // oldest-first iteration; the next key is read before the current one is handed out, so popping the current key
    // inside the loop (the tier's eviction scans do) is safe
    struct It {
        const Lru* l; int32_t cur; int32_t nxt;
        It(const Lru* l_, int32_t c) : l(l_), cur(c), nxt(c >= 0 ? l_->next_[c] : -1) {}
        int64_t operator*() const { return cur; }
        It& operator++() { cur = nxt; nxt = cur >= 0 ? l->next_[cur] : -1; return *this; }
        bool operator==(const It& o) const { return cur == o.cur; }
        bool operator!=(const It& o) const { return cur != o.cur; }
    };
    It begin() const { return It(this, head_); }
    It end() const { return It(this, -1); }

    void unlink(int32_t k) {
        int32_t p = prev_[k], n = next_[k];
        if (p >= 0) next_[p] = n; else head_ = n;
        if (n >= 0) prev_[n] = p; else tail_ = p;
        prev_[k] = next_[k] = -1;
        in_[k] = 0;
        --n_;
    }
    void link_back(int32_t k) {
        prev_[k] = tail_; next_[k] = -1;
        if (tail_ >= 0) next_[tail_] = k; else head_ = k;
        tail_ = k;
        in_[k] = 1;
        ++n_;
    }

private:
    void check(int64_t k) const {
        if (k < 0 || k >= (int64_t)in_.size()) throw py::index_error("Lru key out of range: " + std::to_string(k));
    }
    std::vector<int32_t> prev_, next_;
    std::vector<int64_t> val_;
    std::vector<uint8_t> in_;
    int32_t head_ = -1, tail_ = -1;
    size_t n_ = 0;
};

// ------------------------------------------------------------------------------------------------------------------
// Core: the common part of Tier.ensure (one MoE layer's picks after its routing sync) in one call.
//   For each distinct pick (in ascending expert order, as np.unique): freq += 1, hits += 1;
//   VRAM-resident: touch its VRAM LRU, repair its pointer if it differs from the slot address;
//   RAM-resident, landed (not in flight), not being written back: either an admission candidate (at most
//     admit_per_call per call, freq >= admit_min, and a free VRAM slot of its class or an unprotected LRU victim
//     picked less often), handed back to Python, or read zero-copy: pointer = its RAM slot, touch its RAM LRU;
//   anything else (in flight, NVMe-only, write-back pending): handed back to Python's slow path.
// Python keeps the actions that need torch or futures (reads, copies, write-backs) and the rare paths.
class Core {
public:
    Core(int64_t L, int64_t E, uintptr_t key_cls, uintptr_t vcls, uintptr_t rcls, uintptr_t vram_of, uintptr_t ram_of,
         uintptr_t ptrs_h, uintptr_t vaddr, uintptr_t ram_addr, uintptr_t freq, uintptr_t hits, uintptr_t flags,
         std::vector<Lru*> vram_lru, std::vector<Lru*> ram_lru, double admit_min, int admit_per_call)
        : L_(L), E_(E), key_cls_(p<int64_t>(key_cls)), vcls_(p<int64_t>(vcls)), rcls_(p<int64_t>(rcls)),
          vram_of_(p<int64_t>(vram_of)), ram_of_(p<int64_t>(ram_of)), ptrs_h_(p<int64_t>(ptrs_h)),
          vaddr_(p<int64_t>(vaddr)), ram_addr_(p<int64_t>(ram_addr)), freq_(p<float>(freq)), hits_(p<double>(hits)),
          flags_(p<uint8_t>(flags)), vram_lru_(std::move(vram_lru)), ram_lru_(std::move(ram_lru)),
          admit_min_((float)admit_min), admit_per_call_(admit_per_call) {}

    // returns (changed, slow [(e, k)], admit [(e, k)], vram_hits, zero_copy, ptr_fix, distinct picks)
    // decay: this call is a frequency-decay tick (freq halves after this call's increments, before its decisions,
    // as the Python path). Admission decisions are sequential: an admission without a free slot consumes the first
    // unprotected VRAM LRU entry (which _take_vram_slot evicts), so the next candidate is compared with the next one.
    py::tuple ensure_fast(int64_t li, py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                          std::vector<int64_t> nfree_vram, bool zero_copy, bool decay) {
        if (li < 0 || li >= L_) throw py::index_error("ensure_fast: layer out of range");
        std::vector<int64_t> u(ids.data(), ids.data() + ids.size());
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        int64_t base = li * E_;
        for (int64_t e : u)
            if (e < 0 || e >= E_) throw py::index_error("ensure_fast: expert id out of range");
        // protect: this call's keys (never an eviction victim)
        auto is_protected = [&](int64_t k) { return k >= base && k < base + E_ && std::binary_search(u.begin(), u.end(), k - base); };
        bool changed = false;
        int64_t n_vh = 0, n_zc = 0, n_fix = 0;
        int n_adm = 0;
        py::list slow, admit;
        int64_t* row = ptrs_h_ + base;
        for (int64_t e : u) {
            int64_t k = base + e;
            freq_[k] += 1.0f;
            hits_[k] += 1.0;
        }
        if (decay)
            for (int64_t i = 0, n = L_ * E_; i < n; ++i) freq_[i] *= 0.5f;
        std::vector<int64_t> consumed;            // VRAM victims already claimed by this call's admissions
        for (int64_t e : u) {
            int64_t k = base + e;
            int64_t s = vram_of_[k];
            if (s >= 0) {
                vram_lru_[vcls_[s]]->move_to_end(k);
                ++n_vh;
                if (row[e] != vaddr_[s]) { row[e] = vaddr_[s]; changed = true; ++n_fix; }
                continue;
            }
            int64_t rs = ram_of_[k];
            if (rs < 0 || (flags_[k] & 3)) { slow.append(py::make_tuple(e, k)); continue; }   // NVMe / in flight / wb
            int64_t c = key_cls_[k];
            if (zero_copy) {
                bool adm = false;
                if (n_adm < admit_per_call_ && freq_[k] >= admit_min_) {
                    if (c < (int64_t)nfree_vram.size() && nfree_vram[c] > 0) {
                        adm = true;
                    } else {
                        Lru* lru = vram_lru_[c];
                        for (int32_t v = lru->head(); v >= 0; v = lru->next_of(v))
                            if (!is_protected(v) && std::find(consumed.begin(), consumed.end(), v) == consumed.end()) {
                                adm = freq_[v] < freq_[k];
                                if (adm) consumed.push_back(v);
                                break;
                            }
                    }
                }
                if (!adm) {
                    ram_lru_[c]->move_to_end(k);
                    int64_t addr = ram_addr_[rs];
                    if (row[e] != addr) { row[e] = addr; changed = true; }
                    ++n_zc;
                    continue;
                }
                ++n_adm;
                if (c < (int64_t)nfree_vram.size() && nfree_vram[c] > 0) --nfree_vram[c];
            }
            admit.append(py::make_tuple(e, k));
        }
        return py::make_tuple(changed, slow, admit, n_vh, n_zc, n_fix, (int64_t)u.size());
    }

    // After a layer's routing sync (block_graphs._forward): picks = the layer's picks buffer (picks_layout: sel Q*k |
    // exact Q*k | predicted next layer Q*w1 | ...). Counts the router's substitutions, credits the passed-over exact
    // picks in freq (admission follows what the router wanted), and filters prefetch candidates down to experts in
    // neither VRAM nor RAM nor in flight. Returns (next-layer experts to prefetch, this layer's passed-over experts to
    // prefetch (the first miss_pf per row, all if miss_pf < 0), substituted, picks).
    py::tuple after_sync(int64_t li, int64_t nxt_li, py::array_t<int64_t, py::array::c_style> picks, int64_t Q,
                         int64_t k, int64_t w1, int64_t miss_pf) {
        if (li < 0 || li >= L_ || nxt_li >= L_) throw py::index_error("after_sync: layer out of range");
        if ((int64_t)picks.size() < 2 * Q * k + (nxt_li >= 0 ? Q * w1 : 0))
            throw py::value_error("after_sync: picks buffer too small");
        const int64_t* pk = picks.data();
        const int64_t *sel = pk, *ex = pk + Q * k, *pn = pk + 2 * Q * k;
        int64_t base = li * E_;
        int64_t n_sub = 0;
        std::vector<int64_t> miss;
        for (int64_t t = 0; t < Q; ++t) {
            int64_t taken = 0;
            for (int64_t j = 0; j < k; ++j) {
                int64_t e = ex[t * k + j];
                if (e < 0 || e >= E_) throw py::index_error("after_sync: expert id out of range");
                bool chosen = false;
                for (int64_t i = 0; i < k; ++i) chosen |= sel[t * k + i] == e;
                if (chosen) continue;
                ++n_sub;
                freq_[base + e] += 1.0f;
                if (miss_pf < 0 || taken < miss_pf) { miss.push_back(e); ++taken; }
            }
        }
        auto want = [&](int64_t key) { return vram_of_[key] < 0 && ram_of_[key] < 0 && !(flags_[key] & 1); };
        py::list pf_next, pf_this;
        if (nxt_li >= 0) {
            std::vector<int64_t> v(pn, pn + Q * w1);
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            for (int64_t e : v) {
                if (e < 0 || e >= E_) throw py::index_error("after_sync: predicted expert id out of range");
                if (want(nxt_li * E_ + e)) pf_next.append(e);
            }
        }
        std::sort(miss.begin(), miss.end());
        miss.erase(std::unique(miss.begin(), miss.end()), miss.end());
        for (int64_t e : miss)
            if (want(base + e)) pf_this.append(e);
        return py::make_tuple(pf_next, pf_this, n_sub, Q * k);
    }

private:
    template <typename T> static T* p(uintptr_t a) { return reinterpret_cast<T*>(a); }
    int64_t L_, E_;
    const int64_t *key_cls_, *vcls_, *rcls_, *vram_of_, *ram_of_;
    int64_t* ptrs_h_;
    const int64_t *vaddr_, *ram_addr_;
    float* freq_;
    double* hits_;
    const uint8_t* flags_;
    std::vector<Lru*> vram_lru_, ram_lru_;
    float admit_min_;
    int admit_per_call_;
};

}  // namespace

PYBIND11_MODULE(tier_native, m) {
    py::class_<Lru>(m, "Lru")
        .def(py::init<int64_t>())
        .def("move_to_end", &Lru::move_to_end)
        .def("__setitem__", &Lru::set)
        .def("__getitem__", &Lru::get)
        .def("__contains__", &Lru::contains)
        .def("__len__", &Lru::size)
        .def("__bool__", [](const Lru& l) { return l.size() > 0; })
        .def("pop", &Lru::pop)
        .def("pop", &Lru::pop_default)
        .def("clear", &Lru::clear)
        .def("__iter__", [](const Lru& l) { return py::make_iterator(l.begin(), l.end()); }, py::keep_alive<0, 1>());
    py::class_<Core>(m, "Core")
        .def(py::init<int64_t, int64_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                      uintptr_t, uintptr_t, uintptr_t, uintptr_t, std::vector<Lru*>, std::vector<Lru*>, double, int>(),
             py::keep_alive<1, 15>(), py::keep_alive<1, 16>())
        .def("ensure_fast", &Core::ensure_fast, py::arg("li"), py::arg("ids"), py::arg("nfree_vram"),
             py::arg("zero_copy") = true, py::arg("decay") = false)
        .def("after_sync", &Core::after_sync, py::arg("li"), py::arg("nxt_li"), py::arg("picks"), py::arg("Q"),
             py::arg("k"), py::arg("w1"), py::arg("miss_pf"));
    m.doc() = "native O_DIRECT expert reader for glm_xpu.tier";
    py::class_<Reader>(m, "Reader")
        .def(py::init<int, int, int64_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, double, double, bool>(),
             py::arg("fd"), py::arg("threads"), py::arg("max_piece"), py::arg("res"), py::arg("bonus"),
             py::arg("vram_of"), py::arg("ram_of"), py::arg("dirty"), py::arg("bonus_vram"), py::arg("bonus_ram"),
             py::arg("bonus_on"))
        .def("submit", &Reader::submit, py::arg("key"), py::arg("file_off"), py::arg("n"), py::arg("dst"),
             py::arg("pieces"), py::arg("res_slot") = -1, py::arg("urgent") = false)
        .def("promote", &Reader::promote)
        .def("done", &Reader::done)
        .def("wait", &Reader::wait)
        .def("forget", &Reader::forget)
        .def("pending", &Reader::pending)
        .def("stats", &Reader::stats)
        .def("stop", &Reader::stop, py::call_guard<py::gil_scoped_release>());
}
