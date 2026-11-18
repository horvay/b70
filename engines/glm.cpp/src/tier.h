#pragma once
// The routed-expert tier: VRAM slot pools, a pinned RAM tier the kernels read in place over PCIe (zero-copy), and the
// NVMe store behind them. A port of llm/glm/xpu/glm_xpu/tier.py (decode path) with its native reader / LRU / decision
// core (csrc/tier_native.cpp):
//   - residency is exclusive: an expert is in VRAM or in RAM, never both;
//   - ensure() after a layer's routing sync makes the layer's picks readable: VRAM hits, zero-copy RAM hits, NVMe
//     reads (urgent) into RAM slots, and admissions RAM -> VRAM by decayed pick frequency (at most admit_per_call per
//     layer; the evicted VRAM expert is written back to RAM through a device ring and the copy queue);
//   - begin_demand() starts this layer's NVMe reads before the next layer's prefetches are queued;
//   - prefetch() reads predicted picks of the next layer into RAM slots in the background;
//   - the router's residency bonus (bv for VRAM, br for RAM) lives in bonus_dev, kept by the reader threads too.
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "rt.h"

namespace glm {

class Lru {   // keys 0..n-1 linked through flat arrays; oldest first
public:
    explicit Lru(int64_t n) : prev_(n, -1), next_(n, -1), val_(n, 0), in_(n, 0) {}
    bool contains(int64_t k) const { return in_[k]; }
    size_t size() const { return n_; }
    int32_t head() const { return head_; }
    int32_t next_of(int32_t k) const { return next_[k]; }
    int64_t value(int64_t k) const { return val_[k]; }
    void touch(int64_t k) { if (tail_ != k) { unlink(k); link_back(k); } }
    void set(int64_t k, int64_t v) { val_[k] = v; if (!in_[k]) link_back(k); }
    int64_t pop(int64_t k) { unlink(k); return val_[k]; }

private:
    void unlink(int64_t k);
    void link_back(int64_t k);
    std::vector<int32_t> prev_, next_;
    std::vector<int64_t> val_;
    std::vector<uint8_t> in_;
    int32_t head_ = -1, tail_ = -1;
    size_t n_ = 0;
};

// O_DIRECT record reads on worker threads: page-aligned pieces, a thread-local aligned bounce, memcpy to the
// destination (pinned USM host memory refuses O_DIRECT). Urgent jobs go ahead of queued speculative ones, and 4 threads
// stay free for them. A background job (res_slot >= 0) marks its expert landed when every piece is in and the expert
// still owns that RAM slot.
class Reader {
public:
    struct Hooks {                    // residency publication (atomic word stores)
        const int64_t* ram_of;
        const int64_t* vram_of;
        uint8_t* landed;
        float* bonus;
        uint8_t* dirty;
        float bv, br;
    };
    // direct: every destination is O_DIRECT-capable (memfd) memory, read in place without a bounce buffer
    Reader(int fd, int threads, int max_spec, int64_t max_piece, Hooks h, bool direct);
    ~Reader();
    int64_t submit(int64_t key, int64_t off, int64_t n, void* dst, int pieces, int64_t res_slot, bool urgent);
    void promote(int64_t id);
    bool done(int64_t id);
    void wait(int64_t id);            // throws on a read error; forgets the job
    void forget(int64_t id);
    double t_pread() const { return ns_pread_.load() * 1e-9; }

private:
    struct Job { int64_t key, res_slot; std::atomic<int> left{0}, err{0}; std::atomic<bool> fin{false}; };
    struct Piece { std::shared_ptr<Job> job; int64_t off, n; uint8_t* dst; };
    void run();
    void finish(Job& j);
    int fd_;
    int64_t max_piece_;
    Hooks h_;
    bool direct_ = false;
    std::mutex mu_, mu_done_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<Piece> queue_;
    std::unordered_map<int64_t, std::shared_ptr<Job>> jobs_;
    size_t n_urgent_ = 0;             // pieces at the front of queue_ that belong to urgent jobs
    int spec_running_ = 0, max_spec_ = 1;
    int64_t next_id_ = 1;
    bool stopping_ = false;
    std::vector<std::thread> workers_;
    std::atomic<int64_t> ns_pread_{0};
};

struct TierConfig {
    size_t vram_bytes = 0, ram_bytes = 14ull << 30;
    int readers = 16, pieces = 8;                // reader threads; pieces per speculative read
    int urgent_pieces = 16;                      // pieces per demand read: 16 cut the NVMe wait 21.7 -> 13.9 ms/token
    bool memfd = true;                           // RAM tier / prompt ring as memfd pages (O_DIRECT in place, no bounce)
    int max_spec = 12;                           // threads speculative pieces may occupy (the rest wait for urgent)
    float bonus_vram = 0.1f, bonus_ram = 0.1f;   // router residency bonus (0 = exact routing)
    float admit_min = 4.0f;                      // decayed pick count for VRAM admission
    int admit_per_call = 2;
    int decay_every = 42 * 32;                   // halve the pick counts every ~32 tokens
    bool mtp = false;                            // a small pool for the MTP layer's class
    size_t ring_bytes = (size_t)4608 << 20;      // pinned read-ahead ring for prompts
};

// Host memory the GPU reads in place and NVMe reads land in directly: memfd pages, mapped here, exported in windows of
// at most 64 MB (udmabuf's size limit; a whole number of records each) as dma-bufs and imported into Level Zero as
// allocations over those same pages. (Pinned USM host memory refuses O_DIRECT, so it needs a bounce buffer and a CPU
// copy per read, which doubles host memory traffic beside the GPU's zero-copy reads.)
class HostWindows {
public:
    // n records of rec bytes; false (and nothing allocated) when udmabuf or the import is unavailable
    bool init(size_t n, size_t rec);
    ~HostWindows();
    char* host(size_t i) const { return host_ + (i / per_) * win_ + (i % per_) * rec_; }
    char* dev(size_t i) const { return dev_[i / per_] + (i % per_) * rec_; }
    size_t bytes() const { return dev_.size() * win_; }

private:
    int mfd_ = -1;
    char* host_ = nullptr;
    size_t rec_ = 0, per_ = 0, win_ = 0, mapped_ = 0;
    std::vector<char*> dev_;
};

class Tier {
public:
    Tier(const std::string& store_bin, const TierConfig& cfg);
    ~Tier();
    int E = 0, L = 0, n_moe = 0;
    int kc_of(int li) const { return lay_[li].kc; }
    const int64_t* ptrs(int li) const { return ptrs_dev_.ptr<int64_t>() + (size_t)li * E; }
    const float* bonus(int li) const { return bonus_dev_.ptr<float>() + (size_t)li * E; }
    bool bonus_on() const { return cfg_.bonus_vram > 0 || cfg_.bonus_ram > 0; }

    // fill VRAM, then RAM, with (li, e) in order (hottest first); waits
    void warm(const std::vector<std::pair<int, int>>& order);
    // upload the bonus table if it changed (on q, right before a router that reads it)
    void sync_bonus(sycl::queue& q);
    // after layer li's routing sync: sel / exact [Q*k] picks (exact = the router's wanted picks), pred [Q*w1] for the
    // next MoE layer nli (-1: none). begin_demand + substitution bookkeeping + prefetch; then ensure() before the
    // expert kernels.
    void after_sync(int li, int nli, const int64_t* sel, const int64_t* exact, int nsel, const int64_t* pred, int npred);
    // early (decode): called once before waiting on NVMe reads, with the experts still being read (their pointer rows
    // are set after it returns); every other pick's record is readable from the queue's current point on
    void ensure(sycl::queue& q, int li, const int64_t* sel, int n,
                const std::function<void(const std::vector<int64_t>& late)>& early = {});

    // Prompts (prefill, exact routing, many rows per MoE call): begin_prompt borrows staging slots from the main VRAM
    // pool (their experts are dropped; admissions refill them after the prompt) and starts reading ahead every expert of
    // the next layers that is neither in VRAM nor in RAM into a pinned ring. stage(li, experts) then gives layer li's
    // call its own pointer row: VRAM experts in place, RAM and NVMe ones copied into staging slots (the ring's reads
    // waited for). end_prompt returns the slots.
    // lend(bytes), before begin_prompt: device memory for a prompt's per-row buffers, a run of contiguous slots at the
    // end of the staging class's pool (their experts dropped too), returned by end_prompt. Outside the pool there is
    // only the reserve, which decode's graphs mostly fill (2026-11-02: a 6.9K-token prompt's buffers beside them
    // overcommitted VRAM and the GPU hung).
    void* lend(size_t bytes);
    // dbl (long prompts, where every layer uses nearly all its experts): a second set of E staging slots, so layer
    // li's non-VRAM experts go into set li % 2 on the copy queue ahead of its routing, while the layer before computes
    // (prestage). Falls back to one set when the pool cannot spare 2E slots.
    void begin_prompt(int first_li, bool dbl = false);
    // dbl only, never blocks: copy layer li's experts whose bytes are ready (RAM tier, landed reads) into its set;
    // again as more reads land. stage() copies the rest, waiting for their reads, and puts a barrier on q
    void prestage(int li);
    // dbl only: q has every kernel of layer li's MoE; li's set is free again once they have run
    void moe_done(sycl::queue& q, int li);
    const int64_t* stage(sycl::queue& q, int li, const std::vector<int>& experts);
    void end_prompt();
    // the read-ahead ring (memfd, 4.5 GiB pinned; ~3 s to touch): at load, not on the first prompt
    void prepare_prompts();
    bool in_prompt() const { return prompt_; }

    struct Stats {
        size_t calls = 0, picks = 0, vram_hit = 0, ram_hit = 0, zero_copy = 0, nvme = 0, prefetch = 0, prefetch_used = 0,
               admit = 0, writeback = 0, wb_dropped = 0, substituted = 0,
               early = 0;                 // layers whose ready experts ran before an NVMe wait
        double t_nvme = 0;
        size_t staged_vram = 0, staged_ram = 0, staged_nvme = 0, readahead_used = 0;
        double t_stage_wait = 0;
        double t_host = 0;                 // after_sync + ensure wall time on the decode thread (NVMe waits included)
    } stats;
    std::string summary() const;
    void reset_stats() { stats = Stats{}; }

private:
    struct LayerInfo { size_t off, rec; int kc, cls; };
    TierConfig cfg_;
    std::vector<LayerInfo> lay_;
    int fd_ = -1, C_ = 0;
    std::vector<size_t> crec_;
    // per key (li * E + e)
    std::vector<int> key_cls_;
    std::vector<int64_t> vram_of_, ram_of_;
    std::vector<uint8_t> flags_;          // 1 read in flight, 2 write-back pending
    std::vector<uint8_t> landed_;         // RAM bytes valid (set by the reader for background reads)
    std::vector<float> freq_;
    std::vector<double> hits_;
    float* bonus_h_ = nullptr;            // pinned [L * E]
    uint8_t dirty_ = 1;
    // VRAM slots (global ids; class c's slots are contiguous), RAM slots
    std::vector<char*> vbase_;
    std::vector<int64_t> vaddr_, vram_key_;
    std::vector<int> vcls_;
    std::vector<std::vector<int>> free_vram_;
    std::vector<std::unique_ptr<Lru>> vram_lru_, ram_lru_;
    std::vector<void*> ram_chunks_;                    // pinned USM chunks (fallback without memfd)
    std::vector<std::unique_ptr<HostWindows>> ram_win_;   // memfd RAM tier, one per class
    std::vector<int64_t> ram_addr_, ram_key_;           // ram_addr_: the GPU's address of a slot
    std::vector<char*> ram_host_;                       // the CPU's address of a slot (reads land here)
    std::vector<int> rcls_;
    std::vector<std::vector<int>> free_ram_;
    // reads and write-backs
    std::unique_ptr<Reader> rd_;
    bool direct_ = false;                               // RAM tier and ring are memfd windows
    std::unordered_map<int64_t, int64_t> inflight_;   // key -> reader job
    std::unordered_set<int64_t> demand_;
    struct Wb { sycl::event ev; int ring; };
    std::unordered_map<int64_t, Wb> wb_;
    std::deque<int64_t> wb_order_;                    // keys in write-back order (the copy queue runs in order)
    char* vring_ = nullptr;
    std::vector<int> vring_free_;
    // pointer tables
    int64_t* ptrs_h_ = nullptr;           // pinned [L * E]
    Tensor ptrs_dev_, bonus_dev_;
    size_t tick_ = 0;
    // prompt staging
    bool prompt_ = false;
    int stage_cls_ = 0;                                // the largest main-model record class (set at start)
    std::vector<int> stage_slots_;                     // borrowed VRAM slots (dbl: set 0, slot of expert e at e)
    bool dbl_ = false;
    std::vector<int> stage2_slots_;                    // dbl: set 1
    int pre_li_[2] = {-1, -1};                         // dbl: the layer each set holds
    std::vector<uint8_t> pre_done_[2];                 // dbl: expert e's bytes copied (or under way) into the set
    sycl::event set_free_[2], last_copy_;              // dbl: the set's last layer's kernels ran; the copy queue's last
    bool copied_ = false;                              // dbl: a copy since the last stage() barrier
    std::vector<int> lent_;                            // slots lent for prompt buffers
    Tensor sptrs_dev_;                                 // int64 [L, E]: the staged calls' pointer rows
    int64_t* sptrs_h_ = nullptr;                       // pinned mirror
    std::vector<char*> ring_;                          // read-ahead slots (rec of the staging class): CPU address
    std::vector<char*> ring_dev_;                      // ... and the GPU's
    std::unique_ptr<HostWindows> ring_win_;
    struct RingSlot { int64_t key = -1; int64_t job = -1; sycl::event ev; bool busy = false; };
    std::vector<RingSlot> rs_;
    std::unordered_map<int64_t, int> ring_of_;         // key -> ring slot (read planned or landed)
    int plan_li_ = 0, plan_e_ = 0;                     // read-ahead cursor
    void plan_reads();
    int free_ring_slot(bool wait);

    void set_bonus(int64_t k);
    void ram_put(int64_t k, int rs);
    int take_ram_slot(const std::vector<int64_t>& protect, int c, bool wait);
    int take_vram_slot(sycl::queue& q, const std::vector<int64_t>& protect, int c);
    void write_back(sycl::queue& q, int64_t k, int s, const std::vector<int64_t>& protect);
    void reap_wb();
    void reap_inflight();
    int64_t submit_bg(int64_t k, int rs, bool urgent);
    void prefetch(int li, const std::vector<int64_t>& es, const std::vector<int64_t>& protect);
    bool want_admit(int64_t k, int c, int n_adm, const std::vector<int64_t>& protect, std::vector<int64_t>* consumed);
};

}  // namespace glm
