#include "rt.h"
#include <cstring>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <cstring>

namespace glm {

namespace prop = sycl::ext::oneapi::property::queue;

// A queue on the GPU's copy engine. Under the Level Zero v2 adapter every SYCL queue (copies included) runs on the
// compute engine, so a copy beside kernels stalls them; an immediate command list on a copy-only queue group, wrapped
// as a SYCL queue, does not (Strata, sycl/src/core/copy_queue.cpp). Opt-in (GLM_COPY_ENGINE=1): with it on, every
// process exit logged an xe page fault on the B70 (2026-11-15, 5 in an hour; the hand-made command list's teardown),
// and nothing measured faster with it yet.
static sycl::queue copy_engine_queue(const sycl::context& ctx, const sycl::device& dev) {
    sycl::queue plain{ctx, dev, {sycl::property::queue::in_order{}}};
    if (!(getenv("GLM_COPY_ENGINE") && atoi(getenv("GLM_COPY_ENGINE")) == 1) ||
        dev.get_backend() != sycl::backend::ext_oneapi_level_zero)
        return plain;
    try {
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
        auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
        uint32_t ng = 0;
        if (zeDeviceGetCommandQueueGroupProperties(zdev, &ng, nullptr) != ZE_RESULT_SUCCESS || !ng) return plain;
        std::vector<ze_command_queue_group_properties_t> g(ng);
        for (auto& x : g) { memset(&x, 0, sizeof x); x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES; }
        if (zeDeviceGetCommandQueueGroupProperties(zdev, &ng, g.data()) != ZE_RESULT_SUCCESS) return plain;
        int ord = -1;
        for (uint32_t i = 0; i < ng; i++)
            if ((g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
                !(g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && g[i].numQueues > 0) { ord = (int)i; break; }
        if (ord < 0) return plain;
        ze_command_queue_desc_t qd;
        memset(&qd, 0, sizeof qd);
        qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        qd.ordinal = (uint32_t)ord;
        qd.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
        qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        ze_command_list_handle_t cl = nullptr;
        if (zeCommandListCreateImmediate(zctx, zdev, &qd, &cl) != ZE_RESULT_SUCCESS || !cl) return plain;
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> in(
            cl, dev, sycl::ext::oneapi::level_zero::ownership::keep, sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue cq = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(in, ctx);
        fprintf(stderr, "copy queue: the GPU's copy engine (queue group %d)\n", ord);
        return cq;
    } catch (const std::exception& e) {
        fprintf(stderr, "note: no copy-engine queue (%s)\n", e.what());
        return plain;
    }
}

Dev& Dev::get() {
    static Dev* d = [] {
        auto* d = new Dev;
        d->dev = sycl::device{sycl::gpu_selector_v};
        d->ctx = sycl::context{d->dev};
        // normal priority: a high-priority queue starves every lower one, copies included (tools/prio.cpp)
        // GLM_PROF=2: GPU timestamps on the main queue (phase boundaries as barriers)
        if (getenv("GLM_PROF") && atoi(getenv("GLM_PROF")) >= 2)   // 2: phases, 3: experts, 5: named points
            d->q = sycl::queue{d->ctx, d->dev, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}};
        else
            d->q = sycl::queue{d->ctx, d->dev, {sycl::property::queue::in_order{}}};
        d->low = sycl::queue{d->ctx, d->dev, {sycl::property::queue::in_order{}, prop::priority_low{}}};
        d->copy = copy_engine_queue(d->ctx, d->dev);
        return d;
    }();
    return *d;
}

Tensor Tensor::empty(std::vector<int64_t> shape, DType dt) {
    Tensor t;
    t.shape = std::move(shape);
    t.dt = dt;
    size_t b = t.bytes();
    auto& d = Dev::get();
    t.p = b ? sycl::malloc_device(b, d.q) : nullptr;
    GLM_CHECK(!b || t.p, "device allocation of %zu bytes failed (%.2f GiB in use)", b, d.vram_used / 1073741824.0);
    d.vram_used += b;
    t.own = true;
    return t;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dt) {
    Tensor t = empty(std::move(shape), dt);
    if (t.bytes()) Dev::get().q.memset(t.p, 0, t.bytes()).wait();
    return t;
}

void Tensor::release() {
    if (own && p) {
        auto& d = Dev::get();
        d.q.wait();
        sycl::free(p, d.q);
        d.vram_used -= bytes();
    }
    p = nullptr;
    own = false;
}

void Workspace::init(size_t bytes) {
    auto& d = Dev::get();
    base = static_cast<char*>(sycl::malloc_device(bytes, d.q));
    GLM_CHECK(base, "workspace allocation of %zu bytes failed", bytes);
    d.vram_used += bytes;
    cap = bytes;
    off = 0;
}

bool g_prof_points = getenv("GLM_PROF") && atoi(getenv("GLM_PROF")) == 5;
static std::vector<std::pair<const char*, sycl::event>> g_pp;
static std::map<std::string, double> g_pp_sum;

bool g_sync_points = false;

void prof_point(const char* name) {
    if (g_sync_points) {                // wait at every point and print it (where the GPU stops, when it hangs)
        Dev::get().q.wait();
        fprintf(stderr, "point: %s\n", name);
    }
    if (!g_prof_points) return;
    g_pp.push_back({name, Dev::get().q.ext_oneapi_submit_barrier()});
    if (g_pp.size() > 4096) prof_points_report(0);   // fold in, keep the list short
}

std::string prof_points_report(int tokens) {
    Dev::get().q.wait();
    for (size_t i = 1; i < g_pp.size(); i++) {
        uint64_t a = g_pp[i - 1].second.get_profiling_info<sycl::info::event_profiling::command_end>();
        uint64_t b = g_pp[i].second.get_profiling_info<sycl::info::event_profiling::command_end>();
        g_pp_sum[g_pp[i].first] += (b - a) * 1e-6;
    }
    if (!g_pp.empty()) g_pp.erase(g_pp.begin(), g_pp.end() - 1);
    if (tokens < 0) { g_pp_sum.clear(); return ""; }   // drop everything so far
    if (!tokens) return "";
    std::vector<std::pair<double, std::string>> v;
    for (auto& [k, ms] : g_pp_sum) v.push_back({ms, k});
    std::sort(v.rbegin(), v.rend());
    std::string out;
    char b[160];
    for (auto& [ms, k] : v) {
        snprintf(b, sizeof b, "  %-28s %7.3f ms/token\n", k.c_str(), ms / tokens);
        out += b;
    }
    g_pp_sum.clear();
    g_pp.clear();
    return out;
}

// Through a pinned staging buffer, in pieces: src is often a mapped model file, and the copy engine reading a page that is
// not in memory faults the card (xe "Fault response: Unsuccessful -ENOENT", class bcs, on a cold page cache). Here the
// CPU takes the page faults.
void upload(void* dst, const void* src, size_t bytes) {
    constexpr size_t CH = (size_t)32 << 20;
    auto& q = Dev::get().q;
    static thread_local char* stage = nullptr;
    if (!stage) {
        stage = static_cast<char*>(sycl::malloc_host(CH, q));
        GLM_CHECK(stage, "upload: pinned staging buffer");
    }
    for (size_t o = 0; o < bytes; o += CH) {
        size_t m = std::min(CH, bytes - o);
        memcpy(stage, static_cast<const char*>(src) + o, m);
        q.memcpy(static_cast<char*>(dst) + o, stage, m).wait();
    }
}
void download(void* dst, const void* src, size_t bytes) { Dev::get().q.memcpy(dst, src, bytes).wait(); }

}  // namespace glm
