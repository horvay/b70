#pragma once
// Runtime: the device, its in-order queues, device/host buffers and a per-queue workspace for kernel temporaries.
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace glm {

using half = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define GLM_CHECK(c, ...)                                                                           \
    do {                                                                                            \
        if (!(c)) {                                                                                 \
            char _m[512];                                                                           \
            snprintf(_m, sizeof _m, __VA_ARGS__);                                                   \
            fprintf(stderr, "error: %s:%d: %s\n", __FILE__, __LINE__, _m);                          \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + _m); \
        }                                                                                           \
    } while (0)

enum class DType : uint8_t { F16, BF16, F32, I16, I32, I64, U8 };
inline size_t dsize(DType t) {
    switch (t) {
        case DType::F16: case DType::BF16: case DType::I16: return 2;
        case DType::F32: case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::U8: return 1;
    }
    return 0;
}

// The B70 and its queues: main (normal priority), low (background), copy.
struct Dev {
    sycl::device dev;
    sycl::context ctx;
    sycl::queue q;      // main (decode/prefill), in order
    sycl::queue low;    // background, in order, low priority (gets nothing while q is busy: tools/prio.cpp)
    sycl::queue copy;   // copies beside the main queue (expert write-backs), normal priority
    static Dev& get();
    size_t vram_used = 0;
};

// Device memory with a type and a shape (row major); owns its allocation unless made with view().
struct Tensor {
    void* p = nullptr;
    DType dt = DType::F16;
    std::vector<int64_t> shape;
    bool own = false;

    Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& o) noexcept { *this = std::move(o); }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            release();
            p = o.p; dt = o.dt; shape = std::move(o.shape); own = o.own;
            o.p = nullptr; o.own = false;
        }
        return *this;
    }
    ~Tensor() { release(); }

    static Tensor empty(std::vector<int64_t> shape, DType dt);
    static Tensor zeros(std::vector<int64_t> shape, DType dt);
    static Tensor view(void* p, std::vector<int64_t> shape, DType dt) {
        Tensor t; t.p = p; t.shape = std::move(shape); t.dt = dt; return t;
    }
    Tensor alias() const { return view(p, shape, dt); }
    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
    size_t bytes() const { return numel() * dsize(dt); }
    int64_t size(int i) const { return shape.at(i < 0 ? shape.size() + i : i); }
    template <class T> T* ptr() const { return reinterpret_cast<T*>(p); }
    void release();
};

// Bump allocator for one queue's temporaries. An in-order queue runs ops one after another, so an op may reset it
// on entry and reuse the space: no earlier op is still reading it.
struct Workspace {
    char* base = nullptr;
    size_t cap = 0, off = 0;
    void init(size_t bytes);
    void reset() { off = 0; }
    template <class T> T* take(size_t n) {
        size_t b = (n * sizeof(T) + 255) & ~size_t(255);
        GLM_CHECK(off + b <= cap, "workspace: %zu + %zu > %zu bytes", off, b, cap);
        T* r = reinterpret_cast<T*>(base + off);
        off += b;
        return r;
    }
};

// GLM_PROF=5: named GPU timing points (queue barriers on the profiling queue); the time between consecutive points is
// charged to the later point's name. prof_points_report() sums them and clears.
void prof_point(const char* name);
std::string prof_points_report(int tokens);
extern bool g_prof_points;
extern bool g_sync_points;                  // prof_point waits and prints (GLM_PREFILL_TRACE=3)

void upload(void* dst, const void* src, size_t bytes);           // host -> device, waits
void download(void* dst, const void* src, size_t bytes);         // device -> host, waits

}  // namespace glm
