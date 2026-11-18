// exl3conv: a device (USM) buffer that frees itself.
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace exl3 {

// What this process holds on the card through DBufs and VitWork. With a budget set, an allocation past it throws: past
// the card's real room the driver may not return null, and an out-of-memory can be a Level Zero segfault or a fault
// (notes/toolchain-and-reuse.md §6.1).
struct Vram
{
    inline static size_t used = 0, peak = 0, budget = 0;
    static void take(size_t b)
    {
        if (budget && used + b > budget)
            throw std::runtime_error("device allocation of " + std::to_string(b >> 20) + " MiB would pass the budget: " +
                                     std::to_string(used >> 20) + " of " + std::to_string(budget >> 20) + " MiB in use");
        used += b;
        peak = std::max(peak, used);
    }
    static void give(size_t b) { used -= b; }
};

// Device buffer with an owner.
template <typename T>
struct DBuf
{
    sycl::queue* q = nullptr; T* p = nullptr; size_t n = 0;
    DBuf() = default;
    DBuf(sycl::queue& q_, size_t n_) : q(&q_), n(n_)
    {
        Vram::take(n * sizeof(T));
        p = sycl::malloc_device<T>(n, q_);
        if (!p && n) Vram::give(n * sizeof(T));
        if (!p && n) throw std::runtime_error("device allocation of " + std::to_string(n * sizeof(T) >> 20) + " MiB failed");
    }
    DBuf(DBuf&& o) noexcept : q(o.q), p(o.p), n(o.n) { o.p = nullptr; }
    DBuf& operator=(DBuf&& o) noexcept { reset(); q = o.q; p = o.p; n = o.n; o.p = nullptr; return *this; }
    ~DBuf() { reset(); }
    void reset() { if (p) { sycl::free(p, *q); Vram::give(n * sizeof(T)); } p = nullptr; }
    T* get() const { return p; }
    void upload(const T* h, size_t count) { q->memcpy(p, h, count * sizeof(T)).wait(); }
    void download(T* h, size_t count) const { q->memcpy(h, p, count * sizeof(T)).wait(); }
    void zero() { q->memset(p, 0, n * sizeof(T)).wait(); }
};

} // namespace exl3
