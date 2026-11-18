// exl3conv: SYCL runtime helpers. One in-order queue on the first Level Zero GPU (the B70; the NVIDIA card has no
// Level Zero driver). Creating the queue is what opens the card, so only call make_queue() inside a queue turn.
#pragma once

#include "rt/timer.hpp"

#include <sycl/sycl.hpp>

#include "rt/card.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace rt {

inline sycl::device pick_device()
{
    // Single device allocations above 4 GB (the calibration state is 11 GB) need this before the first query.
    setenv("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1", 0);
    for (auto& p : sycl::platform::get_platforms())
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero)
            for (auto& d : p.get_devices())
                if (d.is_gpu()) return d;
    throw std::runtime_error("no Level Zero GPU found (is ONEAPI_DEVICE_SELECTOR hiding it?)");
}

inline sycl::queue make_queue()
{
    auto dev = pick_device();
    return sycl::queue(dev, sycl::property::queue::in_order());
}

inline void print_device(const sycl::device& d)
{
    std::printf("device: %s\n", d.get_info<sycl::info::device::name>().c_str());
    std::printf("  driver %s, compute units %u, max clock %u MHz\n", d.get_info<sycl::info::device::driver_version>().c_str(),
                d.get_info<sycl::info::device::max_compute_units>(), d.get_info<sycl::info::device::max_clock_frequency>());
    std::printf("  global mem %.1f GiB, max alloc %.1f GiB, local mem %zu KiB\n",
                d.get_info<sycl::info::device::global_mem_size>() / 1073741824.0,
                d.get_info<sycl::info::device::max_mem_alloc_size>() / 1073741824.0,
                (size_t) d.get_info<sycl::info::device::local_mem_size>() / 1024);
    std::printf("  max work-group %zu, sub-group sizes:", d.get_info<sycl::info::device::max_work_group_size>());
    for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", s);
    std::printf("\n  fp16 %s, fp64 %s\n", d.has(sycl::aspect::fp16) ? "yes" : "no", d.has(sycl::aspect::fp64) ? "yes" : "no");
}

} // namespace rt
