// exl3conv: wall-clock timer (host only; rt.hpp re-exports it for SYCL code).
#pragma once
#include <chrono>
namespace rt {
struct Timer
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double s() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
} // namespace rt
