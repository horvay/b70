// exl3conv: the card's VRAM and the host's memory as the kernel reports them. No SYCL, no GPU libraries, nothing allocated on the card.
#pragma once

#include <drm/xe_drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace rt {

// The card's VRAM in bytes, all processes, from the xe memory-region query (as queue/src/card.ts): no GPU libraries and
// nothing allocated. {0, 0} when no render node answers.
struct CardVram { uint64_t used = 0, total = 0; };

inline CardVram card_vram()
{
    // Only the xe node: this machine's other render node is NVIDIA's, which must never see an xe ioctl.
    static int fd = [] {
        namespace fs = std::filesystem;
        std::vector<std::string> nodes;
        for (const auto& e : fs::directory_iterator("/sys/class/drm"))
            if (e.path().filename().string().rfind("renderD", 0) == 0) nodes.push_back(e.path().filename().string());
        std::sort(nodes.begin(), nodes.end());
        for (const auto& n : nodes)
        {
            std::error_code ec;
            const auto drv = fs::canonical("/sys/class/drm/" + n + "/device/driver", ec);
            if (!ec && drv.filename() == "xe") return open(("/dev/dri/" + n).c_str(), O_RDWR | O_CLOEXEC);
        }
        return -1;
    }();
    CardVram v;
    if (fd < 0) return v;
    drm_xe_device_query qy{};
    qy.query = DRM_XE_DEVICE_QUERY_MEM_REGIONS;
    if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &qy) != 0) return v;
    std::vector<uint8_t> buf(qy.size);
    qy.data = (uint64_t) (uintptr_t) buf.data();
    if (ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &qy) != 0) return v;
    const auto* r = (const drm_xe_query_mem_regions*) buf.data();
    for (uint32_t i = 0; i < r->num_mem_regions; ++i)
        if (r->mem_regions[i].mem_class == DRM_XE_MEM_REGION_CLASS_VRAM) { v.used += r->mem_regions[i].used; v.total += r->mem_regions[i].total_size; }
    return v;
}

// Samples card_vram() in the background; take() returns the most VRAM seen since the last take(), in bytes.
class VramWatch
{
public:
    VramWatch() : th_([this] { while (!stop_) { seen(); std::this_thread::sleep_for(std::chrono::milliseconds(100)); } }) {}
    ~VramWatch() { stop_ = true; th_.join(); }
    uint64_t take() { seen(); return peak_.exchange(0); }
private:
    void seen()
    {
        const uint64_t u = card_vram().used;
        uint64_t p = peak_.load();
        while (u > p && !peak_.compare_exchange_weak(p, u)) {}
    }
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> peak_{0};
    std::thread th_;
};

// MemAvailable from /proc/meminfo, in bytes.
inline uint64_t host_mem_available()
{
    std::ifstream f("/proc/meminfo");
    std::string k; uint64_t v; std::string unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return v * 1024;
    return 0;
}

} // namespace rt
