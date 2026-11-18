#include "progress.h"
#include <unistd.h>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include "rt.h"

namespace glm {

namespace {

using Clock = std::chrono::steady_clock;
const bool g_tty = isatty(2);
bool g_running = false, g_pending = false;      // inside a load's progress; g_last is not in the log yet
char g_last[256];
Clock::time_point g_line;                       // when the log last got a line of this progress

}  // namespace

void load_progress(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last, sizeof g_last, fmt, ap);
    va_end(ap);
    if (g_tty) {
        fprintf(stderr, "\r%s", g_last);
        return;
    }
    const auto now = Clock::now();
    if (!g_running) {
        g_running = true;
        g_line = now;
    }
    g_pending = now - g_line < std::chrono::seconds(10);
    if (!g_pending) {
        fprintf(stderr, "%s\n", g_last);
        g_line = now;
    }
}

void load_progress_end(const char* fmt, ...) {
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(g_last, sizeof g_last, fmt, ap);
        va_end(ap);
        fprintf(stderr, "%s%s\n", g_tty ? "\r" : "", g_last);
    } else if (g_tty) {
        fprintf(stderr, "\n");
    } else if (g_pending) {
        fprintf(stderr, "%s\n", g_last);
    }
    g_running = g_pending = false;
}

void ReadMarks::restart(const void* who, double base) {
    std::lock_guard<std::mutex> g(mu_);
    who_ = who;
    base_ = base;
    queued_ = ran_ = 0;
    marks_.clear();
    armed_ = true;
}

void ReadMarks::mark(double tokens) {
    if (!armed_.load(std::memory_order_relaxed)) return;
    // on an in-order queue: done when everything submitted before it is
    sycl::event e = Dev::get().q.ext_oneapi_submit_barrier();
    std::lock_guard<std::mutex> g(mu_);
    queued_ += tokens;
    marks_.emplace_back(std::move(e), queued_);
    if (marks_.size() > 8192) {                 // nobody asks: keep the list short
        ran_ = marks_.front().second;
        marks_.pop_front();
    }
}

double ReadMarks::done(const void* who) {
    std::lock_guard<std::mutex> g(mu_);
    if (who != who_) return -1;
    try {
        while (!marks_.empty() && marks_.front().first.get_info<sycl::info::event::command_execution_status>() ==
                                      sycl::info::event_command_status::complete) {
            ran_ = marks_.front().second;
            marks_.pop_front();
        }
    } catch (const sycl::exception&) {
        // (an event that cannot say: the count stays where it was until the engine's next word)
    }
    return base_ + ran_;
}

ReadMarks& read_marks() {
    static ReadMarks m;
    return m;
}

}  // namespace glm
