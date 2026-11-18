#pragma once
// What a log or a watcher is told while the engine works: a model load's progress line, and how far a prompt's read is.
#include <atomic>
#include <deque>
#include <mutex>
#include <utility>
#include <sycl/sycl.hpp>

namespace glm {

// A load's progress ("loaded layer 12/60 ..."), called again for each step. On a terminal the line rewrites itself. In
// a log file a step is a line of its own at most every 10 s: the file stays short, and a slow load still shows it is
// alive (the client gives up on a start whose log is silent for 10 minutes).
void load_progress(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// The end of that progress: fmt, when given, is its last line; else the last step is.
void load_progress_end(const char* fmt = nullptr, ...) __attribute__((format(printf, 1, 2)));

// How far the prompt being read is, for a watcher on another thread (glm_server's progress lines). prefill() takes a
// pass of thousands of rows through every layer and queues its kernels ahead of the card, so neither its return nor the
// host's place in its loop says where the card is. A model marks each layer it has queued; a mark counts once the card
// has run everything queued before it. Nothing is marked until a watcher has called restart() (the gen tools never do).
class ReadMarks {
public:
    // engine thread: `who` reads its prompt now, with `base` tokens of it read. Earlier marks are dropped.
    void restart(const void* who, double base);
    // prefill: `tokens` more of the prompt are read once the work queued so far is done (rows / layers, after a layer)
    void mark(double tokens);
    // any thread: tokens of who's prompt read so far; -1 when who is not the one reading
    double done(const void* who);

private:
    std::atomic<bool> armed_{false};
    std::mutex mu_;
    const void* who_ = nullptr;
    double base_ = 0, queued_ = 0, ran_ = 0;            // tokens: at restart, marked since, and of those run
    std::deque<std::pair<sycl::event, double>> marks_;  // (everything before this has run, queued_ then)
};
ReadMarks& read_marks();

}  // namespace glm
