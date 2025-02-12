#include <libfork/core.hpp>
#include <mutex>
#include <deque>
#include <optional>
#include <iostream>

template <typename T, lf::scheduler Scheduler>
class FixedCapacityQueue {
public:
    explicit FixedCapacityQueue(Scheduler & scheduler_, size_t capacity_)
        : scheduler(scheduler_), capacity(capacity_) {}

    lf::task<void> push(T value) {
        std::unique_lock lock(mutex);

        while (buffer.size() >= capacity) {
            co_await wait(lock, not_full);
        }

        buffer.push_back(std::move(value));
        notify(not_empty);
    }

    lf::task<std::optional<T>> pop() {
        std::unique_lock lock(mutex);

        while (buffer.empty() && !closed) {
            co_await wait(lock, not_empty);
        }

        if (buffer.empty()) {
            co_return std::nullopt;
        }

        T value = std::move(buffer.front());
        buffer.pop_front();
        notify(not_full);
        co_return value;
    }

    void close() {
        std::lock_guard lock(mutex);
        closed = true;
        notify_all();
    }

private:
    struct Notification {
        std::vector<std::coroutine_handle<>> waiters;
    };

    lf::task<> wait(std::unique_lock<std::mutex> & lock, Notification & note) {
        struct Awaiter {
            bool await_ready() const noexcept { return false; }

            void await_suspend(std::coroutine_handle<> h) {
                note.waiters.push_back(h);
                lock.unlock();
            }

            void await_resume() { lock.lock(); }

            Notification & note;
            std::unique_lock<std::mutex> & lock;
        };
        co_await Awaiter{note, lock};
    }

    void notify(Notification & note) {
        if (!note.waiters.empty()) {
            auto h = note.waiters.back();
            note.waiters.pop_back();
            scheduler.schedule(h);
        }
    }

    void notify_all() {
        for (auto* note : {&not_full, &not_empty}) {
            while (!note->waiters.empty()) {
                auto h = note->waiters.back();
                note->waiters.pop_back();
                scheduler.schedule(h);
            }
        }
    }

    Scheduler & scheduler;
    std::mutex mutex;
    std::deque<T> buffer;
    const size_t capacity;
    bool closed = false;
    Notification not_full;
    Notification not_empty;
};
