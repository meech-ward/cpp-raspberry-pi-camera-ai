module;
#include <array>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
export module CircularFrameBuffer;

template<typename T>
struct FrameData {
    std::shared_ptr<T> data;
    std::chrono::system_clock::time_point timestamp;
};

export template<typename T, size_t BUFFER_SIZE = 2>
class CircularFrameBuffer {
private:
    std::array<FrameData<T>, BUFFER_SIZE> buffer_;
    std::atomic<size_t> write_index_{0};
    mutable std::mutex mutex_;

public:
    void push(T&& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t current_write_index = write_index_.load(std::memory_order_relaxed);
        size_t next_index = (current_write_index + 1) % BUFFER_SIZE;
        buffer_[next_index] = FrameData<T>{
            std::make_shared<T>(std::move(data)),
            std::chrono::system_clock::now()
        };
        write_index_.store(next_index, std::memory_order_release);
    }

    std::shared_ptr<T> getLatest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t current_index = write_index_.load(std::memory_order_acquire);
        return buffer_[current_index].data;
    }

    bool hasData() const {
        return write_index_.load(std::memory_order_acquire) != 0 || buffer_[0].data != nullptr;
    }
};