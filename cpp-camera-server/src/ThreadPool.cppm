module;

#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <functional>

export module ThreadPool;

export class ThreadPool {
 public:
  ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queueMutex);
            this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }
          task();
        }
      });
    }
  }

  template <class F>
  void enqueue(F &&f) {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) {
      worker.join();
    }
  }

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queueMutex;
  std::condition_variable condition;
  std::atomic<bool> stop;
};