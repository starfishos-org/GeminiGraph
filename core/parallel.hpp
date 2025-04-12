#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <unistd.h>
#include <vector>

class ThreadPool {
public:
  ThreadPool(uint32_t thread_count) : stop(false) {
    workers.reserve(thread_count);
    for (uint32_t i = 0; i < thread_count; ++i) {
      pthread_t thread;
      pthread_create(&thread, nullptr, ThreadPool::worker_thread, this);
      workers.push_back(thread);
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (pthread_t &worker : workers) {
      pthread_join(worker, nullptr);
    }
  }

  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      tasks.push(std::move(task));
    }
    condition.notify_one();
  }

private:
  static void *worker_thread(void *arg) {
    auto *pool = static_cast<ThreadPool *>(arg);
    thread_id = pool->thread_id_allocator++;
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(pool->queue_mutex);
        pool->condition.wait(
            lock, [pool] { return pool->stop || !pool->tasks.empty(); });
        if (pool->stop && pool->tasks.empty())
          return nullptr;
        task = std::move(pool->tasks.front());
        pool->tasks.pop();
      }
      task();
    }
  }

  std::vector<pthread_t> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop;

  std::atomic<uint32_t> thread_id_allocator{0};

public:
  static thread_local uint32_t thread_id;
};

class Parallel {
  static uint32_t thread_count;
  static ThreadPool thread_pool;

  static const uint64_t min_chunk_size;

public:
  template <typename Func> static void Invoke(Func func, uint32_t threads) {
    std::atomic<int> remaining_tasks(threads);
    std::mutex completion_mutex;
    std::condition_variable completion_condition;

    for (uint32_t i = 0; i < threads; ++i) {
      thread_pool.enqueue([=, &func, &remaining_tasks, &completion_mutex,
                           &completion_condition] {
        func(i);
        if (--remaining_tasks == 0) {
          std::lock_guard<std::mutex> lock(completion_mutex);
          completion_condition.notify_one();
        }
      });
    }

    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_condition.wait(
        lock, [&remaining_tasks] { return remaining_tasks == 0; });
  }

  template <typename Func>
  static void For(Func func, uint64_t start, uint64_t end, uint64_t incr = 1) {
    // Adjust chunk size to account for 'incr'
    uint64_t chunk_size = std::max(
        ((end - start + incr - 1) / incr + thread_count - 1) / thread_count,
        min_chunk_size);

    uint64_t total_chunks =
        ((end - start + incr - 1) / incr + chunk_size - 1) / chunk_size;

    std::atomic<int> remaining_tasks(total_chunks);
    std::mutex completion_mutex;
    std::condition_variable completion_condition;

    for (uint32_t i = 0; i < total_chunks; ++i) {
      uint64_t per_thread_start = start + i * chunk_size * incr;
      uint64_t per_thread_end =
          std::min(per_thread_start + chunk_size * incr, end);

      thread_pool.enqueue([=, &func, &remaining_tasks, &completion_mutex,
                           &completion_condition] {
        for (uint64_t j = per_thread_start; j < per_thread_end; j += incr) {
          func(j);
        }
        if (--remaining_tasks == 0) {
          std::lock_guard<std::mutex> lock(completion_mutex);
          completion_condition.notify_one();
        }
      });

      if (per_thread_end == end) {
        break;
      }
    }

    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_condition.wait(
        lock, [&remaining_tasks] { return remaining_tasks == 0; });
  }

  template <typename ReduceFunc, typename T>
  static T Reduce(ReduceFunc rfunc, T init, uint64_t start, uint64_t end) {
    uint64_t chunk_size = std::max(
        (end - start + thread_count - 1) / thread_count, min_chunk_size);

    std::vector<T> results(thread_count, init);
    std::atomic<int> remaining_tasks(thread_count);
    std::mutex completion_mutex;
    std::condition_variable completion_condition;

    for (uint32_t i = 0; i < thread_count; ++i) {
      uint64_t per_thread_start = start + i * chunk_size;
      uint64_t per_thread_end = std::min(per_thread_start + chunk_size, end);

      thread_pool.enqueue([=, &rfunc, &results, &remaining_tasks,
                           &completion_mutex, &completion_condition] {
        for (uint64_t j = per_thread_start; j < per_thread_end; ++j) {
          rfunc(results[i], j);
        }
        if (--remaining_tasks == 0) {
          std::lock_guard<std::mutex> lock(completion_mutex);
          completion_condition.notify_one();
        }
      });

      if (per_thread_end == end) {
        break;
      }
    }

    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_condition.wait(
        lock, [&remaining_tasks] { return remaining_tasks == 0; });

    T result = init;
    for (auto &res : results) {
      result = rfunc(result, res);
    }

    return result;
  }

  template <typename Func, typename ReduceFunc, typename T>
  static T Reduce(Func func, ReduceFunc rfunc, T init, uint32_t threads) {
    std::vector<T> results(threads, init);
    std::atomic<int> remaining_tasks(threads);
    std::mutex completion_mutex;
    std::condition_variable completion_condition;

    for (uint32_t t_i = 0; t_i < threads; t_i++) {
      thread_pool.enqueue([&func, &results, t_i, &remaining_tasks,
                           &completion_mutex, &completion_condition]() {
        results[t_i] = func(t_i);
        if (--remaining_tasks == 0) {
          std::lock_guard<std::mutex> lock(completion_mutex);
          completion_condition.notify_one();
        }
      });
    }

    std::unique_lock<std::mutex> lock(completion_mutex);
    completion_condition.wait(
        lock, [&remaining_tasks] { return remaining_tasks == 0; });

    T reducer = init;
    for (uint32_t t_i = 0; t_i < threads; t_i++) {
      rfunc(reducer, results[t_i]);
    }
    return reducer;
  }
};
