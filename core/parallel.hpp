#pragma once
#include "filesystem.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <unistd.h>
#include <vector>
#include <memory>

// 添加宏定义，用于控制 printf 的开关
#define ENABLE_DEBUG_PRINT 0

#if ENABLE_DEBUG_PRINT
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#define RING_BUFFER_SIZE 4096

class ThreadPool {
public:
    ThreadPool()
            : stop(false)
            , thread_id_allocator(0)
            , thread_count(0)
            , task_queues(nullptr)
    {
    }

    ThreadPool(uint32_t thread_count)
            : stop(false)
            , thread_id_allocator(0)
            , thread_count(thread_count)
            , task_queues(new RingBuffer[thread_count])
    {
        for (uint32_t i = 0; i < thread_count; ++i) {
            task_queues[i].init();
        }

        workers.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; ++i) {
            pthread_t thread;
            pthread_create(&thread, nullptr, ThreadPool::worker_thread, this);
            workers.push_back(thread);
        }
    }

    ~ThreadPool()
    {
        destroy();
    }

    void destroy()
    {
        stop = true;
        for (pthread_t &worker : workers) {
            pthread_join(worker, nullptr);
        }
        thread_id_allocator = 0;
        workers.clear();
        delete[] task_queues;
        task_queues = nullptr;
        stop = false;
    }

    void set_thread_count(uint32_t thread_count)
    {
        destroy();
        this->thread_count = thread_count;
        task_queues = new RingBuffer[thread_count];
        for (uint32_t i = 0; i < thread_count; ++i) {
            task_queues[i].init();
        }

        workers.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; ++i) {
            pthread_t thread;
            pthread_create(&thread, nullptr, ThreadPool::worker_thread, this);
            workers.push_back(thread);
        }
    }

    void enqueue(std::function<void()> task)
    {
        static std::atomic<uint32_t> queue_index(0);
        uint32_t index = queue_index.fetch_add(1) % thread_count;
        while (!task_queues[index].push(std::move(task))) {
            // 自旋等待
        }
    }

    void enqueue_to_thread(std::function<void()> task, uint32_t thread_id)
    {
        while (!task_queues[thread_id].push(std::move(task))) {
            // 自旋等待
        }
    }

    void bind_cpu(uint32_t cpu_id)
    {
#ifdef OS_CHCORE
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(cpu_id, &cpu_set);
        sched_setaffinity(-2, sizeof(cpu_set), &cpu_set);
        sched_yield();
        printf("ChCore thread %d bound to CPU\n", cpu_id);
#else
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(cpu_id, &cpu_set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        sched_yield();
#endif
    }

private:
    struct RingBuffer {
        std::function<void()> buffer[RING_BUFFER_SIZE];
        std::atomic<uint32_t> head;
        std::atomic<uint32_t> tail;

        void init()
        {
            head.store(0);
            tail.store(0);
        }

        bool push(std::function<void()> task)
        {
            uint32_t current_tail = tail.load(std::memory_order_relaxed);
            uint32_t next_tail = (current_tail + 1) % RING_BUFFER_SIZE;
            if (next_tail == head.load(std::memory_order_acquire)) {
                return false; // 缓冲区已满
            }
            buffer[current_tail] = std::move(task);
            tail.store(next_tail, std::memory_order_release);
            return true;
        }

        bool pop(std::function<void()> &task)
        {
            uint32_t current_head = head.load(std::memory_order_relaxed);
            if (current_head == tail.load(std::memory_order_acquire)) {
                return false; // 缓冲区为空
            }
            task = std::move(buffer[current_head]);
            head.store((current_head + 1) % RING_BUFFER_SIZE,
                       std::memory_order_release);
            return true;
        }
    };

    static void *worker_thread(void *arg)
    {
        auto *pool = static_cast<ThreadPool *>(arg);
        uint32_t thread_id = pool->thread_id_allocator++;
        pool->bind_cpu(thread_id);
        while (!pool->stop) {
            std::function<void()> task;
            if (pool->task_queues[thread_id].pop(task)) {
                task();
            } else {
                // 自旋等待
            }
        }
        return nullptr;
    }

    std::vector<pthread_t> workers;
    uint32_t thread_count;
    RingBuffer *task_queues;
    std::atomic<bool> stop;
    std::atomic<uint32_t> thread_id_allocator;

public:
    static thread_local uint32_t thread_id;
};

class Parallel {
    static inline unsigned long rdtsc()
    {
        unsigned long hi, lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return lo | (hi << 32);
    }
    static uint32_t thread_count;
    static ThreadPool thread_pool;

    static const uint64_t min_chunk_size;

public:
    static void SetThreadCount(uint32_t threads)
    {
        thread_count = threads;
        thread_pool.set_thread_count(threads);
        DEBUG_PRINT("SetThreadCount called with threads: %u\n", threads);
    }

    template <typename Func> static void Invoke(Func func, uint32_t threads)
    {
        std::atomic<uint32_t> remaining_tasks(threads);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;
        DEBUG_PRINT("Entering invoke, time: %lu\n", rdtsc());

        for (uint32_t i = 0; i < threads; ++i) {
            thread_pool.enqueue([=,
                                 &func,
                                 &remaining_tasks,
                                 &completion_mutex,
                                 &completion_condition] {
                func(i);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            });
        }

        DEBUG_PRINT("Inserted invoke, time: %lu\n", rdtsc());
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks] {
            return remaining_tasks == 0;
        });
        DEBUG_PRINT("Finished invoke, time: %lu\n", rdtsc());
    }

    template <typename Func>
    static void For(Func func, uint64_t start, uint64_t end, uint64_t incr = 1,
                    int64_t user_defined_chunk_size = -1)
    {
        DEBUG_PRINT("Entering for, time: %lu\n", rdtsc());
        uint64_t chunk_size;
        // Adjust chunk size to account for 'incr'
        if (user_defined_chunk_size == -1) {
            chunk_size =
                    std::max(((end - start + incr - 1) / incr + thread_count
                              - 1) / thread_count,
                             min_chunk_size);
        } else {
            chunk_size = user_defined_chunk_size;
        }

        uint64_t total_chunks =
                ((end - start + incr - 1) / incr + chunk_size - 1) / chunk_size;

        std::atomic<uint32_t> remaining_tasks(total_chunks);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;

        for (uint32_t i = 0; i < total_chunks; ++i) {
            uint64_t per_thread_start = start + i * chunk_size * incr;
            uint64_t per_thread_end =
                    std::min(per_thread_start + chunk_size * incr, end);

            thread_pool.enqueue([=,
                                 &func,
                                 &remaining_tasks,
                                 &completion_mutex,
                                 &completion_condition] {
                for (uint64_t j = per_thread_start; j < per_thread_end;
                     j += incr) {
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
        DEBUG_PRINT("Inserted for tasks, time: %lu\n", rdtsc());

        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks] {
            return remaining_tasks == 0;
        });
        DEBUG_PRINT("Finished for tasks, time: %lu\n", rdtsc());
    }

    template <typename ReduceFunc, typename T>
    static T Reduce(ReduceFunc rfunc, T init, uint64_t start, uint64_t end)
    {
        DEBUG_PRINT("Entering reduce, time: %lu\n", rdtsc());

        uint64_t chunk_size =
                std::max((end - start + thread_count - 1) / thread_count,
                         min_chunk_size);

        std::vector<T> results(thread_count, init);
        std::atomic<int> remaining_tasks(thread_count);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;

        for (uint32_t i = 0; i < thread_count; ++i) {
            uint64_t per_thread_start = start + i * chunk_size;
            uint64_t per_thread_end =
                    std::min(per_thread_start + chunk_size, end);

            thread_pool.enqueue([=,
                                 &rfunc,
                                 &results,
                                 &remaining_tasks,
                                 &completion_mutex,
                                 &completion_condition] {
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
        DEBUG_PRINT("Inserted reduce, time: %lu\n", rdtsc());

        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks] {
            return remaining_tasks == 0;
        });

        T result = init;
        for (auto &res : results) {
            result = rfunc(result, res);
        }
        DEBUG_PRINT("Finished reduce, time: %lu\n", rdtsc());

        return result;
    }

    template <typename Func, typename ReduceFunc, typename T>
    static T Reduce(Func func, ReduceFunc rfunc, T init, uint32_t threads)
    {
        DEBUG_PRINT("Entering reduce, time: %lu\n", rdtsc());

        std::vector<T> results(threads, init);
        std::atomic<int> remaining_tasks(threads);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;

        for (uint32_t t_i = 0; t_i < threads; t_i++) {
            thread_pool.enqueue([&func,
                                 &results,
                                 t_i,
                                 &remaining_tasks,
                                 &completion_mutex,
                                 &completion_condition]() {
                results[t_i] = func(t_i);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            });
        }

        DEBUG_PRINT("Inserted reduce, time: %lu\n", rdtsc());

        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks] {
            return remaining_tasks == 0;
        });

        T reducer = init;
        for (uint32_t t_i = 0; t_i < threads; t_i++) {
            rfunc(reducer, results[t_i]);
        }
        DEBUG_PRINT("Finished reduce, time: %lu\n", rdtsc());

        return reducer;
    }
};
