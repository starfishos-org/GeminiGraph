#pragma once
#include "filesystem.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <ctype.h>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <memory>
#ifdef OS_CHCORE
#include <chcore/syscall.h>
#endif

// 添加宏定义，用于控制 printf 的开关
#define ENABLE_DEBUG_PRINT 0

/* Parse bind_cpu file (e.g. "0-7,12-19" -> [0,1,...,7,12,...,19]) */
static int parse_bind_cpu_file(const char *filename, std::vector<int> &out_list) {
  FILE *f = fopen(filename, "r");
  if (!f)
    return -1;
  char buf[1024];
  if (!fgets(buf, sizeof(buf), f)) {
    fclose(f);
    return -1;
  }
  fclose(f);
  char *p = buf;
  while (*p) {
    while (*p && isspace((unsigned char)*p))
      p++;
    if (!*p)
      break;
    long start = strtol(p, &p, 10);
    long end_num = start;
    if (*p == '-') {
      p++;
      end_num = strtol(p, &p, 10);
    }
    if (end_num < start)
      end_num = start;
    for (long cpu = start; cpu <= end_num; ++cpu)
      out_list.push_back((int)cpu);
    while (*p && (*p == ',' || isspace((unsigned char)*p)))
      p++;
  }
  return out_list.empty() ? -1 : 0;
}

#if ENABLE_DEBUG_PRINT
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

#define RING_BUFFER_SIZE 4096

class ThreadPool {
public:
    /* CPU list from gemini_bind_cpu.txt: e.g. 0-7,12-19 -> [0..7,12..19] */
    static std::vector<int> bind_cpu_list;
    /* First CPU of each segment: e.g. [0,12] for machines 0 and 1 */
    static std::vector<int> loader_cpu_list;

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

    void bind_cpu(uint32_t thread_id)
    {
        int cpu_id = (int)thread_id;
        if (!bind_cpu_list.empty() && thread_id < bind_cpu_list.size())
            cpu_id = bind_cpu_list[thread_id];
#ifdef OS_CHCORE
        usys_set_affinity(-2, cpu_id);
        usys_yield();
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

#ifdef OS_CHCORE
class MachineThreadPool {
public:
    MachineThreadPool(int machine_id_, uint32_t threads_this_machine,
                     uint32_t global_thread_start)
        : machine_id(machine_id_), stop(false) {
        workers.reserve(threads_this_machine);
        for (uint32_t i = 0; i < threads_this_machine; ++i) {
            pthread_t thread;
            struct CreateArg {
                MachineThreadPool *pool;
                uint32_t local_idx;
                uint32_t global_start;
            } *ca = new CreateArg{this, i, global_thread_start};
            pthread_create(&thread, nullptr, MachineThreadPool::worker_thread, ca);
            workers.push_back(thread);
        }
    }
    ~MachineThreadPool() { destroy(); }
    void destroy() {
        { std::lock_guard<std::mutex> lock(queue_mutex); stop = true; }
        condition.notify_all();
        for (pthread_t &w : workers)
            pthread_join(w, nullptr);
    }
    void enqueue(std::function<void()> task) {
        { std::lock_guard<std::mutex> lock(queue_mutex); tasks.push(std::move(task)); }
        condition.notify_one();
    }
private:
    static void *worker_thread(void *arg) {
        struct CreateArg { MachineThreadPool *pool; uint32_t local_idx; uint32_t global_start; } *ca =
            static_cast<CreateArg *>(arg);
        MachineThreadPool *pool = ca->pool;
        uint32_t local_idx = ca->local_idx;
        uint32_t global_thread_id = ca->global_start + local_idx;
        delete ca;
        if (!ThreadPool::bind_cpu_list.empty() && global_thread_id < ThreadPool::bind_cpu_list.size()) {
            int cpu = ThreadPool::bind_cpu_list[global_thread_id];
            usys_set_affinity(-2, cpu);
            fprintf(stderr, "[%s:%d] bind_cpu: %d\n", __FILE__, __LINE__, cpu);
            usys_yield();
        }
        ThreadPool::thread_id = global_thread_id;
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(pool->queue_mutex);
                pool->condition.wait(lock, [pool] { return pool->stop || !pool->tasks.empty(); });
                if (pool->stop && pool->tasks.empty())
                    return nullptr;
                task = std::move(pool->tasks.front());
                pool->tasks.pop();
            }
            task();
        }
    }
    int machine_id;
    std::vector<pthread_t> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};
#endif

class Parallel {
    static inline unsigned long rdtsc()
    {
        unsigned long hi, lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return lo | (hi << 32);
    }
    static uint32_t thread_count;
    static ThreadPool thread_pool;
#ifdef OS_CHCORE
    static std::vector<std::unique_ptr<MachineThreadPool>> machine_pools;
    static std::atomic<uint32_t> next_enqueue_machine;
#endif
    static const uint64_t min_chunk_size;

    static void enqueue_impl(std::function<void()> task) {
#ifdef OS_CHCORE
        if (!machine_pools.empty()) {
            uint32_t m = next_enqueue_machine.fetch_add(1) % (uint32_t)machine_pools.size();
            machine_pools[m]->enqueue(std::move(task));
            return;
        }
#endif
        thread_pool.enqueue(std::move(task));
    }
    static void enqueue_to_machine_impl(uint32_t machine_id, std::function<void()> task) {
#ifdef OS_CHCORE
        if (!machine_pools.empty() && machine_id < machine_pools.size()) {
            machine_pools[machine_id]->enqueue(std::move(task));
            return;
        }
#endif
        thread_pool.enqueue(std::move(task));
    }

public:
    static uint32_t get_num_machines() {
#ifdef OS_CHCORE
        if (!ThreadPool::loader_cpu_list.empty())
            return (uint32_t)ThreadPool::loader_cpu_list.size();
#endif
        return 1;
    }
    static uint32_t get_threads_per_socket() {
        uint32_t num_machines = get_num_machines();
        return (num_machines > 0 && thread_count >= num_machines)
            ? (thread_count / num_machines) : thread_count;
    }
    /* Load bind_cpu file before SetThreadCount. Returns 0 on success. */
    static int LoadBindCpuFile(const char *filename) {
        ThreadPool::bind_cpu_list.clear();
        ThreadPool::loader_cpu_list.clear();
        int ret = parse_bind_cpu_file(filename, ThreadPool::bind_cpu_list);
        if (ret == 0 && !ThreadPool::bind_cpu_list.empty()) {
            int prev = ThreadPool::bind_cpu_list[0];
            ThreadPool::loader_cpu_list.push_back(prev);
            for (size_t i = 1; i < ThreadPool::bind_cpu_list.size(); ++i) {
                int cur = ThreadPool::bind_cpu_list[i];
                if (cur != prev + 1)
                    ThreadPool::loader_cpu_list.push_back(cur);
                prev = cur;
            }
        }
        return ret;
    }
#ifdef OS_CHCORE
    static void LimitMachines(uint32_t max_machines) {
        if (max_machines == 0 || ThreadPool::loader_cpu_list.empty())
            return;
        if (max_machines >= ThreadPool::loader_cpu_list.size())
            return;
        int cutoff_cpu = ThreadPool::loader_cpu_list[max_machines];
        auto it = std::lower_bound(ThreadPool::bind_cpu_list.begin(),
                                  ThreadPool::bind_cpu_list.end(), cutoff_cpu);
        ThreadPool::bind_cpu_list.erase(it, ThreadPool::bind_cpu_list.end());
        ThreadPool::loader_cpu_list.resize(max_machines);
    }
#endif
    static void SetThreadCount(uint32_t threads)
    {
        thread_count = threads;
#ifdef OS_CHCORE
        if (!ThreadPool::bind_cpu_list.empty() && !ThreadPool::loader_cpu_list.empty()) {
            uint32_t num_machines = (uint32_t)ThreadPool::loader_cpu_list.size();
            uint32_t tps = (num_machines > 0 && threads >= num_machines)
                ? (threads / num_machines) : threads;
            machine_pools.clear();
            next_enqueue_machine.store(0);
            for (uint32_t m = 0; m < num_machines; ++m) {
                uint32_t global_start = m * tps;
                uint32_t count = (m == num_machines - 1) ? (threads - global_start) : tps;
                machine_pools.push_back(std::unique_ptr<MachineThreadPool>(
                    new MachineThreadPool((int)m, count, global_start)));
            }
            DEBUG_PRINT("SetThreadCount: %u threads, %u machines, %u threads/machine\n",
                        threads, num_machines, tps);
            return;
        }
#endif
        thread_pool.set_thread_count(threads);
        DEBUG_PRINT("SetThreadCount called with threads: %u\n", threads);
    }

    /* Run func(machine_id) on each machine; tasks execute on machine-local threads (NUMA-aware). */
    template <typename Func> static void InvokePerMachine(Func func)
    {
        uint32_t num_machines = get_num_machines();
        if (num_machines == 0) return;
        std::atomic<uint32_t> remaining_tasks(num_machines);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;
        for (uint32_t m = 0; m < num_machines; ++m) {
            auto task = [=, &func, &remaining_tasks, &completion_mutex, &completion_condition] {
                func(m);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            };
            enqueue_to_machine_impl(m, std::move(task));
        }
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks] { return remaining_tasks == 0; });
    }

    /* InvokeOnMachine: enqueue `count` tasks to the given machine; task j runs
     * func(start_thread_id + j). Blocks until all complete. E.g. run 8 threads
     * on machine 0: InvokeOnMachine(0, func, 0, 8). */
    template <typename Func>
    static void InvokeOnMachine(uint32_t machine_id, Func func,
                                uint32_t start_thread_id, uint32_t count) {
        if (count == 0) return;
        std::atomic<uint32_t> remaining_tasks(count);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;
        uint32_t num_machines = get_num_machines();
        if (num_machines == 0) num_machines = 1;
        if (machine_id >= num_machines) machine_id = num_machines - 1;
        for (uint32_t j = 0; j < count; j++) {
            uint32_t tid = start_thread_id + j;
            enqueue_to_machine_impl(machine_id,
                [&func, &remaining_tasks, &completion_mutex, &completion_condition, tid]() {
                    func(tid);
                    if (--remaining_tasks == 0) {
                        std::lock_guard<std::mutex> lock(completion_mutex);
                        completion_condition.notify_one();
                    }
                });
        }
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_condition.wait(lock, [&remaining_tasks, count]() {
            return remaining_tasks == 0;
        });
    }

    template <typename Func> static void Invoke(Func func, uint32_t threads)
    {
        std::atomic<uint32_t> remaining_tasks(threads);
        std::mutex completion_mutex;
        std::condition_variable completion_condition;
        DEBUG_PRINT("Entering invoke, time: %lu\n", rdtsc());

        uint32_t num_machines = get_num_machines();
        uint32_t tps = get_threads_per_socket();
        for (uint32_t i = 0; i < threads; ++i) {
            uint32_t machine_id = (num_machines > 0 && tps > 0) ? (i / tps) : 0;
            if (machine_id >= num_machines)
                machine_id = num_machines > 0 ? num_machines - 1 : 0;
            auto task = [=, &func, &remaining_tasks, &completion_mutex, &completion_condition] {
                func(i);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            };
            enqueue_to_machine_impl(machine_id, std::move(task));
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
                    int64_t user_defined_chunk_size = -1,
                    int fixed_machine_id = -1)
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

        uint32_t num_machines = get_num_machines();
        uint32_t tps_for = get_threads_per_socket();
        for (uint32_t i = 0; i < total_chunks; ++i) {
            uint64_t per_thread_start = start + i * chunk_size * incr;
            uint64_t per_thread_end =
                    std::min(per_thread_start + chunk_size * incr, end);

            uint32_t machine_id;
            if (fixed_machine_id >= 0 && (uint32_t)fixed_machine_id < num_machines) {
                machine_id = (uint32_t)fixed_machine_id;
            } else {
                machine_id = (num_machines > 0 && tps_for > 0)
                    ? (static_cast<uint32_t>(i) / tps_for) : 0;
                if (machine_id >= num_machines && num_machines > 0)
                    machine_id = num_machines - 1;
            }

            auto task = [=, &func, &remaining_tasks, &completion_mutex, &completion_condition] {
                for (uint64_t j = per_thread_start; j < per_thread_end; j += incr)
                    func(j);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            };
            enqueue_to_machine_impl(machine_id, std::move(task));

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

        uint32_t num_machines = get_num_machines();
        uint32_t tps = get_threads_per_socket();
        for (uint32_t i = 0; i < thread_count; ++i) {
            uint64_t per_thread_start = start + i * chunk_size;
            uint64_t per_thread_end =
                    std::min(per_thread_start + chunk_size, end);

            uint32_t machine_id = (num_machines > 0 && tps > 0) ? (i / tps) : 0;
            if (machine_id >= num_machines)
                machine_id = num_machines > 0 ? num_machines - 1 : 0;

            auto task = [=, &rfunc, &results, &remaining_tasks, &completion_mutex, &completion_condition] {
                for (uint64_t j = per_thread_start; j < per_thread_end; ++j)
                    rfunc(results[i], j);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            };
            enqueue_to_machine_impl(machine_id, std::move(task));

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

        uint32_t num_machines = get_num_machines();
        uint32_t tps = get_threads_per_socket();
        for (uint32_t t_i = 0; t_i < threads; t_i++) {
            uint32_t machine_id = (num_machines > 0 && tps > 0) ? (t_i / tps) : 0;
            if (machine_id >= num_machines)
                machine_id = num_machines > 0 ? num_machines - 1 : 0;
            auto task = [&func, &results, t_i, &remaining_tasks, &completion_mutex, &completion_condition]() {
                results[t_i] = func(t_i);
                if (--remaining_tasks == 0) {
                    std::lock_guard<std::mutex> lock(completion_mutex);
                    completion_condition.notify_one();
                }
            };
            enqueue_to_machine_impl(machine_id, std::move(task));
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
