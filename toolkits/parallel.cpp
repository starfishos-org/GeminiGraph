#include "core/parallel.hpp"
#include <thread>

uint32_t thread_local ThreadPool::thread_id;

std::vector<int> ThreadPool::bind_cpu_list;
std::vector<int> ThreadPool::loader_cpu_list;

uint32_t Parallel::thread_count;
ThreadPool Parallel::thread_pool;
#ifdef OS_CHCORE
std::vector<std::unique_ptr<MachineThreadPool>> Parallel::machine_pools;
#endif
const uint64_t Parallel::min_chunk_size = 100;
