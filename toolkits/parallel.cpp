#include "core/parallel.hpp"
#include <thread>

uint32_t thread_local ThreadPool::thread_id;

uint32_t Parallel::thread_count = std::thread::hardware_concurrency();
ThreadPool Parallel::thread_pool(Parallel::thread_count);
const uint64_t Parallel::min_chunk_size = 100;