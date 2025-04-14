#include "core/parallel.hpp"
#include <thread>

uint32_t thread_local ThreadPool::thread_id;

uint32_t Parallel::thread_count;
ThreadPool Parallel::thread_pool;
const uint64_t Parallel::min_chunk_size = 100;
