#include "core/parallel.hpp"
#include <thread>
#include <unordered_map>

uint32_t thread_local ThreadPool::thread_id;

uint32_t Parallel::thread_count;
ThreadPool Parallel::thread_pool;
const uint64_t Parallel::min_chunk_size = 100;

std::unordered_map<std::string, void*> MMapPool::mmap_addr;
std::unordered_map<std::string, int> MMapPool::mmap_fd;