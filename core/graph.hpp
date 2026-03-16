/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <thread>
// #include <omp.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"
#include "parallel.hpp"
#ifdef OS_CHCORE
#include <chcore/syscall.h>
extern "C" {
  extern int usys_print_vmspace_stats(void);
}
#endif

enum ThreadStatus {
  WORKING,
  STEALING
};

enum MessageTag {
  ShuffleGraph,
  PassMessage,
  GatherVertexArray
};

struct ThreadState {
  VertexId curr;
  VertexId end;
  ThreadStatus status;
};

struct MessageBuffer {
  size_t capacity;
  unsigned count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char * data;
  MessageBuffer () {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init (int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char*)malloc(capacity);
  }
  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char * new_data = (char*)realloc(data, new_capacity);
      assert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity;
    }
  }
};

template <typename MsgData>
struct MsgUnit {
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));

template <typename EdgeData = Empty>
class Graph {
public:
  int partition_id;
  int partitions;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;

  bool symmetric;
  VertexId vertices;
  EdgeId edges;
  VertexId ** out_degree_by_socket; // VertexId [sockets][local_range]; numa-aware per socket
  VertexId ** in_degree_by_socket;  // VertexId [sockets][local_range]; numa-aware per socket

  VertexId * partition_offset; // VertexId [partitions+1]
  VertexId * local_partition_offset; // VertexId [sockets+1]

  VertexId owned_vertices;
  EdgeId * outgoing_edges; // EdgeId [sockets]
  EdgeId * incoming_edges; // EdgeId [sockets]

  Bitmap ** incoming_adj_bitmap;
  EdgeId ** incoming_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** incoming_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  Bitmap ** outgoing_adj_bitmap;
  EdgeId ** outgoing_adj_index; // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> ** outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  VertexId * compressed_incoming_adj_vertices;
  CompressedAdjIndexUnit ** compressed_incoming_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  VertexId * compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit ** compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState ** thread_state; // ThreadState* [threads]; numa-aware
  ThreadState ** tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState ** tuned_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer ** local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer *** send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer *** recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware

  Graph(uint32_t thread_count = std::thread::hardware_concurrency()) {
    partition_id = 0;
    partitions = 1;
    // threads = numa_num_configured_cpus();
    threads = thread_count;
    sockets = 1;
    // sockets = numa_num_configured_nodes();
    threads_per_socket = threads / sockets;

    init();
  }

  struct FromBindCpuList {};
  /* OS_CHCORE: Single partition (partition_id=0, partitions=1). Machines become NUMA/sockets.
   * threads = bind_cpu_list.size(), sockets = loader_cpu_list.size(). */
  Graph(FromBindCpuList) {
    partition_id = 0;
    partitions = 1;
#ifdef OS_CHCORE
    int num_sockets = (int)ThreadPool::loader_cpu_list.size();
    if (num_sockets <= 0)
      num_sockets = 1;
    sockets = num_sockets;
#else
    sockets = 1;
#endif
    threads = (int)ThreadPool::bind_cpu_list.size();
    if (threads <= 0)
      threads = 1;
    threads_per_socket = (sockets > 0 && threads >= sockets) ? (threads / sockets) : threads;
    init();
  }

  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  /* NUMA-aware accessors for out_degree/in_degree (per-socket arrays). */
  inline VertexId get_out_degree(VertexId v) const {
    int s = get_local_partition_id(v);
    return out_degree_by_socket[s][v - local_partition_offset[s]];
  }
  inline VertexId get_in_degree(VertexId v) const {
    int s = get_local_partition_id(v);
    return in_degree_by_socket[s][v - local_partition_offset[s]];
  }
  inline VertexId * get_out_degree_ptr(VertexId v) {
    int s = get_local_partition_id(v);
    return &out_degree_by_socket[s][v - local_partition_offset[s]];
  }
  inline VertexId * get_in_degree_ptr(VertexId v) {
    int s = get_local_partition_id(v);
    return &in_degree_by_socket[s][v - local_partition_offset[s]];
  }

  /* Run func(s_i) on each machine; uses Parallel::InvokePerMachine for NUMA-local execution. */
  void run_on_loader_threads(std::function<void(int s_i)> func) {
    if (sockets <= 0) return;
    Parallel::InvokePerMachine([&func](uint32_t m) { func((int)m); });
  }

  void init() {
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    // assert( numa_available() != -1 );
    assert( sizeof(unsigned long) == 8 ); // assume unsigned long is 64-bit

    char nodestring[sockets*2+1];
    nodestring[0] = '0';
    for (int s_i=1;s_i<sockets;s_i++) {
      nodestring[s_i*2-1] = ',';
      nodestring[s_i*2] = '0'+s_i;
    }
    // struct bitmask * nodemask = numa_parse_nodestring(nodestring);
    // numa_set_interleave_mask(nodemask);

    // omp_set_dynamic(0);
    // omp_set_num_threads(threads);
    thread_state = new ThreadState * [threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer * [threads];
    for (int t_i=0;t_i<threads;t_i++) {
      thread_state[t_i] = (ThreadState*)malloc( sizeof(ThreadState));
      local_send_buffer[t_i] = (MessageBuffer*)malloc( sizeof(MessageBuffer));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
    // #pragma omp parallel for
    // for (int t_i=0;t_i<threads;t_i++) {
    //   int s_i = get_socket_id(t_i);
    //   assert(numa_run_on_node(s_i)==0);
    //   #ifdef PRINT_DEBUG_MESSAGES
    //   printf("thread-%d bound to socket-%d\n", t_i, s_i);
    //   #endif
    // }
    #ifdef PRINT_DEBUG_MESSAGES
    // printf("threads=%d*%d\n", sockets, threads_per_socket);
    // printf("interleave on %s\n", nodestring);
    #endif

    // MPI_Comm_rank(MPI_COMM_WORLD, &partition_id);
    // MPI_Comm_size(MPI_COMM_WORLD, &partitions);
    send_buffer = new MessageBuffer ** [partitions];
    recv_buffer = new MessageBuffer ** [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i] = new MessageBuffer * [sockets];
      recv_buffer[i] = new MessageBuffer * [sockets];
      for (int s_i=0;s_i<sockets;s_i++) {
        send_buffer[i][s_i] = (MessageBuffer*)malloc( sizeof(MessageBuffer));
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer*)malloc( sizeof(MessageBuffer));
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 8 * (partitions - 1);

    // MPI_Barrier(MPI_COMM_WORLD);
  }

  // fill a vertex array with a specific value
  template <typename T> void fill_vertex_array(T *array, T value) {
#if 0
    #pragma omp parallel for
    for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
      array[v_i] = value;
    }
#endif
    Parallel::For([array, value](VertexId v_i) { array[v_i] = value; },
                  partition_offset[partition_id],
                  partition_offset[partition_id + 1]);
  }

  // allocate a numa-aware vertex array
  template<typename T>
  T * alloc_vertex_array() {
    char * array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array!=NULL);
    // for (int s_i=0;s_i<sockets;s_i++) {
    //   numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i+1] - local_partition_offset[s_i]), s_i);
    // }
    return (T*)array;
  }

  // deallocate a vertex array
  template <typename T> void dealloc_vertex_array(T *array) {
      // munmap(array, sizeof(T) * vertices);
      // free(array);
      munmap(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template<typename T>
  T * alloc_interleaved_vertex_array() {
    T * array = (T *)malloc( sizeof(T) * vertices );
    assert(array!=NULL);
    return array;
  }

  // dump a vertex array to path
  template<typename T>
  void dump_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      if (partition_id==0) {
        FILE * fout = fopen(path.c_str(), "wb");
        char * buffer = new char [PAGESIZE];
        for (long offset=0;offset<file_length;) {
          if (file_length - offset >= PAGESIZE) {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          } else {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      // MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // restore a vertex array from path
  template<typename T>
  void restore_vertex_array(T * array, std::string path) {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length) {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd!=-1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id+1];
    void * data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET)!=-1);
    while (offset < end_offset) {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes!=-1);
      offset += bytes;
    }
    assert(close(fd)==0);
  }

  // gather a vertex array
  template<typename T>
  void gather_vertex_array(T * array, int root) {
    // if (partition_id!=root) {
    //   MPI_Send(array + partition_offset[partition_id], sizeof(T) * owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    // } else {
    //   for (int i=0;i<partitions;i++) {
    //     if (i==partition_id) continue;
    //     MPI_Status recv_status;
    //     MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
    //     int length;
    //     MPI_Get_count(&recv_status, MPI_CHAR, &length);
    //     assert(length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
    //   }
    // }
  }

  // allocate a vertex subset
  VertexSubset * alloc_vertex_subset() {
    return new VertexSubset(vertices);
  }

  int get_partition_id(VertexId v_i){
    for (int i=0;i<partitions;i++) {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i+1]) {
        return i;
      }
    }
    assert(false);
  }

  int get_local_partition_id(VertexId v_i) const {
    for (int s_i=0;s_i<sockets;s_i++) {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i+1]) {
        return s_i;
      }
    }
    assert(false);
  }

  // load a directed graph and make it undirected
  void load_undirected_from_directed(std::string path, VertexId vertices) {
    double prep_time = 0;
    prep_time -= WTime();

    symmetric = true;

    // MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // NUMA-aware: prelim partition by sockets for bootstrap (same as load_directed)
    std::vector<VertexId> prelim_local_partition_offset_ud(sockets + 1);
    for (int s_i = 0; s_i <= sockets; s_i++) {
      prelim_local_partition_offset_ud[s_i] = (VertexId)s_i * vertices / sockets;
    }
    std::vector<VertexId*> out_degree_prelim_ud(sockets);
    run_on_loader_threads([&](int s_i) {
      VertexId count = prelim_local_partition_offset_ud[s_i + 1] - prelim_local_partition_offset_ud[s_i];
      out_degree_prelim_ud[s_i] = (VertexId*)malloc(sizeof(VertexId) * count);
      for (VertexId v_i = 0; v_i < count; v_i++) {
        out_degree_prelim_ud[s_i][v_i] = 0;
      }
    });
    auto get_prelim_out_degree_ptr_ud = [&](VertexId v) -> VertexId* {
      int s = (vertices > 0) ? (int)((uint64_t)v * sockets / vertices) : 0;
      if (s >= sockets) s = sockets - 1;
      return &out_degree_prelim_ud[s][v - prelim_local_partition_offset_ud[s]];
    };
    auto get_prelim_out_degree_ud = [&](VertexId v) -> VertexId {
      int s = (vertices > 0) ? (int)((uint64_t)v * sockets / vertices) : 0;
      if (s >= sockets) s = sockets - 1;
      return out_degree_prelim_ud[s][v - prelim_local_partition_offset_ud[s]];
    };
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      Parallel::For([&](EdgeId e_i) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(get_prelim_out_degree_ptr_ud(src), 1);
        __sync_fetch_and_add(get_prelim_out_degree_ptr_ud(dst), 1);
      }, 0, curr_read_edges);
    }

    // locality-aware chunking
    partition_offset = new VertexId [partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;
    for (int i=0;i<partitions;i++) {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
      } else {
        EdgeId got_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += get_prelim_out_degree_ud(v_i) + alpha;
          if (got_edges > expected_chunk_size) {
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= get_prelim_out_degree_ud(v_i) + alpha;
      }
    }
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_offset[partition_id+1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    // VertexId * global_partition_offset = new VertexId [partitions + 1];
    // MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1,
    // vid_t, MPI_MAX, MPI_COMM_WORLD); for (int i=0;i<=partitions;i++) {
    //   assert(partition_offset[i] == global_partition_offset[i]);
    // }
    // MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1,
    // vid_t, MPI_MIN, MPI_COMM_WORLD); for (int i=0;i<=partitions;i++) {
    //   assert(partition_offset[i] == global_partition_offset[i]);
    // }
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += get_prelim_out_degree_ud(v_i);
        }
        printf("|V'_%d| = %u |E_%d| = %lu\n", i, partition_offset[i+1] - partition_offset[i], i, part_out_edges);
      }
    }
    // MPI_Barrier(MPI_COMM_WORLD);
    #endif
    // delete [] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId [sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
        part_out_edges += get_prelim_out_degree_ud(v_i);
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_offset[partition_id+1];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_offset[partition_id+1];v_i++) {
            got_edges += get_prelim_out_degree_ud(v_i) + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          remained_amount -= get_prelim_out_degree_ud(v_i) + alpha;
          sub_part_out_edges += get_prelim_out_degree_ud(v_i);
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], partition_id, sub_part_out_edges);
        #endif
      }
    }

    // NUMA-aware: symmetric graph: in_degree = out_degree, each socket allocates and copies from prelim
    out_degree_by_socket = new VertexId* [sockets];
    run_on_loader_threads([&](int s_i) {
      VertexId count = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      out_degree_by_socket[s_i] = (VertexId*)malloc(sizeof(VertexId) * count);
      for (VertexId i = 0; i < count; i++) {
        VertexId v = local_partition_offset[s_i] + i;
        int ps = (vertices > 0) ? (int)((uint64_t)v * sockets / vertices) : 0;
        if (ps >= sockets) ps = sockets - 1;
        out_degree_by_socket[s_i][i] = out_degree_prelim_ud[ps][v - prelim_local_partition_offset_ud[ps]];
      }
    });
    for (int s_i = 0; s_i < sockets; s_i++) {
      free(out_degree_prelim_ud[s_i]);
    }
    in_degree_by_socket = out_degree_by_socket;  // symmetric: in_degree = out_degree

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // constructing symmetric edges
    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];
    outgoing_adj_index = new EdgeId* [sockets];
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap = new Bitmap * [sockets];
    run_on_loader_threads([&](int s_i) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)malloc(sizeof(EdgeId) * (vertices+1));
    });
    {
      // std::thread recv_thread_dst([&]() {
      //   int finished_count = 0;
      //   MPI_Status recv_status;
      //   while (finished_count < partitions) {
      //     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
      //     int i = recv_status.MPI_SOURCE;
      //     assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
      //     int recv_bytes;
      //     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
      //     if (recv_bytes==1) {
      //       finished_count += 1;
      //       char c;
      //       MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //       continue;
      //     }
      //     assert(recv_bytes % edge_unit_size == 0);
      //     int recv_edges = recv_bytes / edge_unit_size;
      //     MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //     // #pragma omp parallel for
      //     for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
      //       VertexId src = recv_buffer[e_i].src;
      //       VertexId dst = recv_buffer[e_i].dst;
      //       assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
      //       int dst_part = get_local_partition_id(dst);
      //       if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
      //         outgoing_adj_bitmap[dst_part]->set_bit(src);
      //         outgoing_adj_index[dst_part][src] = 0;
      //       }
      //       __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
      //     }
      //     recv_outgoing_edges += recv_edges;
      //   }
      // });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData>* recv_buffer,
                                         int recv_edges) {
        for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
          VertexId src = recv_buffer[e_i].src;
          VertexId dst = recv_buffer[e_i].dst;
          assert(dst >= partition_offset[partition_id] &&
                 dst < partition_offset[partition_id + 1]);
          int dst_part = get_local_partition_id(dst);
          if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
            outgoing_adj_bitmap[dst_part]->set_bit(src);
            outgoing_adj_index[dst_part][src] = 0;
          }
          __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
        }
        recv_outgoing_edges += recv_edges;
      };
      for (int i = 0; i < partitions; i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph(
                (EdgeUnit<EdgeData>*)send_buffer[i].data(),
                buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i],
            //          MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph(
                (EdgeUnit<EdgeData>*)send_buffer[i].data(),
                buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i],
        // MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph(
            (EdgeUnit<EdgeData>*)send_buffer[i].data(),
            buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];
    for (int s_i=0;s_i<sockets;s_i++) {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit*)malloc( sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1));
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
      #endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData>*)malloc(unit_size * outgoing_edges[s_i]);
    }
    {
      // std::thread recv_thread_dst([&]() {
      //   int finished_count = 0;
      //   MPI_Status recv_status;
      //   while (finished_count < partitions) {
      //     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
      //     int i = recv_status.MPI_SOURCE;
      //     assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
      //     int recv_bytes;
      //     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
      //     if (recv_bytes==1) {
      //       finished_count += 1;
      //       char c;
      //       MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //       continue;
      //     }
      //     assert(recv_bytes % edge_unit_size == 0);
      //     int recv_edges = recv_bytes / edge_unit_size;
      //     MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //     #pragma omp parallel for
      //     for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
      //       VertexId src = recv_buffer[e_i].src;
      //       VertexId dst = recv_buffer[e_i].dst;
      //       assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
      //       int dst_part = get_local_partition_id(dst);
      //       EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
      //       outgoing_adj_list[dst_part][pos].neighbour = dst;
      //       if (!std::is_same<EdgeData, Empty>::value) {
      //         outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
      //       }
      //     }
      //   }
      // });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData> *recv_buffer,
                                         int recv_edges) {
#if 0
        #pragma omp parallel for
        for (EdgeId e_i = 0; e_i < recv_edges; e_i++) {
          VertexId src = recv_buffer[e_i].src;
          VertexId dst = recv_buffer[e_i].dst;
          assert(dst >= partition_offset[partition_id] &&
                 dst < partition_offset[partition_id + 1]);
          int dst_part = get_local_partition_id(dst);
          EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          outgoing_adj_list[dst_part][pos].neighbour = dst;
          if (!std::is_same<EdgeData, Empty>::value) {
            outgoing_adj_list[dst_part][pos].edge_data =
                recv_buffer[e_i].edge_data;
          }
        }
#endif
        auto func = [&](EdgeId e_i) {
          VertexId src = recv_buffer[e_i].src;
          VertexId dst = recv_buffer[e_i].dst;
          assert(dst >= partition_offset[partition_id] &&
                 dst < partition_offset[partition_id + 1]);
          int dst_part = get_local_partition_id(dst);
          EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          outgoing_adj_list[dst_part][pos].neighbour = dst;
          if (!std::is_same<EdgeData, Empty>::value) {
            outgoing_adj_list[dst_part][pos].edge_data =
                recv_buffer[e_i].edge_data;
          }
        };
        Parallel::For(func, 0, recv_edges);
      };
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph((EdgeUnit<EdgeData>*)send_buffer[i].data(), buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                    buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_dst.join();
    }
    for (int s_i=0;s_i<sockets;s_i++) {
      for (VertexId p_v_i=0;p_v_i<compressed_outgoing_adj_vertices[s_i];p_v_i++) {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
      }
    }
    // MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges = outgoing_edges;
    incoming_adj_index = outgoing_adj_index;
    incoming_adj_list = outgoing_adj_list;
    incoming_adj_bitmap = outgoing_adj_bitmap;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index = compressed_outgoing_adj_index;
    // MPI_Barrier(MPI_COMM_WORLD);

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    tune_chunks();
    tuned_chunks_sparse = tuned_chunks_dense;

    prep_time += WTime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  // transpose the graph
  void transpose() {
    std::swap(out_degree_by_socket, in_degree_by_socket);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  void load_directed(std::string path, VertexId vertices) {
    double prep_time = 0;
    prep_time -= WTime();

    symmetric = false;

    // MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
    #endif

    EdgeId read_edges = edges / partitions;
    if (partition_id==partitions-1) {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> * read_edge_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    // NUMA-aware: prelim partition by sockets (equal split) for bootstrap; each loader thread allocs + inits its range
    std::vector<VertexId> prelim_local_partition_offset(sockets + 1);
    for (int s_i = 0; s_i <= sockets; s_i++) {
      prelim_local_partition_offset[s_i] = (VertexId)s_i * vertices / sockets;
    }
    std::vector<VertexId*> out_degree_prelim(sockets);
    run_on_loader_threads([&](int s_i) {
      VertexId count = prelim_local_partition_offset[s_i + 1] - prelim_local_partition_offset[s_i];
      out_degree_prelim[s_i] = (VertexId*)malloc(sizeof(VertexId) * count);
      for (VertexId v_i = 0; v_i < count; v_i++) {
        out_degree_prelim[s_i][v_i] = 0;
      }
    });
    auto get_prelim_out_degree_ptr = [&](VertexId src) -> VertexId* {
      int s = (vertices > 0) ? (int)((uint64_t)src * sockets / vertices) : 0;
      if (s >= sockets) s = sockets - 1;
      return &out_degree_prelim[s][src - prelim_local_partition_offset[s]];
    };
    auto get_prelim_out_degree = [&](VertexId v) -> VertexId {
      int s = (vertices > 0) ? (int)((uint64_t)v * sockets / vertices) : 0;
      if (s >= sockets) s = sockets - 1;
      return out_degree_prelim[s][v - prelim_local_partition_offset[s]];
    };
    assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes>=0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      Parallel::For([&](EdgeId e_i) {
        VertexId src = read_edge_buffer[e_i].src;
        __sync_fetch_and_add(get_prelim_out_degree_ptr(src), 1);
      }, 0, curr_read_edges);
    }
    // MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId [partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges + EdgeId(vertices) * alpha;
    for (int i=0;i<partitions;i++) {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions==1) {
        partition_offset[i+1] = vertices;
      } else {
        EdgeId got_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<vertices;v_i++) {
          got_edges += get_prelim_out_degree(v_i) + alpha;
          if (got_edges > expected_chunk_size) {
            partition_offset[i+1] = v_i;
            break;
          }
        }
        partition_offset[i+1] = (partition_offset[i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
        remained_amount -= get_prelim_out_degree(v_i) + alpha;
      }
    }
    assert(partition_offset[partitions]==vertices);
    owned_vertices = partition_offset[partition_id+1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    // VertexId * global_partition_offset = new VertexId [partitions + 1];
    // MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    // for (int i=0;i<=partitions;i++) {
    //   assert(partition_offset[i] == global_partition_offset[i]);
    // }
    // MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    // for (int i=0;i<=partitions;i++) {
    //   assert(partition_offset[i] == global_partition_offset[i]);
    // }
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      for (int i=0;i<partitions;i++) {
        EdgeId part_out_edges = 0;
        for (VertexId v_i=partition_offset[i];v_i<partition_offset[i+1];v_i++) {
          part_out_edges += get_prelim_out_degree(v_i);
        }
        printf("|V'_%d| = %u |E^dense_%d| = %lu\n", i, partition_offset[i+1] - partition_offset[i], i, part_out_edges);
      }
    }
    #endif
    // delete [] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId [sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i=partition_offset[partition_id];v_i<partition_offset[partition_id+1];v_i++) {
        part_out_edges += get_prelim_out_degree(v_i);
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i=0;s_i<sockets;s_i++) {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions==1) {
          local_partition_offset[s_i+1] = partition_offset[partition_id+1];
        } else {
          EdgeId got_edges = 0;
          for (VertexId v_i=local_partition_offset[s_i];v_i<partition_offset[partition_id+1];v_i++) {
            got_edges += get_prelim_out_degree(v_i) + alpha;
            if (got_edges > expected_chunk_size) {
              local_partition_offset[s_i+1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i+1] = (local_partition_offset[s_i+1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i=local_partition_offset[s_i];v_i<local_partition_offset[s_i+1];v_i++) {
          remained_amount -= get_prelim_out_degree(v_i) + alpha;
          sub_part_out_edges += get_prelim_out_degree(v_i);
        }
        #ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E^dense_%d_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i+1] - local_partition_offset[s_i], partition_id, s_i, sub_part_out_edges);
        #endif
      }
    }

    // NUMA-aware: each loader thread allocates out_degree_by_socket[s] and copies from prelim
    out_degree_by_socket = new VertexId* [sockets];
    run_on_loader_threads([&](int s_i) {
      VertexId count = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      out_degree_by_socket[s_i] = (VertexId*)malloc(sizeof(VertexId) * count);
      uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, (size_t)count);
      if (nw == 0) nw = 1;
      Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
        size_t chunk = ((size_t)count + nw - 1) / nw;
        VertexId i_beg = (VertexId)((size_t)tid * chunk);
        VertexId i_end = (VertexId)std::min(i_beg + (VertexId)chunk, count);
        for (VertexId i = i_beg; i < i_end; i++) {
          VertexId v = local_partition_offset[s_i] + i;
          int ps = (vertices > 0) ? (int)((uint64_t)v * sockets / vertices) : 0;
          if (ps >= sockets) ps = sockets - 1;
          out_degree_by_socket[s_i][i] = out_degree_prelim[ps][v - prelim_local_partition_offset[ps]];
        }
      }, 0, nw);
    });
    run_on_loader_threads([&](int s_i) {
      free(out_degree_prelim[s_i]);
    });
    in_degree_by_socket = new VertexId* [sockets];
    run_on_loader_threads([&](int s_i) {
      VertexId count = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      in_degree_by_socket[s_i] = (VertexId*)malloc(sizeof(VertexId) * count);
      uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, (size_t)count);
      if (nw == 0) nw = 1;
      Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
        size_t chunk = ((size_t)count + nw - 1) / nw;
        VertexId i_beg = (VertexId)((size_t)tid * chunk);
        VertexId i_end = (VertexId)std::min(i_beg + (VertexId)chunk, count);
        for (VertexId i = i_beg; i < i_end; i++) {
          in_degree_by_socket[s_i][i] = 0;
        }
      }, 0, nw);
    });

    int * buffered_edges = new int [partitions];
    std::vector<char> * send_buffer = new std::vector<char> [partitions];
    for (int i=0;i<partitions;i++) {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> * recv_buffer = new EdgeUnit<EdgeData> [CHUNKSIZE];

    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId [sockets];
    outgoing_adj_index = new EdgeId* [sockets];
    outgoing_adj_list = new AdjUnit<EdgeData>* [sockets];
    outgoing_adj_bitmap = new Bitmap * [sockets];
    run_on_loader_threads([&](int s_i) {
      outgoing_adj_bitmap[s_i] = new Bitmap (vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId*)malloc(sizeof(EdgeId) * (vertices+1));
    });
    {
      // std::thread recv_thread_dst([&]() {
      //   int finished_count = 0;
      //   MPI_Status recv_status;
      //   while (finished_count < partitions) {
      //     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
      //     int i = recv_status.MPI_SOURCE;
      //     assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
      //     int recv_bytes;
      //     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
      //     if (recv_bytes==1) {
      //       finished_count += 1;
      //       char c;
      //       MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //       continue;
      //     }
      //     assert(recv_bytes % edge_unit_size == 0);
      //     int recv_edges = recv_bytes / edge_unit_size;
      //     MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //     // #pragma omp parallel for
      //     for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
      //       VertexId src = recv_buffer[e_i].src;
      //       VertexId dst = recv_buffer[e_i].dst;
      //       assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
      //       int dst_part = get_local_partition_id(dst);
      //       if (!outgoing_adj_bitmap[dst_part]->get_bit(src)) {
      //         outgoing_adj_bitmap[dst_part]->set_bit(src);
      //         outgoing_adj_index[dst_part][src] = 0;
      //       }
      //       __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
      //       __sync_fetch_and_add(&in_degree[dst], 1);
      //     }
      //     recv_outgoing_edges += recv_edges;
      //   }
      // });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData> *recv_buffer,
                                         int recv_edges) {
        // 将边按 dst 所在 socket 分桶，然后在对应 socket 的 loader 线程上串行处理，
        // 避免 cross-socket 线程访问其它 socket 的 outgoing_adj_list。
        std::vector<std::vector<EdgeUnit<EdgeData>>> buckets(sockets);
        buckets.assign(sockets, std::vector<EdgeUnit<EdgeData>>());
        const EdgeId PAR_BUCKET_THRESHOLD = 8192;
        if (recv_edges >= PAR_BUCKET_THRESHOLD && threads > 1) {
          uint32_t nw = (uint32_t)std::min((size_t)threads, (size_t)recv_edges);
          std::vector<std::vector<std::vector<EdgeUnit<EdgeData>>>> thread_buckets(nw);
          for (uint32_t t = 0; t < nw; t++) thread_buckets[t].resize(sockets);
          Parallel::Invoke([&](uint32_t tid) {
            size_t chunk = ((size_t)recv_edges + nw - 1) / nw;
            EdgeId beg = (EdgeId)((size_t)tid * chunk);
            EdgeId end = (EdgeId)std::min(beg + (EdgeId)chunk, (EdgeId)recv_edges);
            for (EdgeId e_i = beg; e_i < end; e_i++) {
              VertexId dst = recv_buffer[e_i].dst;
              if (dst < partition_offset[partition_id] ||
                  dst >= partition_offset[partition_id + 1]) continue;
              int dst_part = get_local_partition_id(dst);
              if (dst_part < 0 || dst_part >= sockets) continue;
              thread_buckets[tid][dst_part].push_back(recv_buffer[e_i]);
            }
          }, nw);
          for (int s = 0; s < sockets; s++) {
            for (uint32_t t = 0; t < nw; t++) {
              auto &tb = thread_buckets[t][s];
              buckets[s].insert(buckets[s].end(), tb.begin(), tb.end());
            }
          }
        } else {
          for (EdgeId e_i = 0; e_i < (EdgeId)recv_edges; e_i++) {
            VertexId dst = recv_buffer[e_i].dst;
            if (dst < partition_offset[partition_id] ||
                dst >= partition_offset[partition_id + 1]) continue;
            int dst_part = get_local_partition_id(dst);
            if (dst_part < 0 || dst_part >= sockets) continue;
            buckets[dst_part].push_back(recv_buffer[e_i]);
          }
        }
        run_on_loader_threads([&](int s_i) {
          auto &bucket = buckets[s_i];
          if (bucket.empty()) return;
          uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, bucket.size());
          if (nw == 0) nw = 1;
          Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
            size_t chunk = (bucket.size() + nw - 1) / nw;
            size_t beg = (size_t)tid * chunk;
            size_t end = std::min(beg + chunk, bucket.size());
            for (size_t idx = beg; idx < end; idx++) {
              const EdgeUnit<EdgeData> &eu = bucket[idx];
              VertexId src = eu.src;
              VertexId dst = eu.dst;
              assert(dst >= partition_offset[partition_id] &&
                     dst < partition_offset[partition_id + 1]);
              if (!outgoing_adj_bitmap[s_i]->get_bit(src)) {
                outgoing_adj_bitmap[s_i]->set_bit(src);
                outgoing_adj_index[s_i][src] = 0;
              }
              __sync_fetch_and_add(&outgoing_adj_index[s_i][src], 1);
              __sync_fetch_and_add(get_in_degree_ptr(dst), 1);
            }
          }, 0, nw);
        });
        recv_outgoing_edges += recv_edges;
      };
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph(
                (EdgeUnit<EdgeData> *)send_buffer[i].data(), buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph(
            (EdgeUnit<EdgeData> *)send_buffer[i].data(), buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_dst.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
      #endif
    }
    compressed_outgoing_adj_vertices = new VertexId [sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit * [sockets];
    run_on_loader_threads([&](int s_i) {
      uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, (size_t)vertices);
      if (nw == 0) nw = 1;
      std::vector<EdgeId> local_edges(nw, 0);
      std::vector<VertexId> local_verts(nw, 0);
      Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
        size_t chunk = ((size_t)vertices + nw - 1) / nw;
        VertexId beg = (VertexId)((size_t)tid * chunk);
        VertexId end = (VertexId)std::min(beg + (VertexId)chunk, (VertexId)vertices);
        EdgeId le = 0;
        VertexId lv = 0;
        for (VertexId v_i = beg; v_i < end; v_i++) {
          if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
            le += outgoing_adj_index[s_i][v_i];
            lv += 1;
          }
        }
        local_edges[tid] = le;
        local_verts[tid] = lv;
      }, 0, nw);
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (uint32_t t = 0; t < nw; t++) {
        outgoing_edges[s_i] += local_edges[t];
        compressed_outgoing_adj_vertices[s_i] += local_verts[t];
      }
      compressed_outgoing_adj_index[s_i] =
          (CompressedAdjIndexUnit*)malloc(
              sizeof(CompressedAdjIndexUnit) *
              (compressed_outgoing_adj_vertices[s_i] + 1));
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
          outgoing_adj_index[s_i][v_i] =
              last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i]
              [compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i]
              [compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      {
        VertexId ncv = compressed_outgoing_adj_vertices[s_i];
        uint32_t nw2 = (uint32_t)std::min((size_t)threads_per_socket, (size_t)ncv);
        if (nw2 == 0) nw2 = 1;
        Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
          size_t chunk = ((size_t)ncv + nw2 - 1) / nw2;
          VertexId p_beg = (VertexId)((size_t)tid * chunk);
          VertexId p_end = (VertexId)std::min(p_beg + (VertexId)chunk, ncv);
          for (VertexId p_v_i = p_beg; p_v_i < p_end; p_v_i++) {
            VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
            outgoing_adj_index[s_i][v_i] =
                compressed_outgoing_adj_index[s_i][p_v_i].index;
            outgoing_adj_index[s_i][v_i+1] =
                compressed_outgoing_adj_index[s_i][p_v_i+1].index;
          }
        }, 0, nw2);
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu sparse mode edges\n",
             partition_id, s_i, outgoing_edges[s_i]);
      #endif
      outgoing_adj_list[s_i] =
          (AdjUnit<EdgeData>*)malloc(unit_size * outgoing_edges[s_i]);
    });
    {
//       std::thread recv_thread_dst([&]() {
//         int finished_count = 0;
//         MPI_Status recv_status;
//         while (finished_count < partitions) {
//           MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
//           int i = recv_status.MPI_SOURCE;
//           assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
//           int recv_bytes;
//           MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
//           if (recv_bytes==1) {
//             finished_count += 1;
//             char c;
//             MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             continue;
//           }
//           assert(recv_bytes % edge_unit_size == 0);
//           int recv_edges = recv_bytes / edge_unit_size;
//           MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// #pragma omp parallel for
//           for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
//             VertexId src = recv_buffer[e_i].src;
//             VertexId dst = recv_buffer[e_i].dst;
//             assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id+1]);
//             int dst_part = get_local_partition_id(dst);
//             EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
//             outgoing_adj_list[dst_part][pos].neighbour = dst;
//             if (!std::is_same<EdgeData, Empty>::value) {
//               outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
//             }
//           }
//         }
//       });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData> *recv_buffer,
                                         int recv_edges) {
        std::vector<std::vector<EdgeUnit<EdgeData>>> buckets(sockets);
        buckets.assign(sockets, std::vector<EdgeUnit<EdgeData>>());
        const EdgeId PAR_BUCKET_THRESHOLD = 8192;
        if (recv_edges >= PAR_BUCKET_THRESHOLD && threads > 1) {
          uint32_t nw = (uint32_t)std::min((size_t)threads, (size_t)recv_edges);
          std::vector<std::vector<std::vector<EdgeUnit<EdgeData>>>> thread_buckets(nw);
          for (uint32_t t = 0; t < nw; t++) thread_buckets[t].resize(sockets);
          Parallel::Invoke([&](uint32_t tid) {
            size_t chunk = ((size_t)recv_edges + nw - 1) / nw;
            EdgeId beg = (EdgeId)((size_t)tid * chunk);
            EdgeId end = (EdgeId)std::min(beg + (EdgeId)chunk, (EdgeId)recv_edges);
            for (EdgeId e_i = beg; e_i < end; e_i++) {
              VertexId dst = recv_buffer[e_i].dst;
              if (dst < partition_offset[partition_id] ||
                  dst >= partition_offset[partition_id + 1]) continue;
              int dst_part = get_local_partition_id(dst);
              if (dst_part < 0 || dst_part >= sockets) continue;
              thread_buckets[tid][dst_part].push_back(recv_buffer[e_i]);
            }
          }, nw);
          for (int s = 0; s < sockets; s++) {
            for (uint32_t t = 0; t < nw; t++) {
              auto &tb = thread_buckets[t][s];
              buckets[s].insert(buckets[s].end(), tb.begin(), tb.end());
            }
          }
        } else {
          for (EdgeId e_i = 0; e_i < (EdgeId)recv_edges; e_i++) {
            VertexId dst = recv_buffer[e_i].dst;
            if (dst < partition_offset[partition_id] ||
                dst >= partition_offset[partition_id + 1]) continue;
            int dst_part = get_local_partition_id(dst);
            if (dst_part < 0 || dst_part >= sockets) continue;
            buckets[dst_part].push_back(recv_buffer[e_i]);
          }
        }
        run_on_loader_threads([&](int s_i) {
          auto &bucket = buckets[s_i];
          if (bucket.empty()) return;
          uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, bucket.size());
          if (nw == 0) nw = 1;
          Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
            size_t chunk = (bucket.size() + nw - 1) / nw;
            size_t beg = (size_t)tid * chunk;
            size_t end = std::min(beg + chunk, bucket.size());
            for (size_t idx = beg; idx < end; idx++) {
              const EdgeUnit<EdgeData> &eu = bucket[idx];
              VertexId src = eu.src;
              VertexId dst = eu.dst;
              assert(dst >= partition_offset[partition_id] &&
                     dst < partition_offset[partition_id + 1]);
              EdgeId pos =
                  __sync_fetch_and_add(&outgoing_adj_index[s_i][src], 1);
              outgoing_adj_list[s_i][pos].neighbour = dst;
              if (!std::is_same<EdgeData, Empty>::value) {
                outgoing_adj_list[s_i][pos].edge_data = eu.edge_data;
              }
            }
          }, 0, nw);
        });
      };
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(), buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_dst.join();
    }
    run_on_loader_threads([&](int s_i) {
      VertexId ncv = compressed_outgoing_adj_vertices[s_i];
      uint32_t nw2 = (uint32_t)std::min((size_t)threads_per_socket, (size_t)ncv);
      if (nw2 > 1) {
        Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
          size_t chunk = ((size_t)ncv + nw2 - 1) / nw2;
          VertexId p_beg = (VertexId)((size_t)tid * chunk);
          VertexId p_end = (VertexId)std::min(p_beg + (VertexId)chunk, ncv);
          for (VertexId p_v_i = p_beg; p_v_i < p_end; p_v_i++) {
            VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
            outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
            outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
          }
        }, 0, nw2);
      } else {
        for (VertexId p_v_i = 0; p_v_i < ncv; p_v_i++) {
          VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
          outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
          outgoing_adj_index[s_i][v_i+1] = compressed_outgoing_adj_index[s_i][p_v_i+1].index;
        }
      }
    });

    EdgeId recv_incoming_edges = 0;
    incoming_edges = new EdgeId [sockets];
    incoming_adj_index = new EdgeId* [sockets];
    incoming_adj_list = new AdjUnit<EdgeData>* [sockets];
    incoming_adj_bitmap = new Bitmap * [sockets];
    run_on_loader_threads([&](int s_i) {
      incoming_adj_bitmap[s_i] = new Bitmap (vertices);
      incoming_adj_bitmap[s_i]->clear();
      incoming_adj_index[s_i] = (EdgeId*)malloc(sizeof(EdgeId) * (vertices+1));
    });
    {
      // std::thread recv_thread_src([&]() {
      //   int finished_count = 0;
      //   MPI_Status recv_status;
      //   while (finished_count < partitions) {
      //     MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
      //     int i = recv_status.MPI_SOURCE;
      //     assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
      //     int recv_bytes;
      //     MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
      //     if (recv_bytes==1) {
      //       finished_count += 1;
      //       char c;
      //       MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //       continue;
      //     }
      //     assert(recv_bytes % edge_unit_size == 0);
      //     int recv_edges = recv_bytes / edge_unit_size;
      //     MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //     // #pragma omp parallel for
      //     for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
      //       VertexId src = recv_buffer[e_i].src;
      //       VertexId dst = recv_buffer[e_i].dst;
      //       assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id+1]);
      //       int src_part = get_local_partition_id(src);
      //       if (!incoming_adj_bitmap[src_part]->get_bit(dst)) {
      //         incoming_adj_bitmap[src_part]->set_bit(dst);
      //         incoming_adj_index[src_part][dst] = 0;
      //       }
      //       __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
      //     }
      //     recv_incoming_edges += recv_edges;
      //   }
      // });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData> *recv_buffer,
                                         int recv_edges) {
        std::vector<std::vector<EdgeUnit<EdgeData>>> buckets(sockets);
        buckets.assign(sockets, std::vector<EdgeUnit<EdgeData>>());
        const EdgeId PAR_BUCKET_THRESHOLD = 8192;
        if (recv_edges >= PAR_BUCKET_THRESHOLD && threads > 1) {
          uint32_t nw = (uint32_t)std::min((size_t)threads, (size_t)recv_edges);
          std::vector<std::vector<std::vector<EdgeUnit<EdgeData>>>> thread_buckets(nw);
          for (uint32_t t = 0; t < nw; t++) thread_buckets[t].resize(sockets);
          Parallel::Invoke([&](uint32_t tid) {
            size_t chunk = ((size_t)recv_edges + nw - 1) / nw;
            EdgeId beg = (EdgeId)((size_t)tid * chunk);
            EdgeId end = (EdgeId)std::min(beg + (EdgeId)chunk, (EdgeId)recv_edges);
            for (EdgeId e_i = beg; e_i < end; e_i++) {
              VertexId src = recv_buffer[e_i].src;
              if (src < partition_offset[partition_id] ||
                  src >= partition_offset[partition_id + 1]) continue;
              int src_part = get_local_partition_id(src);
              if (src_part < 0 || src_part >= sockets) continue;
              thread_buckets[tid][src_part].push_back(recv_buffer[e_i]);
            }
          }, nw);
          for (int s = 0; s < sockets; s++) {
            for (uint32_t t = 0; t < nw; t++) {
              auto &tb = thread_buckets[t][s];
              buckets[s].insert(buckets[s].end(), tb.begin(), tb.end());
            }
          }
        } else {
          for (EdgeId e_i = 0; e_i < (EdgeId)recv_edges; e_i++) {
            VertexId src = recv_buffer[e_i].src;
            if (src < partition_offset[partition_id] ||
                src >= partition_offset[partition_id + 1]) continue;
            int src_part = get_local_partition_id(src);
            if (src_part < 0 || src_part >= sockets) continue;
            buckets[src_part].push_back(recv_buffer[e_i]);
          }
        }
        run_on_loader_threads([&](int s_i) {
          auto &bucket = buckets[s_i];
          if (bucket.empty()) return;
          uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, bucket.size());
          if (nw == 0) nw = 1;
          Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
            size_t chunk = (bucket.size() + nw - 1) / nw;
            size_t beg = (size_t)tid * chunk;
            size_t end = std::min(beg + chunk, bucket.size());
            for (size_t idx = beg; idx < end; idx++) {
              VertexId src = bucket[idx].src;
              VertexId dst = bucket[idx].dst;
              assert(src >= partition_offset[partition_id] &&
                     src < partition_offset[partition_id + 1]);
              if (!incoming_adj_bitmap[s_i]->get_bit(dst)) {
                incoming_adj_bitmap[s_i]->set_bit(dst);
                incoming_adj_index[s_i][dst] = 0;
              }
              __sync_fetch_and_add(&incoming_adj_index[s_i][dst], 1);
            }
          }, 0, nw);
        });
        recv_incoming_edges += recv_edges;
      };
      for (int i = 0; i < partitions; i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                    buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_src.join();
      #ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
      #endif
    }
    compressed_incoming_adj_vertices = new VertexId [sockets];
    compressed_incoming_adj_index = new CompressedAdjIndexUnit * [sockets];
    run_on_loader_threads([&](int s_i) {
      uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, (size_t)vertices);
      if (nw == 0) nw = 1;
      std::vector<EdgeId> local_edges(nw, 0);
      std::vector<VertexId> local_verts(nw, 0);
      Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
        size_t chunk = ((size_t)vertices + nw - 1) / nw;
        VertexId beg = (VertexId)((size_t)tid * chunk);
        VertexId end = (VertexId)std::min(beg + (VertexId)chunk, (VertexId)vertices);
        EdgeId le = 0;
        VertexId lv = 0;
        for (VertexId v_i = beg; v_i < end; v_i++) {
          if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
            le += incoming_adj_index[s_i][v_i];
            lv += 1;
          }
        }
        local_edges[tid] = le;
        local_verts[tid] = lv;
      }, 0, nw);
      incoming_edges[s_i] = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (uint32_t t = 0; t < nw; t++) {
        incoming_edges[s_i] += local_edges[t];
        compressed_incoming_adj_vertices[s_i] += local_verts[t];
      }
      compressed_incoming_adj_index[s_i] = (CompressedAdjIndexUnit*)malloc( sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1));
      compressed_incoming_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i=0;v_i<vertices;v_i++) {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i)) {
          incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
          last_e_i = incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].vertex = v_i;
          compressed_incoming_adj_vertices[s_i] += 1;
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].index = last_e_i;
        }
      }
      {
        VertexId ncv = compressed_incoming_adj_vertices[s_i];
        uint32_t nw2 = (uint32_t)std::min((size_t)threads_per_socket, (size_t)ncv);
        if (nw2 == 0) nw2 = 1;
        Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
          size_t chunk = ((size_t)ncv + nw2 - 1) / nw2;
          VertexId p_beg = (VertexId)((size_t)tid * chunk);
          VertexId p_end = (VertexId)std::min(p_beg + (VertexId)chunk, ncv);
          for (VertexId p_v_i = p_beg; p_v_i < p_end; p_v_i++) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
            incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
          }
        }, 0, nw2);
      }
      #ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
      #endif
      incoming_adj_list[s_i] = (AdjUnit<EdgeData>*)malloc(unit_size * incoming_edges[s_i]);
    });
    {
//       std::thread recv_thread_src([&]() {
//         int finished_count = 0;
//         MPI_Status recv_status;
//         while (finished_count < partitions) {
//           MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
//           int i = recv_status.MPI_SOURCE;
//           assert(recv_status.MPI_TAG == ShuffleGraph && i >=0 && i < partitions);
//           int recv_bytes;
//           MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
//           if (recv_bytes==1) {
//             finished_count += 1;
//             char c;
//             MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             continue;
//           }
//           assert(recv_bytes % edge_unit_size == 0);
//           int recv_edges = recv_bytes / edge_unit_size;
//           MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// #pragma omp parallel for
//           for (EdgeId e_i=0;e_i<recv_edges;e_i++) {
//             VertexId src = recv_buffer[e_i].src;
//             VertexId dst = recv_buffer[e_i].dst;
//             assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id+1]);
//             int src_part = get_local_partition_id(src);
//             EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
//             incoming_adj_list[src_part][pos].neighbour = src;
//             if (!std::is_same<EdgeData, Empty>::value) {
//               incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
//             }
//           }
//         }
//       });
      auto deal_with_shuffle_graph = [&](EdgeUnit<EdgeData> *recv_buffer,
                                         int recv_edges) {
        std::vector<std::vector<EdgeUnit<EdgeData>>> buckets(sockets);
        buckets.assign(sockets, std::vector<EdgeUnit<EdgeData>>());
        const EdgeId PAR_BUCKET_THRESHOLD = 8192;
        if (recv_edges >= PAR_BUCKET_THRESHOLD && threads > 1) {
          uint32_t nw = (uint32_t)std::min((size_t)threads, (size_t)recv_edges);
          std::vector<std::vector<std::vector<EdgeUnit<EdgeData>>>> thread_buckets(nw);
          for (uint32_t t = 0; t < nw; t++) thread_buckets[t].resize(sockets);
          Parallel::Invoke([&](uint32_t tid) {
            size_t chunk = ((size_t)recv_edges + nw - 1) / nw;
            EdgeId beg = (EdgeId)((size_t)tid * chunk);
            EdgeId end = (EdgeId)std::min(beg + (EdgeId)chunk, (EdgeId)recv_edges);
            for (EdgeId e_i = beg; e_i < end; e_i++) {
              VertexId src = recv_buffer[e_i].src;
              if (src < partition_offset[partition_id] ||
                  src >= partition_offset[partition_id + 1]) continue;
              int src_part = get_local_partition_id(src);
              if (src_part < 0 || src_part >= sockets) continue;
              thread_buckets[tid][src_part].push_back(recv_buffer[e_i]);
            }
          }, nw);
          for (int s = 0; s < sockets; s++) {
            for (uint32_t t = 0; t < nw; t++) {
              auto &tb = thread_buckets[t][s];
              buckets[s].insert(buckets[s].end(), tb.begin(), tb.end());
            }
          }
        } else {
          for (EdgeId e_i = 0; e_i < (EdgeId)recv_edges; e_i++) {
            VertexId src = recv_buffer[e_i].src;
            if (src < partition_offset[partition_id] ||
                src >= partition_offset[partition_id + 1]) continue;
            int src_part = get_local_partition_id(src);
            if (src_part < 0 || src_part >= sockets) continue;
            buckets[src_part].push_back(recv_buffer[e_i]);
          }
        }
        run_on_loader_threads([&](int s_i) {
          auto &bucket = buckets[s_i];
          if (bucket.empty()) return;
          uint32_t nw = (uint32_t)std::min((size_t)threads_per_socket, bucket.size());
          if (nw == 0) nw = 1;
          Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
            size_t chunk = (bucket.size() + nw - 1) / nw;
            size_t beg = (size_t)tid * chunk;
            size_t end = std::min(beg + chunk, bucket.size());
            for (size_t idx = beg; idx < end; idx++) {
              const EdgeUnit<EdgeData> &eu = bucket[idx];
              VertexId src = eu.src;
              VertexId dst = eu.dst;
              assert(src >= partition_offset[partition_id] &&
                     src < partition_offset[partition_id + 1]);
              EdgeId pos =
                  __sync_fetch_and_add(&incoming_adj_index[s_i][dst], 1);
              incoming_adj_list[s_i][pos].neighbour = src;
              if (!std::is_same<EdgeData, Empty>::value) {
                incoming_adj_list[s_i][pos].edge_data = eu.edge_data;
              }
            }
          }, 0, nw);
        });
      };
      for (int i=0;i<partitions;i++) {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET)==read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read) {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        } else {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes>=0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i=0;e_i<curr_read_edges;e_i++) {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE) {
            // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(), buffered_edges[i]);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i=0;i<partitions;i++) {
        if (buffered_edges[i]==0) continue;
        // MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        deal_with_shuffle_graph((EdgeUnit<EdgeData> *)send_buffer[i].data(),
                                buffered_edges[i]);
        buffered_edges[i] = 0;
      }
      // for (int i=0;i<partitions;i++) {
      //   char c = 0;
      //   MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      // }
      // recv_thread_src.join();
    }
    run_on_loader_threads([&](int s_i) {
      VertexId ncv = compressed_incoming_adj_vertices[s_i];
      uint32_t nw2 = (uint32_t)std::min((size_t)threads_per_socket, (size_t)ncv);
      if (nw2 > 1) {
        Parallel::InvokeOnMachine((uint32_t)s_i, [&](uint32_t tid) {
          size_t chunk = ((size_t)ncv + nw2 - 1) / nw2;
          VertexId p_beg = (VertexId)((size_t)tid * chunk);
          VertexId p_end = (VertexId)std::min(p_beg + (VertexId)chunk, ncv);
          for (VertexId p_v_i = p_beg; p_v_i < p_end; p_v_i++) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
            incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
          }
        }, 0, nw2);
      } else {
        for (VertexId p_v_i = 0; p_v_i < ncv; p_v_i++) {
          VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
          incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
          incoming_adj_index[s_i][v_i+1] = compressed_incoming_adj_index[s_i][p_v_i+1].index;
        }
      }
    });

    delete [] buffered_edges;
    delete [] send_buffer;
    delete [] read_edge_buffer;
    delete [] recv_buffer;
    close(fin);

    transpose();
    tune_chunks();
    transpose();
    tune_chunks();

    prep_time += WTime();

    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
    #endif
  }

  void tune_chunks() {
    tuned_chunks_dense = new ThreadState * [partitions];
    int current_send_part_id = partition_id;
    for (int step=0;step<partitions;step++) {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense[i] = new ThreadState [threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i=0;t_i<threads;t_i++) {
        tuned_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j==0) {
          VertexId p_v_i = 0;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i]) {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i<compressed_incoming_adj_vertices[s_i]) {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i+1]) {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            remained_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions==1) {
          tuned_chunks_dense[i][t_i].end = end_p_v_i;
        } else {
          EdgeId got_edges = 0;
          for (VertexId p_v_i=last_p_v_i;p_v_i<end_p_v_i;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size) {
              tuned_chunks_dense[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i=tuned_chunks_dense[i][t_i].curr;p_v_i<tuned_chunks_dense[i][t_i].end;p_v_i++) {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i+1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
#ifdef PRINT_CHUNKS_INFO
    printf("[PR][part=%d] CHUNK ALLOCATION (tuned_chunks_dense):\n", partition_id);
    printf("[PR][part=%d]   partition_offset: [", partition_id);
    for (int p = 0; p <= partitions; ++p) printf("%u%s", partition_offset[p], p < partitions ? "," : "]\n");
    printf("[PR][part=%d]   local_partition_offset: [", partition_id);
    for (int s = 0; s <= sockets; ++s) printf("%u%s", local_partition_offset[s], s < sockets ? "," : "]\n");
    for (int p = 0; p < partitions; ++p) {
      printf("[PR][part=%d]   partition %d (machine %d): vertex_range=[%u,%u), compressed_incoming_vertices=%u\n",
             partition_id, p, p, partition_offset[p], partition_offset[p+1], compressed_incoming_adj_vertices[p]);
      int workers = 0;
      for (int t_i = 0; t_i < threads; t_i++) {
        VertexId c = tuned_chunks_dense[p][t_i].curr;
        VertexId e = tuned_chunks_dense[p][t_i].end;
        if (e <= c) continue;
        workers++;
        EdgeId edges = 0;
        for (VertexId p_v_i = c; p_v_i < e; p_v_i++) {
          edges += compressed_incoming_adj_index[p][p_v_i + 1].index -
                   compressed_incoming_adj_index[p][p_v_i].index;
        }
        printf("[PR][part=%d]     thread %d (socket %d): pv_range=[%u,%u), approx_edges=%lu\n",
               partition_id, t_i, get_socket_id(t_i), c, e, (unsigned long)edges);
      }
      printf("[PR][part=%d]     -> %d workers on partition %d (machine %d)\n", partition_id, workers, p, p);
    }
    printf("[PR][part=%d] CHUNK ALLOCATION END\n", partition_id);
#endif
    }
  }

  // process vertices
  template<typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap * active) {
    double stream_time = 0;
    stream_time -= WTime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i=0;t_i<threads;t_i++) {
      int s_i = get_socket_id(t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset[s_i+1] - local_partition_offset[s_i];
      thread_state[t_i]->curr = local_partition_offset[s_i] + partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
      if (s_j == threads_per_socket - 1) {
        thread_state[t_i]->end = local_partition_offset[s_i+1];
      }
      thread_state[t_i]->status = WORKING;
    }
    #if 0
#pragma omp parallel reduction(+ : reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true) {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end) break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset=1;t_offset<threads;t_offset++) {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status!=STEALING) {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr,
          basic_chunk); if (v_i >= thread_state[t_i]->end) continue; unsigned
          long word = active->data[WORD_OFFSET(v_i)]; while (word != 0) {
            if (word & 1) {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      reducer += local_reducer;
    }
#endif
    auto func = [&](int thread_id) {
      R local_reducer = 0;
      while (true) {
        VertexId v_i =
            __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end)
          break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset = 1; t_offset < threads; t_offset++) {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status != STEALING) {
          VertexId v_i =
              __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end)
            continue;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0) {
            if (word & 1) {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      return local_reducer;
    };

    auto rfunc = [](R &a, R &b) { a += b; };

    reducer = Parallel::Reduce<decltype(func),decltype(rfunc), R>(func, rfunc, (R)0, threads);

    R global_reducer;
    // MPI_Datatype dt = get_mpi_data_type<R>();
    // MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    global_reducer = reducer;
    stream_time += WTime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

  template<typename M>
  void flush_local_send_buffer(int t_i) {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos, local_send_buffer[t_i]->data, sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)
  template <typename M> void emit(VertexId vtx, M msg) {
    uint32_t t_i = ThreadPool::thread_id;
    MsgUnit<M> * buffer = (MsgUnit<M>*)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count==local_send_buffer_limit) {
      flush_local_send_buffer<M>(t_i);
    }
  }

  // process edges
  template<typename R, typename M>
  R process_edges(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal, std::function<R(VertexId, M)> dense_slot, Bitmap * active, Bitmap * dense_selective = nullptr) {
    double stream_time = 0;
    stream_time -= WTime();

    for (int t_i=0;t_i<threads;t_i++) {
      local_send_buffer[t_i]->resize( sizeof(MsgUnit<M>) * local_send_buffer_limit );
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    EdgeId active_edges = process_vertices<EdgeId>(
      [&](VertexId vtx){
        return (EdgeId)get_out_degree(vtx);
      },
      active
    );
    bool sparse = (active_edges < edges / 20);
    if (sparse) {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_vertices * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    } else {
      for (int i=0;i<partitions;i++) {
        for (int s_i=0;s_i<sockets;s_i++) {
          recv_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * owned_vertices * sockets );
          send_buffer[i][s_i]->resize( sizeof(MsgUnit<M>) * (partition_offset[i+1] - partition_offset[i]) * sockets );
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    size_t basic_chunk = 64;
    if (sparse) {
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("sparse mode\n");
      }
      #endif
      int * recv_queue = new int [partitions];
      // int recv_queue_size = 0;
      // std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
      // #pragma omp parallel for
      // for (VertexId begin_v_i=partition_offset[partition_id];begin_v_i<partition_offset[partition_id+1];begin_v_i+=basic_chunk) {
      //   VertexId v_i = begin_v_i;
      //   unsigned long word = active->data[WORD_OFFSET(v_i)];
      //   while (word != 0) {
      //     if (word & 1) {
      //       sparse_signal(v_i);
      //     }
      //     v_i++;
      //     word = word >> 1;
      //   }
      // }
      Parallel::For([&](VertexId v_i) {
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0) {
          if (word & 1) {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }, partition_offset[partition_id], partition_offset[partition_id+1], basic_chunk);
      // #pragma omp parallel for
      // for (int t_i=0;t_i<threads;t_i++) {
      //   flush_local_send_buffer<M>(t_i);
      // }
      Parallel::For([&](int t_i) {
        flush_local_send_buffer<M>(t_i);
      }, 0, threads, 1, 1);
      // recv_queue[recv_queue_size] = partition_id;
      // recv_queue_mutex.lock();
      // recv_queue_size += 1;
      // recv_queue_mutex.unlock();
      // std::thread send_thread([&](){
      //   for (int step=1;step<partitions;step++) {
      //     int i = (partition_id - step + partitions) % partitions;
      //     for (int s_i=0;s_i<sockets;s_i++) {
      //       MPI_Send(send_buffer[partition_id][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      //     }
      //   }
      // });
      // std::thread recv_thread([&](){
      //   for (int step=1;step<partitions;step++) {
      //     int i = (partition_id + step) % partitions;
      //     for (int s_i=0;s_i<sockets;s_i++) {
      //       MPI_Status recv_status;
      //       MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
      //       MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
      //       MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //       recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
      //     }
      //     recv_queue[recv_queue_size] = i;
      //     recv_queue_mutex.lock();
      //     recv_queue_size += 1;
      //     recv_queue_mutex.unlock();
      //   }
      // });
      for (int step=0;step<partitions;step++) {
        // while (true) {
        //   recv_queue_mutex.lock();
        //   bool condition = (recv_queue_size<=step);
        //   recv_queue_mutex.unlock();
        //   if (!condition) break;
        //   __asm volatile ("pause" ::: "memory");
        // }
        MessageBuffer **used_buffer;
        used_buffer = send_buffer[partition_id];
        for (int s_i=0;s_i<sockets;s_i++) {
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i=0;t_i<threads;t_i++) {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
            if (s_j == threads_per_socket - 1) {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
          #if 0
#pragma omp parallel reduction(+ : reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true) {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end) break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i>thread_state[thread_id]->end) {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                  local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status==STEALING) continue;
              while (true) {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end) break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i>thread_state[t_i]->end) {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                  VertexId v_i = buffer[b_i].vertex;
                  M msg_data = buffer[b_i].msg_data;
                  if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                  }
                }
              }
            }
            reducer += local_reducer;
          }
#endif
          auto func = [&](uint32_t thread_id) {
            R local_reducer = 0;
            int s_i = get_socket_id(thread_id);
            while (true) {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end) break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i>thread_state[thread_id]->end) {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                  local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status==STEALING) continue;
              while (true) {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end) break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i>thread_state[t_i]->end) {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
                  VertexId v_i = buffer[b_i].vertex;
                  M msg_data = buffer[b_i].msg_data;
                  if (outgoing_adj_bitmap[s_i]->get_bit(v_i)) {
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i+1]));
                  }
                }
              }
            }
            return local_reducer;
          };
          auto rfunc = [](R &a, R &b) { a += b; };
          reducer = Parallel::Reduce<decltype(func), decltype(rfunc), R>(
              func, rfunc, (R)0, threads);
        }
      }
      // send_thread.join();
      // recv_thread.join();
      delete [] recv_queue;
    } else {
      // dense selective bitmap
      if (dense_selective!=nullptr && partitions>1) {
        double sync_time = 0;
        sync_time -= get_time();
        // std::thread send_thread([&](){
        //   for (int step=1;step<partitions;step++) {
        //     int recipient_id = (partition_id + step) % partitions;
        //     MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
        //   }
        // });
        // std::thread recv_thread([&](){
        //   for (int step=1;step<partitions;step++) {
        //     int sender_id = (partition_id - step + partitions) % partitions;
        //     MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //   }
        // });
        // send_thread.join();
        // recv_thread.join();
        // MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
        #ifdef PRINT_DEBUG_MESSAGES
        if (partition_id==0) {
          printf("sync_time = %lf\n", sync_time);
        }
        #endif
      }
      #ifdef PRINT_DEBUG_MESSAGES
      if (partition_id==0) {
        printf("dense mode\n");
      }
      #endif

      // std::thread send_thread([&](){
      //   for (int step=0;step<partitions;step++) {
      //     if (step==partitions-1) {
      //       break;
      //     }
      //     while (true) {
      //       send_queue_mutex.lock();
      //       bool condition = (send_queue_size<=step);
      //       send_queue_mutex.unlock();
      //       if (!condition) break;
      //       __asm volatile ("pause" ::: "memory");
      //     }
      //     int i = send_queue[step];
      //     for (int s_i=0;s_i<sockets;s_i++) {
      //       MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      //     }
      //   }
      // });
      // std::thread recv_thread([&](){
      //   std::vector<std::thread> threads;
      //   for (int step=1;step<partitions;step++) {
      //     int i = (partition_id - step + partitions) % partitions;
      //     threads.emplace_back([&](int i){
      //       for (int s_i=0;s_i<sockets;s_i++) {
      //         MPI_Status recv_status;
      //         MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
      //         MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
      //         MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //         recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
      //       }
      //     }, i);
      //   }
      //   for (int step=1;step<partitions;step++) {
      //     int i = (partition_id - step + partitions) % partitions;
      //     threads[step-1].join();
      //     recv_queue[recv_queue_size] = i;
      //     recv_queue_mutex.lock();
      //     recv_queue_size += 1;
      //     recv_queue_mutex.unlock();
      //   }
      //   recv_queue[recv_queue_size] = partition_id;
      //   recv_queue_mutex.lock();
      //   recv_queue_size += 1;
      //   recv_queue_mutex.unlock();
      // });
      current_send_part_id = partition_id;
      for (int step=0;step<partitions;step++) {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i=0;t_i<threads;t_i++) {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }
        #if 0
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true) {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i) break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i) {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset=1;t_offset<threads;t_offset++) {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status!=STEALING) {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end) break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end) {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
              }
            }
          }
        }
#endif
        {
          auto func = [this, &dense_signal, basic_chunk](uint32_t thread_id) {
            int s_i = get_socket_id(thread_id);
            VertexId final_p_v_i = thread_state[thread_id]->end;
            while (true) {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (begin_p_v_i >= final_p_v_i) break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > final_p_v_i) {
                end_p_v_i = final_p_v_i;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset=1;t_offset<threads;t_offset++) {
              int t_i = (thread_id + t_offset) % threads;
              int s_i = get_socket_id(t_i);
              while (thread_state[t_i]->status!=STEALING) {
                VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (begin_p_v_i >= thread_state[t_i]->end) break;
                VertexId end_p_v_i = begin_p_v_i + basic_chunk;
                if (end_p_v_i > thread_state[t_i]->end) {
                  end_p_v_i = thread_state[t_i]->end;
                }
                for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i ++) {
                  VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                  dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
                }
              }
            }
          };
          
          double dense_pull_invoke_time = 0;
          dense_pull_invoke_time -= get_time();
          Parallel::Invoke(func, threads);
          dense_pull_invoke_time += get_time();
          if (partition_id == 0) {
            printf("[profile][dense_pull_invoke] time=%lf(s)\n", dense_pull_invoke_time);
          }
        }
      }
      
      for (int step=0;step<partitions;step++) {
        // while (true) {
        //   recv_queue_mutex.lock();
        //   bool condition = (recv_queue_size<=step);
        //   recv_queue_mutex.unlock();
        //   if (!condition) break;
        //   __asm volatile ("pause" ::: "memory");
        // }
        MessageBuffer ** used_buffer;
        used_buffer = send_buffer[partition_id];
        for (int t_i=0;t_i<threads;t_i++) {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket  / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j+1);
          if (s_j == threads_per_socket - 1) {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        #if 0
#pragma omp parallel reduction(+ : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true) {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end) break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i>thread_state[thread_id]->end) {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i=begin_b_i;b_i<end_b_i;b_i++) {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
#endif
        auto func = [&](uint32_t thread_id) {
          R local_reducer = 0;
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> * buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true) {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end) {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++) {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          return local_reducer;
        };
        auto rfunc = [](R &a, R &b) { a += b; };
        reducer = Parallel::Reduce<decltype(func), decltype(rfunc), R>(
            func, rfunc, (R)0, threads);
      }
      // send_thread.join();
      // recv_thread.join();
      // delete [] send_queue;
      // delete [] recv_queue;
    }

    R global_reducer;
    // MPI_Datatype dt = get_mpi_data_type<R>();
    // MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    global_reducer = reducer;
    stream_time += WTime();
    #ifdef PRINT_DEBUG_MESSAGES
    if (partition_id==0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
    #endif
    return global_reducer;
  }

};

#endif

