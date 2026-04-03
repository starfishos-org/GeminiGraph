/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "core/graph.hpp"

#include <math.h>
#ifdef OS_CHCORE
#include <chcore/syscall.h>
#endif

const double d = (double)0.85;

void print_data_structure(Graph<Empty> * graph, double * curr, double * next, VertexSubset * active) {
  printf("[PR][part=%d] curr: [%p-%p) next: [%p-%p) active_data: [%p-%p) out_degree: [%p-%p) in_degree: [%p-%p) partition_offset: [%p-%p) local_partition_offset: [%p-%p)"
#ifdef OS_CHCORE
         " incoming_adj_list_replica[m=%d][0]: [%p-%p) incoming_adj_list_replica[m=%d][1]: [%p-%p)"
         " compressed_incoming_adj_index_replica[m=%d][0]: [%p-%p) compressed_incoming_adj_index_replica[m=%d][1]: [%p-%p)"
#endif
         ,
    graph->partition_id,
    (void *)curr, (void *)((char *)curr + graph->vertices * sizeof(double)),
    (void *)next, (void *)((char *)next + graph->vertices * sizeof(double)),
    (void *)active->data, (void *)((char *)active->data + graph->vertices * sizeof(VertexId)),
    /* For global out_degree/in_degree, print the range of socket 0 (if any). */
    (void *)(graph->sockets > 0 ? (void *)graph->out_degree_by_socket[0] : nullptr),
    (void *)(graph->sockets > 0
               ? (void *)((char *)graph->out_degree_by_socket[0]
                          + (graph->local_partition_offset[1] - graph->local_partition_offset[0]) * sizeof(VertexId))
               : nullptr),
    (void *)(graph->sockets > 0 ? (void *)graph->in_degree_by_socket[0] : nullptr),
    (void *)(graph->sockets > 0
               ? (void *)((char *)graph->in_degree_by_socket[0]
                          + (graph->local_partition_offset[1] - graph->local_partition_offset[0]) * sizeof(VertexId))
               : nullptr),
    (void *)graph->partition_offset,
    (void *)((char *)graph->partition_offset + (graph->partitions + 1) * sizeof(VertexId)),
    (void *)graph->local_partition_offset,
    (void *)((char *)graph->local_partition_offset + (graph->sockets + 1) * sizeof(VertexId))
#ifdef OS_CHCORE
    ,
    graph->get_cur_machine_id(),
    (void *)(graph->use_incoming_replica
               ? (void *)graph->incoming_adj_list_replica[graph->get_cur_machine_id()][0]
               : nullptr),
    (void *)(graph->use_incoming_replica
               ? (void *)((char *)graph->incoming_adj_list_replica[graph->get_cur_machine_id()][0]
                          + (long)graph->incoming_edges[0] * graph->unit_size)
               : nullptr),
    graph->get_cur_machine_id(),
    (void *)(graph->use_incoming_replica
               ? (void *)graph->incoming_adj_list_replica[graph->get_cur_machine_id()][1]
               : nullptr),
    (void *)(graph->use_incoming_replica
               ? (void *)((char *)graph->incoming_adj_list_replica[graph->get_cur_machine_id()][1]
                          + (long)graph->incoming_edges[1] * graph->unit_size)
               : nullptr),
    graph->get_cur_machine_id(),
    (void *)(graph->use_incoming_replica
               ? (void *)graph->compressed_incoming_adj_index_replica[graph->get_cur_machine_id()][0]
               : nullptr),
    (void *)(graph->use_incoming_replica
               ? (void *)((char *)graph->compressed_incoming_adj_index_replica[graph->get_cur_machine_id()][0]
                          + (graph->compressed_incoming_adj_vertices[0] + 1)
                            * sizeof(CompressedAdjIndexUnit))
               : nullptr),
    graph->get_cur_machine_id(),
    (void *)(graph->use_incoming_replica
               ? (void *)graph->compressed_incoming_adj_index_replica[graph->get_cur_machine_id()][1]
               : nullptr),
    (void *)(graph->use_incoming_replica
               ? (void *)((char *)graph->compressed_incoming_adj_index_replica[graph->get_cur_machine_id()][1]
                          + (graph->compressed_incoming_adj_vertices[1] + 1)
                            * sizeof(CompressedAdjIndexUnit))
               : nullptr)
#endif
  );
  for (int s_i = 0; s_i < graph->sockets; s_i++) {
    VertexId local_start = graph->local_partition_offset[s_i];
    VertexId local_end = graph->local_partition_offset[s_i + 1];
    VertexId local_count = (local_end > local_start) ? (local_end - local_start) : 0;
    printf(" out_degree_local[%d]=[%p-%p) in_degree_local[%d]=[%p-%p) outgoing_adj_index[%d]=[%p-%p) outgoing_adj_list[%d]=[%p-%p) outgoing_adj_bitmap[%d]=[%p-%p)"
           " incoming_adj_index[%d]=[%p-%p) incoming_adj_list[%d]=[%p-%p) incoming_adj_bitmap[%d]=[%p-%p)"
           " compressed_incoming_adj_index[%d]=[%p-%p) compressed_outgoing_adj_index[%d]=[%p-%p)",
           s_i,
           (void *)(graph->out_degree_by_socket ? (void *)graph->out_degree_by_socket[s_i] : nullptr),
           (void *)(graph->out_degree_by_socket && local_count > 0
                      ? (void *)((char *)graph->out_degree_by_socket[s_i] + local_count * sizeof(VertexId))
                      : nullptr),
           s_i,
           (void *)(graph->in_degree_by_socket ? (void *)graph->in_degree_by_socket[s_i] : nullptr),
           (void *)(graph->in_degree_by_socket && local_count > 0
                      ? (void *)((char *)graph->in_degree_by_socket[s_i] + local_count * sizeof(VertexId))
                      : nullptr),
           s_i,
           (void *)graph->outgoing_adj_index[s_i],
           (void *)((char *)graph->outgoing_adj_index[s_i] + (graph->vertices + 1) * sizeof(EdgeId)),
           s_i,
           (void *)graph->outgoing_adj_list[s_i],
           (void *)((char *)graph->outgoing_adj_list[s_i]
                    + (long)graph->outgoing_edges[s_i] * graph->unit_size),
           s_i,
           (void *)graph->outgoing_adj_bitmap[s_i],
           (void *)((char *)graph->outgoing_adj_bitmap[s_i] + graph->vertices * sizeof(VertexId)),
           s_i,
           (void *)(graph->incoming_adj_index ? (void *)graph->incoming_adj_index[s_i] : nullptr),
           (void *)(graph->incoming_adj_index
                      ? (void *)((char *)graph->incoming_adj_index[s_i] + (graph->vertices + 1) * sizeof(EdgeId))
                      : nullptr),
           s_i,
           (void *)(graph->incoming_adj_list ? (void *)graph->incoming_adj_list[s_i] : nullptr),
           (void *)(graph->incoming_adj_list
                      ? (void *)((char *)graph->incoming_adj_list[s_i]
                                 + (long)graph->incoming_edges[s_i] * graph->unit_size)
                      : nullptr),
           s_i,
           (void *)(graph->incoming_adj_bitmap ? (void *)graph->incoming_adj_bitmap[s_i] : nullptr),
           (void *)(graph->incoming_adj_bitmap
                      ? (void *)((char *)graph->incoming_adj_bitmap[s_i] + graph->vertices * sizeof(VertexId))
                      : nullptr),
           s_i,
           (void *)(graph->compressed_incoming_adj_index ? (void *)graph->compressed_incoming_adj_index[s_i] : nullptr),
           (void *)(graph->compressed_incoming_adj_index
                      ? (void *)((char *)graph->compressed_incoming_adj_index[s_i]
                                 + (graph->compressed_incoming_adj_vertices[s_i] + 1)
                                   * sizeof(CompressedAdjIndexUnit))
                      : nullptr),
           s_i,
           (void *)(graph->compressed_outgoing_adj_index ? (void *)graph->compressed_outgoing_adj_index[s_i] : nullptr),
           (void *)(graph->compressed_outgoing_adj_index
                      ? (void *)((char *)graph->compressed_outgoing_adj_index[s_i]
                                 + (graph->compressed_outgoing_adj_vertices[s_i] + 1)
                                   * sizeof(CompressedAdjIndexUnit))
                      : nullptr));
  }
  printf("\n");
}

void compute(Graph<Empty> * graph, int iterations) {
#ifdef PRINT_VMSPACE_STATS
  usys_print_vmspace_stats();
#endif
  double exec_time = 0;
  exec_time -= get_time();

#ifdef OS_CHCORE
  // 在 ChCore 上，直接把 curr/next 分配在 CXL 共享内存上
  double * curr = graph->alloc_vertex_array_cxl<double>();
  double * next = graph->alloc_vertex_array_cxl<double>();
#else
  double * curr = graph->alloc_vertex_array<double>();
  double * next = graph->alloc_vertex_array<double>();
#endif
  VertexSubset * active = graph->alloc_vertex_subset();
#ifdef PRINT_DATA_STRUCTURE
  print_data_structure(graph, curr, next, active);
#endif
  active->fill();

  double delta = graph->process_vertices<double>(
    [&](VertexId vtx){
      curr[vtx] = (double)1;
      if (graph->get_out_degree(vtx)>0) {
        curr[vtx] /= graph->get_out_degree(vtx);
      }
      return (double)1;
    },
    active
  );
  delta /= graph->vertices;

  for (int i_i=0;i_i<iterations;i_i++) {
    if (graph->partition_id==0) {
      printf("delta(%d)=%lf\n", i_i, delta);
    }
    graph->fill_vertex_array(next, (double)0);
    graph->process_edges<int,double>(
      [&](VertexId src){
        graph->emit(src, curr[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          write_add(&next[dst], msg);
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          // if (src >= graph->vertices) {
          //   printf("[BUG] src >= graph->vertices\n");
          //   exit(1);
          // }
          // printf("[BUG] curr = %p, src = %u, vertices = %u\n", curr, src, graph->vertices);
          // printf("[BUG] curr[src] = %lf\n", (long long)curr + src * sizeof(double));
          sum += curr[src];
        }
        graph->emit(dst, sum);
      },
      [&](VertexId dst, double msg) {
        write_add(&next[dst], msg);
        return 0;
      },
      active
    );
    if (i_i==iterations-1) {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          return 0;
        },
        active
      );
    } else {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          if (graph->get_out_degree(vtx)>0) {
            next[vtx] /= graph->get_out_degree(vtx);
            return fabs(next[vtx] - curr[vtx]) * graph->get_out_degree(vtx);
          }
          return fabs(next[vtx] - curr[vtx]);
        },
        active
      );
    }
    delta /= graph->vertices;
    std::swap(curr, next);
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  double pr_sum = graph->process_vertices<double>(
    [&](VertexId vtx) {
      return curr[vtx];
    },
    active
  );
  if (graph->partition_id==0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(curr, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (curr[v_i] > curr[max_v_i]) max_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, curr[max_v_i]);
  }

#ifdef PRINT_VMSPACE_STATS
  usys_print_vmspace_stats();
#endif

  graph->dealloc_vertex_array(curr);
  graph->dealloc_vertex_array(next);
  delete active;
}

void bind_cpu(uint32_t cpu_id) {
  #ifdef OS_CHCORE
      cpu_set_t cpu_set;
      CPU_ZERO(&cpu_set);
      CPU_SET(cpu_id, &cpu_set);
      sched_setaffinity(-2, sizeof(cpu_set), &cpu_set);
      sched_yield();
  #else
      cpu_set_t cpu_set;
      CPU_ZERO(&cpu_set);
      CPU_SET(cpu_id, &cpu_set);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
      sched_yield();
  #endif
    }

int main(int argc, char ** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <file> <vertices> <iterations> [stage1 threads] [stage2 threads]\n", argv[0]);
    fprintf(stderr, "  (Linux) Or for ChCore: <file> <vertices> <iterations> [machines]\n");
    fprintf(stderr, "  [machines]: For ChCore, limit to first N machine segments from gemini_bind_cpu.txt (0=use all).\n");
    exit(EXIT_FAILURE);
  }

  Graph<Empty> *graph;

#ifdef OS_CHCORE
  uint32_t requested_machines = (argc >= 5) ? (uint32_t)std::atoi(argv[4]) : 0;
  if (Parallel::LoadBindCpuFile("gemini_bind_cpu.txt") != 0) {
    fprintf(stderr, "error: must provide gemini_bind_cpu.txt (e.g. 0-7,12-19)\n");
    exit(EXIT_FAILURE);
  }
  if (requested_machines > 0)
    Parallel::LimitMachines(requested_machines);

  int main_cpu = ThreadPool::bind_cpu_list.empty() ? 0 : ThreadPool::bind_cpu_list[0];
  usys_set_affinity(-2, main_cpu);

  uint32_t thread_count = (uint32_t)ThreadPool::bind_cpu_list.size();
  DEBUG_PRINT("thread_count: %u\n", thread_count);
  Parallel::SetThreadCount(thread_count);
  graph = new Graph<Empty>(Graph<Empty>::FromBindCpuList());
#else
  bind_cpu(0);
  uint32_t thread_count1 = (argc > 4) ? std::atoi(argv[4]) : std::thread::hardware_concurrency();
  uint32_t thread_count2 = (argc > 5) ? std::atoi(argv[5]) : thread_count1;
  Parallel::SetThreadCount(thread_count1);
  graph = new Graph<Empty>(thread_count2);
#endif

  graph->load_directed(argv[1], std::atoi(argv[2]));
  int iterations = std::atoi(argv[3]);

#ifndef OS_CHCORE
  Parallel::SetThreadCount(thread_count2);
#endif
  for (int run=0;run<1;run++) {
    compute(graph, iterations);
  }

  delete graph;
  printf("Pagerank is done.\n");
  return 0;
}
