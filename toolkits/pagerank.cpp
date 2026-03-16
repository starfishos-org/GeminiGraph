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

void compute(Graph<Empty> * graph, int iterations) {
  double exec_time = 0;
  exec_time -= get_time();

  double * curr = graph->alloc_vertex_array<double>();
  double * next = graph->alloc_vertex_array<double>();
  VertexSubset * active = graph->alloc_vertex_subset();
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
    usys_print_vmspace_stats();
    compute(graph, iterations);
    usys_print_vmspace_stats();
  }

  delete graph;
  printf("Pagerank is done.\n");
  return 0;
}
