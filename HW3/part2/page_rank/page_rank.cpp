#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <iostream>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence) {

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  bool converged = false;
  double global_diff;
  double *score_new = new double[numNodes];
  std::vector<Vertex> no_outgoing;
  double sum_no_outgoing_score;

  for (int i = 0; i < numNodes; ++i) {
    if (!outgoing_size(g, i)) {
      no_outgoing.push_back(i);
    }
  }

  while(!converged) {
      std::fill(score_new, score_new+numNodes, 0.0);
      sum_no_outgoing_score = 0.0;
      #pragma omp parallel for reduction(+:sum_no_outgoing_score)
      for(int i = 0; i < no_outgoing.size(); ++i) {
        sum_no_outgoing_score += damping * solution[no_outgoing.at(i)] / numNodes;
      }
      global_diff = 0.0;
      #pragma omp parallel for reduction(+:global_diff)
      for (int i = 0; i < numNodes; ++i) {
          const Vertex* in_begin = incoming_begin(g, i);
          const Vertex* in_end = incoming_end(g, i);
          for (const Vertex* v=in_begin; v!=in_end; v++) {
              Vertex in_node = *v;
              unsigned int num_outgoing = outgoing_size(g, in_node);
              score_new[i] += solution[in_node] / num_outgoing;
          }
          score_new[i] = damping * score_new[i] + (1-damping) / numNodes + sum_no_outgoing_score;
          global_diff += abs(score_new[i] - solution[i]);
      }
      std::copy(score_new, score_new+numNodes, solution);
      converged = (global_diff < convergence);
  }


  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
