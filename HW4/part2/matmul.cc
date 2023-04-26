#include <mpi.h>
#include <iostream>
#define TH 800*800*800

void show_matrix(int *mat_ptr, int m, int n) {
//  std::cout << which << " matrix : " << std::endl;
  for(int i=0; i<m; ++i) {
    for(int j=0; j<n; ++j) {
      std::cout << mat_ptr[i*n+j] << " ";
    }
    std::cout << std::endl;
  }
}

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr,
                        int **b_mat_ptr) {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#ifdef DEBUG
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name, world_rank, world_size);
#endif

  int a_size, b_size;
  if(world_rank == 0) {
    std::cin >> *n_ptr >> *m_ptr >> *l_ptr;

#ifdef DEBUG
    std::cout << *n_ptr << " " << *m_ptr << " " << *n_ptr << std::endl;
#endif

    a_size = (*n_ptr) * (*m_ptr);
    *a_mat_ptr = new int[a_size];
    int ptr = 0;
    for (int i = 0; i < a_size; ++i) {
      std::cin >> (*a_mat_ptr)[ptr++];
    }

#ifdef DEBUG
    show_matrix(*a_mat_ptr, *n_ptr, *m_ptr, 'A');
#endif

    b_size = (*m_ptr) * (*l_ptr);
    *b_mat_ptr = new int[b_size];
    ptr = 0;
    for (int i = 0; i < b_size; ++i) {
      std::cin >> (*b_mat_ptr)[ptr++];
    }
//    std::cout << " Finished reading, before sending ... " << std::endl;
#ifdef DEBUG
    show_matrix(*b_mat_ptr, *m_ptr, *l_ptr, 'B');
#endif


    MPI_Request requestsA[world_size * 3];
    for (int i = 1; i < world_size; ++i) {
      MPI_Isend(m_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, requestsA + i * 3);
      MPI_Isend(n_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, requestsA + i * 3 + 1);
      MPI_Isend(l_ptr, 1, MPI_INT, i, 0, MPI_COMM_WORLD, requestsA + i * 3 + 2);
    }
    MPI_Waitall((world_size - 1) * 3, requestsA + 3, MPI_STATUS_IGNORE);

    int nml = (*n_ptr) * (*m_ptr) * (*l_ptr);
    if(nml > TH) {
      MPI_Request requestsB[world_size * 2];
      for (int i = 1; i < world_size; ++i) {
        MPI_Isend(*a_mat_ptr, a_size, MPI_INT, i, 0, MPI_COMM_WORLD, requestsB + i * 2);
        MPI_Isend(*b_mat_ptr, b_size, MPI_INT, i, 0, MPI_COMM_WORLD, requestsB + i * 2 + 1);
      }
      MPI_Waitall((world_size - 1) * 2, requestsB + 2, MPI_STATUS_IGNORE);
    }

  } else if(world_rank > 0) {

    MPI_Request requestsA[3];
    MPI_Irecv(m_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, requestsA);
    MPI_Irecv(n_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, requestsA+1);
    MPI_Irecv(l_ptr, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, requestsA+2);
    MPI_Waitall(3, requestsA, MPI_STATUS_IGNORE);

    int nml = (*n_ptr) * (*m_ptr) * (*l_ptr);
    if(nml > TH) {
      a_size = (*n_ptr) * (*m_ptr);
      b_size = (*m_ptr) * (*l_ptr);
      *a_mat_ptr = new int[a_size];
      *b_mat_ptr = new int[b_size];

      MPI_Request requestsB[2];
      MPI_Irecv(*a_mat_ptr, a_size, MPI_INT, 0, 0, MPI_COMM_WORLD, requestsB);
      MPI_Irecv(*b_mat_ptr, b_size, MPI_INT, 0, 0, MPI_COMM_WORLD, requestsB + 1);
      MPI_Waitall(2, requestsB, MPI_STATUS_IGNORE);
    } else {
      *a_mat_ptr = nullptr;
      *b_mat_ptr = nullptr;
    }

#ifdef DEBUG
    show_matrix(*a_mat_ptr, *n_ptr, *m_ptr, 'A');
    show_matrix(*b_mat_ptr, *m_ptr, *l_ptr, 'B');
#endif

  }
}

void matrix_multiply(const int n, const int m, const int l, const int *a_mat,
                     const int *b_mat) {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int workload = n/world_size;
  int start, end;
  bool only_one_proc = (n * m * l <= TH);
  if(world_rank == 0) {
    if(only_one_proc) {
      workload = n;
      start = 0;
      end = n;
    } else {
      workload += n % world_size;
      start = 0;
      end = (world_rank + 1) * workload;
    }
  } else if(world_rank > 0) {
    if(only_one_proc) {
      workload = 0;
      start = 0;
      end = 0;
    } else {
      start = (world_rank * workload + n % world_size);
      end = ((world_rank + 1) * workload + n % world_size);
    }
  }

  int *c_mat;
  if(workload) {
    c_mat = new int[workload * l]{0};
  }
  for(int i=start; i<end; ++i) {
    for(int j=0; j<m; ++j) {
      for(int k=0; k<l; ++k) {
        c_mat[(i-start) * l + k] += a_mat[i * m + j] * b_mat[j * l + k];
      }
    }
  }

  if(world_rank > 0 && !only_one_proc) {
    MPI_Send(c_mat, workload*l, MPI_INT, 0, 0, MPI_COMM_WORLD);
  } else if(world_rank == 0) {
    show_matrix(c_mat, workload, l);
    if(!only_one_proc) {
      for (int i = 1; i < world_size; ++i) {
        MPI_Recv(c_mat, n / world_size * l, MPI_INT, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        show_matrix(c_mat, n / world_size, l);
      }
    }
  }
}

void destruct_matrices(int *a_mat, int *b_mat) {
  delete [] a_mat;
  delete [] b_mat;
}
