#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // TODO: MPI init
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  long long int workload = tosses / world_size;
  long long int number_in_circle = 0;
  double x, y, distance_squared;
  unsigned int seed = time(NULL);
  seed *= (world_rank + 1);
  srand(seed);

  // TODO: binary tree reduction
  for (long long int i = 0; i < workload; ++i) {
    x = (double)rand_r(&seed) / RAND_MAX;
    y = (double)rand_r(&seed) / RAND_MAX;
    distance_squared = x * x + y * y;
    if (distance_squared <= 1) {
      ++number_in_circle;
    }
  }

  for (int i = 2; i <= world_size; i *= 2) {
    if ((world_rank % i) == i / 2) {
      MPI_Send(&number_in_circle, 1, MPI_LONG_LONG, world_rank - (i / 2), 0,
               MPI_COMM_WORLD);
    }
    if (world_rank % i == 0) {
      long long int count;
      MPI_Recv(&count, 1, MPI_LONG_LONG, world_rank + (i / 2), 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      number_in_circle += count;
    }
  }

  if (world_rank == 0) {
    // TODO: PI result
    pi_result = 4 * number_in_circle / ((double)tosses);
    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---

#ifdef STORE_TIME
    char *time_info;
    FILE *fp = fopen("elapsed.csv", "a");
    asprintf(&time_info, "pi_block_tree_proc%d,%lf\n", world_size,
             end_time - start_time);
    fputs(time_info, fp);
    free(time_info);
    fclose(fp);
#endif
  }

  MPI_Finalize();
  return 0;
}
