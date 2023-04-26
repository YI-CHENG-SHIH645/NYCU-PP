#include "FastRand.h"
#include "pcg_random.hpp"
#include <iostream>
#include <pthread.h>
#include <random>
#include <emmintrin.h>
#include <stdint.h>

typedef long long int ll;

typedef struct alignas(64) WorkInfo {
  int num_threads;
  int thread_id;
  ll number_of_tosses;
  ll number_in_circle;
} WorkInfo;

void *cal_nic(void *ptr);

//double randDouble(float low, float high);

double *gen_rand(ll num);

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "pass in 2 arguments." << std::endl;
    exit(1);
  }
  int num_threads = atoi(argv[1]);
  ll number_of_tosses = atoll(argv[2]);
  ll number_in_circle = 0;
  double pi_estimate;

  if (num_threads > 1) {
    // Parallel
    pthread_t thread_pool[num_threads];
    WorkInfo work_info[num_threads];
    for (int i = 0; i < num_threads; ++i) {
      work_info[i] = {.num_threads = num_threads, .thread_id = i, .number_of_tosses = number_of_tosses};
      /* Create independent threads each of which will execute function */
      pthread_create(&thread_pool[i], NULL, cal_nic, (void *) &(work_info[i]));
    }

    for (int i = 0; i < num_threads; ++i) {
      /* Wait till threads are complete before main continues. */
      pthread_join(thread_pool[i], NULL);
    }

    for (int i = 0; i < num_threads; ++i) {
      number_in_circle += work_info[i].number_in_circle;
    }
  } else {
    // Serial
    double x, y, distance_squared;
    double *random_double = gen_rand(number_of_tosses*2);
    for (ll toss = 0; toss < number_of_tosses; toss++) {
      x = random_double[toss];
      y = random_double[toss+number_of_tosses];
      distance_squared = x * x + y * y;
      if (distance_squared <= 1)
        number_in_circle++;
    }
  }

  pi_estimate = 4 * number_in_circle / ((double) number_of_tosses);
  std::cout << pi_estimate << std::endl;

  return 0;
}

//double randDouble(float low, float high) {
//  thread_local pcg_extras::seed_seq_from<std::random_device> seed_source;
//  thread_local pcg32_fast rng(seed_source);
//  thread_local std::uniform_real_distribution<double> urd;
//  return urd(rng, decltype(urd)::param_type{low, high});
//}

double *gen_rand(ll num) {
  fastrand fr;
  uint32_t prngSeed[8];
  uint16_t *sptr = (uint16_t *) prngSeed;
  pcg_extras::seed_seq_from<std::random_device> seed_source;
  pcg32_fast rng(seed_source);
  for (uint8_t i = 0; i < 8; i++) {
    prngSeed[i] = rng();
  }
  InitFastRand(sptr[0], sptr[1],
               sptr[2], sptr[3],
               sptr[4], sptr[5],
               sptr[6], sptr[7],
               sptr[8], sptr[9],
               sptr[10], sptr[11],
               sptr[12], sptr[13],
               sptr[14], sptr[15],
               &fr);
  double *res = new double[num];
  uint8_t k = 4;
  for (ll i = 0; i < (ll) ceil(num / 4.0); i++) {
    if (i + 1 >= (ll) ceil(num / 4.0))
      k = (uint8_t) (num % 4);
    FastRand_SSE(&fr);
    for (uint8_t j = 0; j < k; j++) {
      res[4 * i + j] = ((double) fr.res[j] / 4294967295.0) * 2 - 1;
    }
  }

  return res;
}

void *cal_nic(void *ptr) {
  WorkInfo *work_info = (WorkInfo *) ptr;
  int num_threads = work_info->num_threads;
  int thread_id = work_info->thread_id;
  ll number_of_tosses = work_info->number_of_tosses;

  ll workload = number_of_tosses / num_threads;
  if ((thread_id == num_threads - 1) && (number_of_tosses % num_threads)) {
    workload += (number_of_tosses % num_threads);
  }
  double *random_double = gen_rand(workload*2);

  double x, y, distance_squared;
  for (ll i = 0; i < workload; ++i) {
    x = random_double[i];
    y = random_double[i + workload];
    distance_squared = x * x + y * y;
    if (distance_squared <= 1) {
      ++work_info->number_in_circle;
    }
  }

  return NULL;
}
