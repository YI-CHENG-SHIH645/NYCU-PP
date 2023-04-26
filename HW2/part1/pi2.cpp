#include "FastRand.h"
#include "pcg_random.hpp"
#include <iostream>
#include <pthread.h>
#include <random>

typedef long long int ll;

typedef struct {
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
  thread_local fastrand fr;
  thread_local uint32_t prngSeed[8];
  thread_local uint16_t *sptr = (uint16_t *) prngSeed;
  thread_local pcg_extras::seed_seq_from<std::random_device> seed_source;
  thread_local pcg32_fast rng(seed_source);
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
  thread_local double *res = new double[num];
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
  ll start = workload * thread_id;
  if ((thread_id == num_threads - 1) && (number_of_tosses % num_threads)) {
    workload += (number_of_tosses % num_threads);
  }
  ll end = start + workload;
  double *random_double = gen_rand(workload*2);

  double *in_circle_arr = new double[2];
  double bound[2] = { 1., 1. };
  double dis[2];
  //  double x1, y1, x2, y2;

  __m128d x, y, distance_squared, in_circle_vec, bound_vec = _mm_load_pd(bound);
  for (ll i = 0; i < end-start; i+=2) {
    x = _mm_load_pd(random_double+i);
    //    x1 = random_double[i]; x2 = random_double[i+1];
    y = _mm_load_pd(random_double+i+workload);
    //    y1 = random_double[i+workload]; y2 = random_double[i+workload+1];

    distance_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));
    in_circle_vec = _mm_cmple_pd(distance_squared, bound_vec);
    _mm_store_pd(in_circle_arr, in_circle_vec);
    //    _mm_store_pd(dis, distance_squared);
    if(in_circle_arr[0])
      ++work_info->number_in_circle;
    if(in_circle_arr[1])
      ++work_info->number_in_circle;
  }
  //  std::cout << "x: " << x1 << " " << x2 << std::endl;
  //  std::cout << "y: " << y1 << " " << y2 << std::endl;
  //  std::cout << "dis: " << dis[0] << " " << dis[1] << std::endl;
  //  std::cout << work_info->number_in_circle << " ---" << std::endl;
  //  std::cout << in_circle_arr[0] << std::endl;

  return NULL;
}
