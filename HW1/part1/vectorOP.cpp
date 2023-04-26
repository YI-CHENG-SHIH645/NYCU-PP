#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_int y;
  __pp_vec_float base, result;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float one_f = _pp_vset_float(1.f);
  __pp_vec_float upper_bound = _pp_vset_float(9.999999f);
  __pp_mask maskAll, maskExpZero, maskStillNeedCal;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones, it doesn't add lanes
    maskAll = _pp_init_ones((N-i < VECTOR_WIDTH) ? N-i : VECTOR_WIDTH);

    // All zeros, it doesn't add lanes
    maskExpZero = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    // Utilized Vector Lanes += cnt_bits(maskAll) --- 1/0
    // Total Vector Lanes += VECTOR_WIDTH --- 1
    _pp_vload_int(y, exponents + i, maskAll); // y = values[i];

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 2/0
    // Total Vector Lanes += VECTOR_WIDTH ---2
    _pp_veq_int(maskExpZero, y, zero, maskAll); // zero power mask

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 3/0
    // Total Vector Lanes += VECTOR_WIDTH --- 3
    _pp_vmove_float(result, one_f, maskExpZero); // zero power mask set to 1.f

    // Utilized Vector Lanes += VECTOR_WIDTH --- 3/1
    // Total Vector Lanes += VECTOR_WIDTH --- 4
    maskStillNeedCal = _pp_mask_not(maskExpZero);

    // Utilized Vector Lanes += VECTOR_WIDTH --- 3/2
    // Total Vector Lanes += VECTOR_WIDTH --- 5
    maskStillNeedCal = _pp_mask_and(maskStillNeedCal, maskAll); // mask for those still need calculation

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 4/2
    // Total Vector Lanes += VECTOR_WIDTH --- 6
    _pp_vload_float(base, values + i, maskStillNeedCal);

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 5/2
    // Total Vector Lanes += VECTOR_WIDTH --- 7
    _pp_vload_float(result, values + i, maskStillNeedCal); // set base value

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 6/2
    // Total Vector Lanes += VECTOR_WIDTH --- 8
    _pp_vsub_int(y, y, one, maskStillNeedCal); // power-1

    // Utilized Vector Lanes += cnt_bits(maskAll) --- 7/2
    // Total Vector Lanes += VECTOR_WIDTH --- 9
    _pp_vgt_int(maskStillNeedCal, y, zero, maskStillNeedCal); // reset mask

    // ******** The above instructions won't affect the Vector Utilization
    //          no matter what the VECTOR_WIDTH is given s=10000 (or some other constant) ********


    // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    // Utilized Vector Lanes += VECTOR_WIDTH
    // Total Vector Lanes += VECTOR_WIDTH
    while(_pp_cntbits(maskStillNeedCal)) {
      // Utilized Vector Lanes += cnt_bits(maskAll)
      // Total Vector Lanes += VECTOR_WIDTH
      _pp_vmult_float(result, result, base, maskStillNeedCal); // current_result * base

      // Utilized Vector Lanes += cnt_bits(maskAll)
      // Total Vector Lanes += VECTOR_WIDTH
      _pp_vsub_int(y, y, one, maskStillNeedCal); // power-1

      // Utilized Vector Lanes += cnt_bits(maskAll)
      // Total Vector Lanes += VECTOR_WIDTH ---
      _pp_vgt_int(maskStillNeedCal, y, zero, maskStillNeedCal); // reset mask
//      _pp_vlt_float(maskStillNeedCal, result, upper_bound, maskStillNeedCal); // mask0 those result already exceeds
    }

    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    maskStillNeedCal = _pp_mask_not(maskExpZero); // won't affect the Vector Utilization
    maskStillNeedCal = _pp_mask_and(maskStillNeedCal, maskAll); // won't affect the Vector Utilization

    // if the data generated is fixed, then the following instructions won't affect Vector Utilization
    _pp_vgt_float(maskStillNeedCal, result, upper_bound, maskStillNeedCal);
    _pp_vmove_float(result, upper_bound, maskStillNeedCal);

    // Write results back to memory, won't affect the Vector Utilization
    _pp_vstore_float(output + i, result, maskAll);

  }

}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float data_slice;
  __pp_vec_float cur_sum = _pp_vset_float(0.f);
  __pp_mask maskAll = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(data_slice, values + i, maskAll);
    _pp_vadd_float(cur_sum, cur_sum, data_slice, maskAll);
  }
  for (int j = 1; j < VECTOR_WIDTH; j *= 2)
  {
    _pp_hadd_float(cur_sum, cur_sum);
    _pp_interleave_float(data_slice, cur_sum);
    cur_sum = data_slice;
  }

  return cur_sum.value[0];
}
