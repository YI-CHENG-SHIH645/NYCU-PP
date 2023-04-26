/*****************************************************************************
	
	Super-Fast MWC1616 Pseudo-Random Number Generator 
	for Intel/AMD Processors (using SSE or SSE4 instruction set)
	Copyright (c) 2012, Ivan Dimkovic
	All rights reserved.

	Redistribution and use in source and binary forms, with or without 
	modification, are permitted provided that the following conditions are met:

	Redistributions of source code must retain the above copyright notice, 
	this list of conditions and the following disclaimer.
	
	Redistributions in binary form must reproduce the above copyright notice, 
	this list of conditions and the following disclaimer in the documentation 
	and/or other materials provided with the distribution.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
	THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
	PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
	OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "FastRand.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  fastrand fr;

  uint32_t prngSeed[8];
  uint16_t *sptr = (uint16_t *) prngSeed;

  //
  // Randomize the seed values

  for (uint8_t i = 0; i < 8; i++) {
    prngSeed[i] = rand();
  }

  //
  // Initialize the PRNG

  InitFastRand(sptr[0], sptr[1],
               sptr[2], sptr[3],
               sptr[4], sptr[5],
               sptr[6], sptr[7],
               sptr[8], sptr[9],
               sptr[10], sptr[11],
               sptr[12], sptr[13],
               sptr[14], sptr[15],
               &fr);

  double res[4];
  for (uint32_t i = 0; i < 25000000; i++) {
    FastRand_SSE(&fr);

    for (uint8_t j = 0; j < 4; j++) {
      res[j] = ((double) fr.res[j] / 4294967295.0) * 2 - 1;
    }
  }
  printf("%f\n", res[3]);
  return 0;
}
