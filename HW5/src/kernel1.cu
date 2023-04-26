#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 400
// assert z*z <= 1024
// valid common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25
#define BLOCK_SIZE 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX,
                             float stepY, int *img_d, int maxIterations,
                             int resX, int resY) {
  // To avoid error caused by the floating number, use the following pseudocode
  //
  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;
  float x = lowerX + thisX * stepX;
  float y = lowerY + thisY * stepY;
  int index = thisY * resX + thisX;

  float z_re = x, z_im = y;
  int i;
  for (i = 0; i < maxIterations; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;
    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = x + new_re;
    z_im = y + new_im;
  }
  img_d[index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  // 1, 1, -2, -1, 1200x1600, 1600, 1200, 256
  float stepX = (upperX - lowerX) / resX; // 3./1600
  float stepY = (upperY - lowerY) / resY; // 2./1200
  size_t byte_size = resX * resY * sizeof(int);


  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  int *img_h, *img_d;
  img_h = (int *)malloc(byte_size);
  cudaMalloc((void **)&img_d, byte_size);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlock(resX / threadsPerBlock.x,
                resY / threadsPerBlock.y); // 1600/16, 1200/16

  mandelKernel<<<numBlock, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY,
                                              img_d, maxIterations, resX, resY);


  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  cudaMemcpy(img_h, img_d, byte_size, cudaMemcpyDeviceToHost);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


  memcpy(img, img_h, byte_size);

  free(img_h);
  cudaFree(img_d);
}
