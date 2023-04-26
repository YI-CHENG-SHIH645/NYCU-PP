#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 400
// assert z*z <= 1024
// valid common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25
#define BLOCK_SIZE 16

__global__ void mandelKernel(float lowerX, float lowerY, float stepX,
                             float stepY, size_t pitch, int *img_d,
                             int maxIterations, int resX, int resY) {
  // To avoid error caused by the floating number, use the following pseudocode
  //
  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
  int thisY = blockIdx.y * blockDim.y + threadIdx.y;
  float x = lowerX + thisX * stepX;
  float y = lowerY + thisY * stepY;

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
  // doc here :
  // http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html#g80d689bc903792f906e49be4a0b6d8db
  *((int *)((char *)img_d + thisY * pitch) + thisX) = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  // 1, 1, -2, -1, 1200x1600, 1600, 1200, 256
  float stepX = (upperX - lowerX) / resX; // 3./1600
  float stepY = (upperY - lowerY) / resY; // 2./1200
  size_t byte_size = resX * resY * sizeof(int);


  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  size_t byte_per_row = resX * sizeof(int);
  size_t pitch;

  int *img_h, *img_d;
  cudaHostAlloc((void **)&img_h, byte_size, cudaHostAllocDefault);
  cudaMallocPitch((void **)&img_d, &pitch, byte_per_row, (size_t)resY);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlock(resX / threadsPerBlock.x,
                resY / threadsPerBlock.y); // 1600/16, 1200/16

  mandelKernel<<<numBlock, threadsPerBlock>>>(
      lowerX, lowerY, stepX, stepY, pitch, img_d, maxIterations, resX, resY);


  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  cudaMemcpy2D(img_h, byte_per_row, img_d, pitch, byte_per_row, resY,
               cudaMemcpyDeviceToHost);
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


  memcpy(img, img_h, byte_size);

  cudaFreeHost(img_h);
  cudaFree(img_d);
}
