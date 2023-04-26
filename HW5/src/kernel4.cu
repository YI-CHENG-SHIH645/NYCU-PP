//#include <cuda.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
//
//// common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 400
//// assert z*z <= 1024
//// valid common factors z: 1, 2, 4, 5, 8, 10, 16, 20, 25
//#define BLOCK_SIZE 8
//
//__global__ void mandelKernel(float lowerX, float lowerY, float stepX,
//                             float stepY, int *img_d, int maxIterations,
//                             int resX, int resY, int m, int n) {
//  // To avoid error caused by the floating number, use the following pseudocode
//  //
//  int thisX = blockIdx.x * blockDim.x + threadIdx.x;
//  int thisY = blockIdx.y * blockDim.y + threadIdx.y + m * 1200/n;
//  if (thisX >= resX || thisY >= resY)
//    return;
//  int index = thisY * resX + thisX;
//  float x = lowerX + thisX * stepX;
//  float y = lowerY + thisY * stepY;
//
//  float z_re = x, z_im = y;
//  int i;
//  for (i = 0; i < maxIterations; ++i) {
//    if (z_re * z_re + z_im * z_im > 4.f)
//      break;
//    float new_re = z_re * z_re - z_im * z_im;
//    float new_im = 2.f * z_re * z_im;
//    z_re = x + new_re;
//    z_im = y + new_im;
//  }
//  img_d[index] = i;
//}
//
//// Host front-end function that allocates the memory and launches the GPU kernel
//void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
//            int resX, int resY, int maxIterations) {
//  // 1, 1, -2, -1, 1200x1600, 1600, 1200, 256
//  float stepX = (upperX - lowerX) / resX; // 3./1600
//  float stepY = (upperY - lowerY) / resY; // 2./1200
//  size_t byte_size = resX * resY * sizeof(int);
//
//  // allocation
//  int *img_h, *img_d;
//  cudaHostAlloc((void **)&img_h, byte_size, cudaHostAllocDefault);
//  cudaMalloc((void **)&img_d, byte_size);
//
//  int nStreams = 2;
//  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
//  dim3 numBlock(resX / threadsPerBlock.x, // 200
//                resY / threadsPerBlock.y / nStreams); // 150 / nStreams
//
//  cudaStream_t streams[nStreams];
//  for (int i = 0; i < nStreams; ++i) { cudaStreamCreate(&streams[i]); }
//
//  size_t size = resX * resY / nStreams;
//  size_t bs = byte_size / nStreams;
//  for (int i = 0; i < nStreams; i++) {
//    size_t offset = i * size;
//    mandelKernel<<<numBlock, threadsPerBlock, 0, streams[i]>>>(lowerX, lowerY, stepX, stepY,
//                                                               img_d, maxIterations,
//                                                               resX, resY, i, nStreams);
//    cudaMemcpyAsync(img_h+offset, img_d+offset, bs, cudaMemcpyDeviceToHost, streams[i]);
//  }
//
//  for (int i = 0; i < nStreams; ++i) {
//    cudaStreamSynchronize(streams[i]);
//    cudaStreamDestroy(streams[i]);
//  }
//
//  memcpy(img, img_h, byte_size);
//
//  cudaFreeHost(img_h);
//  cudaFree(img_d);
//}

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

  int index = thisY * resX + thisX;
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
  img_d[index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img,
            int resX, int resY, int maxIterations) {
  // 1, 1, -2, -1, 1200x1600, 1600, 1200, 256
  float stepX = (upperX - lowerX) / resX; // 3./1600
  float stepY = (upperY - lowerY) / resY; // 2./1200
  size_t byte_size = resX * resY * sizeof(int);

  // allocation
  int *img_d;
  cudaMallocManaged((void**)&img_d, byte_size);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlock(resX / threadsPerBlock.x,
                resY / threadsPerBlock.y); // 1600/16, 1200/16

  // kernel computation
  mandelKernel<<<numBlock, threadsPerBlock>>>(lowerX, lowerY, stepX, stepY,
                                              img_d, maxIterations, resX, resY);

  // host <- device
  cudaMemcpy(img, img_d, byte_size, cudaMemcpyDeviceToHost);

  cudaFree(img_d);
}
